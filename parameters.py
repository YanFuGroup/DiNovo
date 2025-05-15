import numpy as np
import scipy.stats
# import matplotlib.pyplot as plt
import math

aa_to_mass_dict={
    'A':71.037114,
    'R':156.101111,
    'N':114.042927,
    #'N(+.98)': 115.02695,#Deamidation
    'D':115.026943,
    'C(+57.02)': 160.03065,#Carbamidomethylation  C 103.009185
    'E':129.042593,
    'Q':128.058578,
    #'Q(+.98)': 129.0426,#Deamidation
    'G':57.021464,
    'H':137.058912,
    #'I':113.084064,
    'L':113.084064,
    'K':128.094963,
    'M':131.040485,
    'M(+15.99)': 147.0354,#Oxidation
    'F':147.068414,
    'P':97.052764,
    'S':87.032028,
    'T':101.047679,
    'W':186.079313,
    'Y':163.063329,
    'V':99.068414
}
aa_to_mass_sorted_dict = dict(sorted(aa_to_mass_dict.items(), key=lambda x: x[1]))
# print(aa_to_mass_sorted_dict)
'''
[71.037114,156.101111,114.042927,115.02695,115.026943,160.03065,129.042593,128.058578,
129.0426,57.021464,137.058912,113.084064,113.084064,128.094963,131.040485,147.0354,147.068414,97.052764,
87.032028,101.047679,186.079313,163.063329,99.068414]
'''

atom_mass={
    'H': 1.0078,
    'PROTON':1.00727646677,
    'H2O':18.010565,
    'NH3':17.02655,
    'CO':27.99492,
    'CHO':29.00272,
    'Isotope' : 1.00335
}

Iminium_mass_np = np.array(list(aa_to_mass_dict.values())) - atom_mass['CO'] + atom_mass['PROTON']
# print('Iminium_mass_np',Iminium_mass_np)
Iminium_ion_tolerance = 0.01

modifications = ['Carbamidomethyl[C]','Oxidation[M]']
modifications_in_aa_dict_keys = ['C(+57.02)' , 'M(+15.99)' ]

aa_list = list(aa_to_mass_dict.keys())
mass_list = list(aa_to_mass_dict.values())
vocab_size = len(mass_list)
mass_AA_min = aa_to_mass_dict["G"] # 57.02146
mass_AA_min_round = int(round(mass_AA_min * 1000)) # 57.02146

aa_to_index_dict = dict([(aa,i) for i,aa in enumerate(aa_list)])

ppm_or_dalton = 1

#'''
peaks_delta_mass_error = 0.015 #the tolerance of -K/K/R/-R
peaks_sum_mass_error = peaks_delta_mass_error
filter_confidence = 2 #peaks with confidence larger than 2 will retain

isotope_tolerance = 0.015
peptide_tolerance = 0.015
fragment_tolerance = 0.015
validation_fragment_tolerance = 0.015
build_aa_permutation_tolerance = 0.01
build_aa_permutation_max_mass = 500
build_aa_permutation_max_length = 10
build_aa_tag_max_mass = 1000
knapsack_MZ_MAX = 3000
mass_resolution = 1000
lower_mass_resolution = 100
peaks_merge_tolerance = 0.015
filter_threshold = 1000
'''
peaks_delta_mass_error = 0.015 #the tolerance of -K/K/R/-R
peaks_sum_mass_error = peaks_delta_mass_error
filter_confidence = 2 #peaks with confidence larger than 2 will retain

fragment_tolerance = 20
validation_fragment_tolerance = 20
build_aa_permutation_tolerance = 0.01
build_aa_permutation_max_mass = 500
build_aa_permutation_max_length = 8
build_aa_tag_max_mass = 1000
knapsack_MZ_MAX = 3000
mass_resolution = 1000
lower_mass_resolution = 100
peaks_merge_tolerance = 0.015
filter_threshold = 1000
'''


#output parameters
topk_maxscore_subseq = 5
topk_max_score_subseq_on_Nterm = 10
topk_longest_initial_path = 2
topk_maxscore_initial_path = 2
breakpoint_topk_maxscore_initial_path = 1
output_topk = 20
validation_topk_output = 20

extract_factor = 0.6
extract_top_N = lambda x: x if x<= 150 else int(round(extract_factor*x))

#score function
ion_type_2charge_dict = {
    'y+':True,
    'y2+':False,
    'y_H2O':True,
    'y_NH3':True,
    'b+':True,
    'b2+':False,
    'b_H2O':True,
    'b_NH3':True,
    'a+':True,
    'a2+':False,
    'a_H2O':False,
    'a_NH3':False,
}
ion_type_3charge_dict = ion_type_2charge_dict.copy()
ion_type_3charge_dict['y2+'] = True
ion_type_3charge_dict['b2+'] = True

Dinovo_match_type_transfer = {
    'A1:K-K': 'A_KK',
    'A2:R-R': 'A_RR',
    'A3:X-X': 'A',
    'B: R-K': 'B',
    'C: K-R': 'C',
    'D: K-X': 'D',
    'E: R-X': 'E',
    'F: X-K': 'F',
    'G: X-R': 'G',
}

def ion_type_duplicate_function(charge):
    if charge <= 2:
        ion_type_duplicate_dict = ion_type_2charge_dict.copy()
        #ion_type_duplicate_dict['a+'] = True
        ion_type_duplicate_dict['b_H2O'] = True
        ion_type_duplicate_dict['b_NH3'] = True
        ion_type_duplicate_dict['y_H2O'] = True
        ion_type_duplicate_dict['y_NH3'] = True
    else:
        ion_type_duplicate_dict = ion_type_3charge_dict.copy()
        #ion_type_duplicate_dict['a+'] = True
        ion_type_duplicate_dict['b_H2O'] = True
        ion_type_duplicate_dict['b_NH3'] = True
        ion_type_duplicate_dict['y_H2O'] = True
        ion_type_duplicate_dict['y_NH3'] = True

    return ion_type_duplicate_dict

score_scale_factor = 1.25

exp_pdf_parameter = 7
intensity_pdf = lambda x : scipy.stats.expon.pdf( x , scale = 1/exp_pdf_parameter)

boundary_of_narrow_and_middle = 0.0025
boundary_of_middle_and_wide = 0.01
narrow_norm_pdf_parameter = 0.01
narrow_guass_pdf = scipy.stats.norm(0, narrow_norm_pdf_parameter)
def delta_mass_pdf(x):
    if x > boundary_of_middle_and_wide:
        return  math.sqrt(narrow_guass_pdf.pdf(boundary_of_middle_and_wide))
    elif x > boundary_of_narrow_and_middle:
        return math.sqrt(narrow_guass_pdf.pdf(x))
    else:
        return math.sqrt(narrow_guass_pdf.pdf(boundary_of_narrow_and_middle))


x = list(np.arange(0,0.02,0.0001))
y = [delta_mass_pdf(item) for item in x]

'''
#画图代码
plt.figure()
plt.plot(x,y)
plt.show()
'''