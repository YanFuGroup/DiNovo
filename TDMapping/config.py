
topk_peaks = 200
fasta_file = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\fasta\yeast\uniprot-download_true_format_fasta_query__28yeast_29_20AND_20_28mode-2023.02.27-07.17.47.02_con.fasta"

# 计算离子覆盖率时用到的参数
if_use_ppm = True
fragment_tolerance_ppm = 20
fragment_tolerance_da = 0.02

modifications = ['Carbamidomethyl[C]', 'Oxidation[M]']
modifications_in_aa_dict_keys = ['C(+57.02)', 'M(+15.99)']
pure_aa_list = ['C', 'M']
modifications_other = ["Acetyl[ProteinN-term]"]
modifications_other_in_aa_dict_keys = ["B(+42.01)"]
modifications_other_mass = [42.010565]

modifications_DiNovo = ['Carbamidomethyl[C]', 'Oxidation[M]']
modifications_MirrorNovo = ['Carbamidomethylation[C]', 'Oxidation[M]']

aa_to_mass_dict = {
    'A': 71.037114,
    'R': 156.101111,
    'N': 114.042927,
    # 'N(+.98)': 115.02695,  # Deamidation
    'D': 115.026943,
    'C(+57.02)': 160.03065,  # Carbamidomethylation
    'E': 129.042593,
    'Q': 128.058578,
    # 'Q(+.98)': 129.0426,  # Deamidation
    'G': 57.021464,
    'H': 137.058912,
    'I': 113.084064,
    'L': 113.084064,
    'K': 128.094963,
    'M': 131.040485,
    'M(+15.99)': 147.0354,  # Oxidation
    'F': 147.068414,
    'P': 97.052764,
    'S': 87.032028,
    'T': 101.047679,
    'W': 186.079313,
    'Y': 163.063329,
    'V': 99.068414
}

atom_mass = {
    'H': 1.0078,
    'PROTON': 1.00727646677,
    'H2O': 18.010565,
    'NH3': 17.02655,
    'CO': 27.99492,
    'CHO': 29.00272,
    'Isotope': 1.00335
}

