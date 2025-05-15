import heapq
import parameters
import math
import numpy as np
import os


#########################################################################################################################################
#the functions of building aa permutations

def updata_dict(dict,mass, aa_list):
    mass_round = int(round(mass * parameters.mass_resolution))
    if mass_round in dict.keys():
        dict[mass_round].append(''.join(aa_list))
    else:
        dict[mass_round] = [''.join(aa_list)]

def write_to_file(dict , path):
    mass_round_list = list(dict.keys())
    aa_permutations_list = list(dict.values())

    shift_number = int(round(parameters.build_aa_permutation_tolerance * parameters.mass_resolution))
    mass_round_to_index_list = [-1] * (parameters.build_aa_permutation_max_mass * parameters.mass_resolution + shift_number)
    mass_round_to_deltamass_list = [1] * (parameters.build_aa_permutation_max_mass * parameters.mass_resolution + shift_number)
    multi_aa_permutations_mass = []
    unit_mass = float(1/parameters.mass_resolution)
    for i,mass_round in enumerate(mass_round_list):
        for j in range(shift_number + 1):
            if j == 0:
                if mass_round_to_index_list[mass_round] == -1:
                    mass_round_to_index_list[mass_round] = i
                    mass_round_to_deltamass_list[mass_round] = 0
                else:
                    multi_aa_permutations_mass.append(mass_round)
                    if type(mass_round_to_index_list[mass_round]) == list:
                        mass_round_to_index_list[mass_round].append(i)
                        mass_round_to_deltamass_list[mass_round] = 0
                    else:
                        mass_round_to_index_list[mass_round] = [mass_round_to_index_list[mass_round], i]
                        mass_round_to_deltamass_list[mass_round] = 0
            else:
                if mass_round_to_index_list[mass_round - j] == -1 :
                    mass_round_to_index_list[mass_round - j] = i
                    mass_round_to_deltamass_list[mass_round - j] = j * unit_mass
                else:
                    multi_aa_permutations_mass.append(mass_round - j)
                    if type(mass_round_to_index_list[mass_round - j]) == list:
                        mass_round_to_index_list[mass_round - j].append(i)
                        mass_round_to_deltamass_list[mass_round - j] = min(j * unit_mass , mass_round_to_deltamass_list[mass_round - j])
                    else:
                        mass_round_to_index_list[mass_round - j] = [mass_round_to_index_list[mass_round - j] , i]
                        mass_round_to_deltamass_list[mass_round - j] = min(j * unit_mass , mass_round_to_deltamass_list[mass_round - j])

                if mass_round_to_index_list[mass_round + j] == -1:
                    mass_round_to_index_list[mass_round + j] = i
                    mass_round_to_deltamass_list[mass_round + j] = j * unit_mass
                else:
                    multi_aa_permutations_mass.append(mass_round + j)
                    if type(mass_round_to_index_list[mass_round + j]) == list:
                        mass_round_to_index_list[mass_round + j].append(i)
                        mass_round_to_deltamass_list[mass_round + j] = min(j * unit_mass , mass_round_to_deltamass_list[mass_round + j])
                    else:
                        mass_round_to_index_list[mass_round + j] = [mass_round_to_index_list[mass_round + j] , i]
                        mass_round_to_deltamass_list[mass_round + j] = min(j * unit_mass , mass_round_to_deltamass_list[mass_round + j])


    np.savez(path,
             aa_permutations = aa_permutations_list,
             mass_to_index = mass_round_to_index_list,
             mass_to_deltamass = mass_round_to_deltamass_list)

def read_from_file(filepath):
    data = np.load(filepath,allow_pickle=True)
    aa_permutations_list = data['aa_permutations']
    mass_round_to_index_list = data['mass_to_index']
    mass_round_to_deltamass_list = data['mass_to_deltamass']

    return aa_permutations_list,mass_round_to_index_list,mass_round_to_deltamass_list


def searchGraph_in_aa_permutations():  # 搜索树
    mass_to_aa_permutation_dict = {}

    graph_aa_list = list(parameters.aa_to_mass_sorted_dict.keys())
    graph_mass_list = list(parameters.aa_to_mass_sorted_dict.values())

    for i,start_aa in enumerate(graph_aa_list):

        aa_list = [start_aa]
        index_list = [i]
        start_mass = parameters.aa_to_mass_sorted_dict[start_aa]
        mass_list = [start_mass]

        updata_dict(mass_to_aa_permutation_dict,start_mass,start_aa)

        current_mass = start_mass

        generatePath_in_aa_permutations(graph_aa_list , graph_mass_list, aa_list, index_list, mass_list , current_mass , mass_to_aa_permutation_dict)  # 生成路径

    return mass_to_aa_permutation_dict

def generatePath_in_aa_permutations(graph_aa_list, graph_mass_list, aa_list, index_list, mass_list , current_mass , mass_to_aa_permutation_dict):  # 生成路径
    for index, aa in enumerate(graph_aa_list):
        new_aa_mass = graph_mass_list[index]
        if current_mass + new_aa_mass <= parameters.build_aa_permutation_max_mass:
            updata_dict(mass_to_aa_permutation_dict,current_mass+new_aa_mass,aa_list+[aa])
            generatePath_in_aa_permutations(graph_aa_list, graph_mass_list, aa_list+[aa], index_list+[index], mass_list+[new_aa_mass] , current_mass+new_aa_mass , mass_to_aa_permutation_dict)
        else:
            break

def build_aa_permutations_file_function():
    # aa_permutations_table
    current_path = os.getcwd()
    path_list = os.listdir(current_path)
    flag = False
    for filename in path_list:
        if filename == f"mass_and_aa_permutations_{parameters.build_aa_permutation_max_mass}Da_{parameters.build_aa_permutation_max_length}aa_{parameters.build_aa_permutation_tolerance}tolerance.npz":
            flag = True
            npz_path = os.path.join(current_path, filename)

    if flag == True:
        aa_permutations_list, \
        mass_round_to_index_list,\
        mass_round_to_deltamass_list = read_from_file(npz_path)
    else:
        save_path = os.path.join(current_path,
                                 f"mass_and_aa_permutations_{parameters.build_aa_permutation_max_mass}Da_{parameters.build_aa_permutation_max_length}aa_{parameters.build_aa_permutation_tolerance}tolerance.npz")
        mass_to_aa_permutation_dict = searchGraph_in_aa_permutations()
        write_to_file(mass_to_aa_permutation_dict, save_path)
        aa_permutations_list, \
        mass_round_to_index_list ,\
        mass_round_to_deltamass_list = read_from_file(save_path)

    return aa_permutations_list, mass_round_to_index_list, mass_round_to_deltamass_list

#########################################################################################################################################
#build pair peaks between trypsin and lys

def build_signal_index_function(try_mz_np , lys_mz_np ,delta_mass_target, sum_mass_target):

    try_signal_N_index_dict = {}
    try_signal_C_index_dict = {}
    for i, try_peaks_mz in enumerate(try_mz_np):
        single_peak_delta_mass = try_peaks_mz - lys_mz_np
        single_peak_sum_mass = try_peaks_mz + lys_mz_np

        single_peak_delta_mass_Temp = single_peak_delta_mass - delta_mass_target
        # 第一行代表与-K或-R的误差，第二行代表与K或R的误差
        single_peak_delta_mass_tag = (abs(single_peak_delta_mass_Temp) <= parameters.peaks_delta_mass_error).astype(np.int)
        single_peak_delta_mass_tag[0, :] *= -1
        single_peak_delta_mass_tag = single_peak_delta_mass_tag.sum(axis=0)

        single_peak_sum_mass_Temp = single_peak_sum_mass - sum_mass_target
        # 第一行代表与-K或-R的误差，第二行代表与K或R的误差
        single_peak_sum_mass_tag = (
                abs(single_peak_sum_mass_Temp) <= parameters.peaks_sum_mass_error).astype(
            np.int)
        single_peak_sum_mass_tag[0, :] *= -1
        single_peak_sum_mass_tag = single_peak_sum_mass_tag.sum(axis=0)

        for j, tag in enumerate(single_peak_delta_mass_tag):
            if tag == -1 or single_peak_sum_mass_tag[j] == -1:
                if not i in try_signal_N_index_dict.keys():
                    try_signal_N_index_dict[i] = [j]
                else:
                    try_signal_N_index_dict[i].append(j)
            elif tag == 1 or single_peak_sum_mass_tag[j] == 1:
                if not i in try_signal_C_index_dict.keys():
                    try_signal_C_index_dict[i] = [j]
                else:
                    try_signal_C_index_dict[i].append(j)

    return try_signal_N_index_dict , try_signal_C_index_dict

def get_target_value(try_pepmass , lys_pepmass , match_type):
    '''
    类型A还有点问题，以后再单独考虑
    :param try_pepmass: single_charge MH
    :param lys_pepmass: single_charge MH
    :param match_type:
    :return:
    '''
    if match_type == 'A1:K-K':
        average_pepmass = (try_pepmass + lys_pepmass) / 2.0
        delta_mass = np.array([[-parameters.aa_to_mass_dict['K']], [parameters.aa_to_mass_dict['K']]])
        sum_mass = np.array([[average_pepmass + parameters.atom_mass['PROTON'] - parameters.aa_to_mass_dict['K']],
                            [average_pepmass + parameters.atom_mass['PROTON'] + parameters.aa_to_mass_dict['K']]])

    elif match_type == 'A2:R-R':
        average_pepmass = (try_pepmass + lys_pepmass) / 2.0
        delta_mass = np.array([[-parameters.aa_to_mass_dict['R']], [parameters.aa_to_mass_dict['R']]])
        sum_mass = np.array([[average_pepmass + parameters.atom_mass['PROTON'] - parameters.aa_to_mass_dict['R']],
                                  [average_pepmass + parameters.atom_mass['PROTON'] + parameters.aa_to_mass_dict['R']]])

    elif match_type == 'B: R-K':
        delta_mass = np.array([[-parameters.aa_to_mass_dict['K']], [parameters.aa_to_mass_dict['R']]])
        sum_mass = np.array([[(try_pepmass - parameters.aa_to_mass_dict['R'] + lys_pepmass -
                                 parameters.aa_to_mass_dict['K']) / 2.0 + parameters.atom_mass['PROTON']],
                               [(try_pepmass + parameters.aa_to_mass_dict['K'] + lys_pepmass +
                                 parameters.aa_to_mass_dict['R']) / 2.0 + parameters.atom_mass['PROTON']]])

    elif match_type == 'C: K-R':
        delta_mass = np.array([[-parameters.aa_to_mass_dict['R']], [parameters.aa_to_mass_dict['K']]])
        sum_mass = np.array([[(try_pepmass - parameters.aa_to_mass_dict['K'] + lys_pepmass -
                                 parameters.aa_to_mass_dict['R']) / 2.0 + parameters.atom_mass['PROTON']],
                               [(try_pepmass + parameters.aa_to_mass_dict['R'] + lys_pepmass +
                                 parameters.aa_to_mass_dict['K']) / 2.0 + parameters.atom_mass['PROTON']]])

    elif match_type == 'D: K-X':
        delta_mass = np.array([[0], [parameters.aa_to_mass_dict['K']]])
        sum_mass = np.array(
            [[(try_pepmass - parameters.aa_to_mass_dict['K'] + lys_pepmass) / 2.0 + parameters.atom_mass['PROTON']],
             [(try_pepmass + lys_pepmass + parameters.aa_to_mass_dict['K']) / 2.0 + parameters.atom_mass['PROTON']]])

    elif match_type == 'E: R-X':
        delta_mass = np.array([[0], [parameters.aa_to_mass_dict['R']]])
        sum_mass = np.array(
            [[(try_pepmass - parameters.aa_to_mass_dict['R'] + lys_pepmass) / 2.0 + parameters.atom_mass['PROTON']],
             [(try_pepmass + lys_pepmass + parameters.aa_to_mass_dict['R']) / 2.0 + parameters.atom_mass['PROTON']]])

    elif match_type == 'F: X-K':
        delta_mass = np.array([[-parameters.aa_to_mass_dict['K']], [0]])
        sum_mass = np.array(
            [[(try_pepmass + lys_pepmass - parameters.aa_to_mass_dict['K']) / 2.0 + parameters.atom_mass['PROTON']],
             [(try_pepmass + parameters.aa_to_mass_dict['K'] + lys_pepmass) / 2.0 + parameters.atom_mass['PROTON']]])

    elif match_type == 'G: X-R':
        delta_mass = np.array([[-parameters.aa_to_mass_dict['R']], [0]])
        sum_mass = np.array(
            [[(try_pepmass + lys_pepmass - parameters.aa_to_mass_dict['R']) / 2.0 + parameters.atom_mass['PROTON']],
             [(try_pepmass + parameters.aa_to_mass_dict['R'] + lys_pepmass) / 2.0 + parameters.atom_mass['PROTON']]])
    else:
        assert False , " unexpected match type !"

    return delta_mass , sum_mass

#########################################################################################################################################3
#de novo
def generatePath(try_signal_C_mass_np , try_signal_C_intensity_np , adjacen_matrix_list, adjacen_edge_score_matrix_list , mass_list , intensity_list , index_list , current_index , edge_score_list , paths_mass_list , paths_intensity_list , paths_index_list , paths_edge_score_list):  # 生成路径
    if mass_list[-1] == try_signal_C_mass_np[-1]:
        paths_mass_list.append(mass_list)
        paths_intensity_list.append(intensity_list)
        paths_index_list.append(index_list)
        paths_edge_score_list.append(edge_score_list)
    else:
        for next_index, next_mass in enumerate(try_signal_C_mass_np[current_index+1:]):
            if next_mass - mass_list[-1] > parameters.build_aa_permutation_max_mass:
                break
            if adjacen_matrix_list[current_index][current_index + next_index + 1] != [-1]:
                generatePath(try_signal_C_mass_np , try_signal_C_intensity_np , adjacen_matrix_list, adjacen_edge_score_matrix_list , mass_list + [next_mass] , intensity_list + [try_signal_C_intensity_np[current_index + next_index + 1]] , index_list + [current_index + next_index + 1], current_index + next_index + 1 , edge_score_list + [adjacen_edge_score_matrix_list[current_index][current_index + next_index + 1]] , paths_mass_list , paths_intensity_list , paths_index_list , paths_edge_score_list)

def searchGraph(try_signal_C_mass_np , try_signal_C_intensity_np , adjacen_matrix_list , adjacen_edge_score_matrix_list):  # 搜索树

    assert len(try_signal_C_mass_np) == len(try_signal_C_intensity_np)
    paths_mass_list = []
    paths_intensity_list = []
    paths_index_list = []
    paths_edge_score_list = []
    mass_list = [try_signal_C_mass_np[0]]
    intensity_list = [try_signal_C_intensity_np[0]]
    index_list = [0]
    edge_score_list = []

    generatePath(try_signal_C_mass_np , try_signal_C_intensity_np , adjacen_matrix_list , adjacen_edge_score_matrix_list , mass_list , intensity_list , index_list ,0 , edge_score_list , paths_mass_list , paths_intensity_list , paths_index_list , paths_edge_score_list)

    return paths_mass_list , paths_intensity_list ,paths_index_list , paths_edge_score_list

def search_result_path(candidate_path_list , candidate_sumscore_list , candidate_aa_score_list):
    assert len(candidate_path_list) == len(candidate_sumscore_list) == len(candidate_aa_score_list)
    current_index = -1
    path_list = []
    sumscore = 0
    aa_score_list = []
    result_path = []
    result_sumscore = []
    result_aa_score = []

    generate_result_path( candidate_path_list , candidate_sumscore_list , candidate_aa_score_list , current_index  , path_list , sumscore , aa_score_list , result_path ,result_sumscore ,result_aa_score)

    return result_path , result_sumscore ,result_aa_score

def generate_result_path( candidate_path_list , candidate_sumscore_list , candidate_aa_score_list, current_index  , path_list , sumscore ,  aa_score_list ,result_path ,result_sumscore, result_aa_score):  # 生成路径
    if current_index == len(candidate_path_list)-1 and path_list[-1] in candidate_path_list[-1]:
        result_path.append(path_list)
        result_sumscore.append(sumscore)
        result_aa_score.append(aa_score_list)
    else:
        for i,next_subseq in enumerate(candidate_path_list[current_index + 1]):
            generate_result_path( candidate_path_list , candidate_sumscore_list , candidate_aa_score_list, current_index + 1 , path_list + [next_subseq] , sumscore+ candidate_sumscore_list[current_index+1][i] , aa_score_list + candidate_aa_score_list[current_index+1][i] , result_path ,result_sumscore ,result_aa_score)

def de_novo_candidate_path_function( initial_mass_path ,  initial_index_path ,
                                     original_try_mz_np , original_lys_mz_np ,
                                     original_try_intensity_np ,original_lys_intensity_np ,
                                     try_pepmass ,
                                     adjacen_matrix_list, match_type,
                                     aa_permutations_list , try_charge , lys_charge,
                                     index_to_subseq_and_score_list ,seq_to_score_list, try_mz_to_score_dict, lys_mz_to_score_dict
                                     #try_mz_to_internal_score_dict , lys_mz_to_internal_score_dict
                                     ):
    if try_charge <= 2 :
        try_ion_type_dict = parameters.ion_type_2charge_dict.copy()
    else:
        try_ion_type_dict = parameters.ion_type_3charge_dict.copy()
    if lys_charge <= 2:
        lys_ion_type_dict = parameters.ion_type_2charge_dict.copy()
    else:
        lys_ion_type_dict = parameters.ion_type_3charge_dict.copy()

    candidate_path_list = []
    length = len(initial_index_path)
    for i,start_index in enumerate(initial_index_path):
        if i >= length - 1:
            break
        next_index = initial_index_path[i + 1]
        aa_permutations_index = adjacen_matrix_list[start_index][next_index]
        aa_permutations = []
        for aa_index in aa_permutations_index:
            if aa_index == -1:
                assert False , 'the initial path is not continuous.'
            else:
                aa_permutations = aa_permutations + aa_permutations_list[aa_index]
        candidate_path_list.append(aa_permutations)

    sumscore_list = []
    aa_score_list = []
    new_candidate_path_list = []
    for i,sub_seq_list in enumerate(candidate_path_list):
        if i >= length - 1:
            break
        if i == 0:
            start_or_end_tag = 0
        elif i == len(candidate_path_list) - 1:
            start_or_end_tag = -1
        else:
            start_or_end_tag = 1
        prefix_index = initial_index_path[i]
        end_index = initial_index_path[i+1]
        if end_index in index_to_subseq_and_score_list[prefix_index].keys():
            new_subseq_list, sub_aa_score_list, sub_sumscore_sort_list = index_to_subseq_and_score_list[prefix_index][end_index]
            sumscore_list.append(sub_sumscore_sort_list)
            aa_score_list.append(sub_aa_score_list)
            new_candidate_path_list.append(new_subseq_list)
        else:
            prefix_mass = initial_mass_path[i]
            end_mass = initial_mass_path[i+1]
            sub_sumscore = []
            sub_score = []
            for j,sub_seq in enumerate(sub_seq_list):
                if sub_seq in seq_to_score_list[prefix_index].keys():
                    site_score , sumscore = seq_to_score_list[prefix_index][sub_seq]
                    sub_score.append(site_score)
                    sub_sumscore.append(sumscore)
                else:
                    site_score ,\
                    sumscore, \
                    try_mz_to_score_dict, \
                    lys_mz_to_score_dict = scoring_for_subseq(original_try_mz_np , original_lys_mz_np ,
                                                                        original_try_intensity_np ,original_lys_intensity_np ,
                                                                        prefix_mass , end_mass ,
                                                                        sub_seq , try_pepmass ,
                                                                        match_type , try_charge, lys_charge,
                                                                        try_ion_type_dict , lys_ion_type_dict ,
                                                                        try_mz_to_score_dict , lys_mz_to_score_dict,
                                                                       #try_mz_to_internal_score_dict , lys_mz_to_internal_score_dict,start_or_end_tag
                                                                       )
                    sub_score.append(site_score)
                    sub_sumscore.append(sumscore)
                    seq_to_score_list[prefix_index][sub_seq] = (site_score , sumscore)

            sub_sumscore_np = np.array(sub_sumscore)
            sub_seq_number = len(sub_sumscore_np)

            if i == len(candidate_path_list) - 1:
                topk_maxscore_subseq = 3 * parameters.topk_maxscore_subseq
            else:
                topk_maxscore_subseq = parameters.topk_maxscore_subseq
            #topk_maxscore_subseq = parameters.topk_maxscore_subseq
            if topk_maxscore_subseq < sub_seq_number:
                max_sub_score_index = np.argpartition(sub_sumscore_np , -topk_maxscore_subseq)[-topk_maxscore_subseq:]
            else:
                max_sub_score_index = np.arange(sub_seq_number)
            '''
            max_sumscore = sub_sumscore_np[sub_sumscore_index[0]]
            i = 0
            sub_length = len(sub_sumscore_np)
            while True:
                if i > sub_length - 1:
                    break
                else:
                    if sub_sumscore_np[sub_sumscore_index[i]] == max_sumscore:
                        i = i + 1
                    else:
                        if i < parameters.topk_maxscore_subseq:
                            max_sumscore = sub_sumscore_np[sub_sumscore_index[i]]
                            i = i + 1
                        else:
                            break
            max_sub_score_index = sub_sumscore_index[:i]
            '''
            sub_sumscore_sort_np = sub_sumscore_np[max_sub_score_index]
            topN_sort_index = np.argsort(sub_sumscore_sort_np)[::-1]
            sub_sumscore_sort_np = np.array([sub_sumscore_np[max_sub_score_index[i]] for i in topN_sort_index])
            sub_aa_score_list = [sub_score[max_sub_score_index[i]] for i in topN_sort_index]
            new_subseq_list = [sub_seq_list[max_sub_score_index[i]] for i in topN_sort_index]
            '''
            if len(sub_sumscore_sort_np) != len(set(sub_sumscore_sort_np)):#It means there are repeatitive values in sub_sumscore_sort_np
                sub_sumscore = []
                sub_score = []
                for j , sub_seq in enumerate(new_subseq_list):
                    site_score, sumscore = scoring_for_subseq(original_try_mz_np, original_lys_mz_np,
                                                              original_try_intensity_np, original_lys_intensity_np,
                                                              peaks_tag_try,
                                                              prefix_mass, end_mass,
                                                              sub_seq, try_pepmass,
                                                              match_type, try_charge,
                                                              ion_type_duplicate_dict)
                    sub_score.append(site_score)
                    sub_sumscore.append(sumscore)

                sub_sumscore_sort_np = np.array(sub_sumscore)
                sub_aa_score_list = sub_score
            '''

            sub_sumscore_sort_list = list(sub_sumscore_sort_np)
            sumscore_list.append(sub_sumscore_sort_list)
            aa_score_list.append(sub_aa_score_list)
            new_candidate_path_list.append(new_subseq_list)

            index_to_subseq_and_score_list[prefix_index][end_index] = (new_subseq_list , sub_aa_score_list , sub_sumscore_sort_list)

    return new_candidate_path_list , sumscore_list , aa_score_list , index_to_subseq_and_score_list, seq_to_score_list , try_mz_to_score_dict , lys_mz_to_score_dict#, try_mz_to_internal_score_dict , lys_mz_to_internal_score_dict


def generate_adjacen_matrix(try_signal_C_mass_np , mass_round_to_index_list , mass_round_to_deltamass_list):
    signal_peaks_number = len(try_signal_C_mass_np)
    adjacen_matrix_01_np = np.zeros((signal_peaks_number,signal_peaks_number))
    adjacen_matrix_list = []
    adjacen_edge_score_matrix_list = []
    for i,start_mass in enumerate(try_signal_C_mass_np):
        if i != signal_peaks_number - 1:
            tag_list = [[-1]] * (i+1)
            edge_score_list = [0] * (i+1)
            delta_mass_np = start_mass - try_signal_C_mass_np[i+1:]
            for j,delta_mass in enumerate(delta_mass_np):
                delta_mass = abs(delta_mass)
                if delta_mass > parameters.build_aa_permutation_max_mass:
                    break
                mass_round = int(round(delta_mass * parameters.mass_resolution))
                index = mass_round_to_index_list[mass_round]
                if index == -1:
                    tag_list.append([-1])
                    edge_score_list.append(0)
                elif type(index) == list:
                    tag_list.append(index)
                    edge_score_list.append(round(parameters.delta_mass_pdf(mass_round_to_deltamass_list[mass_round])/4 , 2))
                    adjacen_matrix_01_np[i , j + i + 1] = 1
                    adjacen_matrix_01_np[j + i + 1 , i] = 1
                else:
                    tag_list.append([index])
                    edge_score_list.append(round(parameters.delta_mass_pdf(mass_round_to_deltamass_list[mass_round])/4 , 2))
                    adjacen_matrix_01_np[i , j + i + 1] = 1
                    adjacen_matrix_01_np[j + i + 1 , i] = 1

            adjacen_matrix_list.append(tag_list)
            adjacen_edge_score_matrix_list.append(edge_score_list)

    return adjacen_matrix_list , adjacen_matrix_01_np , adjacen_edge_score_matrix_list

def de_novo_initial_path_function(try_signal_N_index_dict ,
                                 try_signal_C_index_dict ,
                                try_mz_np, try_intensity_np,
                                original_try_mz_np,
                                original_lys_mz_np,
                                original_try_intensity_np,
                                original_lys_intensity_np,
                                 try_pepmass,
                                 aa_permutations_list ,
                                 mass_round_to_index_list ,
                                 mass_round_to_deltamass_list ,
                                 match_type , try_charge , lys_charge):

    try_signal_N_index = list(try_signal_N_index_dict.keys())
    try_signal_C_index = list(try_signal_C_index_dict.keys())

    try_signal_N_mass_np = try_mz_np[try_signal_N_index]
    try_signal_C_mass_np = try_mz_np[try_signal_C_index]
    try_signal_N_intensity_np = try_intensity_np[try_signal_N_index]
    try_signal_C_intensity_np = try_intensity_np[try_signal_C_index]

#########################################################################################################3
#transfer mass from N/C term to C/N term

    try_signal_N_mass_np ,  try_signal_N_intensity_np = merge_aion_mass_function(try_signal_N_mass_np ,  try_signal_N_intensity_np)


    try_signal_transfer_Cmass_np = try_pepmass - try_signal_N_mass_np + parameters.atom_mass['PROTON']
    try_signal_transfer_Cintensity_np = try_signal_N_intensity_np

    try_signal_C_mass_np = np.append(try_signal_C_mass_np , try_signal_transfer_Cmass_np)
    try_signal_C_intensity_np = np.append(try_signal_C_intensity_np , try_signal_transfer_Cintensity_np)

    try_signal_C_mass_sort_index = np.argsort(try_signal_C_mass_np)
    try_signal_C_mass_np = try_signal_C_mass_np[try_signal_C_mass_sort_index]
    try_signal_C_intensity_np = try_signal_C_intensity_np[try_signal_C_mass_sort_index]

    assert len(try_signal_C_mass_np) == len(try_signal_C_intensity_np)

##########################################################################################################
#add start index and end index ;
#merge similai mass;
    start_mass = parameters.atom_mass['H2O'] + parameters.atom_mass['PROTON']
    end_mass = try_pepmass
    if match_type == 'A1:K-K' or match_type == 'C: K-R' or match_type == 'D: K-X':
        try_signal_C_mass_np = np.append(start_mass + parameters.aa_to_mass_dict['K'], try_signal_C_mass_np)
        try_signal_C_intensity_np = np.append(1 , try_signal_C_intensity_np)
    elif match_type == 'A2:R-R' or match_type == 'B: R-K' or match_type == 'E: R-X':
        try_signal_C_mass_np = np.append(start_mass + parameters.aa_to_mass_dict['R'], try_signal_C_mass_np)
        try_signal_C_intensity_np = np.append(1, try_signal_C_intensity_np)
    elif match_type == 'A3:X-X' or match_type == 'F: X-K' or match_type == 'G: X-R':
        try_signal_C_mass_np = np.append(start_mass, try_signal_C_mass_np)
        try_signal_C_intensity_np = np.append(1, try_signal_C_intensity_np)
    else:
        assert False, 'unexpected match type!' + match_type

    try_signal_C_mass_np = np.append(try_signal_C_mass_np, end_mass)
    try_signal_C_intensity_np = np.append(try_signal_C_intensity_np , 1)

    try_signal_C_mass_np ,  try_signal_C_intensity_np = merge_similar_mass_function(try_signal_C_mass_np ,  try_signal_C_intensity_np)

    original_try_intensity_np = local_rank_of_intensity_function(original_try_mz_np , original_try_intensity_np)
    original_lys_intensity_np = local_rank_of_intensity_function(original_lys_mz_np , original_lys_intensity_np)

    signal_peaks_number = len(try_signal_C_mass_np)

###################################################################################################################
#generate adjacen_matrix and initial paths

    adjacen_matrix,adjacen_matrix_01np , adjacen_edge_score_matrix_list = generate_adjacen_matrix(try_signal_C_mass_np,mass_round_to_index_list,mass_round_to_deltamass_list)
    path_index,L = find_k_longest_initial_path_function(try_signal_C_intensity_np,try_signal_C_mass_np,adjacen_matrix_01np,adjacen_edge_score_matrix_list,parameters.topk_maxscore_initial_path)
    paths_mass_list,paths_index_list = pdag_path_to_seq_function_for_denovo(path_index,try_signal_C_mass_np)

    #print(np.array(paths_index_list)[score_sort_index])
    #print(paths_sum_intensity_np[score_sort_index])

    #adjacen_matrix_list , adjacen_matrix_01_np , adjacen_edge_score_matrix_list = pdag_algorithm.generate_adjacen_matrix(try_signal_C_mass_np , mass_round_to_index_list , mass_round_to_deltamass_list)
    #path, max_score_L = pdag_algorithm.find_k_longest_path_function(try_signal_C_intensity_np, try_signal_C_mass_np, adjacen_matrix_01_np)

    #print(path)
    #print(max_score_L[-1])



    #de novo sequence and generate candidate paths.
    #print('candidate-adj_time',candidate-adj_time)


    result_seq_list = []
    result_aa_score_list = []
    result_seq_score_list = []

    index_to_subseq_and_score_list = [{} for i in range(signal_peaks_number - 1)]
    seq_to_score_list = [{} for i in range(signal_peaks_number - 1)]
    try_mz_to_score_dict = {}
    lys_mz_to_score_dict = {}
    try_mz_to_internal_score_dict = {}
    lys_mz_to_internal_score_dict = {}
    output_number_current = 0
    max_score = 0
    max_score_number = 0
    i = 0
    for i in range(len(paths_index_list)):
        initial_mass_path = paths_mass_list[i]
        initial_index_path = paths_index_list[i]
        #print(paths_score_np[index],initial_index_path)
        #print(paths_edge_score_list[index])
        #print('initial_mass_path',initial_mass_path)
        candidate_path_list , \
        sumscore_list  , \
        aa_score_list , \
        index_to_subseq_and_score_list_Temp, \
        seq_to_score_list , \
        try_mz_to_score_dict, \
        lys_mz_to_score_dict = de_novo_candidate_path_function( initial_mass_path , initial_index_path ,
                                                                     original_try_mz_np , original_lys_mz_np ,
                                                                     original_try_intensity_np ,original_lys_intensity_np ,
                                                                     try_pepmass ,
                                                                     adjacen_matrix, match_type ,
                                                                     aa_permutations_list , try_charge , lys_charge,
                                                                     index_to_subseq_and_score_list,seq_to_score_list,
                                                                     try_mz_to_score_dict,lys_mz_to_score_dict
                                                                     #try_mz_to_internal_score_dict,lys_mz_to_internal_score_dict
                                                                    )
        index_to_subseq_and_score_list = index_to_subseq_and_score_list_Temp
        #print(candidate_path_list)
        #print(sumscore_list)


        result_path, result_sumscore, result_aa_score = find_k_longest_candidate_path_function(candidate_path_list,
                                                                                                    sumscore_list,
                                                                                                    aa_score_list,
                                                                                                    parameters.output_topk)
        '''
        for index in range(len(result_sumscore)):
            path_list = result_path[index]
            if i == 0:
                result_seq_list.append(path_list)
                result_aa_score_list.append(result_aa_score[index])
                result_seq_score_list.append(round(result_sumscore[index], 2))
            else:
                if path_list not in result_seq_list:
                    result_seq_list.append(path_list)
                    result_aa_score_list.append(result_aa_score[index])
                    result_seq_score_list.append(round(result_sumscore[index], 2))
        '''

        #result_path1 , result_sumscore1 , result_aa_score1  = search_result_path(candidate_path_list , sumscore_list , aa_score_list)


        if len(result_sumscore) <= parameters.output_topk:
            for index in range(len(result_sumscore)):
                path_list = result_path[index]
                if i == 0:
                    result_seq_list.append(path_list)
                    result_aa_score_list.append(result_aa_score[index])
                    result_seq_score_list.append(round(result_sumscore[index], 2))
                else:
                    if path_list not in result_seq_list:
                        result_seq_list.append(path_list)
                        result_aa_score_list.append(result_aa_score[index])
                        result_seq_score_list.append(round(result_sumscore[index], 2))
        else:
            sort_index = np.argpartition(np.array(result_sumscore), -parameters.output_topk)[-parameters.output_topk:]
            for j, index in enumerate(sort_index):
                if j < parameters.output_topk:
                    path_list = result_path[index]
                    if i == 0:
                        result_seq_list.append(path_list)
                        result_aa_score_list.append(result_aa_score[index])
                        result_seq_score_list.append(round(result_sumscore[index], 2))
                    else:
                        if path_list not in result_seq_list:
                            result_seq_list.append(path_list)
                            result_aa_score_list.append(result_aa_score[index])
                            result_seq_score_list.append(round(result_sumscore[index], 2))
                else:
                    break

        #print(result_seq_list)
        #print(result_seq_score_list)
        #print(result_aa_score_list)
        #print('#####')


        '''
        for j,path in enumerate(result_path):
            path_list = transfer_seq_str_to_list(''.join(path))
            if path_list not in result_seq_list:
                result_seq_list.append(path_list)
                result_aa_score_list.append(result_aa_score[j])
                seq_score = round(result_sumscore[j], 2)
                result_seq_score_list.append(seq_score)
                if seq_score > max_score:
                    max_score_number = 1
                    max_score = seq_score
                elif seq_score == max_score:
                    max_score_number += 1
        '''
        #output_number_current = len(result_seq_list)

    #print('end_candidate-candidate',end_candidate-candidate)

    result_seq_norepeat_list = result_seq_list
    result_seq_score_norepeat_list = result_seq_score_list
    result_aa_score_norepeat_list = result_aa_score_list

    ##################################################################################################################
    #sort the candidate path
    result_seq_score_norepeat_np = np.array(result_seq_score_norepeat_list)
    output_index = np.argsort(result_seq_score_norepeat_np)[::-1][:parameters.output_topk]

    result_seq = [result_seq_norepeat_list[index] for index in output_index]
    result_aa_score = [result_aa_score_norepeat_list[index] for index in output_index]
    result_seq_score = result_seq_score_norepeat_np[output_index]

    for i,path in enumerate(result_seq):
        if match_type == 'A1:K-K' or match_type == 'C: K-R' or match_type == 'D: K-X':
            path = (['K'] + path)[::-1]
            aa_score =  ([0] + result_aa_score[i])[::-1]
        elif match_type == 'A2:R-R' or match_type == 'B: R-K' or match_type == 'E: R-X':
            path = (['R'] + path)[::-1]
            aa_score =  ([0] + result_aa_score[i])[::-1]
        elif match_type == 'A3:X-X' or match_type == 'F: X-K' or match_type == 'G: X-R':
            path = path[::-1]
            aa_score = result_aa_score[i][::-1]
        else:
            assert False, 'unexpected match type!'
        result_seq[i] = path
        result_aa_score[i] = aa_score

##################################################################################################################3
    #Rescore
    '''
    if max_score_number >= 2 and max_score_number <= parameters.output_topk: #最大打分肽段数可能大于10（还未处理）

        # Rescore conditions
        flag = True
        length = len(result_seq[0])
        for i in range(1,max_score_number):
            if length != len(result_seq[i]):
                flag = False

        if flag == True:
            max_score_seq_list = [result_seq[index] for index in range(max_score_number)]
            max_score_lysseq_list = [result_lysseq[index] for index in range(max_score_number)]
            if_rescore , Rescore_index_list = Rescore_function(original_try_mz_np,original_try_intensity_np,original_lys_mz_np,original_lys_intensity_np,
                                            max_score_seq_list , max_score_lysseq_list)
            if if_rescore == True:
                Rescore_index_list = Rescore_index_list + list(range(max_score_number,parameters.output_topk))
                result_seq = [result_seq[index] for index in Rescore_index_list]
                result_aa_score = [result_aa_score[index] for index in Rescore_index_list]
                result_seq_score = result_seq_score[Rescore_index_list]
    '''

    return result_seq , result_aa_score ,result_seq_score
#########################################################################

def scoring_for_subseq(try_mz_np , lys_mz_np ,
                       try_intensity_np ,lys_intensity_np ,
                       prefix_mass , end_mass ,
                       subseq , try_pepmass,
                       match_type , try_charge, lys_charge,
                       try_ion_type_dict , lys_ion_type_dict,
                       try_mz_to_score_dict, lys_mz_to_score_dict,
                       #try_mz_to_internal_score_dict , lys_mz_to_internal_score_dict,start_or_end_tag
                       ):

    try_ion_type_tag_list = list(try_ion_type_dict.values())
    try_ion_type_list = list(try_ion_type_dict.keys())
    lys_ion_type_tag_list = list(lys_ion_type_dict.values())
    lys_ion_type_list = list(lys_ion_type_dict.keys())
    subseq_string = subseq

    if 'AG' in subseq_string:
        index = subseq_string.find('AG')
        if '(' in subseq_string:
            subseq, index = transfer_seq_str_to_list(subseq, index)
        else:
            subseq = transfer_seq_str_to_list(subseq)

    elif 'GA' in subseq_string:
        index = subseq_string.find('GA')
        if '(' in subseq_string:
            subseq, index = transfer_seq_str_to_list(subseq, index)
        else:
            subseq = transfer_seq_str_to_list(subseq)

    elif 'GG' in subseq_string:
        index = subseq_string.find('GG')
        if '(' in subseq_string:
            subseq, index = transfer_seq_str_to_list(subseq, index)
        else:
            subseq = transfer_seq_str_to_list(subseq)

    else:
        subseq = transfer_seq_str_to_list(subseq)

    # subseq = transfer_seq_str_to_list(subseq)

    length = len(subseq)
    site_score_np = np.zeros(length)
    site_signal_num_np = np.zeros(length)

    try_peptide_residue_mass = [parameters.aa_to_mass_dict[aa] for aa in subseq]
    try_peptide_residue_cumsummass = np.cumsum(try_peptide_residue_mass)

    for i, ion_type in enumerate(try_ion_type_list):

        if ion_type == 'y+' and try_ion_type_tag_list[i] == True:
            try_theoretical_single_charge_y_ion = try_peptide_residue_cumsummass + prefix_mass
            try_theoretical_single_charge_y_ion_score, try_mz_to_score_dict, try_theoretical_single_charge_y_ion_if_match = match_theoretical_and_real_ions_function(
                try_theoretical_single_charge_y_ion, try_mz_np, try_intensity_np, try_mz_to_score_dict)
            if match_type == 'A1:K-K' or match_type == 'C: K-R' or match_type == 'D: K-X':
                pairtry_theoretical_single_charge_y_ion = try_theoretical_single_charge_y_ion - \
                                                          parameters.aa_to_mass_dict['K']
            elif match_type == 'A2:R-R' or match_type == 'B: R-K' or match_type == 'E: R-X':
                pairtry_theoretical_single_charge_y_ion = try_theoretical_single_charge_y_ion - \
                                                          parameters.aa_to_mass_dict['R']
            elif match_type == 'A3:X-X' or match_type == 'F: X-K' or match_type == 'G: X-R':
                pairtry_theoretical_single_charge_y_ion = try_theoretical_single_charge_y_ion
            else:
                assert False, 'Unexpected match type!'
            pairtry_theoretical_single_charge_y_ion_score, lys_mz_to_score_dict, pairtry_theoretical_single_charge_y_ion_if_match = match_theoretical_and_real_ions_function(
                pairtry_theoretical_single_charge_y_ion, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
            site_score_np = site_score_np + try_theoretical_single_charge_y_ion_score + pairtry_theoretical_single_charge_y_ion_score
            site_signal_num_np = site_signal_num_np + try_theoretical_single_charge_y_ion_if_match + pairtry_theoretical_single_charge_y_ion_if_match

        if ion_type == 'y2+':
            if try_ion_type_tag_list[i] == True:
                try_theoretical_two_charge_y_ion = (try_theoretical_single_charge_y_ion + parameters.atom_mass[
                    'PROTON']) / 2.0
                if lys_ion_type_tag_list[i] == False:
                    try_theoretical_two_charge_y_ion_score, try_mz_to_score_dict, try_theoretical_two_charge_y_ion_if_match = match_theoretical_and_real_ions_function(
                        try_theoretical_two_charge_y_ion, try_mz_np, try_intensity_np, try_mz_to_score_dict)
                    site_score_np = site_score_np + try_theoretical_two_charge_y_ion_score
                    site_signal_num_np = site_signal_num_np + try_theoretical_two_charge_y_ion_if_match
                else:
                    if match_type == 'A1:K-K' or match_type == 'C: K-R' or match_type == 'D: K-X':
                        pairtry_theoretical_two_charge_y_ion = try_theoretical_two_charge_y_ion - \
                                                               parameters.aa_to_mass_dict['K'] / 2.0
                    elif match_type == 'A2:R-R' or match_type == 'B: R-K' or match_type == 'E: R-X':
                        pairtry_theoretical_two_charge_y_ion = try_theoretical_two_charge_y_ion - \
                                                               parameters.aa_to_mass_dict['R'] / 2.0
                    elif match_type == 'A3:X-X' or match_type == 'F: X-K' or match_type == 'G: X-R':
                        pairtry_theoretical_two_charge_y_ion = try_theoretical_two_charge_y_ion
                    else:
                        assert False, 'Unexpected match type!'
                    try_theoretical_two_charge_y_ion_score, try_mz_to_score_dict, try_theoretical_two_charge_y_ion_if_match = match_theoretical_and_real_ions_function(
                        try_theoretical_two_charge_y_ion, try_mz_np, try_intensity_np, try_mz_to_score_dict)
                    pairtry_theoretical_two_charge_y_ion_score, lys_mz_to_score_dict, pairtry_theoretical_two_charge_y_ion_if_match = match_theoretical_and_real_ions_function(
                        pairtry_theoretical_two_charge_y_ion, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
                    site_score_np = site_score_np + try_theoretical_two_charge_y_ion_score + pairtry_theoretical_two_charge_y_ion_score
                    site_signal_num_np = site_signal_num_np + try_theoretical_two_charge_y_ion_if_match + pairtry_theoretical_two_charge_y_ion_if_match
            else:
                if lys_ion_type_tag_list[i] == True:
                    try_theoretical_two_charge_y_ion = (try_theoretical_single_charge_y_ion + parameters.atom_mass[
                        'PROTON']) / 2.0
                    if match_type == 'A1:K-K' or match_type == 'C: K-R' or match_type == 'D: K-X':
                        pairtry_theoretical_two_charge_y_ion = try_theoretical_two_charge_y_ion - \
                                                               parameters.aa_to_mass_dict['K'] / 2.0
                    elif match_type == 'A2:R-R' or match_type == 'B: R-K' or match_type == 'E: R-X':
                        pairtry_theoretical_two_charge_y_ion = try_theoretical_two_charge_y_ion - \
                                                               parameters.aa_to_mass_dict['R'] / 2.0
                    elif match_type == 'A3:X-X' or match_type == 'F: X-K' or match_type == 'G: X-R':
                        pairtry_theoretical_two_charge_y_ion = try_theoretical_two_charge_y_ion
                    else:
                        assert False, 'Unexpected match type!'
                    pairtry_theoretical_two_charge_y_ion_score, lys_mz_to_score_dict, pairtry_theoretical_two_charge_y_ion_if_match = match_theoretical_and_real_ions_function(
                        pairtry_theoretical_two_charge_y_ion, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
                    site_score_np = site_score_np + pairtry_theoretical_two_charge_y_ion_score
                    site_signal_num_np = site_signal_num_np + pairtry_theoretical_two_charge_y_ion_if_match

        if ion_type == 'y_H2O' and try_ion_type_tag_list[i] == True:
            try_theoretical_single_charge_y_H2O = try_theoretical_single_charge_y_ion - parameters.atom_mass['H2O']
            try_theoretical_single_charge_y_H2O_score, try_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                try_theoretical_single_charge_y_H2O, try_mz_np, try_intensity_np, try_mz_to_score_dict)
            if match_type == 'A1:K-K' or match_type == 'C: K-R' or match_type == 'D: K-X':
                pairtry_theoretical_single_charge_y_H2O = try_theoretical_single_charge_y_H2O - \
                                                          parameters.aa_to_mass_dict['K']
            elif match_type == 'A2:R-R' or match_type == 'B: R-K' or match_type == 'E: R-X':
                pairtry_theoretical_single_charge_y_H2O = try_theoretical_single_charge_y_H2O - \
                                                          parameters.aa_to_mass_dict['R']
            elif match_type == 'A3:X-X' or match_type == 'F: X-K' or match_type == 'G: X-R':
                pairtry_theoretical_single_charge_y_H2O = try_theoretical_single_charge_y_H2O
            else:
                assert False, 'Unexpected match type!'
            pairtry_theoretical_single_charge_y_H2O_score, lys_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                pairtry_theoretical_single_charge_y_H2O, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
            site_score_np = site_score_np + 0.25 * try_theoretical_single_charge_y_H2O_score + 0.25 * pairtry_theoretical_single_charge_y_H2O_score

        if ion_type == 'y_NH3' and try_ion_type_tag_list[i] == True:
            try_theoretical_single_charge_y_NH3 = try_theoretical_single_charge_y_ion - parameters.atom_mass['NH3']
            try_theoretical_single_charge_y_NH3_score, try_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                try_theoretical_single_charge_y_NH3, try_mz_np, try_intensity_np, try_mz_to_score_dict)
            if match_type == 'A1:K-K' or match_type == 'C: K-R' or match_type == 'D: K-X':
                pairtry_theoretical_single_charge_y_NH3 = try_theoretical_single_charge_y_NH3 - \
                                                          parameters.aa_to_mass_dict['K']
            elif match_type == 'A2:R-R' or match_type == 'B: R-K' or match_type == 'E: R-X':
                pairtry_theoretical_single_charge_y_NH3 = try_theoretical_single_charge_y_NH3 - \
                                                          parameters.aa_to_mass_dict['R']
            elif match_type == 'A3:X-X' or match_type == 'F: X-K' or match_type == 'G: X-R':
                pairtry_theoretical_single_charge_y_NH3 = try_theoretical_single_charge_y_NH3
            else:
                assert False, 'Unexpected match type!'
            pairtry_theoretical_single_charge_y_NH3_score, lys_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                pairtry_theoretical_single_charge_y_NH3, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
            site_score_np = site_score_np + 0.25 * try_theoretical_single_charge_y_NH3_score + 0.25 * pairtry_theoretical_single_charge_y_NH3_score

        if ion_type == 'b+' and try_ion_type_tag_list[i] == True:
            try_theoretical_single_charge_b_ion = try_pepmass - try_theoretical_single_charge_y_ion + \
                                                  parameters.atom_mass['PROTON']
            try_theoretical_single_charge_b_ion_score, try_mz_to_score_dict, try_theoretical_single_charge_b_ion_if_match = match_theoretical_and_real_ions_function(
                try_theoretical_single_charge_b_ion, try_mz_np, try_intensity_np, try_mz_to_score_dict)
            if match_type == 'A1:K-K' or match_type == 'B: R-K' or match_type == 'F: X-K':
                pairtry_theoretical_single_charge_b_ion = try_theoretical_single_charge_b_ion + \
                                                          parameters.aa_to_mass_dict['K']
            elif match_type == 'A2:R-R' or match_type == 'C: K-R' or match_type == 'G: X-R':
                pairtry_theoretical_single_charge_b_ion = try_theoretical_single_charge_b_ion + \
                                                          parameters.aa_to_mass_dict['R']
            elif match_type == 'A3:X-X' or match_type == 'D: K-X' or match_type == 'E: R-X':
                pairtry_theoretical_single_charge_b_ion = try_theoretical_single_charge_b_ion
            else:
                assert False, 'Unexpected match type!'
            pairtry_theoretical_single_charge_b_ion_score, lys_mz_to_score_dict, pairtry_theoretical_single_charge_b_ion_if_match = match_theoretical_and_real_ions_function(
                pairtry_theoretical_single_charge_b_ion, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
            site_score_np = site_score_np + try_theoretical_single_charge_b_ion_score + pairtry_theoretical_single_charge_b_ion_score
            site_signal_num_np = site_signal_num_np + try_theoretical_single_charge_b_ion_if_match + pairtry_theoretical_single_charge_b_ion_if_match

        if ion_type == 'b2+':
            if try_ion_type_tag_list[i] == True:
                try_theoretical_two_charge_b_ion = (try_theoretical_single_charge_b_ion + parameters.atom_mass[
                    'PROTON']) / 2.0
                if lys_ion_type_tag_list[i] == False:
                    try_theoretical_two_charge_b_ion_score, try_mz_to_score_dict, try_theoretical_two_charge_b_ion_if_match = match_theoretical_and_real_ions_function(
                        try_theoretical_two_charge_b_ion, try_mz_np, try_intensity_np, try_mz_to_score_dict)
                    site_score_np = site_score_np + try_theoretical_two_charge_b_ion_score
                    site_signal_num_np = site_signal_num_np + try_theoretical_two_charge_b_ion_if_match
                else:
                    if match_type == 'A1:K-K' or match_type == 'B: R-K' or match_type == 'F: X-K':
                        pairtry_theoretical_two_charge_b_ion = try_theoretical_two_charge_b_ion + \
                                                               parameters.aa_to_mass_dict['K'] / 2.0
                    elif match_type == 'A2:R-R' or match_type == 'C: K-R' or match_type == 'G: X-R':
                        pairtry_theoretical_two_charge_b_ion = try_theoretical_two_charge_b_ion + \
                                                               parameters.aa_to_mass_dict['R'] / 2.0
                    elif match_type == 'A3:X-X' or match_type == 'D: K-X' or match_type == 'E: R-X':
                        pairtry_theoretical_two_charge_b_ion = try_theoretical_two_charge_b_ion
                    else:
                        assert False, 'Unexpected match type!'
                    try_theoretical_two_charge_b_ion_score, try_mz_to_score_dict, try_theoretical_two_charge_b_ion_if_match = match_theoretical_and_real_ions_function(
                        try_theoretical_two_charge_b_ion, try_mz_np, try_intensity_np, try_mz_to_score_dict)
                    pairtry_theoretical_two_charge_b_ion_score, lys_mz_to_score_dict, pairtry_theoretical_two_charge_b_ion_if_match = match_theoretical_and_real_ions_function(
                        pairtry_theoretical_two_charge_b_ion, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
                    site_score_np = site_score_np + try_theoretical_two_charge_b_ion_score + pairtry_theoretical_two_charge_b_ion_score
                    site_signal_num_np = site_signal_num_np + try_theoretical_two_charge_b_ion_if_match + pairtry_theoretical_two_charge_b_ion_if_match
            else:
                if lys_ion_type_tag_list[i] == True:
                    try_theoretical_two_charge_b_ion = (try_theoretical_single_charge_b_ion + parameters.atom_mass[
                        'PROTON']) / 2.0
                    if match_type == 'A1:K-K' or match_type == 'B: R-K' or match_type == 'F: X-K':
                        pairtry_theoretical_two_charge_b_ion = try_theoretical_two_charge_b_ion + \
                                                               parameters.aa_to_mass_dict['K'] / 2.0
                    elif match_type == 'A2:R-R' or match_type == 'C: K-R' or match_type == 'G: X-R':
                        pairtry_theoretical_two_charge_b_ion = try_theoretical_two_charge_b_ion + \
                                                               parameters.aa_to_mass_dict['R'] / 2.0
                    elif match_type == 'A3:X-X' or match_type == 'D: K-X' or match_type == 'E: R-X':
                        pairtry_theoretical_two_charge_b_ion = try_theoretical_two_charge_b_ion
                    else:
                        assert False, 'Unexpected match type!'
                    pairtry_theoretical_two_charge_b_ion_score, lys_mz_to_score_dict, pairtry_theoretical_two_charge_b_ion_if_match = match_theoretical_and_real_ions_function(
                        pairtry_theoretical_two_charge_b_ion, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
                    site_score_np = site_score_np + pairtry_theoretical_two_charge_b_ion_score
                    site_signal_num_np = site_signal_num_np + pairtry_theoretical_two_charge_b_ion_if_match

        if ion_type == 'b_H2O' and try_ion_type_tag_list[i] == True:
            try_theoretical_single_charge_b_H2O = try_theoretical_single_charge_b_ion - parameters.atom_mass['H2O']
            try_theoretical_single_charge_b_H2O_score, try_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                try_theoretical_single_charge_b_H2O, try_mz_np, try_intensity_np, try_mz_to_score_dict)
            if match_type == 'A1:K-K' or match_type == 'B: R-K' or match_type == 'F: X-K':
                pairtry_theoretical_single_charge_b_H2O = try_theoretical_single_charge_b_H2O + \
                                                          parameters.aa_to_mass_dict['K']
            elif match_type == 'A2:R-R' or match_type == 'C: K-R' or match_type == 'G: X-R':
                pairtry_theoretical_single_charge_b_H2O = try_theoretical_single_charge_b_H2O + \
                                                          parameters.aa_to_mass_dict['R']
            elif match_type == 'A3:X-X' or match_type == 'D: K-X' or match_type == 'E: R-X':
                pairtry_theoretical_single_charge_b_H2O = try_theoretical_single_charge_b_H2O
            else:
                assert False, 'Unexpected match type!'
            pairtry_theoretical_single_charge_b_H2O_score, lys_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                pairtry_theoretical_single_charge_b_H2O, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
            site_score_np = site_score_np + 0.25 * try_theoretical_single_charge_b_H2O_score + 0.25 * pairtry_theoretical_single_charge_b_H2O_score

        if ion_type == 'b_NH3' and try_ion_type_tag_list[i] == True:
            try_theoretical_single_charge_b_NH3 = try_theoretical_single_charge_b_ion - parameters.atom_mass['NH3']
            try_theoretical_single_charge_b_NH3_score, try_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                try_theoretical_single_charge_b_NH3, try_mz_np, try_intensity_np, try_mz_to_score_dict)
            if match_type == 'A1:K-K' or match_type == 'B: R-K' or match_type == 'F: X-K':
                pairtry_theoretical_single_charge_b_NH3 = try_theoretical_single_charge_b_NH3 + \
                                                          parameters.aa_to_mass_dict['K']
            elif match_type == 'A2:R-R' or match_type == 'C: K-R' or match_type == 'G: X-R':
                pairtry_theoretical_single_charge_b_NH3 = try_theoretical_single_charge_b_NH3 + \
                                                          parameters.aa_to_mass_dict['R']
            elif match_type == 'A3:X-X' or match_type == 'D: K-X' or match_type == 'E: R-X':
                pairtry_theoretical_single_charge_b_NH3 = try_theoretical_single_charge_b_NH3
            else:
                assert False, 'Unexpected match type!'
            pairtry_theoretical_single_charge_b_NH3_score, lys_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                pairtry_theoretical_single_charge_b_NH3, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
            site_score_np = site_score_np + 0.25 * try_theoretical_single_charge_b_NH3_score + 0.25 * pairtry_theoretical_single_charge_b_NH3_score

        if ion_type == 'a+' and try_ion_type_tag_list[i] == True:
            try_theoretical_single_charge_a_ion = try_theoretical_single_charge_b_ion - parameters.atom_mass['CO']
            try_theoretical_single_charge_a_ion_score, try_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                try_theoretical_single_charge_a_ion, try_mz_np, try_intensity_np, try_mz_to_score_dict)
            if match_type == 'A1:K-K' or match_type == 'B: R-K' or match_type == 'F: X-K':
                pairtry_theoretical_single_charge_a_ion = try_theoretical_single_charge_a_ion + \
                                                          parameters.aa_to_mass_dict['K']
            elif match_type == 'A2:R-R' or match_type == 'C: K-R' or match_type == 'G: X-R':
                pairtry_theoretical_single_charge_a_ion = try_theoretical_single_charge_a_ion + \
                                                          parameters.aa_to_mass_dict['R']
            elif match_type == 'A3:X-X' or match_type == 'D: K-X' or match_type == 'E: R-X':
                pairtry_theoretical_single_charge_a_ion = try_theoretical_single_charge_a_ion
            else:
                assert False, 'Unexpected match type!'
            pairtry_theoretical_single_charge_a_ion_score, lys_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                pairtry_theoretical_single_charge_a_ion, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
            site_score_np = site_score_np + 0.25 * try_theoretical_single_charge_a_ion_score + 0.25 * pairtry_theoretical_single_charge_a_ion_score

        if ion_type == 'a2+':
            if try_ion_type_tag_list[i] == True:
                try_theoretical_two_charge_a_ion = (try_theoretical_single_charge_a_ion + parameters.atom_mass[
                    'PROTON']) / 2.0
                if lys_ion_type_tag_list[i] == False:
                    try_theoretical_two_charge_a_ion_score, try_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                        try_theoretical_two_charge_a_ion, try_mz_np, try_intensity_np, try_mz_to_score_dict)
                    site_score_np = site_score_np + try_theoretical_two_charge_a_ion_score
                else:
                    if match_type == 'A1:K-K' or match_type == 'B: R-K' or match_type == 'F: X-K':
                        pairtry_theoretical_two_charge_a_ion = try_theoretical_two_charge_a_ion + \
                                                               parameters.aa_to_mass_dict['K'] / 2.0
                    elif match_type == 'A2:R-R' or match_type == 'C: K-R' or match_type == 'G: X-R':
                        pairtry_theoretical_two_charge_a_ion = try_theoretical_two_charge_a_ion + \
                                                               parameters.aa_to_mass_dict['R'] / 2.0
                    elif match_type == 'A3:X-X' or match_type == 'D: K-X' or match_type == 'E: R-X':
                        pairtry_theoretical_two_charge_a_ion = try_theoretical_two_charge_a_ion
                    else:
                        assert False, 'Unexpected match type!'
                    try_theoretical_two_charge_a_ion_score, try_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                        try_theoretical_two_charge_a_ion, try_mz_np, try_intensity_np, try_mz_to_score_dict)
                    pairtry_theoretical_two_charge_a_ion_score, lys_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                        pairtry_theoretical_two_charge_a_ion, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
                    site_score_np = site_score_np + try_theoretical_two_charge_a_ion_score + pairtry_theoretical_two_charge_a_ion_score
            else:
                if lys_ion_type_tag_list[i] == True:
                    try_theoretical_two_charge_a_ion = (try_theoretical_single_charge_a_ion + parameters.atom_mass[
                        'PROTON']) / 2.0
                    if match_type == 'A1:K-K' or match_type == 'B: R-K' or match_type == 'F: X-K':
                        pairtry_theoretical_two_charge_a_ion = try_theoretical_two_charge_a_ion + \
                                                               parameters.aa_to_mass_dict['K'] / 2.0
                    elif match_type == 'A2:R-R' or match_type == 'C: K-R' or match_type == 'G: X-R':
                        pairtry_theoretical_two_charge_a_ion = try_theoretical_two_charge_a_ion + \
                                                               parameters.aa_to_mass_dict['R'] / 2.0
                    elif match_type == 'A3:X-X' or match_type == 'D: K-X' or match_type == 'E: R-X':
                        pairtry_theoretical_two_charge_a_ion = try_theoretical_two_charge_a_ion
                    else:
                        assert False, 'Unexpected match type!'
                    pairtry_theoretical_two_charge_a_ion_score, lys_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                        pairtry_theoretical_two_charge_a_ion, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
                    site_score_np = site_score_np + pairtry_theoretical_two_charge_a_ion_score

        if ion_type == 'a_H2O' and try_ion_type_tag_list[i] == True:
            try_theoretical_single_charge_a_H2O = try_theoretical_single_charge_a_ion - parameters.atom_mass['H2O']
            try_theoretical_single_charge_a_H2O_score, try_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                try_theoretical_single_charge_a_H2O, try_mz_np, try_intensity_np, try_mz_to_score_dict)
            if match_type == 'A1:K-K' or match_type == 'B: R-K' or match_type == 'F: X-K':
                pairtry_theoretical_single_charge_a_H2O = try_theoretical_single_charge_a_H2O + \
                                                          parameters.aa_to_mass_dict['K']
            elif match_type == 'A2:R-R' or match_type == 'C: K-R' or match_type == 'G: X-R':
                pairtry_theoretical_single_charge_a_H2O = try_theoretical_single_charge_a_H2O + \
                                                          parameters.aa_to_mass_dict['R']
            elif match_type == 'A3:X-X' or match_type == 'D: K-X' or match_type == 'E: R-X':
                pairtry_theoretical_single_charge_a_H2O = try_theoretical_single_charge_a_H2O
            else:
                assert False, 'Unexpected match type!'
            pairtry_theoretical_single_charge_a_H2O_score, lys_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                pairtry_theoretical_single_charge_a_H2O, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
            site_score_np = site_score_np + try_theoretical_single_charge_a_H2O_score + pairtry_theoretical_single_charge_a_H2O_score

        if ion_type == 'a_NH3' and try_ion_type_tag_list[i] == True:
            try_theoretical_single_charge_a_NH3 = try_theoretical_single_charge_a_ion - parameters.atom_mass['NH3']
            try_theoretical_single_charge_a_NH3_score, try_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                try_theoretical_single_charge_a_NH3, try_mz_np, try_intensity_np, try_mz_to_score_dict)
            if match_type == 'A1:K-K' or match_type == 'B: R-K' or match_type == 'F: X-K':
                pairtry_theoretical_single_charge_a_NH3 = try_theoretical_single_charge_a_NH3 + \
                                                          parameters.aa_to_mass_dict['K']
            elif match_type == 'A2:R-R' or match_type == 'C: K-R' or match_type == 'G: X-R':
                pairtry_theoretical_single_charge_a_NH3 = try_theoretical_single_charge_a_NH3 + \
                                                          parameters.aa_to_mass_dict['R']
            elif match_type == 'A3:X-X' or match_type == 'D: K-X' or match_type == 'E: R-X':
                pairtry_theoretical_single_charge_a_NH3 = try_theoretical_single_charge_a_NH3
            else:
                assert False, 'Unexpected match type!'
            pairtry_theoretical_single_charge_a_NH3_score, lys_mz_to_score_dict, _ = match_theoretical_and_real_ions_function(
                pairtry_theoretical_single_charge_a_NH3, lys_mz_np, lys_intensity_np, lys_mz_to_score_dict)
            site_score_np = site_score_np + try_theoretical_single_charge_a_NH3_score + pairtry_theoretical_single_charge_a_NH3_score

    if 'AG' in subseq_string:
        sum_score = (sum(site_score_np) + (parameters.score_scale_factor - 1) * site_score_np[index]) / length
    elif 'GA' in subseq_string:
        sum_score = (sum(site_score_np) + (parameters.score_scale_factor - 1) * site_score_np[index]) / length
    elif 'GG' in subseq_string:
        sum_score = (sum(site_score_np) + (parameters.score_scale_factor - 1) * site_score_np[index]) / length
    else:
        sum_score = sum(site_score_np) / length
    site_score_list = list(site_score_np)

    return site_score_list, sum_score, try_mz_to_score_dict, lys_mz_to_score_dict  # ,try_mz_to_internal_score_dict, lys_mz_to_internal_score_dict


def match_theoretical_and_real_ions_function(theoretical_mz_np , real_mz_np ,real_intensity_np, mz_to_score_dict):

    theoretical_score_np = np.zeros_like(theoretical_mz_np)
    site_if_match_np = np.zeros_like(theoretical_mz_np)
    for i, theoretical_mass in enumerate(theoretical_mz_np):
        if theoretical_mass in mz_to_score_dict.keys():
            if mz_to_score_dict[theoretical_mass] != 0:
                theoretical_score_np[i] = mz_to_score_dict[theoretical_mass]
                site_if_match_np[i] = 1
        else:
            index , min_delta_mass = find_nearest(real_mz_np , theoretical_mass)
            mul = 2 if theoretical_mass <= 205 else 1
            if parameters.ppm_or_dalton == 0 :
                min_delta_mass_ppm = min_delta_mass/theoretical_mass * 1000000
                if min_delta_mass_ppm <= mul * parameters.fragment_tolerance:
                    score = round(real_intensity_np[index] + parameters.delta_mass_pdf(min_delta_mass) , 4)
                    theoretical_score_np[i] = score
                    site_if_match_np[i] = 1
                    mz_to_score_dict[theoretical_mass] = score
                else:
                    mz_to_score_dict[theoretical_mass] = 0
            else:
                if min_delta_mass <= mul * parameters.fragment_tolerance:
                    score = round(real_intensity_np[index] + parameters.delta_mass_pdf(min_delta_mass) , 4)
                    theoretical_score_np[i] = score
                    site_if_match_np[i] = 1
                    mz_to_score_dict[theoretical_mass] = score
                else:
                    mz_to_score_dict[theoretical_mass] = 0

    return theoretical_score_np , mz_to_score_dict , site_if_match_np

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) < abs(value - array[idx])):
        return idx-1,abs(value - array[idx-1])
    else:
        return idx,abs(value - array[idx])

###########################################################################################################################################
#preprocess spectra
def remove_isotope_peaks(original_try_mz_list, original_try_intensity_list, original_lys_mz_list, original_lys_intensity_list,
                         original_try_isotope_cluster, original_try_isotope_charge, original_lys_isotope_cluster, original_lys_isotope_charge):
    visited_tag = np.zeros_like(original_try_isotope_cluster)
    new_original_try_mz_list = []
    new_original_try_intensity_list = []
    new_try_isotope_charge = []
    try_tag_list = []
    for i , cluster_index in enumerate(original_try_isotope_cluster):
        if cluster_index == 0:
            new_original_try_mz_list.append(original_try_mz_list[i])
            new_original_try_intensity_list.append(original_try_intensity_list[i])
            new_try_isotope_charge.append(0)
            try_tag_list.append(0)
        else:
            if visited_tag[cluster_index] == 0:
                charge = original_try_isotope_charge[i]
                new_peaks_mz = original_try_mz_list[i] * charge - (charge - 1) * parameters.atom_mass['PROTON']
                new_original_try_mz_list.append(new_peaks_mz)
                new_original_try_intensity_list.append(original_try_intensity_list[i])
                new_try_isotope_charge.append(1)
                try_tag_list.append(0.5)
                visited_tag[cluster_index] = 1

    try_remove_isotope_mz_list = new_original_try_mz_list
    try_remove_isotope_intensity_list = new_original_try_intensity_list

    visited_tag = np.zeros_like(original_lys_isotope_cluster)
    new_original_lys_mz_list = []
    new_original_lys_intensity_list = []
    new_lys_isotope_charge = []
    lys_tag_list = []
    for i, cluster_index in enumerate(original_lys_isotope_cluster):
        if cluster_index == 0:
            new_original_lys_mz_list.append(original_lys_mz_list[i])
            new_original_lys_intensity_list.append(original_lys_intensity_list[i])
            new_lys_isotope_charge.append(0)
            lys_tag_list.append(0)
        else:
            if visited_tag[cluster_index] == 0:
                charge = original_lys_isotope_charge[i]
                new_peaks_mz = original_lys_mz_list[i] * charge - (charge - 1) * parameters.atom_mass['PROTON']
                new_original_lys_mz_list.append(new_peaks_mz)
                new_original_lys_intensity_list.append(original_lys_intensity_list[i])
                new_lys_isotope_charge.append(1)
                lys_tag_list.append(0.5)
                visited_tag[cluster_index] = 1

    lys_remove_isotope_mz_list = new_original_lys_mz_list
    lys_remove_isotope_intensity_list = new_original_lys_intensity_list


    return try_remove_isotope_mz_list ,try_remove_isotope_intensity_list ,\
           lys_remove_isotope_mz_list ,lys_remove_isotope_intensity_list


def correct_pepmass_function(mz_np , intensity_np, pepmass):
    '''

    :param mz_np:
    :param intensity_np:
    :param pepmass: MH+
    :return:
    '''
    pair_peaks = []
    pepmass_mz_exp = [pepmass]
    target_mz_theoretical = pepmass + parameters.atom_mass['PROTON']
    for i, pre_mz in enumerate(mz_np):
        sum_mz = pre_mz + mz_np[i + 1:]
        tag = (abs(sum_mz - target_mz_theoretical) <= parameters.peaks_sum_mass_error).astype(int)
        for j, post_mz in enumerate(mz_np[i + 1:]):
            if tag[j] == 1:
                pair_peaks.append((pre_mz, post_mz))
                pepmass_mz_exp.append(sum_mz[j] - parameters.atom_mass['PROTON'])
                intensity_np[i] *= 1.5
                intensity_np[i+j+1] *= 1.5

    return intensity_np , np.mean(pepmass_mz_exp)

def normalized_intensity_function(intensity_list):
    '''
    normalized intensity by their rank and map to exp_function.
    :param intensity_list:
    :return:
    '''

    intensity_np = np.array(intensity_list)
    minus_intensity_np = (-1) * intensity_np
    rank = np.argsort(np.argsort(minus_intensity_np)) + 1
    normalized_intensity_np = rank/len(rank)

    normalized_intensity_np = parameters.intensity_pdf(normalized_intensity_np)
    normalized_intensity_list = list(normalized_intensity_np)

    return normalized_intensity_list

def merge_similar_mass_function(mass_np,intensity_np : np.ndarray = None):
    '''
    assert: the mass_np is increasing one by one !!!!
            the position of intensity is accordance with the mass_np.
    :param mass_np:
    :param intensity_np:
    :return:
    '''
    if intensity_np is not None:
        new_mass_list = []
        new_intensity_list = []
        length = len(mass_np)
        i = 0
        while i < length-1:
            mass = mass_np[i]
            next_mass = mass_np[i+1]
            if abs(mass - next_mass) <= parameters.peaks_merge_tolerance:
                delta_mass = abs(mass - mass_np[i+1:])
                similar_mass = np.where(delta_mass<= parameters.peaks_merge_tolerance , mass_np[i+1:] , 0)
                similar_number = len(np.nonzero(similar_mass)[0])
                average_mass = float(sum(mass_np[i:i + similar_number + 1])/(similar_number + 1))
                new_mass_list.append(average_mass)
                sum_intensity = float(max(intensity_np[i:i + similar_number + 1]))
                new_intensity_list.append(sum_intensity)
                i = i + similar_number + 1
            else:
                new_mass_list.append(mass)
                new_intensity_list.append(intensity_np[i])
                i = i + 1
                continue

        new_mass_list.append(mass_np[-1])
        new_intensity_list.append(intensity_np[-1])

        new_mass_np = np.array(new_mass_list)
        new_intensity_np = np.array(new_intensity_list)

        return new_mass_np, new_intensity_np

    else:
        new_mass_list = []
        length = len(mass_np)
        i = 0
        while i < length - 1:
            mass = mass_np[i]
            next_mass = mass_np[i + 1]
            if abs(mass - next_mass) <= parameters.peaks_merge_tolerance:
                delta_mass = abs(mass - mass_np[i + 1:])
                similar_mass = np.where(delta_mass <= parameters.peaks_merge_tolerance, mass_np[i + 1:], 0)
                similar_number = len(np.nonzero(similar_mass)[0])
                average_mass = float(sum(mass_np[i:i + similar_number + 1]) / (similar_number + 1))
                new_mass_list.append(average_mass)
                i = i + similar_number + 1
            else:
                new_mass_list.append(mass)
                i = i + 1
                continue

        new_mass_list.append(mass_np[-1])

        new_mass_np = np.array(new_mass_list)
        return new_mass_np

def merge_aion_mass_function(mass_np,intensity_np):
    '''
    assert: the mass_np is increasing one by one !!!!
            the position of intensity is accordance with the mass_np.
            please confirm that had already uesd merge_similar_mass_function before using this function.
    :param mass_np:
    :param intensity_np:
    :return:
    '''
    assert len(mass_np) == len(intensity_np)
    new_mass_list = []
    new_intensity_list = []
    length = len(mass_np)

    if length == 0 :
        return np.array(new_mass_list) , np.array(new_intensity_list)

    i = 0
    while i < length-1:
        mass = mass_np[i]
        j = 1
        while True:

            if i+j > length -1 :
                i = i + 1
                break

            next_mass = mass_np[i+j]
            if abs(mass - next_mass) <= parameters.atom_mass['CO'] - parameters.peaks_merge_tolerance:
                j = j + 1
                continue
            elif abs(mass - next_mass) <= parameters.atom_mass['CO'] + parameters.peaks_merge_tolerance:
                average_mass = float( (mass + parameters.atom_mass['CO'] + next_mass)/2.0)
                sum_intensity = float(max(intensity_np[i],intensity_np[i + j]) + math.sqrt(min(intensity_np[i],intensity_np[i + j])))
                mass_np[i + j] = average_mass
                intensity_np[i + j] = sum_intensity
                i = i + 1
                break
            else:
                new_mass_list.append(mass)
                new_intensity_list.append(intensity_np[i])
                i = i + 1
                break

    new_mass_list.append(mass_np[-1])
    new_intensity_list.append(intensity_np[-1])

    new_mass_np = np.array(new_mass_list)
    new_intensity_np = np.array(new_intensity_list)

    return new_mass_np , new_intensity_np

def local_rank_of_intensity_function(real_mz_np , real_intensity_np):

    peaks_number = len(real_mz_np)
    local_rank_np = np.zeros(peaks_number)

    for i,mz_center in enumerate(real_mz_np):
        intensity_center = real_intensity_np[i]
        heigher_peaks_number = 0
        total_peaks_number = 1
        lower_index = i - 1
        upper_index = i + 1
        while True:
            if lower_index < 0:
                break
            if real_mz_np[lower_index] < mz_center - 100:
                break
            else:
                if real_intensity_np[lower_index] > intensity_center:
                    heigher_peaks_number = heigher_peaks_number + 1
                total_peaks_number = total_peaks_number + 1
                lower_index = lower_index - 1
        while True:
            if upper_index > peaks_number - 1:
                break
            if real_mz_np[upper_index] > mz_center + 100:
                break
            else:
                if real_intensity_np[upper_index] > intensity_center:
                    heigher_peaks_number = heigher_peaks_number + 1
                total_peaks_number = total_peaks_number + 1
                upper_index = upper_index + 1

        local_rank_np[i] = heigher_peaks_number + 1

    local_rank_np = local_rank_np/peaks_number
    local_score_np = parameters.intensity_pdf(local_rank_np)

    return local_score_np


###############################################################################################################################################
#transfer the form of result

def transfer_seq_str_to_list(sequence : str , target_index = None):
    if target_index == None:
        if '(' in sequence:
            i = 0
            length = len(sequence)
            sequence_list = []
            while True:
                if i > length - 1:
                    break
                if i == length - 1:
                    sequence_list.append(sequence[i])
                    break
                if sequence[i+1] != '(':
                    sequence_list.append(sequence[i])
                    i = i + 1
                else:
                    j = 1
                    while sequence[i+j] != ')':
                        j = j + 1
                    sequence_list.append(sequence[i:i+j+1])
                    i = i + j + 1
        else:
            sequence_list = list(sequence)

        return sequence_list

    else:
        if '(' in sequence:
            i = 0
            length = len(sequence)
            sequence_list = []
            while True:
                if i == target_index:
                    return_index = len(sequence_list)
                if i > length - 1:
                    break
                if i == length - 1:
                    sequence_list.append(sequence[i])
                    break
                if sequence[i+1] != '(':
                    sequence_list.append(sequence[i])
                    i = i + 1
                else:
                    j = 1
                    while sequence[i+j] != ')':
                        j = j + 1
                    sequence_list.append(sequence[i:i+j+1])
                    i = i + j + 1
            return sequence_list , return_index
        else:
            sequence_list = list(sequence)
            return sequence_list , target_index


################################################################################################################################
#search
def find_predecessor_function(current_index , adjacen_matrix_01_np):
    return np.nonzero(adjacen_matrix_01_np[current_index][:current_index])[0]

class My_PriorityQueue(object):
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        """
        队列由 (priority, index, item) 形式组成
        priority 增加 "-" 号是因为 heappush 默认是最小堆
        index 是为了当两个对象的优先级一致时，按照插入顺序排列
        """
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        """
        弹出优先级最高的对象
        """
        return heapq.heappop(self._queue)[-1]

    def qsize(self):
        return len(self._queue)

    def empty(self):
        return True if not self._queue else False

def pdag_for_find_k_longest_initial_path_function(MaxPQ, try_signal_C_intensity_np, adjacen_matrix_01_np, adjacen_edge_score_matrix_list, path_index , cnt, L, prefix, v , k):
    if k < cnt[v]:
        return True
    if cnt[v] == 0:
        predecessor_index_np = find_predecessor_function(v, adjacen_matrix_01_np)
        if len(predecessor_index_np) != 0:
            for u in predecessor_index_np:
                pdag_for_find_k_longest_initial_path_function(MaxPQ, try_signal_C_intensity_np, adjacen_matrix_01_np, adjacen_edge_score_matrix_list,path_index , cnt, L, prefix, u , 0)
                cost = L[u,0] + try_signal_C_intensity_np[v] + adjacen_edge_score_matrix_list[u][v]
                MaxPQ[v].push((cost,u,0),cost)
        else:
            return False
    else:
        if v == 0:
            return False
        (u,j) = prefix[v][k-1]
        if pdag_for_find_k_longest_initial_path_function(MaxPQ, try_signal_C_intensity_np, adjacen_matrix_01_np, adjacen_edge_score_matrix_list, path_index , cnt, L, prefix, u , j+1) == True:
            cost = L[u,j+1] + try_signal_C_intensity_np[v] + adjacen_edge_score_matrix_list[u][v]
            MaxPQ[v].push((cost,u,j+1),cost)
        elif MaxPQ[v].empty():
            return False
    (cost,u,j) = MaxPQ[v].pop()
    L[v,k] = cost
    cnt[v] = cnt[v] + 1
    prefix[v][k] = (u,j)
    return True

def find_k_longest_initial_path_function(try_signal_C_intensity_np, try_signal_C_mass_np, adjacen_matrix_01_np, adjacen_edge_score_matrix_list , topk):
    signal_peaks_number = len(try_signal_C_mass_np)
    MaxPQ = [My_PriorityQueue() for i in range(signal_peaks_number)]
    path_index = np.zeros((topk , signal_peaks_number))
    cnt = np.zeros(signal_peaks_number)
    prefix = [[(-1,-1) for j in range(topk)] for i in range(signal_peaks_number)]
    #L = np.ones((signal_peaks_number,topk)) * -float('inf')
    L = np.zeros((signal_peaks_number,topk))
    L[0,0] = 1
    cnt[0] = 1

    end_tag = False
    for i in range(topk):
        if pdag_for_find_k_longest_initial_path_function(MaxPQ, try_signal_C_intensity_np, adjacen_matrix_01_np, adjacen_edge_score_matrix_list, path_index , cnt, L, prefix, signal_peaks_number - 1 , i) == False:
            break
        else:
            n = 0
            u = signal_peaks_number - 1
            j = i
            while u != 0:
                path_index[i , n] = u
                n += 1
                (u,j) = prefix[u][j]
                if (-1,-1) == (u , j):
                    end_tag = True
                    path_index[i] = np.zeros(signal_peaks_number)
                    break
            if end_tag == True:
                break


    return path_index,L

def get_aa_permutations(aa_permutations_list,adjacen_matrix_list , current_index , next_index):

    aa_permutations = []
    for index in adjacen_matrix_list[current_index][next_index]:
        if index == -1:
            assert False , 'the initial path is not continuous.'
        else:
            aa_permutations = aa_permutations + aa_permutations_list[index]

    return aa_permutations

def pdag_path_to_seq_function_for_denovo(pdag_path , graph_try_C_mz_np):
    path_mass_list = []
    path_index_list = []
    for i,path in enumerate(pdag_path):
        if len(np.nonzero(path)[0]) == 0:
            break
        end_index = path.tolist().index(0)
        path = path[:end_index + 1][::-1].astype(int)
        path_index_list.append(list(path))
        path_mass_list.append(graph_try_C_mz_np[path].tolist())
    return path_mass_list,path_index_list

def pdag_for_find_k_longest_candidate_path_function(MaxPQ, predecessor_index,  peaks_score_list,  cnt, L, prefix, v , k):
    if k < cnt[v]:
        return True
    if cnt[v] == 0:
        predecessor_index_np = predecessor_index[v]
        if len(predecessor_index_np) != 0:
            for u in predecessor_index_np:
                pdag_for_find_k_longest_candidate_path_function(MaxPQ, predecessor_index,  peaks_score_list,  cnt, L, prefix, u , 0)
                cost = L[u,0] + peaks_score_list[v]
                MaxPQ[v].push((cost,u,0),cost)
        else:
            return False
    else:
        if v == 0:
            return False
        (u,j) = prefix[v][k-1]
        if pdag_for_find_k_longest_candidate_path_function(MaxPQ, predecessor_index, peaks_score_list,  cnt, L, prefix, u , j+1) == True:
            cost = L[u,j+1] + peaks_score_list[v]
            MaxPQ[v].push((cost,u,j+1),cost)
        elif MaxPQ[v].empty():
            return False
    (cost,u,j) = MaxPQ[v].pop()
    L[v,k] = cost
    cnt[v] = cnt[v] + 1
    prefix[v][k] = (u,j)
    return True

def find_k_longest_candidate_path_function(candidate_path_list,sumscore_list,aa_score_list,topk):
    candidate_subseq_number_list = [len(item) for item in candidate_path_list]

    peaks_subseq_list = []
    peaks_score_list = []
    peaks_aa_score_list = []
    for i,item in enumerate(sumscore_list):
        peaks_score_list.extend(item)
        peaks_subseq_list.extend(candidate_path_list[i])
        peaks_aa_score_list.extend(aa_score_list[i])
    peaks_score_list = [0] + peaks_score_list + [0]
    peaks_subseq_list = ['O'] + peaks_subseq_list + ['O']
    peaks_aa_score_list = [[0]] + peaks_aa_score_list + [[0]]

    peaks_number = sum(candidate_subseq_number_list) + 2
    assert peaks_number == len(peaks_score_list)
    candidate_subseq_number_cumsum = np.cumsum(candidate_subseq_number_list)
    candidate_subseq_number_cumsum = np.append(0,candidate_subseq_number_cumsum)
    predecessor_index = {}
    for i,index in enumerate(candidate_subseq_number_cumsum):
        if i == 0:
            predecessor_index[0] = []
        else:
            if i == 1:
                last = [0]
                last_end = 0
            else:
                last_start = candidate_subseq_number_cumsum[i - 2] + 1
                last_end = candidate_subseq_number_cumsum[i - 1]
                last = list(range(last_start, last_end + 1, 1))
            current_start = last_end  + 1
            current_end = candidate_subseq_number_cumsum[i]
            current = list(range(current_start , current_end + 1 , 1))
            for current_index in current:
                predecessor_index[current_index] = last

            if i == len(candidate_subseq_number_cumsum) - 1:
                predecessor_index[peaks_number - 1] = current

    MaxPQ = [My_PriorityQueue() for i in range(peaks_number)]
    path_index = np.zeros((topk , peaks_number))
    cnt = np.zeros(peaks_number)
    prefix = [[(-1,-1) for j in range(topk)] for i in range(peaks_number)]
    L = np.ones((peaks_number,topk)) * -float('inf')
    L[0,0] = 0
    cnt[0] = 1

    end_tag = False
    for i in range(topk):
        if pdag_for_find_k_longest_candidate_path_function(MaxPQ, predecessor_index, peaks_score_list,  cnt, L, prefix, peaks_number - 1 , i) == False:
            break
        else:
            n = 0
            u = peaks_number - 1
            j = i
            while u != 0:
                path_index[i , n] = u
                n += 1
                (u,j) = prefix[u][j]
                if (-1,-1) == (u , j):
                    end_tag = True
                    path_index[i] = np.zeros(peaks_number)
                    break
            if end_tag == True:
                break

    path_index_list = []
    result_path = []
    result_sumscore = []
    result_aa_score = []
    for i, path in enumerate(path_index):
        if len(np.nonzero(path)[0]) == 0:
            break
        end_index = path.tolist().index(0)
        path = path[:end_index + 1][::-1].astype(int)
        path_index_list.append(list(path))
        result_path_Temp = []
        result_sumscore_Temp = 0
        result_aa_score_Temp = []
        for i,index in enumerate(path):
            if i == 0:
                continue
            if i == len(path) - 1:
                continue
            if '(' in peaks_subseq_list[index]:
                Temp = transfer_seq_str_to_list(peaks_subseq_list[index])
                result_path_Temp.extend(Temp)
            else:
                result_path_Temp.extend(peaks_subseq_list[index])
            result_sumscore_Temp += peaks_score_list[index]
            result_aa_score_Temp.extend(peaks_aa_score_list[index])
        result_path.append(result_path_Temp)
        result_sumscore.append(result_sumscore_Temp)
        result_aa_score.append(result_aa_score_Temp)

    return result_path , result_sumscore , result_aa_score