import os
import pickle
import numpy as np
import parameters

input_pfind_path = r"D:\ResearchAssistant@AMSS\DiNovoData\YeastRes0315\Yeast_try_pFind-Filtered[extract][v0315].spectra"
input_pfind_path = r"D:\MSData\Ecoli[v315]\pFind-Filtered_trypsin[extract][v0315].spectra"
input_pfind_path = r"D:\ResearchAssistant@AMSS\DiNovoData\Ecoli_trypsin-lysargiNase\trypsin-Res.spectra"
Pnovom_result = False
input_Pnovom_result_path = f"E:\\Fugroup\\pnovom\\pnovom\\data\\chigou\\data\\MC2155_nolabel\\pnovom_inputmgf\\Dinovo_result\\results.txt"
input_Pnovom_result_path = r"E:\MyPythonWorkSpace\DiNovo\test20230506-m\[DiNovo]pNovoMRes.txt"
Dinovo_result = True
input_Dinovo_result_path = r"E:\MyPythonWorkSpace\DiNovo\test20230321-99\validation\[DiNovo]MirrorSequencing[OnlyTP].candidate"
input_Dinovo_result_path = r"E:\MyPythonWorkSpace\DiNovo\test20230506\[DiNovo]MirrorSequencing.candidate"
input_Dinovo_result_path = r"D:\ResearchAssistant@AMSS\DiNovoData\Ecoli_trypsin-lysargiNase-new\[2-3]mirrorNovo.res"
def cal_Pnovom_result(Pnovom_result_path, pfind_path, pfind_location_dict):
    Mirror_number = 0
    candidate_0_number = 0
    illegal_number = 0
    topk_if_sequence_match_cum_np = np.zeros(parameters.output_topk)  # 累计topk测对数量(1~10)
    top1_aa_match_number = 0
    top1_target_length_number = 0
    top1_target_length_number_more_than_0_candidate = 0
    top1_denovo_length_number = 0
    with open(Pnovom_result_path, "r") as f:
        while True:

            line = f.readline()

            if len(line) <= 1:
                break

            assert line.startswith('======')

            line = f.readline().split('\t')

            tryTitle = line[0].lstrip('=').split('@')[0]

            target_sequence = find_target_sequence(pfind_path, tryTitle, pfind_location_dict)
            if target_sequence != 'NULL':  #can't find the sequence in pfind_file/unexpected mod can be found in sequence
                cal_tag = True
                Mirror_number += 1
                candidate_seq_number = int(line[-1])
                if candidate_seq_number == 0:
                    candidate_0_number += 1
                    top1_target_length_number += len(target_sequence)
                    continue
            else:
                illegal_number += 1
                cal_tag = False

            for i in range(candidate_seq_number):
                line = f.readline().split('\t')
                if cal_tag == False:
                    continue
                else:
                    if i == 0:
                        denovo_sequence = list(line[0])[1:]
                        denovo_sequence = transfer_mod_function(denovo_sequence)
                        num_match, if_seq_match = match_seq_function(target_sequence, denovo_sequence)
                        top1_aa_match_number += num_match
                        top1_denovo_length_number += len(denovo_sequence)
                        top1_target_length_number += len(target_sequence)
                        top1_target_length_number_more_than_0_candidate += len(target_sequence)
                        if if_seq_match:
                            topk_if_sequence_match_cum_np = topk_if_sequence_match_cum_np + 1
                            cal_tag = False
                    else:
                        denovo_sequence = list(line[0])[1:]
                        denovo_sequence = transfer_mod_function(denovo_sequence)
                        num_match, if_seq_match = match_seq_function(target_sequence, denovo_sequence)
                        if if_seq_match:
                            topk_if_sequence_match_cum_np[i:] = topk_if_sequence_match_cum_np[i:] + 1
                            cal_tag = False

        topk_peptide_recall = topk_if_sequence_match_cum_np / Mirror_number if Mirror_number > 0 else 0
        top1_aa_recall = top1_aa_match_number / top1_target_length_number if top1_target_length_number > 0 else 0
        top1_aa_precision = top1_aa_match_number / top1_denovo_length_number if top1_denovo_length_number > 0 else 0

        with open(Pnovom_result_path + ".accuracy_stat.txt", "a+") as fw:
            fw.write('=' * 20)
            fw.write('top 1 aa performance on full dataset')
            fw.write('=' * 20)
            fw.write('\n')
            string = f"top1_Pnovom_aa_recall_total : {top1_aa_match_number} / {top1_target_length_number} = {top1_aa_recall}" + '\n' + \
                     f"top1_Pnovom_aa_precision_total : {top1_aa_match_number} / {top1_denovo_length_number} = {top1_aa_precision}" + '\n' + \
                     f"Pnovom without sequence : {candidate_0_number}" + '\n' + \
                     f"Pnovom illegal sequence : {illegal_number}" + '\n'
            fw.write(string)

            fw.write('=' * 20)
            fw.write('topk peptide recall on full dataset')
            fw.write('=' * 20)
            fw.write('\n')
            string = f"topk : peptide_recall"
            fw.write(string)
            fw.write('\n')
            for i in range(parameters.output_topk):
                string = f" {i + 1}  : {topk_if_sequence_match_cum_np[i]} / {Mirror_number} = {topk_peptide_recall[i]} " + '\n'
                fw.write(string)


            topk_peptide_recall = topk_if_sequence_match_cum_np / (Mirror_number - candidate_0_number)
            top1_aa_recall = top1_aa_match_number / top1_target_length_number_more_than_0_candidate

            fw.write('=' * 20)
            fw.write('top 1 aa performance on dataset more than 0 sequence')
            fw.write('=' * 20)
            fw.write('\n')
            string = f"top1_Pnovom_aa_recall_total : {top1_aa_match_number} / {top1_target_length_number_more_than_0_candidate} = {top1_aa_recall}" + '\n' + \
                     f"top1_Pnovom_aa_precision_total : {top1_aa_match_number} / {top1_denovo_length_number} = {top1_aa_precision}" + '\n'
            fw.write(string)

            fw.write('=' * 20)
            fw.write('topk peptide recall on dataset more than 0 sequence')
            fw.write('=' * 20)
            fw.write('\n')
            string = f"topk : peptide_recall"
            fw.write(string)
            fw.write('\n')
            for i in range(parameters.output_topk):
                string = f" {i + 1}  : {topk_if_sequence_match_cum_np[i]} / {Mirror_number - candidate_0_number} = {topk_peptide_recall[i]} " + '\n'
                fw.write(string)

def cal_Dinovo_result(Dinovo_result_path, pfind_path, pfind_location_dict):
    Mirror_number = 0
    candidate_0_number = 0
    illegal_number = 0
    topk_if_sequence_match_cum_np = np.zeros(parameters.output_topk)  # 累计topk测对数量(1~10)
    top1_aa_match_number = 0
    top1_target_length_number = 0
    top1_target_length_number_more_than_0_candidate = 0
    top1_denovo_length_number = 0
    with open(Dinovo_result_path, "r") as f:
        header = f.readline()
        header = f.readline()
        cnt,cal_tag = 0,True
        last_line_tag = ''
        while True:
            cnt, result_line = cnt+1, f.readline()
            if len(result_line) <= 1:
                if last_line_tag != '':
                    if cal_tag == True:
                        candidate_0_number += 1
                        top1_target_length_number += len(target_sequence)
                break
            result_line = str(result_line).split('\t')
            if result_line[0] == '':
                if cal_tag == False:
                    last_line_tag = result_line[0]
                    continue
                else:
                    rank = int(result_line[1])
                    if rank == 1:
                        denovo_sequence, denovo_mod = result_line[3], result_line[4]
                        denovo_sequence = transfer_seq_str_to_list(denovo_sequence, denovo_mod)
                        # print("[res line num]", cnt, "\n\tpFind:", target_sequence, "\n\tDiNovo:", denovo_sequence)
                        num_match, if_seq_match = match_seq_function(target_sequence, denovo_sequence)
                        top1_aa_match_number += num_match
                        top1_denovo_length_number += len(denovo_sequence)
                        top1_target_length_number += len(target_sequence)
                        top1_target_length_number_more_than_0_candidate += len(target_sequence)
                        if if_seq_match:
                            topk_if_sequence_match_cum_np = topk_if_sequence_match_cum_np + 1
                            cal_tag = False
                    else:
                        denovo_sequence, denovo_mod = result_line[3], result_line[4]
                        denovo_sequence = transfer_seq_str_to_list(denovo_sequence, denovo_mod)
                        num_match, if_seq_match = match_seq_function(target_sequence, denovo_sequence)
                        if if_seq_match:
                            topk_if_sequence_match_cum_np[rank - 1:] = topk_if_sequence_match_cum_np[rank - 1:] + 1
                            cal_tag = False
                    last_line_tag = result_line[0]
            else:

                if last_line_tag != '' and cal_tag == True:
                    candidate_0_number += 1
                    top1_target_length_number += len(target_sequence)

                tryTitle = result_line[0]
                target_sequence = find_target_sequence(pfind_path, tryTitle, pfind_location_dict)
                if target_sequence != 'NULL':
                    cal_tag = True
                    Mirror_number += 1
                else:
                    illegal_number += 1
                    cal_tag = False
                last_line_tag = result_line[0]

        topk_peptide_recall = topk_if_sequence_match_cum_np / Mirror_number if Mirror_number > 0 else ["NaN"] * len(topk_if_sequence_match_cum_np)
        top1_aa_recall = top1_aa_match_number / top1_target_length_number if top1_target_length_number > 0 else "NaN"
        top1_aa_precision = top1_aa_match_number / top1_denovo_length_number if top1_denovo_length_number > 0 else "NaN"

        with open(Dinovo_result_path + ".accuracy_stat.txt", "a+") as fw:
            fw.write('=' * 20)
            fw.write('top 1 aa performance on full dataset')
            fw.write('=' * 20)
            fw.write('\n')
            string = f"top1_Dinovo_aa_recall_total : {top1_aa_match_number} / {top1_target_length_number} = {top1_aa_recall}" + '\n' + \
                     f"top1_Dinovo_aa_precision_total : {top1_aa_match_number} / {top1_denovo_length_number} = {top1_aa_precision}" + '\n' + \
                     f"Dinovo without sequence : {candidate_0_number}" + '\n' + \
                     f"Dinovo illegal sequence : {illegal_number}" + '\n'
            fw.write(string)

            fw.write('=' * 20)
            fw.write('topk peptide recall on full dataset')
            fw.write('=' * 20)
            fw.write('\n')
            string = f"topk : peptide_recall"
            fw.write(string)
            fw.write('\n')
            for i in range(parameters.output_topk):
                string = f" {i + 1}  : {topk_if_sequence_match_cum_np[i]} / {Mirror_number} = {topk_peptide_recall[i]} " + '\n'
                fw.write(string)

            topk_peptide_recall = topk_if_sequence_match_cum_np / (Mirror_number - candidate_0_number)
            top1_aa_recall = top1_aa_match_number / top1_target_length_number_more_than_0_candidate if top1_target_length_number_more_than_0_candidate > 0 else "NaN"

            fw.write('=' * 20)
            fw.write('top 1 aa performance on dataset more than 0 candidate')
            fw.write('=' * 20)
            fw.write('\n')
            string = f"top1_Dinovo_aa_recall_total : {top1_aa_match_number} / {top1_target_length_number_more_than_0_candidate} = {top1_aa_recall}" + '\n' + \
                     f"top1_Dinovo_aa_precision_total : {top1_aa_match_number} / {top1_denovo_length_number} = {top1_aa_precision}" + '\n'
            fw.write(string)

            fw.write('=' * 20)
            fw.write('topk peptide recall on dataset more than 0 candidate')
            fw.write('=' * 20)
            fw.write('\n')
            string = f"topk : peptide_recall"
            fw.write(string)
            fw.write('\n')
            for i in range(parameters.output_topk):
                string = f" {i + 1}  : {topk_if_sequence_match_cum_np[i]} / {Mirror_number - candidate_0_number} = {topk_peptide_recall[i]} " + '\n'
                fw.write(string)

def transfer_mod_function(sequence):
    for i, aa in enumerate(sequence):
        if aa == 'J':
            sequence[i] = 'M(+15.99)'
        if aa == 'C':
            sequence[i] = 'C(+57.02)'
    return sequence

def transfer_seq_str_to_list(sequence : str, mod:str):
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
    print(sequence_list)
    if mod:
        mod_list = mod.split(";")[:-1]
        print(sequence, mod)
        for mod_s in mod_list:
            site_s, mod_name = mod_s.split(",")
            sequence_list[int(site_s) - 1] += mod_name
    return sequence_list

def find_target_sequence(pfind_path, tryTitle , pfind_location_dict):

    with open(pfind_path, 'r') as result_f:
        if tryTitle not in pfind_location_dict.keys():
            return 'NULL'
        else:
            location = pfind_location_dict[tryTitle]
            result_f.seek(location)
            line = str(result_f.readline()).split()
            assert line[0] == tryTitle
            try_sequence = line[5]
            try_sequence = try_sequence.replace('I','L')
            try_sequence = list(try_sequence)
            modification = line[10]
            if len(modification) != 1:
                modification = modification.split(';')
                modification = modification[:-1]
                for item in modification:
                    site = int(item.split(',')[0])
                    modification_type = item.split(',')[1]
                    if modification_type not in parameters.modifications:
                        return 'NULL'
                    for i, mod in enumerate(parameters.modifications):
                        if modification_type == mod:
                            try_sequence[site - 1] = parameters.modifications_in_aa_dict_keys[i]
            return try_sequence


def match_seq_function(target, predicted):

    num_match = 0
    target_len = len(target)
    predicted_len = len(predicted)
    target_mass = [103.009185 if x == 'C' else parameters.aa_to_mass_dict[x] for x in target]
    target_mass_cum = np.cumsum(target_mass)
    predicted_mass = [parameters.aa_to_mass_dict[x] for x in predicted]
    predicted_mass_cum = np.cumsum(predicted_mass)

    i = 0
    j = 0
    while i < target_len and j < predicted_len:
        if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.05:
            if target[i] == predicted[j]:
                num_match += 1
            i += 1
            j += 1
        elif target_mass_cum[i] < predicted_mass_cum[j]:
            i += 1
        else:
            j += 1

    if_seq_match = num_match == target_len and num_match == predicted_len

    return num_match , if_seq_match

def build_location_function(pfind_path):

    # 读取txt文件，创建result的location字典
    location_file = os.path.join(pfind_path + "_location")

    if os.path.exists(location_file)== False:

        result_location_dict = {}
        line = True
        count = 0
        with open(pfind_path, 'r') as f:
            header = f.readline()
            while line:
                current_location = f.tell()
                line = str(f.readline()).split()
                count += 1
                if len(line) == 0:
                    print('ends with empty line :', count)
                    break
                title = line[0]
                result_location_dict[title] = current_location

        location_file = pfind_path + "_location"
        with open(location_file, 'wb') as fw:
            pickle.dump(result_location_dict, fw)

    else:
        result_location_dict = {}
        with open(location_file, 'rb') as fr:
            result_location_dict = pickle.load(fr)

    return result_location_dict

if __name__ == "__main__":

    pfind_location_dict = build_location_function(input_pfind_path)
    if Dinovo_result:
        cal_Dinovo_result(input_Dinovo_result_path, input_pfind_path, pfind_location_dict)
    if Pnovom_result:
        cal_Pnovom_result(input_Pnovom_result_path, input_pfind_path, pfind_location_dict)




