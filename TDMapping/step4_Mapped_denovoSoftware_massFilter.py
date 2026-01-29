import os
import pickle
import pandas as pd
import numpy as np
import bisect
import re
from collections import defaultdict
from pyteomics import mztab
import config
from tqdm import tqdm
import multiprocessing
import common_function
from common_function import build_fasta_dict_function, build_index_dict_function, generate_reverse_fasta, \
    build_mgf_location_function, transfer_str_to_list_seq, if_evident_function, parallel_get_confident_info, \
    mapped_function_single, get_fasta_coverage_np_single, write_dict_to_csv, read_dict_from_csv, \
    merge_target_and_decoy_fasta_seq, calculate_mass

# CasaNovo/PEAKS/PointNovo/pNovo3
topk_peaks = config.topk_peaks
# singleSpec_tool_list = ["pNovo3","PEAKS","PointNovo","CasaNovo","CasaNovo","pNovo3","PEAKS","PointNovo","CasaNovo","CasaNovo"]
# singleSpec_tool_list = ["CasaNovo","CasaNovo","CasaNovo","CasaNovo"]
singleSpec_tool_list = ["GCNovo","GCNovo"]
fasta_file = config.fasta_file
# massFilter_list = [746,795,810,770,770,777,822,857,775,775]
massFilter_list = [748, 764]
SingleSpec_folder_list = [
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pnovo_output\lysC\result",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\peaks_deepnovo_output\lysC\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pointnovo_output\lysC\23charge\test",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\9species_model\lysC\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\massiveKB_model\lysC\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\GraphNovo_output\lysC",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\PrimeNovo_output\lysC",
    r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MirrorNovo_output\single\try",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pnovo_output\lysN\result",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\peaks_deepnovo_output\lysN\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pointnovo_output\lysN\23charge\test",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\9species_model\lysN\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\massiveKB_model\lysN\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\GraphNovo_output\lysN"
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\PrimeNovo_output\lysN",
    r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MirrorNovo_output\single\lys",
]
peaks_suffix_name = common_function.peaks_suffix_name

# 计算离子覆盖率时用到的参数
if_use_ppm = config.if_use_ppm
fragment_tolerance_ppm = config.fragment_tolerance_ppm
fragment_tolerance_da = config.fragment_tolerance_da
modifications = config.modifications
modifications_in_aa_dict_keys = config.modifications_in_aa_dict_keys
pure_aa_list = config.pure_aa_list
modifications_other = config.modifications_other
modifications_other_in_aa_dict_keys = config.modifications_other_in_aa_dict_keys
modifications_other_mass = config.modifications_other_mass
modifications_DiNovo = config.modifications_DiNovo
aa_to_mass_dict = config.aa_to_mass_dict
atom_mass = config.atom_mass


def generate_peaks_ori_title(row, filepath_to_title_dict):
    return filepath_to_title_dict[(row["Source File"].lstrip("[SCAN]").rstrip(f"{peaks_suffix_name}"), row["Scan"])]
    # return row["Source File"].lstrip("[SCAN]").rstrip(f"{peaks_suffix_name}") + "." + str(row["Scan"])[:-1] + "." + str(row["Scan"])[:-1] + "." + str(row["z"]) + "." + str(row["Scan"])[-1] + ".dta"


def build_location_function_PEAKS(file_path):
    print("loading location of peaks res...")
    # 如果mgf是pfind格式，那么就要这样转
    data_df = pd.read_csv(file_path)
    # 读取txt文件，创建result的location字典
    location_file = os.path.join(file_path + "_location")

    if os.path.exists(location_file) == False:

        result_location_dict = {}
        title_count = 0
        with open(file_path, 'r') as f:
            line = f.readline()  # header
            while True:
                if title_count % 10000 == 0:
                    print(f"\r", title_count, end="")
                current_location = f.tell()
                line = f.readline().split(",")
                if len(line) <= 1:
                    print(f"loading {title_count} specs of peaks...")
                    break
                title = line[-1].strip()
                if title not in result_location_dict.keys():
                    result_location_dict[title] = current_location
                    title_count += 1
                else:
                    print(title, result_location_dict[title])
                    assert False

        location_file = file_path + "_location"
        with open(location_file, 'wb') as fw:
            pickle.dump(result_location_dict, fw)

    else:
        result_location_dict = {}
        with open(location_file, 'rb') as fr:
            result_location_dict = pickle.load(fr)

    return result_location_dict


def build_location_function_pnovo(file_path):
    # if if_merge:
    #
    #     # 读取txt文件，创建result的location字典
    #     location_file = os.path.join(file_path + "_location")
    #
    #     if os.path.exists(location_file) == False:
    #
    #         result_location_dict = {}
    #         count = 0
    #         title_count = 0
    #         with open(file_path, 'r') as f:
    #
    #             while True:
    #                 current_location = f.tell()
    #                 line = f.readline().split()
    #                 if len(line) == 0:
    #                     count += 1
    #                     if count >= 10:
    #                         print("title_count: ", title_count)
    #                         break
    #                     else:
    #                         continue
    #                 count = 0
    #                 if line[0] == 'S':
    #                     title = line[1]
    #                     result_location_dict[title] = current_location
    #                     title_count += 1
    #
    #         location_file = file_path + "_location"
    #         with open(location_file, 'wb') as fw:
    #             pickle.dump(result_location_dict, fw)
    #
    #     else:
    #         result_location_dict = {}
    #         with open(location_file, 'rb') as fr:
    #             result_location_dict = pickle.load(fr)
    #
    #     return result_location_dict
    #
    # else:

    # 读取txt文件，创建result的location字典
    location_file = os.path.join(file_path + "_location")

    if os.path.exists(location_file) == False:

        result_location_dict = {}
        count = 0
        title_count = 0
        with open(file_path, 'r') as f:

            while True:
                current_location = f.tell()
                line = f.readline().split()
                if len(line) == 0:
                    count += 1
                    if count >= 10:
                        print("title_count: ", title_count)
                        break
                    else:
                        continue
                count = 0
                if line[0][0] == 'S':
                    title = line[1]
                    result_location_dict[title] = current_location
                    title_count += 1

        location_file = file_path + "_location"
        with open(location_file, 'wb') as fw:
            pickle.dump(result_location_dict, fw)

    else:
        result_location_dict = {}
        with open(location_file, 'rb') as fr:
            result_location_dict = pickle.load(fr)

    return result_location_dict


def read_seqs_from_casanovo(location, fi, title):
    #######################################################################################################################################################
    # #找到第一条符合母离子的肽段。如果该肽段含有未知修饰，就返回[""],否则返回该肽段
    fi.seek(location)
    tag = False
    while True:
        line = fi.readline().split("\t")
        if tag == False:
            assert title in line[-1], f"{title} != {line[0]}"
            tag = True
        else:
            if title not in line[-1]:
                return ['']
        seq, if_precursor_proper = line[0], line[-5]
        if if_precursor_proper == "True":
            seq = seq.replace("C+57.021", "C(+57.02)")
            seq = seq.replace("M+15.995", "M(+15.99)")
            seq = seq.replace("I", "L")
            # 如果+和(的数量不匹配，说明有未知修饰
            if seq.count("(") != seq.count("+") or "-" in seq:
                return ['']
            else:
                seq_list = transfer_str_to_list_seq(seq)
                return seq_list


#######################################################################################################################################################
# #找到第一条符合母离子且不含有未知修饰的肽段，如果都有位未知修饰就返回[""]
# fi.seek(location)
# tag = False
# while True:
#     line = fi.readline().split("\t")
#     if tag == False:
#         assert title in line[-1], f"{title} != {line[0]}"
#         tag = True
#     else:
#         if title not in line[-1]:
#             return ['']
#     seq, if_precursor_proper = line[0], line[-5]
#     if if_precursor_proper == "True":
#         seq = seq.replace("C+57.021","C(+57.02)")
#         seq = seq.replace("M+15.995","M(+15.99)")
#         seq = seq.replace("I","L")
#         #如果+和(的数量不匹配，说明有未知修饰
#         if seq.count("(") != seq.count("+") or "-" in seq:
#             continue
#         else:
#             seq_list = transfer_str_to_list_seq(seq)
#             return seq_list
#######################################################################################################################################################
# #找到第一条不含有未知修饰的肽段，如果都有位未知修饰就返回[""]


def if_precursor_proper(exp_mz, cal_mz, tol_ppm):
    if abs(exp_mz - cal_mz) / cal_mz * 1e6 <= tol_ppm:
        return True
    else:
        return False


def get_RunIndex_to_filepath_from_mztab(mztabFile):
    RunIndex_to_file_dict = {}
    with open(mztabFile, "r") as f:
        while True:
            line = f.readline().strip()
            if line.startswith("PSM"):
                break
            else:
                if line.startswith("MTD"):
                    line = line.split("\t")
                    if line[1].startswith("ms_run[") and line[1].endswith("]-location"):
                        RunIndex = int(line[1].lstrip("ms_run[").rstrip("]-location"))
                        file = line[2].lstrip("file:////").split("/")[-1]
                        RunIndex_to_file_dict[RunIndex] = file
    return RunIndex_to_file_dict


def extract_RunIndex(s):
    start = s.find('[') + 1
    end = s.find(']')
    return int(s[start:end])


def build_location_function_casanovo(file_path):
    print("loading location of casanovo res...")
    # 读取txt文件，创建result的location字典
    location_file = os.path.join(file_path + "_location")

    if os.path.exists(location_file) == False:

        result_location_dict = {}
        title_count = 0
        with open(file_path, 'r') as f:
            line = f.readline()  # header
            while True:
                if title_count % 10000 == 0:
                    print(f"\r", title_count, end="")
                current_location = f.tell()
                line = f.readline().split("\t")
                if len(line) <= 1:
                    print(f"loading {title_count} specs of peaks...")
                    break
                title = line[-1].strip()
                if title not in result_location_dict.keys():
                    result_location_dict[title] = current_location
                    title_count += 1
                else:
                    continue

        location_file = file_path + "_location"
        with open(location_file, 'wb') as fw:
            pickle.dump(result_location_dict, fw)

    else:
        result_location_dict = {}
        with open(location_file, 'rb') as fr:
            result_location_dict = pickle.load(fr)

    return result_location_dict


def build_location_function_pointnovo(file_path):
    print("loading location of pointnovo res...")
    # 读取txt文件，创建result的location字典
    location_file = os.path.join(file_path + "_location")

    if os.path.exists(location_file) == False:

        result_location_dict = {}
        title_count = 0
        with open(file_path, 'r') as f:
            line = f.readline()  # header
            while True:
                if title_count % 10000 == 0:
                    print(f"\r", title_count, end="")
                current_location = f.tell()
                line = f.readline().split("\t")
                if len(line) <= 1:
                    print(f"loading {title_count} specs of peaks...")
                    break
                title = line[0].strip()
                if title not in result_location_dict.keys():
                    result_location_dict[title] = current_location
                    title_count += 1
                else:
                    print(title, result_location_dict[title])
                    assert False

        location_file = file_path + "_location"
        with open(location_file, 'wb') as fw:
            pickle.dump(result_location_dict, fw)

    else:
        result_location_dict = {}
        with open(location_file, 'rb') as fr:
            result_location_dict = pickle.load(fr)

    return result_location_dict


def transfer_mod_function(type, sequence):
    if type == 'pnovom':
        for i, aa in enumerate(sequence):
            if aa == 'J':
                sequence[i] = 'M(+15.99)'
            if aa == 'C':
                sequence[i] = 'C(+57.02)'
            if aa == "I":
                sequence[i] = 'L'
        return sequence
    if type == 'pnovo':  # PNovo结果中I统一变成了L，所以这里不用再转换了
        for i, aa in enumerate(sequence):
            if aa == 'a':
                sequence[i] = 'M(+15.99)'
            if aa == 'C':
                sequence[i] = 'C(+57.02)'
        if sequence[0] == '0':
            sequence[0] = modifications_other_in_aa_dict_keys[0]
            sequence[1] = sequence[1].upper()
        if sequence[1] == 'C':
            sequence[1] = 'C(+57.02)'
        return sequence

    assert False, "unconpletment"


def read_seqs_from_pnovores(location, fi, title):
    seqs = []
    scores = []
    fi.seek(location)
    line = fi.readline().split()

    # if if_merge:
    #     assert title == line[1] and 'S' == line[0]
    # else:
    assert title == line[1] and 'S' == line[0][0]

    while True:
        line = fi.readline().split()
        if len(line) == 0:
            seq = ['']
            break
        else:
            seq = line[1]
            seq = transfer_mod_function('pnovo', list(seq))
            break

    return seq


def read_seqs_from_PEAKS(location, fi, title):
    fi.seek(location)
    line = fi.readline().split(",")
    assert title in line[-1], f"{title} != {line[-1]}"
    seq = line[2]
    seq = seq.replace("I", "L")
    seq_list = transfer_str_to_list_seq(seq)
    return seq_list


def read_seqs_from_pointnovo(location, fi, title):
    fi.seek(location)
    line = fi.readline().split("\t")
    assert title in line[0], f"{title} != {line[0]}"
    seq = line[2]
    if "N(Deamidation)" in seq or "Q(Deamidation)" in seq:
        return ['']
    seq = seq.replace("I", "L")
    seq = seq.replace("M(Oxidation)", "M(+15.99)")
    seq = seq.replace("C(Carbamidomethylation)", "C(+57.02)")
    seq_list = seq.split(",")
    return seq_list


if __name__ == "__main__":
    for i in range(len(singleSpec_tool_list)):

        SingleSpec_folder = SingleSpec_folder_list[i]
        singleSpec_tool = singleSpec_tool_list[i]
        massFilter = massFilter_list[i]

        fasta_seq = build_fasta_dict_function(fasta_file)
        fasta_seq_first_4aa_index = build_index_dict_function(fasta_seq)
        fasta_seq_decoy = generate_reverse_fasta(fasta_seq)
        fasta_seq_first_4aa_index_decoy = build_index_dict_function(fasta_seq_decoy)

        if not os.path.exists(SingleSpec_folder + f"\\title_to_denovo_info.res"):
            assert False
        else:
            print(f"loading title_to_denovo_info[{massFilter}Da].res file...")
            title_to_denovo_df = pd.read_csv(SingleSpec_folder + "\\title_to_denovo_info.res", sep="\t")
        if "Pep_mass_denovo" not in title_to_denovo_df.columns:
            title_to_denovo_df["Pep_mass_denovo"] = title_to_denovo_df.apply(lambda x: calculate_mass(x["denovoSeq"], if_delete_first_aa=False), axis=1)

        if not os.path.exists(SingleSpec_folder + f"\\fasta_coverage_denovo_info[{massFilter}Da].res"):
            seq_to_unique_dict, fasta_coverage_denovo_dict, denovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
                fasta_seq, title_to_denovo_df, "denovo", False, massFilter)
            write_dict_to_csv(fasta_coverage_denovo_dict,
                              SingleSpec_folder + f"\\fasta_coverage_denovo_info[{massFilter}Da].res",
                              fasta_seq, None, denovo_proteinName_to_uniqueSeq_dict)
        else:
            fasta_coverage_denovo_dict, denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                SingleSpec_folder + f"\\fasta_coverage_denovo_info[{massFilter}Da].res")

        if not os.path.exists(SingleSpec_folder + f"\\fasta_coverage_Evidentdenovo_info[{massFilter}Da].res"):
            _, fasta_coverage_Evidentdenovo_dict, Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
                fasta_seq, title_to_denovo_df, "denovo", True, massFilter)
            write_dict_to_csv(fasta_coverage_Evidentdenovo_dict,
                              SingleSpec_folder + f"\\fasta_coverage_Evidentdenovo_info[{massFilter}Da].res", fasta_seq, None,
                              Evidentdenovo_proteinName_to_uniqueSeq_dict)
        else:
            fasta_coverage_Evidentdenovo_dict, Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                SingleSpec_folder + f"\\fasta_coverage_Evidentdenovo_info[{massFilter}Da].res")

        #######################################################################################################################################################
        # 谱图肽段水平
        evident_spec_num_denovo = 0
        evident_find_spec_num_denovo = 0
        NonEvident_spec_num_denovo = 0
        NonEvident_find_spec_num_denovo = 0
        denovoSeq_to_info_dict = {}
        dbSeq_to_info_dict = {}
        for index, row in title_to_denovo_df.iterrows():
            if index % 1000 == 0:
                print(f"\r", index, "/", len(title_to_denovo_df), end="")
            title, denovo_seq, ifEvident, ifFind = row["title"], row["denovoSeq"], row["ifEvident_denovo"], row["ifFind_denovo"]
            ifFindDecoy = row["ifFindDecoy_denovo"]
            pep_mass = row["Pep_mass_denovo"]
            if pep_mass < massFilter:
                continue

            if ifEvident == True:
                evident_spec_num_denovo += 1
                if (ifFind == True) and (ifFindDecoy == False):
                    evident_find_spec_num_denovo += 1
            else:
                NonEvident_spec_num_denovo += 1
                if (ifFind == True) and (ifFindDecoy == False):
                    NonEvident_find_spec_num_denovo += 1
            if denovo_seq == denovo_seq:  # 不为Na
                if denovo_seq not in denovoSeq_to_info_dict:
                    denovoSeq_to_info_dict[denovo_seq] = [ifEvident, ifFind, ifFindDecoy,
                                                          [(title, ifEvident, ifFind, ifFindDecoy)]]
                else:
                    ifEvident_new = denovoSeq_to_info_dict[denovo_seq][0] | ifEvident
                    ifFind_new = denovoSeq_to_info_dict[denovo_seq][1] | ifFind
                    ifFindDecoy_new = denovoSeq_to_info_dict[denovo_seq][2] | ifFindDecoy
                    denovoSeq_to_info_dict[denovo_seq] = [ifEvident_new, ifFind_new, ifFindDecoy_new,
                                                          denovoSeq_to_info_dict[denovo_seq][3] + [
                                                              (title, ifEvident, ifFind, ifFindDecoy)]]

        intsection_info_fout = open(SingleSpec_folder + f"\\info[{massFilter}Da].res", "a+")
        total_spec_num = len(title_to_denovo_df)
        string = f"total_spec_num\t{len(title_to_denovo_df)}\n"
        intsection_info_fout.write(string)
        string = f"denovoSeq_evident_spec_num\t{evident_spec_num_denovo}\n"
        intsection_info_fout.write(string)
        string = f"denovoSeq_evident_find_spec_num\t{evident_find_spec_num_denovo}\n"
        intsection_info_fout.write(string)
        string = f"denovoSeq_NonEvident_spec_num\t{NonEvident_spec_num_denovo}\n"
        intsection_info_fout.write(string)
        string = f"denovoSeq_NonEvident_find_spec_num\t{NonEvident_find_spec_num_denovo}\n"
        intsection_info_fout.write(string)
        string = f"denovoSeq_find_spec_num\t{evident_find_spec_num_denovo + NonEvident_find_spec_num_denovo}\n"
        intsection_info_fout.write(string)

        # 肽段
        denovoSeq_df = pd.read_csv(SingleSpec_folder + "\\denovoSeq_to_ifEvidentFind.res", sep="\t")
        denovoSeq_df["ifTry"] = True
        denovoSeq_df["Pep_mass"] = denovoSeq_df["denovoSeq"].apply(lambda seq: calculate_mass(seq, if_delete_first_aa=False))
        denovoSeq_df.to_csv(SingleSpec_folder + "\\denovoSeq_to_ifEvidentFind.res", sep="\t", index=False)

        denovoSeq_Find_set = set(denovoSeq_df[(denovoSeq_df["ifFind"] == True) & (denovoSeq_df["ifFindDecoy"] == False) & (denovoSeq_df["Pep_mass"] >= massFilter)]["denovoSeq"])
        denovoSeq_Find_Evident_set = set(denovoSeq_df[
                    (denovoSeq_df["ifFind"] == True) & (denovoSeq_df["ifFindDecoy"] == False)
                    & (denovoSeq_df["ifEvident"] == True) & (denovoSeq_df["Pep_mass"] >= massFilter)]["denovoSeq"])
        string = f"denovoSeq_Find_num\t{len(denovoSeq_Find_set)}\n"
        intsection_info_fout.write(string)
        string = f"denovoSeq_Find_Evident_num\t{len(denovoSeq_Find_Evident_set)}\n"
        intsection_info_fout.write(string)

        # 谱图
        denovo_find_spec_set = set(title_to_denovo_df[
                (title_to_denovo_df["ifFind_denovo"] == True) & (title_to_denovo_df["ifFindDecoy_denovo"] == False)
                & (title_to_denovo_df["Pep_mass_denovo"] >= massFilter)]["title"])
        denovo_find_spec_evident_set = set(title_to_denovo_df[
                (title_to_denovo_df["ifFind_denovo"] == True) & (title_to_denovo_df["ifFindDecoy_denovo"] == False) &
                (title_to_denovo_df["ifEvident_denovo"] == True) & (title_to_denovo_df["Pep_mass_denovo"] >= massFilter)]["title"])
        string = f"denovo_find_spec_num\t{len(denovo_find_spec_set)}\n"
        intsection_info_fout.write(string)
        string = f"denovo_find_spec_evident_num\t{len(denovo_find_spec_evident_set)}\n"
        intsection_info_fout.write(string)








