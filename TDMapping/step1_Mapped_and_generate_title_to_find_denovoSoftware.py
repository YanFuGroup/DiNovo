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
from common_function import build_fasta_dict_function, build_index_dict_function, generate_reverse_fasta, build_mgf_location_function, transfer_str_to_list_seq, if_evident_function, parallel_get_confident_info,\
    mapped_function_single, get_fasta_coverage_np_single, write_dict_to_csv, read_dict_from_csv, merge_target_and_decoy_fasta_seq,\
    build_singleSpectra_location_function, read_seqs_from_DiNovoSingleRes

#CasaNovo/PEAKS/PointNovo/pNovo3/GraphNovo/PrimeNovo
#如果是graphNovo，一定要给scan_mgf
topk_peaks = config.topk_peaks
# singleSpec_tool_list = ["pNovo3","PEAKS","PointNovo","CasaNovo","CasaNovo","pNovo3","PEAKS","PointNovo","CasaNovo","CasaNovo"]
singleSpec_tool_list = ["GCNovo","GCNovo"]
fasta_file = config.fasta_file
SingleSpec_file_list = [
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pnovo_output\lysC\result\pNovo[23charge].res",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\peaks_deepnovo_output\lysC\23charge\denovo.csv",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pointnovo_output\lysC\23charge\test\features.test.csv.deepnovo_denovo",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\9species_model\lysC\23charge\casanovo_output.mztab",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\massiveKB_model\lysC\23charge\casanovo_output.mztab",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\GraphNovo_output\lysC\Sample 1.deepnovo.csv",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\PrimeNovo_output\try\denovo.tsv",
    r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MirrorNovo_output\single\try\Yeast_trypsin.txt.beamsearch[23charge]",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pnovo_output\lysN\result\pNovo[23charge].res",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\peaks_deepnovo_output\lysN\23charge\denovo.csv",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pointnovo_output\lysN\23charge\test\features.test.csv.deepnovo_denovo",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\9species_model\lysN\23charge\casanovo_output.mztab",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\massiveKB_model\lysN\23charge\casanovo_output.mztab"
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\PrimeNovo_output\lys\denovo.tsv"
    r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MirrorNovo_output\single\lys\Yeast_lysargiNase.txt.beamsearch[23charge]"
]
GCNovo_tag = [0,1]
mgf_list = [
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\mgf\lysC\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\mgf\lysC\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\mgf\lysC\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\mgf\lysC\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\mgf\lysC\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\mgf\lysC\scan",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\mgf\try\23charge",
    r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\mgf\try\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\mgf\lysN\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\mgf\lysN\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\mgf\lysN\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\mgf\lysN\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\mgf\lysN\23charge"
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\mgf\lys\23charge"
    r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\mgf\lys\23charge"
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


def build_location_function_PrimeNovo(file_path):
    print("loading location of PrimeNovo res...")
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


def build_location_function_PEAKS(file_path):
    print("loading location of peaks or graphnovo res...")
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
    seq = seq.replace("I","L")
    seq_list = transfer_str_to_list_seq(seq)
    return seq_list

def read_seqs_from_graphnovo(location, fi, title):
    fi.seek(location)
    line = fi.readline().split(",")
    assert title in line[-1], f"{title} != {line[-1]}"
    seq = line[2]
    seq = seq.replace("I","L")
    seq_list = transfer_str_to_list_seq(seq)
    return seq_list

def read_seqs_from_primenovo(location, fi, title):
    fi.seek(location)
    line = fi.readline().split("\t")
    assert title in line[0], f"{title} != {line[0]}"
    seq = line[1]
    seq = seq.replace("I","L")
    seq = seq.replace("C[+57.021]", "C(+57.02)")
    seq = seq.replace("M[+15.995]", "M(+15.99)")
    # 如果+和(的数量不匹配，说明有未知修饰
    if "[" in seq or "-" in seq:
        return ['']
    else:
        seq_list = transfer_str_to_list_seq(seq)
        return seq_list

def read_seqs_from_pointnovo(location, fi, title):
    fi.seek(location)
    line = fi.readline().split("\t")
    assert title in line[0], f"{title} != {line[0]}"
    seq = line[2]
    if "N(Deamidation)" in seq or "Q(Deamidation)" in seq:
        return ['']
    seq = seq.replace("I","L")
    seq = seq.replace("M(Oxidation)","M(+15.99)")
    seq = seq.replace("C(Carbamidomethylation)","C(+57.02)")
    seq_list = seq.split(",")
    return seq_list

def build_mgf_location_function_for_scanmgf_graphnovo(mgf_folder):

    if not os.path.exists(os.path.join(mgf_folder, "MGF_index.txt")):

        # 读取mgf文件，创建location字典
        path_list = os.listdir(mgf_folder)
        # path_list.sort()  # 对读取的路径进行排序

        spectrum_location_dict = {}
        filepathAndOrderNum_to_title = {}
        filepathAndScan_to_title = {}
        detected_file_count = 0
        for file_path in path_list:  # 挨个读取mgf文件
            if file_path[-4:] != ".mgf":
                continue

            suffix_name = file_path.split('_')[-1]

            print(f"Building location for: " + file_path)
            file_path = os.path.join(mgf_folder, file_path)
            detected_file_count += 1

            #统一换行符为\n,避免不同操作系统带来的影响
            with open(file_path, 'r') as file:
                content = file.read()
                content = content.replace('\r\n', '\n').replace('\r', '\n')
            with open(file_path, 'w') as file:
                file.write(content)

            # line = True
            count = 0
            line_num = 0
            current_location = 0
            spectrum_location = 0
            with open(file_path, 'r') as f:
                # for line in f:
                while True:
                    # current_location = f.tell()
                    line = f.readline()
                    line_num += 1

                    if len(line) == 0:
                        print('ends with empty line : ' + str(count))
                        break

                    if line[0] == "B" and line[:5] == "BEGIN":
                        spectrum_location = current_location
                        count += 1

                    elif line[0] == "C" and line[:6] == "CHARGE":
                        charge = int(line[7:].rstrip("\n").rstrip("+"))

                    elif line[0] == "T" and line[:5] == "TITLE":
                        title = line[6:-1]

                    elif line[0] == 'S' and line[:5] == "SCANS":
                        scan = int(line[6:].strip())

                    elif line[0] == "E" and line[:3] == "END":
                        spectrum_location_dict[title] = (file_path, spectrum_location, charge, scan)
                        filepathAndScan_to_title[(file_path, scan)] = title
                        # read_one_spec(title,spectrum_location_dict[title])

                    current_location += len(line) + 1
                    # print(repr(line), len(line))

        with open(os.path.join(mgf_folder, "MGF_index.txt"), 'w') as f:
            for key in spectrum_location_dict.keys():
                f.write(key + "\t" + spectrum_location_dict[key][0] + "\t" + str(spectrum_location_dict[key][1]) + "\t" + str(spectrum_location_dict[key][2]) + "\t" + str(spectrum_location_dict[key][3])  + "\n")
        with open(os.path.join(mgf_folder, 'filepathAndOrderNum_to_title'), 'wb') as fw:
            pickle.dump(filepathAndOrderNum_to_title, fw)
        with open(os.path.join(mgf_folder, 'filepathAndScan_to_title'), 'wb') as fw:
            pickle.dump(filepathAndScan_to_title, fw)

        return spectrum_location_dict, suffix_name, len(spectrum_location_dict), True, filepathAndOrderNum_to_title, filepathAndScan_to_title
    else:
        print("Loading location for mgf files...")
        spectrum_location_dict = {}
        file_path_dict = {}
        detected_spectrum_count = 0
        with open(os.path.join(mgf_folder, "MGF_index.txt"), 'r') as f:
            for line in f:
                key, file_path, location, charge, scan = line.strip().split("\t")
                spectrum_location_dict[key] = (file_path, int(location), int(charge),  int(scan))
                file_path_dict[file_path] = 1
                detected_spectrum_count += 1

        filepathAndOrderNum_to_title_file = os.path.join(mgf_folder, 'filepathAndOrderNum_to_title')
        with open(filepathAndOrderNum_to_title_file, 'rb') as fr:
            filepathAndOrderNum_to_title = pickle.load(fr)
        filepathAndScan_to_title_file = os.path.join(mgf_folder, 'filepathAndScan_to_title')
        with open(filepathAndScan_to_title_file, 'rb') as fr:
            filepathAndScan_to_title = pickle.load(fr)

        # 读取mgf文件，创建location字典
        path_list = os.listdir(mgf_folder)
        for file_path in path_list:  # 挨个读取mgf文件
            if file_path[-4:] != ".mgf":
                continue
            suffix_name = file_path.split('_')[-1]

        return spectrum_location_dict, suffix_name, detected_spectrum_count, False, filepathAndOrderNum_to_title, filepathAndScan_to_title

def generate_graphnovo_ori_title(row, filepath_to_title_dict, mgf_folder):
    return filepath_to_title_dict[(row["Source File"].rstrip(".mgf"), row["Scan"])]


if __name__ == "__main__":
    fasta_seq = build_fasta_dict_function(fasta_file)
    fasta_seq_first_4aa_index = build_index_dict_function(fasta_seq)
    fasta_seq_decoy = generate_reverse_fasta(fasta_seq)
    fasta_seq_first_4aa_index_decoy = build_index_dict_function(fasta_seq_decoy)

    for i in range(len(SingleSpec_file_list)):
        SingleSpec_tool = singleSpec_tool_list[i]
        SingleSpec_file = SingleSpec_file_list[i]
        mgf = mgf_list[i]
        print(f"processing {SingleSpec_tool}...")

        if SingleSpec_tool != "GraphNovo":
            spectrum_location_dict, suffix_name, filepathAndOrderNum_to_title_dict, _ = build_mgf_location_function(mgf + "\\")
        else:
            spectrum_location_dict, suffix_name, filepathAndOrderNum_to_title_dict, filepathAndScan_to_title_dict = build_mgf_location_function(mgf + "\\", True)

        if SingleSpec_tool == "pNovo3":
            SingleSpec_title_to_location = build_location_function_pnovo(SingleSpec_file)
        elif SingleSpec_tool == "GCNovo":
            if i in GCNovo_tag:
                SingleSpec_title_to_location = build_singleSpectra_location_function(SingleSpec_file,
                                                                                     "DiNovo_MirrorNovo")
            else:
                SingleSpec_title_to_location = build_singleSpectra_location_function(SingleSpec_file, "MirrorNovo")
        elif SingleSpec_tool == "PrimeNovo":
            SingleSpec_title_to_location = build_location_function_PrimeNovo(SingleSpec_file)
        elif SingleSpec_tool == "PEAKS":
            print("transfer try peaks res to my format...")
            data_df = pd.read_csv(SingleSpec_file)
            if "title" not in data_df.columns:
                data_df["title"] = data_df.apply(lambda x: generate_peaks_ori_title(x, filepathAndOrderNum_to_title_dict), axis=1)
                data_df.sort_values(by='title', inplace=True)
                SingleSpec_file = SingleSpec_file.replace(".csv", "[myFormat].csv")
                data_df.to_csv(SingleSpec_file, index=False)
            else:
                SingleSpec_file = SingleSpec_file.replace(".csv", "[myFormat].csv")
            SingleSpec_title_to_location = build_location_function_PEAKS(SingleSpec_file)
        elif SingleSpec_tool == "GraphNovo":
            print("transfer try graphnovo res to my format...")
            data_df = pd.read_csv(SingleSpec_file)
            if "title" not in data_df.columns:
                # graphnovo的scan号：如果在mgf中有scan=，那么就是该值
                data_df["title"] = data_df.apply(lambda x: generate_graphnovo_ori_title(x, filepathAndScan_to_title_dict, mgf),axis=1)
                data_df.sort_values(by='title', inplace=True)
                SingleSpec_file = SingleSpec_file.replace(".csv", "[myFormat].csv")
                data_df.to_csv(SingleSpec_file, index=False)
            else:
                SingleSpec_file = SingleSpec_file.replace(".csv", "[myFormat].csv")
            SingleSpec_title_to_location = build_location_function_PEAKS(SingleSpec_file)
        elif SingleSpec_tool == "PointNovo":
            SingleSpec_title_to_location = build_location_function_pointnovo(SingleSpec_file)
        elif SingleSpec_tool == "CasaNovo":
            RunIndex_to_file_dict = get_RunIndex_to_filepath_from_mztab(SingleSpec_file)
            myFormat_denovo_path = SingleSpec_file.replace(".mztab", "[myFormat].csv")

            # try
            if os.path.exists(myFormat_denovo_path):
                psm_data_df = pd.read_csv(myFormat_denovo_path, sep="\t")
            else:
                # 使用Pyteomics的mzTab模块打开并解析文件
                mztab_file = mztab.MzTab(SingleSpec_file)
                # 提取文件中的所有PSM（Peptide-Spectrum Match）信息
                for data in mztab_file:
                    if data[0] == "PSM":
                        psm_data_df = pd.DataFrame(data[1])
                        psm_data_df["Precursor_proper"] = psm_data_df.apply(
                            lambda x: if_precursor_proper(x["exp_mass_to_charge"], x["calc_mass_to_charge"], 20), axis=1)
                        psm_data_df.to_csv(myFormat_denovo_path, sep="\t", index=False, header=True)
            if "Title" not in psm_data_df.columns:
                psm_data_df["RunIndex"] = psm_data_df.spectra_ref.map(lambda x: extract_RunIndex(x))
                psm_data_df["FileTmp"] = psm_data_df.RunIndex.map(
                    lambda x: RunIndex_to_file_dict[x].rstrip(peaks_suffix_name))
                psm_data_df["OrderNum"] = psm_data_df.spectra_ref.map(lambda x: int(x[x.find('=') + 1:]))
                # 注意OrderNum是从0开始的，而location_dict是从1开始的
                psm_data_df["Title"] = psm_data_df.apply(
                    lambda x: filepathAndOrderNum_to_title_dict[(x["FileTmp"], x["OrderNum"] + 1)], axis=1)
                psm_data_df.to_csv(myFormat_denovo_path, sep="\t", index=False, header=True)
            SingleSpec_title_to_location = build_location_function_casanovo(myFormat_denovo_path)
            SingleSpec_file = myFormat_denovo_path
        else:
            assert False

        if not os.path.exists(os.path.dirname(SingleSpec_file) + "\\title_to_denovo_info.res"):
            fout = open(os.path.dirname(SingleSpec_file) + "\\title_to_denovo_info.res", "a+")
            fout.write("title\tdenovoSeq\tsite_tag_denovo\tsite_peaknum_denovo\tsite_score_denovo\tpep_scoremean_denovo\tpep_scoresum_denovo\tifEvident_denovo\tifFind_denovo\n")
            SingleSpec_fin = open(SingleSpec_file, 'r')
            print("reading mgf files ...")
            denovo_psm_list = []
            for filename in os.listdir(mgf):
                if filename.endswith(".mgf"):
                    file = mgf + f"\\{filename}"
                    print(f"reading {file}...")
                    with open(file, 'r') as f:
                        while True:
                            line = f.readline()
                            if len(line) == 0:
                                break
                            if line.startswith("TITLE="):
                                title = line.split('=')[1].rstrip()
                                if title in SingleSpec_title_to_location.keys():
                                    if SingleSpec_tool == "pNovo3":
                                        denovo_seq = read_seqs_from_pnovores(SingleSpec_title_to_location[title],
                                                                             SingleSpec_fin, title)
                                    elif SingleSpec_tool == "GCNovo":
                                        if title in SingleSpec_title_to_location.keys():
                                            if i in GCNovo_tag:
                                                denovo_seq, score = read_seqs_from_DiNovoSingleRes(
                                                    SingleSpec_title_to_location[title], SingleSpec_fin, title, "DiNovo_MirrorNovo")
                                            else:
                                                denovo_seq, score = read_seqs_from_DiNovoSingleRes(
                                                    SingleSpec_title_to_location[title], SingleSpec_fin, title, "MirrorNovo")
                                        else:
                                            denovo_seq = ['']
                                    elif SingleSpec_tool == "PEAKS":
                                        denovo_seq = read_seqs_from_PEAKS(SingleSpec_title_to_location[title],
                                                                          SingleSpec_fin, title)
                                    elif SingleSpec_tool == "PointNovo":
                                        denovo_seq = read_seqs_from_pointnovo(SingleSpec_title_to_location[title],
                                                                              SingleSpec_fin, title)
                                    elif SingleSpec_tool == "CasaNovo":
                                        denovo_seq = read_seqs_from_casanovo(SingleSpec_title_to_location[title],
                                                                             SingleSpec_fin, title)
                                    elif SingleSpec_tool == "GraphNovo":
                                        denovo_seq = read_seqs_from_graphnovo(SingleSpec_title_to_location[title],
                                                                          SingleSpec_fin, title)
                                    elif SingleSpec_tool == "PrimeNovo":
                                        denovo_seq = read_seqs_from_primenovo(SingleSpec_title_to_location[title],
                                                                             SingleSpec_fin, title)
                                    else:
                                        assert False
                                else:
                                    denovo_seq = ['']
                                denovo_psm_list.append((title, denovo_seq))
            denovo_res_dict = parallel_get_confident_info(denovo_psm_list, spectrum_location_dict, suffix_name, mgf)
            for i in range(len(denovo_psm_list)):
                title = denovo_psm_list[i][0]
                denovo_illegal = True
                if denovo_psm_list[i][1] != ['']:
                    denovoSeq = denovo_psm_list[i][1]
                    denovoSeq_str = "".join(denovo_psm_list[i][1])
                    denovo_psm = title + "@" + denovoSeq_str
                    denovo_illegal = False
                if not denovo_illegal:
                    (site_tag_denovo,
                     site_peaknum_denovo,
                     site_score_denovo,
                     pep_scoremean_denovo,
                     pep_scoresum_denovo,
                     intensity_list_denovo,
                     site_intensitytemp_denovo) = denovo_res_dict[denovo_psm]["site_tag"], denovo_res_dict[denovo_psm][
                        "site_peaknum"], denovo_res_dict[denovo_psm]["site_score"], denovo_res_dict[denovo_psm][
                        "pepscore_mean"], denovo_res_dict[denovo_psm]["pepscore_sum"], denovo_res_dict[denovo_psm][
                        "intensity_list"], denovo_res_dict[denovo_psm]["site_intensitytemp"]
                    sort_intensity_list_denovo = sorted(intensity_list_denovo)
                    rank = len(sort_intensity_list_denovo) - np.array(
                        [bisect.bisect_left(sort_intensity_list_denovo, i) for i in site_intensitytemp_denovo]) + 1
                    site_tag_denovo = np.array(
                        [1 if tag == 1 and rank[i] <= topk_peaks else 0 for i, tag in enumerate(site_tag_denovo)])
                    miss_num = np.sum(site_tag_denovo == 0)
                    coverage = 1 - miss_num / len(site_tag_denovo)
                    ifEvident_denovo = if_evident_function(coverage)
                    # print(site_tag_denovo, type(site_tag_denovo), miss_num, coverage, ifEvident_denovo)
                    # input()


                if denovo_illegal:
                    denovoSeq_str = ''
                    site_tag_denovo = ''
                    site_peaknum_denovo = ''
                    site_score_denovo = ''
                    pep_scoremean_denovo = ''
                    pep_scoresum_denovo = ''
                    ifEvident_denovo = ''

                try:
                    fout.write(f"{title}\t{denovoSeq_str}\t{''.join(list(map(str, site_tag_denovo)))}\t{','.join(list(map(str, site_peaknum_denovo)))}\t{','.join(list(map(str, site_score_denovo)))}\t{pep_scoremean_denovo}\t{pep_scoresum_denovo}\t{ifEvident_denovo}\t{False}\n")
                except:
                    print(site_tag_denovo)
                    print(''.join(list(map(str, site_tag_denovo))))
                    print(site_score_denovo)
                    print(''.join(list(map(str, site_score_denovo))))
                    assert False

            fout.close()
            title_to_denovo_df = pd.read_csv(os.path.dirname(SingleSpec_file) + "\\title_to_denovo_info.res", sep="\t")
        else:
            print("loading title_to_denovo_info.res file...")
            title_to_denovo_df = pd.read_csv(os.path.dirname(SingleSpec_file) + "\\title_to_denovo_info.res", sep="\t")

        if not "ifFind_denovo" in title_to_denovo_df.columns:
            seq_str_set = set(title_to_denovo_df["denovoSeq"])
            top1_res_dict = mapped_function_single(seq_str_set, fasta_seq, fasta_seq_first_4aa_index)
            title_to_denovo_df["ifFind_denovo"] = title_to_denovo_df["denovoSeq"].apply(
                lambda x: top1_res_dict[x][1] if x == x else False)
            title_to_denovo_df["FindLocation_denovo"] = title_to_denovo_df["denovoSeq"].apply(
                lambda x: top1_res_dict[x][3] if x == x else "")
            top1_res_dict_decoy = mapped_function_single(seq_str_set, fasta_seq_decoy, fasta_seq_first_4aa_index_decoy)
            title_to_denovo_df["ifFindDecoy_denovo"] = title_to_denovo_df["denovoSeq"].apply(
                lambda x: top1_res_dict_decoy[x][2] if x == x else False)
            title_to_denovo_df["FindLocationDecoy_denovo"] = title_to_denovo_df["denovoSeq"].apply(
                lambda x: top1_res_dict_decoy[x][4] if x == x else "")
            title_to_denovo_df.to_csv(os.path.dirname(SingleSpec_file) + "\\title_to_denovo_info.res", sep="\t", index=False)
            write_dict_to_csv(fasta_coverage_denovo_dict, os.path.dirname(SingleSpec_file) + "\\fasta_coverage_denovo_info.res",
                              fasta_seq, None, denovo_proteinName_to_uniqueSeq_dict)
        else:
            print("ifFind_denovo has been calculated!")
            title_to_denovo_df = pd.read_csv(os.path.dirname(SingleSpec_file) + "\\title_to_denovo_info.res", sep="\t")

        if not os.path.exists(os.path.dirname(SingleSpec_file) + "\\fasta_coverage_denovo_info.res"):
            seq_to_unique_dict, fasta_coverage_denovo_dict, denovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
                fasta_seq, title_to_denovo_df, "denovo", False, 0.0)
            title_to_denovo_df["Seq_Unique_denovo"] = title_to_denovo_df["denovoSeq"].apply(
                lambda x: seq_to_unique_dict[x] if x == x else "")
            write_dict_to_csv(fasta_coverage_denovo_dict,
                              os.path.dirname(SingleSpec_file) + "\\fasta_coverage_denovo_info.res",
                              fasta_seq, None, denovo_proteinName_to_uniqueSeq_dict)
        else:
            fasta_coverage_denovo_dict, denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                os.path.dirname(SingleSpec_file) + "\\fasta_coverage_denovo_info.res")

        if not os.path.exists(os.path.dirname(SingleSpec_file) + "\\fasta_coverage_Evidentdenovo_info.res"):
            _, fasta_coverage_Evidentdenovo_dict, Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
                fasta_seq, title_to_denovo_df, "denovo", True, 0.0)
            write_dict_to_csv(fasta_coverage_Evidentdenovo_dict,
                              os.path.dirname(SingleSpec_file) + "\\fasta_coverage_Evidentdenovo_info.res", fasta_seq, None,
                              Evidentdenovo_proteinName_to_uniqueSeq_dict)
        else:
            fasta_coverage_Evidentdenovo_dict, Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                os.path.dirname(SingleSpec_file) + "\\fasta_coverage_Evidentdenovo_info.res")

        #######################################################################################################################################################
        #谱图肽段水平
        evident_spec_num_denovo = 0
        evident_find_spec_num_denovo = 0
        NonEvident_spec_num_denovo = 0
        NonEvident_find_spec_num_denovo = 0
        denovoSeq_to_info_dict = {}
        dbSeq_to_info_dict = {}
        title_to_denovo_df = pd.read_csv(os.path.dirname(SingleSpec_file) + "\\title_to_denovo_info.res", sep="\t")
        for index, row in title_to_denovo_df.iterrows():
            if index % 1000 == 0:
                print(f"\r", index, "/", len(title_to_denovo_df), end="")
            title, denovo_seq, ifEvident, ifFind = row["title"], row["denovoSeq"], row["ifEvident_denovo"], row["ifFind_denovo"]
            ifFindDecoy = row["ifFindDecoy_denovo"]
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

        denovoSeq_info_fw = open(os.path.dirname(SingleSpec_file) + "\\denovoSeq_to_ifEvidentFind.res", "w")
        header = "denovoSeq\tifEvident\tifFind\tifFindDecoy\ttitles\n"
        denovoSeq_info_fw.write(header)
        for key in denovoSeq_to_info_dict:
            string = f"{key}\t{denovoSeq_to_info_dict[key][0]}\t{denovoSeq_to_info_dict[key][1]}\t{denovoSeq_to_info_dict[key][2]}\t{denovoSeq_to_info_dict[key][3]}\n"
            denovoSeq_info_fw.write(string)
        denovoSeq_info_fw.flush()
        denovoSeq_info_fw.close()

        intsection_info_fout = open(os.path.dirname(SingleSpec_file) + "\\info.res", "a+")
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

        #肽段
        denovoSeq_df = pd.read_csv(os.path.dirname(SingleSpec_file) + "\\denovoSeq_to_ifEvidentFind.res", sep="\t")
        denovoSeq_Find_set = set(denovoSeq_df[(denovoSeq_df["ifFind"] == True) & (denovoSeq_df["ifFindDecoy"] == False)]["denovoSeq"])
        denovoSeq_Find_Evident_set = set(denovoSeq_df[(denovoSeq_df["ifFind"] == True) & (denovoSeq_df["ifFindDecoy"] == False) & (denovoSeq_df["ifEvident"] == True)]["denovoSeq"])
        string = f"denovoSeq_Find_num\t{len(denovoSeq_Find_set)}\n"
        intsection_info_fout.write(string)
        string = f"denovoSeq_Find_Evident_num\t{len(denovoSeq_Find_Evident_set)}\n"
        intsection_info_fout.write(string)

        #谱图
        denovo_find_spec_set = set(title_to_denovo_df[(title_to_denovo_df["ifFind_denovo"] == True) & (title_to_denovo_df["ifFindDecoy_denovo"] == False)]["title"])
        denovo_find_spec_evident_set = set(title_to_denovo_df[(title_to_denovo_df["ifFind_denovo"] == True) & (title_to_denovo_df["ifFindDecoy_denovo"] == False) & (title_to_denovo_df["ifEvident_denovo"] == True)]["title"])
        string = f"denovo_find_spec_num\t{len(denovo_find_spec_set)}\n"
        intsection_info_fout.write(string)
        string = f"denovo_find_spec_evident_num\t{len(denovo_find_spec_evident_set)}\n"
        intsection_info_fout.write(string)








