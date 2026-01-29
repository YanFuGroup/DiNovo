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
import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn3
import seaborn as sns
from common_function import build_fasta_dict_function, build_index_dict_function, generate_reverse_fasta, build_mgf_location_function, transfer_str_to_list_seq, if_evident_function, parallel_get_confident_info,\
    mapped_function_single, get_fasta_coverage_np_single, write_dict_to_csv, read_dict_from_csv, get_confident_info,\
    plot_aa_intsection_venn3, plot_aa_intsection_venn2, merge_fasta_coverage_dict, merge_proteinName_to_uniqueSeq,\
    generate_try_lys_seq_from_RseqAndMatchtype, transfer_mod_function, find_location_in_fasta_single, find_location_in_fasta_single,\
    generate_DiNovoPairs_to_matchtype_dict,merge_target_and_decoy_fasta_seq,calculate_mass,calculate_mass_mirror_from_specdf,calculate_mass_mirror,\
    generate_Best_title_to_denovo_df

#CasaNovo/PEAKS/PointNovo/pNovo3
topk_peaks = config.topk_peaks
DiNovo_tool = "MirrorNovo"#DiNovo_MirrorNovo/DiNovo_pNovoM2/MirrorNovo/pNovoM2
DiNovo_merge = False
fasta_file = config.fasta_file
DiNovo_res_folder = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MirrorNovo_output\mirror\20240525\23charge"
DiNovoPairs_file = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\spectraPairs\20240525\[DiNovo]SpectralPairs[23charge].res"
fig_name1 = "Try"
fig_name2 = "Lys"
massFilter1 = 748
massFilter2 = 764
massFilter12 = 795

# 计算离子覆盖率时用到的参数
if_use_ppm = config.if_use_ppm
fragment_tolerance_ppm = config.fragment_tolerance_ppm
fragment_tolerance_da = config.fragment_tolerance_da
modifications = config.modifications
modifications_in_aa_dict_keys = config.modifications_in_aa_dict_keys
pure_aa_list = config.pure_aa_list
modifications_other  = config.modifications_other
modifications_other_in_aa_dict_keys  = config.modifications_other_in_aa_dict_keys
modifications_other_mass  = config.modifications_other_mass
modifications_DiNovo  = config.modifications_DiNovo
aa_to_mass_dict = config.aa_to_mass_dict
atom_mass = config.atom_mass

def generate_Rseq_lys_seq_from_try_seq(try_seq, match_type):
    flag = False
    if type(try_seq) == str:
        try_seq = transfer_str_to_list_seq(try_seq)
        flag = True

    if match_type == "A1:K-K" or match_type == "B: R-K":
        lys_seq = ["K"] + try_seq[:-1]
        Rseq = ["K"] + try_seq
    elif match_type == "F: X-K":
        lys_seq = ["K"] + try_seq
        Rseq = ["K"] + try_seq
    elif match_type == "A2:R-R" or match_type == "C: K-R":
        lys_seq = ["R"] + try_seq[:-1]
        Rseq = ["R"] + try_seq
    elif match_type == "G: X-R":
        lys_seq = ["R"] + try_seq
        Rseq = ["R"] + try_seq
    elif match_type == "D: K-X" or match_type == "E: R-X":
        lys_seq = try_seq[:-1]
        Rseq = try_seq
    else:
        print(try_seq,match_type)
        assert False

    if flag:
        lys_seq = ''.join(lys_seq)
        Rseq = ''.join(Rseq)
    return Rseq,lys_seq

def parallel_get_confident_info_mirror(psm_list, try_mgf_location_dict, try_suffix_name, try_mgf_path,
                                       lys_mgf_location_dict, lys_suffix_name, lys_mgf_path):
    print("Start to calculate coverage info")
    # try
    with multiprocessing.Pool(12) as p:
        try_res_list = list(tqdm(p.imap(get_confident_info,
                                        [(psm[1], psm[4], try_mgf_location_dict, try_suffix_name, try_mgf_path) for psm
                                         in psm_list], chunksize=100000), total=len(psm_list)))
    assert len(try_res_list) == len(psm_list)

    try_res_dict = {}
    for res_item in try_res_list:
        try_res_dict[res_item["PSM"]] = res_item

    # lys
    with multiprocessing.Pool(12) as p:
        lys_res_list = list(tqdm(p.imap(get_confident_info,
                                        [(psm[2], psm[5], lys_mgf_location_dict, lys_suffix_name, lys_mgf_path) for psm
                                         in psm_list], chunksize=100000), total=len(psm_list)))
    assert len(lys_res_list) == len(psm_list)

    lys_res_dict = {}
    for res_item in lys_res_list:
        lys_res_dict[res_item["PSM"]] = res_item

    return try_res_dict, lys_res_dict


def merge_try_and_lys_info(try_info_dict, lys_info_dict):
    try_site_tag = try_info_dict["site_tag"]
    try_site_peaknum = try_info_dict["site_peaknum"]
    try_site_score = try_info_dict["site_score"]
    try_intensitytemp = try_info_dict["site_intensitytemp"]
    lys_site_tag = lys_info_dict["site_tag"]
    lys_site_peaknum = lys_info_dict["site_peaknum"]
    lys_site_score = lys_info_dict["site_score"]
    lys_intensitytemp = lys_info_dict["site_intensitytemp"]

    if match_type == "A1:K-K" or match_type == "A2:R-R" or match_type == "B: R-K" or match_type == "C: K-R":
        site_tag = np.zeros_like(try_site_tag)
        site_peaknum = np.zeros_like(try_site_peaknum)
        site_score = np.zeros_like(try_site_score)
        site_intensitytemp_A = np.zeros_like(try_intensitytemp)
        site_tag[:-1] = try_site_tag[:-1] + lys_site_tag[1:]
        site_tag[-1] = try_site_tag[-1]
        site_peaknum[:-1] = try_site_peaknum[:-1] + lys_site_peaknum[1:]
        site_peaknum[-1] = try_site_peaknum[-1]
        site_score[:-1] = try_site_score[:-1] + lys_site_score[1:]
        site_score[-1] = try_site_score[-1]
        site_intensitytemp_A[:-1] = try_intensitytemp[:-1] + lys_intensitytemp[1:]
        site_intensitytemp_A[-1] = try_intensitytemp[-1]
    elif match_type == "F: X-K" or match_type == "G: X-R":
        site_tag = np.zeros_like(try_site_tag)
        site_peaknum = np.zeros_like(try_site_peaknum)
        site_score = np.zeros_like(try_site_score)
        site_intensitytemp_A = np.zeros_like(try_intensitytemp)
        site_tag = try_site_tag + lys_site_tag[1:]
        site_peaknum = try_site_peaknum + lys_site_peaknum[1:]
        site_score = try_site_score + lys_site_score[1:]
        site_intensitytemp_A = try_intensitytemp + lys_intensitytemp[1:]
    elif match_type == "D: K-X" or match_type == "E: R-X":
        # print(top1_Rseq, try_denovo_seq, lys_denovo_seq)
        # print(len(try_site_tag),try_site_tag)
        # print(len(lys_site_tag),lys_site_tag)
        site_tag = np.zeros_like(try_site_tag)
        site_peaknum = np.zeros_like(try_site_peaknum)
        site_score = np.zeros_like(try_site_score)
        site_intensitytemp_A = np.zeros_like(try_intensitytemp) 
        site_tag[:-1] = try_site_tag[:-1] + lys_site_tag
        site_tag[-1] = try_site_tag[-1]
        site_peaknum[:-1] = try_site_peaknum[:-1] + lys_site_peaknum
        site_peaknum[-1] = try_site_peaknum[-1]
        site_score[:-1] = try_site_score[:-1] + lys_site_score
        site_score[-1] = try_site_score[-1]
        site_intensitytemp_A[:-1] = try_intensitytemp[:-1] + lys_intensitytemp
        site_intensitytemp_A[-1] = try_intensitytemp[-1]
    else:
        print(match_type)
        assert False

    site_tag = np.array([1 if tag != 0 else 0 for tag in site_tag])
    pep_scoremean = np.round(np.mean(site_score), 2)
    pep_scoresum = np.round(np.sum(site_score), 2)
    return {
        "site_tag": site_tag,
        "site_peaknum": site_peaknum,
        "site_score": site_score,
        "pep_scoremean": pep_scoremean,
        "pep_scoresum": pep_scoresum,
        "site_intensitytemp_A": site_intensitytemp_A,
    }

def read_seqs_from_DiNovoSingleRes(location, fi, title, fasta_seq = None, index_first_4aa = None ):

    if fasta_seq == None:
        #只选择top1结果
        fi.seek(location)
        line = fi.readline().strip()
        title_temp = line.split("\t")[0]
        assert title == title_temp,f"title:{title}, title_temp:{title_temp}"
        line = fi.readline()
        line = fi.readline()
        if "END" in line:
            seq = ['']
            score = 0.0
        else:
            line = line.strip().split("\t")
            seq = line[2]
            score = float(line[3])
            seq = seq.replace("C(Carbamidomethylation)","C(+57.02)")
            seq = seq.replace("M(Oxidation)","M(+15.99)")
            seq = seq.replace("I","L")
            seq = seq.split(",")
    else:
        #选择topn中第一条能够回贴到数据库的结果
        fi.seek(location)
        line = fi.readline().strip()
        title_temp = line.split("\t")[0]
        assert title == title_temp, f"title:{title}, title_temp:{title_temp}"
        line = fi.readline()
        line = fi.readline()
        if "END" in line:
            seq = ['']
        else:
            find_tag_this_spectrum = False
            while "END" not in line:
                if find_tag_this_spectrum:
                    break
                line = line.strip().split("\t")
                seq_ori = line[2]
                seq = line[2].replace(',','')
                seq = seq.replace("C(Carbamidomethylation)", "C")
                seq = seq.replace("M(Oxidation)", "M")
                seq = seq.replace("I", "L")
                assert "(" not in seq
                find_tag = find_location_in_fasta_single(seq, fasta_seq, index_first_4aa)
                if find_tag:
                    seq = seq_ori.replace("C(Carbamidomethylation)", "C(+57.02)")
                    seq = seq.replace("M(Oxidation)", "M(+15.99)")
                    seq = seq.replace("I", "L")
                    seq = seq.split(",")
                    find_tag_this_spectrum = True
                else:
                    seq = ['']
                line = fi.readline()

    return seq, score


def build_singleSpectra_location_function(file_path,software):
    if software == "DiNovo":
        if if_merge:
            location_file = os.path.join(file_path + "_location")
            if os.path.exists(location_file) == False:
                print("building location for ", file_path)
                result_location_dict = {}
                with open(file_path, "r") as fin:
                    location = fin.tell()
                    while True:
                        line = fin.readline().strip()
                        if len(line) == 0:
                            break
                        line = line.split("\t")
                        assert line[0].startswith("S")
                        title = line[1].strip()
                        try_title = title.split("@")[0]
                        candidate_num = int(line[-1].strip())
                        result_location_dict[try_title] = location
                        for i in range(candidate_num):
                            fin.readline()
                        location = fin.tell()
                location_file = file_path + "_location"
                with open(location_file, 'wb') as fw:
                    pickle.dump(result_location_dict, fw)

            else:
                result_location_dict = {}
                with open(location_file, 'rb') as fr:
                    result_location_dict = pickle.load(fr)
            return result_location_dict

        else:
            location_file = os.path.join(file_path + "_location")
            if os.path.exists(location_file) == False:
                print("building location for ", file_path)
                result_location_dict = {}
                with open(file_path, "r") as fin:
                    location = fin.tell()
                    while True:
                        line = fin.readline().strip()
                        if len(line) == 0:
                            break
                        line = line.split("\t")
                        assert line[0].startswith("S")
                        title = line[1].strip()
                        try_title = title.split("@")[0]
                        candidate_num = int(line[-1].strip())
                        result_location_dict[try_title] = location
                        for i in range(candidate_num):
                            fin.readline()
                        location = fin.tell()
                location_file = file_path + "_location"
                with open(location_file, 'wb') as fw:
                    pickle.dump(result_location_dict, fw)
            else:
                result_location_dict = {}
                with open(location_file, 'rb') as fr:
                    result_location_dict = pickle.load(fr)
            return result_location_dict

    else:#MirrorNovo
        assert software == "MirrorNovo"
        location_file = os.path.join(file_path + "_location")
        if os.path.exists(location_file) == False:
            print("building location for ", file_path)
            result_location_dict = {}
            with open(file_path, "r") as fin:
                while True:
                    line = fin.readline().strip()
                    if len(line) == 0:
                        break
                    assert "BEGIN" in line
                    location = fin.tell()
                    line = fin.readline()
                    title = line.split("\t")[0]
                    result_location_dict[title] = location
                    while True:
                        line = fin.readline()
                        if "END" in line:
                            break
            location_file = file_path + "_location"
            with open(location_file, 'wb') as fw:
                pickle.dump(result_location_dict, fw)
        else:
            result_location_dict = {}
            with open(location_file, 'rb') as fr:
                result_location_dict = pickle.load(fr)
        return result_location_dict

def generate_RseqTD_from_trylysTD(try_TD, lys_TD):
    if try_TD == "T" and lys_TD == "T":
        return "T"
    else:
        if try_TD == "D" and lys_TD == "D":
            return "D"
        else:
            if try_TD == "NoMatch" or lys_TD == "NoMatch":
                return "NoMatch"
            else:
                return "TD"

def mapped_function_mirror(Seq_str_set, fasta_seq_dict, first_index_4aa, mapped_version):
    '''

    :param Seq_str_set:
    :param fasta_seq_dict:
    :param first_index_4aa:
    :param mapped_version: 1:原来的回贴方式，分开回帖   2：先整体回贴，整体回帖不上再分开回帖
    :return:
    '''
    seq_str_dict = {}
    try_seq_str_list = []
    lys_seq_str_list = []
    for denovo_seq, match_type in Seq_str_set:
        if denovo_seq != denovo_seq or denovo_seq == "":
            continue
        else:
            try_seq, lys_seq = generate_try_lys_seq_from_RseqAndMatchtype(denovo_seq, match_type)
            try_seq_str_list.append(try_seq)
            lys_seq_str_list.append(lys_seq)
            seq_str_dict[denovo_seq + "@" + match_type] = (try_seq, lys_seq, denovo_seq)
            # if denovo_seq == "KLLAVLC(+57.02)":
            #     print(try_seq, lys_seq, match_type)
    try_seq_str_set = set(try_seq_str_list)
    lys_seq_str_set = set(lys_seq_str_list)

    if mapped_version == 1:
        assert False,"还没有针对target和decoy合在一起的做法做改动"
        processing_workers = 24
        with multiprocessing.Pool(processing_workers) as p:
            res_dict_list = list(tqdm(p.imap(find_location_in_fasta_single,
                                             [(seq, fasta_seq_dict, first_index_4aa) for seq in try_seq_str_set],
                                             chunksize=1000000), total=len(try_seq_str_set)))
            try_res_dict = {seq_dict["seq"]: [seq_dict["pure_seq"], seq_dict["find"], seq_dict["location"]] for seq_dict
                            in res_dict_list}  # seq: [find, location, if_unique]

        with multiprocessing.Pool(processing_workers) as p:
            res_dict_list = list(tqdm(p.imap(find_location_in_fasta_single,
                                             [(seq, fasta_seq_dict, first_index_4aa) for seq in lys_seq_str_set],
                                             chunksize=1000000), total=len(lys_seq_str_set)))
            lys_res_dict = {seq_dict["seq"]: [seq_dict["pure_seq"], seq_dict["find"], seq_dict["location"]] for seq_dict
                            in res_dict_list}  # seq: [find, location, if_unique]

        res_dict = {}
        for denovo_seq_and_match_type in seq_str_dict:
            try_seq, lys_seq, denovo_seq = seq_str_dict[denovo_seq_and_match_type]
            tryFind = try_res_dict[try_seq][1]
            lysFind = lys_res_dict[lys_seq][1]
            if tryFind and lysFind:
                res_dict[denovo_seq_and_match_type] = ["", True,
                                                       try_res_dict[try_seq][2] + "||||" + lys_res_dict[lys_seq][2]]
            else:
                res_dict[denovo_seq_and_match_type] = ["", False,
                                                       try_res_dict[try_seq][2] + "||||" + lys_res_dict[lys_seq][2]]
        return res_dict

    elif mapped_version == 2:
        processing_workers = 24
        with multiprocessing.Pool(processing_workers) as p:
            res_dict_list = list(tqdm(p.imap(find_location_in_fasta_single,
                                             [(seq_str_dict[item][2], fasta_seq_dict, first_index_4aa) for item in
                                              seq_str_dict.keys()], chunksize=1000000), total=len(seq_str_dict)))
            res_dict = {seq_dict["seq"]: [seq_dict["pure_seq"], seq_dict["target_find"] | seq_dict["decoy_find"], seq_dict["target_location"], seq_dict["decoy_location"], seq_dict["TD_tag"]] for seq_dict in res_dict_list}  # seq: [find, location, if_unique]

        return_res_dict = {}
        for denovo_seq_and_match_type in seq_str_dict:
            denovo_seq = seq_str_dict[denovo_seq_and_match_type][2]
            ifFind = res_dict[denovo_seq][1]
            if not ifFind:
                try_seq, lys_seq = seq_str_dict[denovo_seq_and_match_type][0], seq_str_dict[denovo_seq_and_match_type][1]
                try_this_psm_res_dict = find_location_in_fasta_single((try_seq, fasta_seq_dict, first_index_4aa))
                try_ifFind = try_this_psm_res_dict["target_find"] | try_this_psm_res_dict["decoy_find"]
                
                lys_this_psm_res_dict = find_location_in_fasta_single((lys_seq, fasta_seq_dict, first_index_4aa))
                lys_ifFind = lys_this_psm_res_dict["target_find"] | lys_this_psm_res_dict["decoy_find"]

                TD_temp = generate_RseqTD_from_trylysTD(try_this_psm_res_dict["TD_tag"],
                                                        lys_this_psm_res_dict["TD_tag"])
                if try_ifFind and lys_ifFind:
                    return_res_dict[denovo_seq_and_match_type] = [res_dict[denovo_seq][0], True,
                                                                  try_this_psm_res_dict["target_location"] + "||||" + lys_this_psm_res_dict["target_location"],
                                                                  try_this_psm_res_dict["decoy_location"] + "||||" + lys_this_psm_res_dict["decoy_location"],
                                                                  TD_temp]
                elif try_ifFind and (not lys_ifFind):
                    assert TD_temp == "NoMatch"
                    return_res_dict[denovo_seq_and_match_type] = [res_dict[denovo_seq][0], False,
                                                                  try_this_psm_res_dict["target_location"] + "||||",
                                                                  try_this_psm_res_dict["decoy_location"] + "||||",
                                                                  TD_temp]
                elif (not try_ifFind) and lys_ifFind:
                    assert TD_temp == "NoMatch"
                    return_res_dict[denovo_seq_and_match_type] = [res_dict[denovo_seq][0], False,
                                                                  "||||" + lys_this_psm_res_dict["target_location"],
                                                                  "||||" + lys_this_psm_res_dict["decoy_location"],
                                                                  TD_temp]
                else:
                    assert TD_temp == "NoMatch"
                    return_res_dict[denovo_seq_and_match_type] = [res_dict[denovo_seq][0], False, "||||","||||", TD_temp]
            else:  # ifFind
                return_res_dict[denovo_seq_and_match_type] = res_dict[denovo_seq]

        return return_res_dict
    elif mapped_version == 3:
        assert False,"还没有针对target和decoy合在一起的做法做改动"
        processing_workers = 24
        with multiprocessing.Pool(processing_workers) as p:
            res_dict_list = list(tqdm(p.imap(find_location_in_fasta_single,
                                             [(seq_str_dict[item][2], fasta_seq_dict, first_index_4aa) for item in
                                              seq_str_dict.keys()], chunksize=1000000), total=len(seq_str_dict)))
            res_dict = {seq_dict["seq"]: [seq_dict["pure_seq"], seq_dict["find"], seq_dict["location"]] for seq_dict in
                        res_dict_list}  # seq: [find, location, if_unique]

        return_res_dict = {}
        for denovo_seq_and_match_type in seq_str_dict:
            denovo_seq = seq_str_dict[denovo_seq_and_match_type][2]
            ifFind = res_dict[denovo_seq][1]
            if not ifFind:
                try_seq, lys_seq = seq_str_dict[denovo_seq_and_match_type][0], seq_str_dict[denovo_seq_and_match_type][
                    1]
                try_this_psm_res_dict = find_location_in_fasta_single((try_seq, fasta_seq_dict, first_index_4aa))
                try_ifFind = try_this_psm_res_dict["find"]
                lys_this_psm_res_dict = find_location_in_fasta_single((lys_seq, fasta_seq_dict, first_index_4aa))
                lys_ifFind = lys_this_psm_res_dict["find"]
                if try_ifFind and lys_ifFind:
                    return_res_dict[denovo_seq_and_match_type] = [res_dict[denovo_seq][0], False,
                                                                  try_this_psm_res_dict["location"] + "||||" +
                                                                  lys_this_psm_res_dict["location"]]
                elif try_ifFind and (not lys_ifFind):
                    return_res_dict[denovo_seq_and_match_type] = [res_dict[denovo_seq][0], False,
                                                                  try_this_psm_res_dict["location"] + "||||"]
                elif (not try_ifFind) and lys_ifFind:
                    return_res_dict[denovo_seq_and_match_type] = [res_dict[denovo_seq][0], False,
                                                                  "||||" + lys_this_psm_res_dict["location"]]
                else:
                    return_res_dict[denovo_seq_and_match_type] = [res_dict[denovo_seq][0], False, "||||"]
            else:  # ifFind
                return_res_dict[denovo_seq_and_match_type] = res_dict[denovo_seq]

        return return_res_dict
    else:
        assert False

def get_fasta_coverage_np_mirror(fasta_seq_dict, df, Evident_select = False, mass_threshold = 0.0):

    #seq_to_if_unique_dict
    seq_to_if_unique_dict = {}
    seq_to_if_split_dict = {}
    proteinName_to_uniqueSeq_dict = {}

    # init
    coverage_fasta_dict = {}
    for key in fasta_seq_dict:
        seq = fasta_seq_dict[key]
        length = len(seq)
        coverage_fasta_dict[key] = [np.zeros(length, dtype=int), np.zeros(length, dtype=int), np.zeros(length, dtype=float), 0, 0, 0]  # site_tag, site_score, unique peptide num, non-unique peptide num,peptide num,

    for index, row in df.iterrows():

            seq = row[f"denovoSeqID"]
            if seq != seq or seq == "":
                seq_to_if_unique_dict[seq] = False
                seq_to_if_split_dict[seq] = ""
                continue

            try:
                only_seq = seq.split("@")[0]
                matchtype = seq.split("@")[1]
                try_only_seq, lys_only_seq = generate_try_lys_seq_from_RseqAndMatchtype(only_seq, matchtype)
                ifFind, location, site_score = row[f"ifFind_denovo"], row[f"FindLocation_denovo"], row[f"site_score_denovo"]
                TDTag = row["TDTag_denovo"]
                site_score = np.array(site_score.split(","),dtype=float)
                FindFinal = ifFind and (TDTag == "T")

                if FindFinal:
                    if "||||" in location:
                        split_seq = True
                        seq_to_if_split_dict[seq] = True
                    else:
                        split_seq = False
                        seq_to_if_split_dict[seq] = False
                else:
                    seq_to_if_split_dict[seq] = ""

            except:
                assert False
            if Evident_select and (mass_threshold != 0.0):
                pep_mass, ifEvident = row[f"Pep_mass_denovo"], row[f"ifEvident_denovo"]
                if pep_mass < mass_threshold:
                    continue
                if not ifEvident:
                    continue
            else:
                if (not Evident_select) and (mass_threshold != 0.0):
                    pep_mass = row[f"Pep_mass_denovo"]
                    if pep_mass < mass_threshold:
                        continue
                else:
                    if Evident_select and (mass_threshold == 0.0):
                        ifEvident = row[f"ifEvident_denovo"]
                        if not ifEvident:
                            continue
                    else:
                        pass

            if FindFinal:
                if not split_seq:
                    location_split = location.split("&")
                    protein_name = {}
                    for item in location_split:
                        location = item.split("$")
                        coverage_fasta_dict[location[0]][0][int(location[1]):int(location[2])] = 1
                        coverage_fasta_dict[location[0]][1][int(location[1]):int(location[2])] += 1
                        # coverage_fasta_dict[location[0]][2][int(location[1]):int(location[2])] = np.maximum(coverage_fasta_dict[location[0]][1][int(location[1]):int(location[2])], site_score)
                        protein_name[location[0]] = 1
                    if len(protein_name) == 1:
                        coverage_fasta_dict[list(protein_name.keys())[0]][3] += 1
                        coverage_fasta_dict[list(protein_name.keys())[0]][5] += 1
                        seq_to_if_unique_dict[seq] = True
                        this_protein_name = list(protein_name.keys())[0]
                        if this_protein_name not in proteinName_to_uniqueSeq_dict:
                            proteinName_to_uniqueSeq_dict[this_protein_name] = [only_seq]
                        else:
                            proteinName_to_uniqueSeq_dict[this_protein_name].append(only_seq)
                    else:
                        for protein in protein_name:
                            coverage_fasta_dict[protein][4] += 1
                            coverage_fasta_dict[protein][5] += 1
                        seq_to_if_unique_dict[seq] = False
                else: #split_seq
                    try_unique = False
                    lys_unique = False
                    try_location = location.split("||||")[0]
                    lys_location = location.split("||||")[1]
                    #try
                    location_split = try_location.split("&")
                    protein_name = {}
                    for item in location_split:
                        location = item.split("$")
                        coverage_fasta_dict[location[0]][0][int(location[1]):int(location[2])] = 1
                        coverage_fasta_dict[location[0]][1][int(location[1]):int(location[2])] += 1
                        # coverage_fasta_dict[location[0]][2][int(location[1]):int(location[2])] = np.maximum(coverage_fasta_dict[location[0]][1][int(location[1]):int(location[2])], site_score)
                        protein_name[location[0]] = 1
                    if len(protein_name) == 1:
                        coverage_fasta_dict[list(protein_name.keys())[0]][3] += 1
                        coverage_fasta_dict[list(protein_name.keys())[0]][5] += 1
                        try_unique = True
                        this_protein_name = list(protein_name.keys())[0]
                        if this_protein_name not in proteinName_to_uniqueSeq_dict:
                            proteinName_to_uniqueSeq_dict[this_protein_name] = [try_only_seq]
                        else:
                            proteinName_to_uniqueSeq_dict[this_protein_name].append(try_only_seq)
                    else:
                        for protein in protein_name:
                            coverage_fasta_dict[protein][4] += 1
                            coverage_fasta_dict[protein][5] += 1
                        try_unique = False
                    #lys
                    location_split = lys_location.split("&")
                    protein_name = {}
                    for item in location_split:
                        location = item.split("$")
                        coverage_fasta_dict[location[0]][0][int(location[1]):int(location[2])] = 1
                        coverage_fasta_dict[location[0]][1][int(location[1]):int(location[2])] += 1
                        # coverage_fasta_dict[location[0]][2][int(location[1]):int(location[2])] = np.maximum(coverage_fasta_dict[location[0]][1][int(location[1]):int(location[2])], site_score)
                        protein_name[location[0]] = 1
                    if len(protein_name) == 1:
                        coverage_fasta_dict[list(protein_name.keys())[0]][3] += 1
                        coverage_fasta_dict[list(protein_name.keys())[0]][5] += 1
                        lys_unique = True
                        this_protein_name = list(protein_name.keys())[0]
                        if this_protein_name not in proteinName_to_uniqueSeq_dict:
                            proteinName_to_uniqueSeq_dict[this_protein_name] = [lys_only_seq]
                        else:
                            proteinName_to_uniqueSeq_dict[this_protein_name].append(lys_only_seq)
                    else:
                        for protein in protein_name:
                            coverage_fasta_dict[protein][4] += 1
                            coverage_fasta_dict[protein][5] += 1
                        lys_unique = False
                    seq_to_if_unique_dict[seq] = str(try_unique) + "@" + str(lys_unique)
            else:
                seq_to_if_unique_dict[seq] = False

    #去重
    proteinName_to_uniqueSeq_dict = {key: list(set(proteinName_to_uniqueSeq_dict[key])) for key in proteinName_to_uniqueSeq_dict}

    return seq_to_if_unique_dict, seq_to_if_split_dict, coverage_fasta_dict, proteinName_to_uniqueSeq_dict

def merge_proteinName_to_uniqueSeq(proteinName_to_uniqueSeq_dict_list):
    merged_proteinName_to_uniqueSeq_dict = {}
    for proteinName_to_uniqueSeq_dict in proteinName_to_uniqueSeq_dict_list:
        for key in proteinName_to_uniqueSeq_dict:
            if key not in merged_proteinName_to_uniqueSeq_dict:
                merged_proteinName_to_uniqueSeq_dict[key] = proteinName_to_uniqueSeq_dict[key]
            else:
                merged_proteinName_to_uniqueSeq_dict[key] += proteinName_to_uniqueSeq_dict[key]

    #去重
    merged_proteinName_to_uniqueSeq_dict = {key: list(set(merged_proteinName_to_uniqueSeq_dict[key])) for key in merged_proteinName_to_uniqueSeq_dict}

    return merged_proteinName_to_uniqueSeq_dict

def transfer_Rseq_to_trySeq(Rseq):
    if Rseq[0] == 'K' or Rseq[0] == 'R':
        return Rseq[1:]
    else:
        return Rseq


def transfer_Rseq_to_lysSeq(Rseq):
    if Rseq[-1] == 'K' or Rseq[-1] == 'R':
        return Rseq[:-1]
    else:
        return Rseq


def remove_try_KR_function(target_seq):
    assert type(target_seq) == str
    if target_seq[-1] == "K" or target_seq[-1] == "R":
        target_seq_NoKR = target_seq[:-1]
    else:
        target_seq_NoKR = target_seq
    return target_seq_NoKR


def remove_lys_KR_function(target_seq):
    if type(target_seq) != str:
        print(target_seq)
        assert False
    if target_seq[0] == "K" or target_seq[0] == "R":
        target_seq_NoKR = target_seq[1:]
    else:
        if target_seq.startswith("B(+42.01)K") or target_seq.startswith("B(+42.01)R"):  # string
            target_seq_NoKR = target_seq.lstrip("B(+42.01)K").lstrip("B(+42.01)R")
        else:
            target_seq_NoKR = target_seq
    return target_seq_NoKR

def generate_targetRseq_function(target_seq_try, target_seq_lys):
    # 合并，生成target_Rseq
    if target_seq_try != target_seq_try and target_seq_lys != target_seq_lys:  # 如果都是nan
        assert False

    if target_seq_try != target_seq_try:  # 如果是nan
        target_Rseq = target_seq_lys
    elif target_seq_lys != target_seq_lys:  # 如果是nan
        target_Rseq = target_seq_try
    else:  # 正常情况
        if len(target_seq_try) == len(target_seq_lys):
            if target_seq_try == target_seq_lys:
                target_Rseq = target_seq_try
            else:
                assert target_seq_try != target_seq_lys and target_seq_try[:-1] == target_seq_lys[1:]
                target_Rseq = target_seq_lys + target_seq_try[-1]
        elif len(target_seq_try) == len(target_seq_lys) + 1:
            assert target_seq_try[:-1] == target_seq_lys
            target_Rseq = target_seq_try
        elif len(target_seq_try) + 1 == len(target_seq_lys):
            assert target_seq_try == target_seq_lys[1:]
            target_Rseq = target_seq_lys
        else:
            assert target_seq_lys.startswith('B(+42.01)K') or target_seq_lys.startswith('B(+42.01)R')
            if target_seq_try[-1] == "K" or target_seq_try[-1] == "R":
                target_Rseq = target_seq_lys + target_seq_try[-1]
            elif target_seq_try == target_seq_lys.lstrip('B(+42.01)K').lstrip('B(+42.01)R'):
                target_Rseq = target_seq_lys
            else:
                assert False
            print("target_seq_try: ", target_seq_try)
            print("target_seq_lys: ", target_seq_lys)
            print("target_Rseq: ", target_Rseq)

    return target_Rseq

def find_new_Rseq(row, df):
    if row["tag"] == 3:
        return ''
    elif row["tag"] != row["tag"]:
        return ''
    elif row["tag"] == 1:
        res = list(df[df["Rseq_try"] == row["denovo_Rseq"]]["denovo_Rseq"])
        res = ','.join(res) if res else ''
        return res
    elif row["tag"] == 2:
        res = list(df[df["Rseq_lys"] == row["denovo_Rseq"]]["denovo_Rseq"])
        res = ','.join(res) if res else ''
        return res
    else:
        assert False

def get_maxscore_info(df):
    tryTitle_to_BestRow = {}
    lysTitle_to_BestRow = {}
    for index,row in df.iterrows():
        tryTitle = row["A_title"]
        lysTitle = row["B_title"]
        if tryTitle not in tryTitle_to_BestRow:
            tryTitle_to_BestRow[tryTitle] = row
        else:
            if row["MirrorFinderScore"] < tryTitle_to_BestRow[tryTitle]["MirrorFinderScore"]:
                # assert re.sub(r'[^a-zA-Z]', '', row["A_denovoSeq"]) == re.sub(r'[^a-zA-Z]', '', tryTitle_to_BestRow[tryTitle]["A_denovoSeq"]),f"row:{row['A_denovoSeq']}, tryTitle_to_BestRow[tryTitle]:{tryTitle_to_BestRow[tryTitle]['A_denovoSeq']}"
                tryTitle_to_BestRow[tryTitle] = row
        if lysTitle not in lysTitle_to_BestRow:
            lysTitle_to_BestRow[lysTitle] = row
        else:
            if row["MirrorFinderScore"] < lysTitle_to_BestRow[lysTitle]["MirrorFinderScore"]:
                # assert row["B_denovoSeq"] == lysTitle_to_BestRow[lysTitle]["B_denovoSeq"], f"row:{row['B_denovoSeq']}, lysTitle_to_BestRow[lysTitle]:{lysTitle_to_BestRow[lysTitle]['B_denovoSeq']}"
                lysTitle_to_BestRow[lysTitle] = row
    return tryTitle_to_BestRow, lysTitle_to_BestRow

def generate_Find_tag(row):
    ifFind = row["ifFind_denovo"]
    if ifFind:
        TD_Tag = row["TDTag_denovo"]
        if TD_Tag == "NoMatch":
            return False, False
        else:
            if TD_Tag == "T":
                return True,False
            elif TD_Tag == "D":
                return False,True
            else:
                assert False
    else:
        return False,False

def generate_title_to_res_dict(df):
    title_to_res_dict = {}
    for index, row in df.iterrows():
        title = row["title"]
        title_to_res_dict[title] = row
    return title_to_res_dict

def filter_conflict_result(df, title_to_DiNovoTry_res, title_to_DiNovoLys_res):
    df = df.copy()
    #新增全为False的列
    df["ifFilter"] = False
    df["match_type_class"] = 0
    debug_fout = open("debug.txt","w")
    debug_fout.write("A_title\tB_title\tA_seq_from_single\tA_seq_from_mirror\tB_seq_from_mirror\tB_seq_from_single\tclass\n")
    for index, row in df.iterrows():
        A_title = row["A_title"]
        B_title = row["B_title"]
        A_seq_from_mirror = re.sub(r'[^a-zA-Z]', '',row["A_denovoSeq"])
        B_seq_from_mirror = re.sub(r'[^a-zA-Z]', '',row["B_denovoSeq"])
        if A_title in title_to_DiNovoTry_res:
            A_seq_from_single_find = True
            A_seq_from_single = re.sub(r'[^a-zA-Z]', '',title_to_DiNovoTry_res[A_title]["denovoSeq"])
        else:           
            A_seq_from_single_find = False
        
        if B_title in title_to_DiNovoLys_res:
            B_seq_from_single_find = True
            B_seq_from_single = re.sub(r'[^a-zA-Z]', '',title_to_DiNovoLys_res[B_title]["denovoSeq"])
        else:           
            B_seq_from_single_find = False

        #判断是否保留
        if A_seq_from_single_find and B_seq_from_single_find:
            if A_seq_from_single == A_seq_from_mirror and B_seq_from_single == B_seq_from_mirror:
                df.loc[index, "ifFilter"] = False
                df.loc[index, "match_type_class"] = 1
            else:
                df.loc[index, "ifFilter"] = True
                if A_seq_from_single != A_seq_from_mirror and B_seq_from_single == B_seq_from_mirror:
                    df.loc[index, "match_type_class"] = 2
                    debug_fout.write(f"{A_title}\t{B_title}\t{A_seq_from_single}\t{A_seq_from_mirror}\t{B_seq_from_mirror}\t{B_seq_from_single}\t{2}\n")
                elif A_seq_from_single == A_seq_from_mirror and B_seq_from_single != B_seq_from_mirror:
                    df.loc[index, "match_type_class"] = 3
                    debug_fout.write(f"{A_title}\t{B_title}\t{A_seq_from_single}\t{A_seq_from_mirror}\t{B_seq_from_mirror}\t{B_seq_from_single}\t{3}\n")
                elif A_seq_from_single == B_seq_from_single and A_seq_from_single != A_seq_from_mirror:
                    df.loc[index, "match_type_class"] = 4
                    debug_fout.write(f"{A_title}\t{B_title}\t{A_seq_from_single}\t{A_seq_from_mirror}\t{B_seq_from_mirror}\t{B_seq_from_single}\t{4}\n")
                elif A_seq_from_single != A_seq_from_mirror and B_seq_from_single != B_seq_from_mirror and A_seq_from_single != B_seq_from_single:
                    df.loc[index, "match_type_class"] = 5
                    debug_fout.write(f"{A_title}\t{B_title}\t{A_seq_from_single}\t{A_seq_from_mirror}\t{B_seq_from_mirror}\t{B_seq_from_single}\t{5}\n")
                else:
                    assert False, (A_seq_from_single, A_seq_from_mirror, B_seq_from_mirror,B_seq_from_single)
                # print(A_seq_from_single, A_seq_from_mirror, B_seq_from_mirror,B_seq_from_single)
        else:
            if A_seq_from_single_find and (not B_seq_from_single_find):
                if A_seq_from_single == A_seq_from_mirror:
                    df.loc[index, "ifFilter"] = False
                    df.loc[index, "match_type_class"] = 6
                else:
                    df.loc[index, "ifFilter"] = True
                    df.loc[index, "match_type_class"] = 7
                    debug_fout.write(f"{A_title}\t{B_title}\t{A_seq_from_single}\t{A_seq_from_mirror}\t{B_seq_from_mirror}\t{None}\t{7}\n")
                    # print(A_seq_from_single, A_seq_from_mirror, B_seq_from_mirror, B_seq_from_single)
            else:
                if (not A_seq_from_single_find) and B_seq_from_single_find:
                    if B_seq_from_single == B_seq_from_mirror:
                        df.loc[index, "ifFilter"] = False
                        df.loc[index, "match_type_class"] = 8
                    else:
                        df.loc[index, "ifFilter"] = True
                        df.loc[index, "match_type_class"] = 9
                        debug_fout.write(f"{A_title}\t{B_title}\t{None}\t{A_seq_from_mirror}\t{B_seq_from_mirror}\t{B_seq_from_single}\t{8}\n")
                        # print(A_seq_from_single, A_seq_from_mirror, B_seq_from_mirror, B_seq_from_single)
                else:
                    df.loc[index, "ifFilter"] = False
                    df.loc[index, "match_type_class"] = 10

    return df

            



if __name__ == "__main__":

    # DiNovo_res_folder_list = [
    #     r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MirrorNovo_output\mirror\20240525\23charge",
    #     r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\MirrorNovo_output\mirror\20240525\23charge"
    # ]
    # DiNovoPairs_list = [
    #     r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\spectraPairs\20240525\[DiNovo]SpectralPairs[23charge].res",
    #     r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\spectraPairs\20240525\[DiNovo]SpectralPairs[23charge].res"
    # ]
    # for index,DiNovo_res_folder in enumerate(DiNovo_res_folder_list):
    #     DiNovoTryLys_file = DiNovo_res_folder + "\\DiNovoTryLys_title_to_denovo_info.res"
    #     DiNovoTryLys_title_to_denovo_df = pd.read_csv(DiNovoTryLys_file, sep="\t")
    #
    #     DiNovoPairs_file = DiNovoPairs_list[index]
    #     (try_title_header_index,
    #      lys_title_header_index,
    #      match_type_header_index,
    #      titlesPair_to_location_dict) = generate_DiNovoPairs_to_matchtype_dict(DiNovoPairs_file)
    #
    #     DiNovoTryLys_title_to_denovo_df["Pep_mass_denovo"] = DiNovoTryLys_title_to_denovo_df.apply(lambda row: calculate_mass_mirror_from_specdf(row, titlesPair_to_location_dict), axis=1)
    #     DiNovoTryLys_title_to_denovo_df.to_csv(DiNovo_res_folder + "\\[withMass]DiNovoTryLys_title_to_denovo_info.res", sep="\t", index=False)
    # print(123)
    # input()


    fasta_seq = build_fasta_dict_function(fasta_file)
    fasta_seq_first_4aa_index = build_index_dict_function(fasta_seq)
    fasta_seq_decoy = generate_reverse_fasta(fasta_seq)
    fasta_seq_first_4aa_index_decoy = build_index_dict_function(fasta_seq_decoy)
    fasta_seq_targetAnddecoy = merge_target_and_decoy_fasta_seq(fasta_seq, fasta_seq_decoy)
    fasta_seq_first_4aa_index_targetAnddecoy = build_index_dict_function(fasta_seq_targetAnddecoy)

    (try_title_header_index,
     lys_title_header_index,
     match_type_header_index,
     titlesPair_to_location_dict) = generate_DiNovoPairs_to_matchtype_dict(DiNovoPairs_file)
    ###################################################################################################################################################################################################################################

    if DiNovo_tool == "MirrorNovo" or DiNovo_tool == "pNovoM2":
        if not os.path.exists(DiNovo_res_folder + "\\DiNovoTryLys_title_to_denovo_info.res"):
            assert False
        else:
            print("loading DiNovo targetRseq file...")
            DiNovoTryLys_file = DiNovo_res_folder + "\\DiNovoTryLys_title_to_denovo_info.res"
            DiNovoTryLys_title_to_denovo_df = pd.read_csv(DiNovoTryLys_file, sep="\t")
        DiNovoTryLys_title_to_denovo_df.loc[:,"A_title"] = DiNovoTryLys_title_to_denovo_df["title"].apply(lambda x:x.split("@")[0])
        DiNovoTryLys_title_to_denovo_df.loc[:,"B_title"] = DiNovoTryLys_title_to_denovo_df["title"].apply(lambda x:x.split("@")[1])
        DiNovoTryLys_title_to_denovo_df["Pep_mass_denovo"] = DiNovoTryLys_title_to_denovo_df.apply(lambda row: calculate_mass_mirror_from_specdf(row, titlesPair_to_location_dict), axis=1)
        DiNovoTryLys_title_to_denovo_df = DiNovoTryLys_title_to_denovo_df[DiNovoTryLys_title_to_denovo_df["Pep_mass_denovo"] >= massFilter12]

        if not os.path.exists(DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res"):
            assert False
        else:
            print("loading DiNovoTry_title_to_denovo_info.res file...")
            DiNovoTry_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res", sep="\t")
        DiNovoTry_title_to_denovo_df["Pep_mass_denovo"] = DiNovoTry_title_to_denovo_df.apply(lambda row: calculate_mass(row["denovoSeq"], if_delete_first_aa=False), axis=1)
        DiNovoTry_title_to_denovo_df = DiNovoTry_title_to_denovo_df[DiNovoTry_title_to_denovo_df["Pep_mass_denovo"] >= massFilter1]

        if not os.path.exists(DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res"):
            assert False
        else:
            print("loading DiNovoLys_title_to_denovo_info.res file...")
            DiNovoLys_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res", sep="\t")
        DiNovoLys_title_to_denovo_df["Pep_mass_denovo"] = DiNovoLys_title_to_denovo_df.apply(lambda row: calculate_mass(row["denovoSeq"], if_delete_first_aa=False), axis=1)
        DiNovoLys_title_to_denovo_df = DiNovoLys_title_to_denovo_df[DiNovoLys_title_to_denovo_df["Pep_mass_denovo"] >= massFilter2]

    if not os.path.exists(DiNovo_res_folder + f"\\DiNovoTryLys_fasta_coverage_denovo_info[{massFilter12}Da].res"):
        seq_to_unique_dict, seq_to_split_dict, DiNovoTryLys_fasta_coverage_denovo_dict, DiNovoTryLys_denovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_mirror( fasta_seq, DiNovoTryLys_title_to_denovo_df , False, massFilter12)
        write_dict_to_csv(DiNovoTryLys_fasta_coverage_denovo_dict,
                          DiNovo_res_folder + f"\\DiNovoTryLys_fasta_coverage_denovo_info[{massFilter12}Da].res", fasta_seq,
                          None, DiNovoTryLys_denovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTryLys_fasta_coverage_denovo_dict, DiNovoTryLys_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + f"\\DiNovoTryLys_fasta_coverage_denovo_info[{massFilter12}Da].res")

    if not os.path.exists(DiNovo_res_folder + f"\\DiNovoTryLys_fasta_coverage_Evidentdenovo_info[{massFilter12}Da].res"):
        _, _, DiNovoTryLys_fasta_coverage_Evidentdenovo_dict, DiNovoTryLys_Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_mirror(
            fasta_seq,
            DiNovoTryLys_title_to_denovo_df,
        True, massFilter12)
        write_dict_to_csv(DiNovoTryLys_fasta_coverage_Evidentdenovo_dict,
                          DiNovo_res_folder + f"\\DiNovoTryLys_fasta_coverage_Evidentdenovo_info[{massFilter12}Da].res",
                          fasta_seq, None, DiNovoTryLys_Evidentdenovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTryLys_fasta_coverage_Evidentdenovo_dict, DiNovoTryLys_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + f"\\DiNovoTryLys_fasta_coverage_Evidentdenovo_info[{massFilter12}Da].res")

    if not os.path.exists(DiNovo_res_folder + f"\\DiNovoTry_fasta_coverage_denovo_info[{massFilter1}Da].res"):
        seq_to_unique_dict, DiNovoTry_fasta_coverage_denovo_dict, DiNovoTry_denovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoTry_title_to_denovo_df,
            "denovo", False, massFilter1)
        write_dict_to_csv(DiNovoTry_fasta_coverage_denovo_dict,
                          DiNovo_res_folder + f"\\DiNovoTry_fasta_coverage_denovo_info[{massFilter1}Da].res", fasta_seq, None,
                          DiNovoTry_denovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTry_fasta_coverage_denovo_dict, DiNovoTry_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + f"\\DiNovoTry_fasta_coverage_denovo_info[{massFilter1}Da].res")

    if not os.path.exists(DiNovo_res_folder + f"\\DiNovoTry_fasta_coverage_Evidentdenovo_info[{massFilter1}Da].res"):
        _, DiNovoTry_fasta_coverage_Evidentdenovo_dict, DiNovoTry_Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoTry_title_to_denovo_df,
            "denovo", True, massFilter1)
        write_dict_to_csv(DiNovoTry_fasta_coverage_Evidentdenovo_dict,
                          DiNovo_res_folder + f"\\DiNovoTry_fasta_coverage_Evidentdenovo_info[{massFilter1}Da].res",
                          fasta_seq, None, DiNovoTry_Evidentdenovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTry_fasta_coverage_Evidentdenovo_dict, DiNovoTry_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + f"\\DiNovoTry_fasta_coverage_Evidentdenovo_info[{massFilter1}Da].res")

    if not os.path.exists(DiNovo_res_folder + f"\\DiNovoLys_fasta_coverage_denovo_info[{massFilter2}Da].res"):
        seq_to_unique_dict, DiNovoLys_fasta_coverage_denovo_dict, DiNovoLys_denovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoLys_title_to_denovo_df,
            "denovo", False, massFilter2)
        write_dict_to_csv(DiNovoLys_fasta_coverage_denovo_dict,
                          DiNovo_res_folder + f"\\DiNovoLys_fasta_coverage_denovo_info[{massFilter2}Da].res", fasta_seq, None,
                          DiNovoLys_denovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoLys_fasta_coverage_denovo_dict, DiNovoLys_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + f"\\DiNovoLys_fasta_coverage_denovo_info[{massFilter2}Da].res")

    if not os.path.exists(DiNovo_res_folder + f"\\DiNovoLys_fasta_coverage_Evidentdenovo_info[{massFilter2}Da].res"):
        _, DiNovoLys_fasta_coverage_Evidentdenovo_dict, DiNovoLys_Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoLys_title_to_denovo_df,
            "denovo", True, massFilter2)
        write_dict_to_csv(DiNovoLys_fasta_coverage_Evidentdenovo_dict,
                          DiNovo_res_folder + f"\\DiNovoLys_fasta_coverage_Evidentdenovo_info[{massFilter2}Da].res",
                          fasta_seq, None, DiNovoLys_Evidentdenovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoLys_fasta_coverage_Evidentdenovo_dict, DiNovoLys_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + f"\\DiNovoLys_fasta_coverage_Evidentdenovo_info[{massFilter2}Da].res")


    #生成TryBest和LysBest的title_to_denovo_info以及fasta_coverage
    intsection_info_fout = open(DiNovo_res_folder + f"\\info[{massFilter1}DaU{massFilter2}DaU{massFilter12}Da].res", "w")
    print("原始镜像谱共计",len(DiNovoTryLys_title_to_denovo_df),"对")
    intsection_info_fout.write("原始镜像谱共计" + str(len(DiNovoTryLys_title_to_denovo_df)) + "对\n")
    DiNovoTryLysFind_title_to_denovo_df = DiNovoTryLys_title_to_denovo_df[(DiNovoTryLys_title_to_denovo_df["ifFind_denovo"] == True) & (DiNovoTryLys_title_to_denovo_df["TDTag_denovo"] == "T")].copy()
    print("镜像谱回贴后共计",len(DiNovoTryLysFind_title_to_denovo_df),"对")
    intsection_info_fout.write("镜像谱回贴后共计" + str(len(DiNovoTryLysFind_title_to_denovo_df)) + "对\n")
    DiNovoTryLysFind_title_to_denovo_df.loc[:,"A_title"] = DiNovoTryLysFind_title_to_denovo_df["title"].apply(lambda x:x.split("@")[0])
    DiNovoTryLysFind_title_to_denovo_df.loc[:,"B_title"] = DiNovoTryLysFind_title_to_denovo_df["title"].apply(lambda x:x.split("@")[1])
    DiNovoTryLysFind_title_to_denovo_df.loc[:,"A_denovoSeq"] = DiNovoTryLysFind_title_to_denovo_df.apply(lambda row: generate_try_lys_seq_from_RseqAndMatchtype(row["denovoSeq"],row["match_type"])[0],axis=1)
    DiNovoTryLysFind_title_to_denovo_df.loc[:,"B_denovoSeq"] = DiNovoTryLysFind_title_to_denovo_df.apply(lambda row: generate_try_lys_seq_from_RseqAndMatchtype(row["denovoSeq"],row["match_type"])[1],axis=1)
    DiNovoTryFind_title_to_denovo_df = DiNovoTry_title_to_denovo_df[(DiNovoTry_title_to_denovo_df["ifFind_denovo"] == True) & (DiNovoTry_title_to_denovo_df["ifFindDecoy_denovo"] == False)]
    DiNovoLysFind_title_to_denovo_df = DiNovoLys_title_to_denovo_df[(DiNovoLys_title_to_denovo_df["ifFind_denovo"] == True) & (DiNovoLys_title_to_denovo_df["ifFindDecoy_denovo"] == False)]
    #过滤有矛盾的结果
    DiNovoTryFind_title_to_resRow = generate_title_to_res_dict(DiNovoTryFind_title_to_denovo_df)
    DiNovoLysFind_title_to_resRow = generate_title_to_res_dict(DiNovoLysFind_title_to_denovo_df)
    DiNovoTryLysFindFilter_title_to_denovo_df = filter_conflict_result(DiNovoTryLysFind_title_to_denovo_df,DiNovoTryFind_title_to_resRow,DiNovoLysFind_title_to_resRow)
    DiNovoTryLysFindFilter_title_to_denovo_df.to_csv(DiNovo_res_folder + f"\\DiNovoTryLysFindFilter_title_to_denovo_info[{massFilter1}DaU{massFilter2}DaU{massFilter12}Da].res", sep="\t", index=False)
    for i in range(10):
        type_class = i + 1
        print(f"类型{type_class}有{len(DiNovoTryLysFindFilter_title_to_denovo_df[DiNovoTryLysFindFilter_title_to_denovo_df['match_type_class'] == type_class])}对")
        intsection_info_fout.write(f"类型{type_class}有{len(DiNovoTryLysFindFilter_title_to_denovo_df[DiNovoTryLysFindFilter_title_to_denovo_df['match_type_class'] == type_class])}对\n")
    print(f"镜像谱回贴矛盾有{len(DiNovoTryLysFindFilter_title_to_denovo_df[DiNovoTryLysFindFilter_title_to_denovo_df['ifFilter'] == True])}对")
    intsection_info_fout.write(f"镜像谱回贴矛盾有{len(DiNovoTryLysFindFilter_title_to_denovo_df[DiNovoTryLysFindFilter_title_to_denovo_df['ifFilter'] == True])}对\n")
    DiNovoTryLysFindFilter_title_to_denovo_df = DiNovoTryLysFindFilter_title_to_denovo_df[DiNovoTryLysFindFilter_title_to_denovo_df["ifFilter"] == False]
    print(f"镜像谱回贴并去除矛盾后还剩{len(DiNovoTryLysFindFilter_title_to_denovo_df)}对")
    intsection_info_fout.write(f"镜像谱回贴并去除矛盾后还剩{len(DiNovoTryLysFindFilter_title_to_denovo_df)}对\n")

    #选择打分最高的PSM
    if not os.path.exists(DiNovo_res_folder + f"\\DiNovoTryBest_title_to_denovo_info[{massFilter1}DaU{massFilter12}Da].res"):
        DiNovoTryFindFilter_title_to_BestRow, DiNovoLysFindFilter_title_to_BestRow = get_maxscore_info(DiNovoTryLysFindFilter_title_to_denovo_df)
        A_titles = set(DiNovoTryFind_title_to_denovo_df["title"]) | set(DiNovoTryFindFilter_title_to_BestRow.keys())
        B_titles = set(DiNovoLysFind_title_to_denovo_df["title"]) | set(DiNovoLysFindFilter_title_to_BestRow.keys())
        (DiNovoTryBest_title_to_denovo_df,
         A_count1, A_count2, A_count21, A_count22, A_count3, A_count31, A_count32) = generate_Best_title_to_denovo_df(A_titles, DiNovoTryFind_title_to_denovo_df, DiNovoTryFindFilter_title_to_BestRow, "A")
        (DiNovoLysBest_title_to_denovo_df,
         B_count1, B_count2, B_count21, B_count22, B_count3, B_count31, B_count32) = generate_Best_title_to_denovo_df(B_titles, DiNovoLysFind_title_to_denovo_df, DiNovoLysFindFilter_title_to_BestRow, "B")
        DiNovoTryBest_title_to_denovo_df.to_csv(DiNovo_res_folder + f"\\DiNovoTryBest_title_to_denovo_info[{massFilter1}DaU{massFilter12}Da].res", sep="\t", index=False)
        DiNovoLysBest_title_to_denovo_df.to_csv(DiNovo_res_folder + f"\\DiNovoLysBest_title_to_denovo_info[{massFilter2}DaU{massFilter12}Da].res", sep="\t", index=False)
        #输出并写入文件
        print("Try共计鉴定到谱图数:", len(A_titles))
        intsection_info_fout.write("TryBest:\n")
        print("无单谱有镜像谱图：",A_count1)
        intsection_info_fout.write("无单谱有镜像谱图：" + str(A_count1) + "\n")
        print("有单谱有镜像谱图：",A_count2, "有单谱有镜像谱图且一致：",A_count21, "有单谱有镜像谱图且不一致：",A_count22)
        intsection_info_fout.write("有单谱有镜像谱图：" + str(A_count2) + "有单谱有镜像谱图且一致：" + str(A_count21) + "有单谱有镜像谱图且不一致：" + str(A_count22) + "\n")
        print("无镜像谱图：",A_count3, "无镜像谱有单谱：", A_count31, "无镜像谱无单谱：", A_count32)
        intsection_info_fout.write("无镜像谱图：" + str(A_count3) + "无镜像谱有单谱：" + str(A_count31) + "无镜像谱无单谱：" + str(A_count32) + "\n")
        print("Lys共计鉴定到普图数:", len(B_titles))
        intsection_info_fout.write("LysBest:\n")
        print("无单谱有镜像谱图：",B_count1)
        intsection_info_fout.write("无单谱有镜像谱图：" + str(B_count1) + "\n")
        print("有单谱有镜像谱图：",B_count2, "有单谱有镜像谱图且一致：",B_count21, "有单谱有镜像谱图且不一致：",B_count22)
        intsection_info_fout.write("有单谱有镜像谱图：" + str(B_count2) + "有单谱有镜像谱图且一致：" + str(B_count21) + "有单谱有镜像谱图且不一致：" + str(B_count22) + "\n")
        print("无镜像谱图：",B_count3, "无镜像谱有单谱：", B_count31, "无镜像谱无单谱：", B_count32)
        intsection_info_fout.write("无镜像谱图：" + str(B_count3) + "无镜像谱有单谱：" + str(B_count31) + "无镜像谱无单谱：" + str(B_count32) + "\n")
    else:
        DiNovoTryBest_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + f"\\DiNovoTryBest_title_to_denovo_info[{massFilter1}DaU{massFilter12}Da].res", sep="\t")
        DiNovoLysBest_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + f"\\DiNovoLysBest_title_to_denovo_info[{massFilter2}DaU{massFilter12}Da].res", sep="\t")

    if not os.path.exists(DiNovo_res_folder + f"\\DiNovoTryBest_fasta_coverage_denovo_info[{massFilter1}DaU{massFilter12}Da].res"):
        seq_str_set = set(DiNovoTryBest_title_to_denovo_df["denovoSeq"])
        top1_res_dict = mapped_function_single(seq_str_set, fasta_seq, fasta_seq_first_4aa_index)
        DiNovoTryBest_title_to_denovo_df["ifFind_denovo"] = DiNovoTryBest_title_to_denovo_df["denovoSeq"].apply(lambda x: top1_res_dict[x][1] if x == x else False)
        DiNovoTryBest_title_to_denovo_df["FindLocation_denovo"] = DiNovoTryBest_title_to_denovo_df["denovoSeq"].apply(lambda x: top1_res_dict[x][3] if x == x else "")
        top1_res_dict_decoy = mapped_function_single(seq_str_set, fasta_seq_decoy, fasta_seq_first_4aa_index_decoy)
        DiNovoTryBest_title_to_denovo_df["ifFindDecoy_denovo"] = DiNovoTryBest_title_to_denovo_df["denovoSeq"].apply(lambda x: top1_res_dict_decoy[x][2] if x == x else False)
        DiNovoTryBest_title_to_denovo_df["FindLocationDecoy_denovo"] = DiNovoTryBest_title_to_denovo_df["denovoSeq"].apply(lambda x: top1_res_dict_decoy[x][4] if x == x else "")
        _, DiNovoTryBest_fasta_coverage_denovo_dict, DiNovoTryBest_denovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoTryBest_title_to_denovo_df,
            "denovo", False, 0.0)
        DiNovoTryBest_title_to_denovo_df.to_csv(DiNovo_res_folder + f"\\DiNovoTryBest_title_to_denovo_info[{massFilter1}DaU{massFilter12}Da].res",index=False, sep ="\t")
        write_dict_to_csv(DiNovoTryBest_fasta_coverage_denovo_dict,
                          DiNovo_res_folder + f"\\DiNovoTryBest_fasta_coverage_denovo_info[{massFilter1}DaU{massFilter12}Da].res",
                          fasta_seq, None, DiNovoTryBest_denovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTryBest_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + f"\\DiNovoTryBest_title_to_denovo_info[{massFilter1}DaU{massFilter12}Da].res", sep="\t")
        DiNovoTryBest_fasta_coverage_denovo_dict, DiNovoTryBest_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + f"\\DiNovoTryBest_fasta_coverage_denovo_info[{massFilter1}DaU{massFilter12}Da].res")

    if not os.path.exists(DiNovo_res_folder + f"\\DiNovoTryBest_fasta_coverage_Evidentdenovo_info[{massFilter1}DaU{massFilter12}Da].res"):
        _, DiNovoTryBest_fasta_coverage_Evidentdenovo_dict, DiNovoTryBest_Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoTryBest_title_to_denovo_df,
            "denovo", True, 0.0)
        write_dict_to_csv(DiNovoTryBest_fasta_coverage_Evidentdenovo_dict,
                          DiNovo_res_folder + f"\\DiNovoTryBest_fasta_coverage_Evidentdenovo_info[{massFilter1}DaU{massFilter12}Da].res",
                          fasta_seq, None, DiNovoTryBest_Evidentdenovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTryBest_fasta_coverage_Evidentdenovo_dict, DiNovoTryBest_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + f"\\DiNovoTryBest_fasta_coverage_Evidentdenovo_info[{massFilter1}DaU{massFilter12}Da].res")

    if not os.path.exists(DiNovo_res_folder + f"\\DiNovoLysBest_fasta_coverage_denovo_info[{massFilter2}DaU{massFilter12}Da].res"):
        seq_str_set = set(DiNovoLysBest_title_to_denovo_df["denovoSeq"])
        top1_res_dict = mapped_function_single(seq_str_set, fasta_seq, fasta_seq_first_4aa_index)
        DiNovoLysBest_title_to_denovo_df["ifFind_denovo"] = DiNovoLysBest_title_to_denovo_df["denovoSeq"].apply(lambda x: top1_res_dict[x][1] if x == x else False)
        DiNovoLysBest_title_to_denovo_df["FindLocation_denovo"] = DiNovoLysBest_title_to_denovo_df["denovoSeq"].apply(lambda x: top1_res_dict[x][3] if x == x else "")
        top1_res_dict_decoy = mapped_function_single(seq_str_set, fasta_seq_decoy, fasta_seq_first_4aa_index_decoy)
        DiNovoLysBest_title_to_denovo_df["ifFindDecoy_denovo"] = DiNovoLysBest_title_to_denovo_df["denovoSeq"].apply(lambda x: top1_res_dict_decoy[x][2] if x == x else False)
        DiNovoLysBest_title_to_denovo_df["FindLocationDecoy_denovo"] = DiNovoLysBest_title_to_denovo_df["denovoSeq"].apply(lambda x: top1_res_dict_decoy[x][4] if x == x else "")
        _, DiNovoLysBest_fasta_coverage_denovo_dict, DiNovoLysBest_denovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoLysBest_title_to_denovo_df,
            "denovo", False, 0.0)
        DiNovoLysBest_title_to_denovo_df.to_csv(DiNovo_res_folder + f"\\DiNovoLysBest_title_to_denovo_info[{massFilter2}DaU{massFilter12}Da].res",index=False, sep ="\t")
        write_dict_to_csv(DiNovoLysBest_fasta_coverage_denovo_dict,
                          DiNovo_res_folder + f"\\DiNovoLysBest_fasta_coverage_denovo_info[{massFilter2}DaU{massFilter12}Da].res",
                          fasta_seq, None, DiNovoLysBest_denovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoLysBest_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + f"\\DiNovoLysBest_title_to_denovo_info[{massFilter2}DaU{massFilter12}Da].res", sep="\t")
        DiNovoLysBest_fasta_coverage_denovo_dict, DiNovoLysBest_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + f"\\DiNovoLysBest_fasta_coverage_denovo_info[{massFilter2}DaU{massFilter12}Da].res")

    if not os.path.exists(DiNovo_res_folder + f"\\DiNovoLysBest_fasta_coverage_Evidentdenovo_info[{massFilter2}DaU{massFilter12}Da].res"):
        _, DiNovoLysBest_fasta_coverage_Evidentdenovo_dict, DiNovoLysBest_Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoLysBest_title_to_denovo_df,
            "denovo", True, 0.0)
        write_dict_to_csv(DiNovoLysBest_fasta_coverage_Evidentdenovo_dict,
                          DiNovo_res_folder + f"\\DiNovoLysBest_fasta_coverage_Evidentdenovo_info[{massFilter2}DaU{massFilter12}Da].res",
                          fasta_seq, None, DiNovoLysBest_Evidentdenovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoLysBest_fasta_coverage_Evidentdenovo_dict, DiNovoLysBest_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + f"\\DiNovoLysBest_fasta_coverage_Evidentdenovo_info[{massFilter2}DaU{massFilter12}Da].res")

    # 把DiNovo镜像谱和单谱合并在一起
    plot_aa_intsection_venn3(DiNovoTry_fasta_coverage_denovo_dict, DiNovoLys_fasta_coverage_denovo_dict,
                             DiNovoTryLys_fasta_coverage_denovo_dict,
                             DiNovo_res_folder + f"\\intsection_denovo_AA[DiNovoNei][{massFilter1}DaU{massFilter2}DaU{massFilter12}Da].png", fig_name1,
                             fig_name2, fig_name1 + "@" + fig_name2)
    DiNovo_fasta_coverage_denovo_dict = merge_fasta_coverage_dict(
        [DiNovoTry_fasta_coverage_denovo_dict, DiNovoLys_fasta_coverage_denovo_dict,
         DiNovoTryLys_fasta_coverage_denovo_dict])
    DiNovo_denovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
        [DiNovoTry_denovo_proteinName_to_uniqueSeq_dict, DiNovoLys_denovo_proteinName_to_uniqueSeq_dict,
         DiNovoTryLys_denovo_proteinName_to_uniqueSeq_dict])
    write_dict_to_csv(DiNovo_fasta_coverage_denovo_dict,
                      DiNovo_res_folder + f"\\Union_fasta_coverage_denovo_info[DiNovoNei][{massFilter1}DaU{massFilter2}DaU{massFilter12}Da].res", fasta_seq,
                      None, DiNovo_denovo_proteinName_to_uniqueSeq_dict)

    plot_aa_intsection_venn3(DiNovoTry_fasta_coverage_Evidentdenovo_dict,
                             DiNovoLys_fasta_coverage_Evidentdenovo_dict,
                             DiNovoTryLys_fasta_coverage_Evidentdenovo_dict,
                             DiNovo_res_folder + f"\\intsection_Evidentdenovo_AA[DiNovoNei][{massFilter1}DaU{massFilter2}DaU{massFilter12}Da].png", fig_name1,
                             fig_name2, fig_name1 + "@" + fig_name2)
    DiNovo_fasta_coverage_Evidentdenovo_dict = merge_fasta_coverage_dict(
        [DiNovoTry_fasta_coverage_Evidentdenovo_dict, DiNovoLys_fasta_coverage_Evidentdenovo_dict,
         DiNovoTryLys_fasta_coverage_Evidentdenovo_dict])
    DiNovo_Evidentdenovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
        [DiNovoTry_Evidentdenovo_proteinName_to_uniqueSeq_dict,
         DiNovoLys_Evidentdenovo_proteinName_to_uniqueSeq_dict,
         DiNovoTryLys_Evidentdenovo_proteinName_to_uniqueSeq_dict])
    write_dict_to_csv(DiNovo_fasta_coverage_Evidentdenovo_dict,
                      DiNovo_res_folder + f"\\Union_fasta_coverage_Evidentdenovo_info[DiNovoNei][{massFilter1}DaU{massFilter2}DaU{massFilter12}Da].res",
                      fasta_seq, None, DiNovo_Evidentdenovo_proteinName_to_uniqueSeq_dict)

    #TryBest和LysBest合并
    plot_aa_intsection_venn2(DiNovoTryBest_fasta_coverage_denovo_dict, DiNovoLysBest_fasta_coverage_denovo_dict,
                             DiNovo_res_folder + f"\\intsection_denovo_AA[DiNovoNeiBest][{massFilter1}DaU{massFilter12}DaU{massFilter2}DaU{massFilter12}Da].png", fig_name1,
                             fig_name2)
    DiNovoBest_fasta_coverage_denovo_dict = merge_fasta_coverage_dict(
        [DiNovoTryBest_fasta_coverage_denovo_dict, DiNovoLysBest_fasta_coverage_denovo_dict])
    DiNovoBest_denovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
        [DiNovoTryBest_denovo_proteinName_to_uniqueSeq_dict, DiNovoLysBest_denovo_proteinName_to_uniqueSeq_dict])
    write_dict_to_csv(DiNovoBest_fasta_coverage_denovo_dict,
                      DiNovo_res_folder + f"\\Union_fasta_coverage_denovo_info[DiNovoNeiBest][{massFilter1}DaU{massFilter12}DaU{massFilter2}DaU{massFilter12}Da].res", fasta_seq,
                      None, DiNovoBest_denovo_proteinName_to_uniqueSeq_dict)

    plot_aa_intsection_venn2(DiNovoTryBest_fasta_coverage_Evidentdenovo_dict,
                             DiNovoLysBest_fasta_coverage_Evidentdenovo_dict,
                             DiNovo_res_folder + f"\\intsection_Evidentdenovo_AA[DiNovoNeiBest][{massFilter1}DaU{massFilter12}DaU{massFilter2}DaU{massFilter12}Da].png", fig_name1,
                             fig_name2)
    DiNovoBest_fasta_coverage_Evidentdenovo_dict = merge_fasta_coverage_dict(
        [DiNovoTryBest_fasta_coverage_Evidentdenovo_dict, DiNovoLysBest_fasta_coverage_Evidentdenovo_dict])
    DiNovoBest_Evidentdenovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
        [DiNovoTryBest_Evidentdenovo_proteinName_to_uniqueSeq_dict,
         DiNovoLysBest_Evidentdenovo_proteinName_to_uniqueSeq_dict])
    write_dict_to_csv(DiNovoBest_fasta_coverage_Evidentdenovo_dict,
                      DiNovo_res_folder + f"\\Union_fasta_coverage_Evidentdenovo_info[DiNovoNeiBest][{massFilter1}DaU{massFilter12}DaU{massFilter2}DaU{massFilter12}Da].res",
                      fasta_seq, None, DiNovoBest_Evidentdenovo_proteinName_to_uniqueSeq_dict)

    #######################################################################
    #肽段
    DiNovoTryLys_evident_spec_num_denovo = 0
    DiNovoTryLys_NonEvident_spec_num_denovo = 0
    DiNovoTryLys_evident_find_spec_num_denovo = 0
    DiNovoTryLys_NonEvident_find_spec_num_denovo = 0
    DiNovoTryLys_denovoSeq_to_info_dict = {}
    for index, row in DiNovoTryLys_title_to_denovo_df.iterrows():
        if index % 1000 == 0:
            print(f"\r", index, "/", len(DiNovoTryLys_title_to_denovo_df), end="")
        title, denovo_seq, ifEvident, ifFind = row["title"], row["denovoSeq"], row["ifEvident_denovo"], row["ifFind_denovo"]
        TDTag = row["TDTag_denovo"]
        assert TDTag in set(["T","D","TD","NoMatch"]), f"TDTag不在范围内{TDTag}啊"
        if ifEvident:
            DiNovoTryLys_evident_spec_num_denovo += 1
            if ifFind and TDTag == "T":
                DiNovoTryLys_evident_find_spec_num_denovo += 1
        else:
            DiNovoTryLys_NonEvident_spec_num_denovo += 1
            if ifFind and TDTag == "T":
                DiNovoTryLys_NonEvident_find_spec_num_denovo += 1
        if denovo_seq == denovo_seq:
            if denovo_seq not in DiNovoTryLys_denovoSeq_to_info_dict:
                DiNovoTryLys_denovoSeq_to_info_dict[denovo_seq] = [ifEvident, ifFind, TDTag,
                                                                       [(title, ifEvident, ifFind, TDTag)]]
            else:
                ifEvident_new = DiNovoTryLys_denovoSeq_to_info_dict[denovo_seq][0] | ifEvident
                ifFind_new = DiNovoTryLys_denovoSeq_to_info_dict[denovo_seq][1] | ifFind
                if TDTag != DiNovoTryLys_denovoSeq_to_info_dict[denovo_seq][2]:
                    if len(TDTag) < len(DiNovoTryLys_denovoSeq_to_info_dict[denovo_seq][2]):
                        pass
                    elif len(TDTag) > len(DiNovoTryLys_denovoSeq_to_info_dict[denovo_seq][2]):
                        TDTag = DiNovoTryLys_denovoSeq_to_info_dict[denovo_seq][2]
                    else:
                        print("TDTag不一致")
                        print(title, denovo_seq, ifEvident, ifFind, TDTag)
                        print(DiNovoTryLys_denovoSeq_to_info_dict[denovo_seq])
                        assert False
                DiNovoTryLys_denovoSeq_to_info_dict[denovo_seq] = [ifEvident_new, ifFind_new, TDTag,
                                                                       DiNovoTryLys_denovoSeq_to_info_dict[
                                                                           denovo_seq][3] + [
                                                                           (title, ifEvident, ifFind, TDTag)]]
    
    intsection_info_fout = open(DiNovo_res_folder + f"\\info[{massFilter1}DaU{massFilter2}DaU{massFilter12}Da].res", "a+")
    DiNovoTryLys_total_spec_num = len(DiNovoTryLys_title_to_denovo_df)
    string = f"DiNovoTryLys_total_spec_num\t{DiNovoTryLys_total_spec_num}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryLys_denovoSeq_evident_spec_num\t{DiNovoTryLys_evident_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryLys_denovoSeq_evident_find_spec_num\t{DiNovoTryLys_evident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryLys_denovoSeq_NonEvident_spec_num\t{DiNovoTryLys_NonEvident_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryLys_denovoSeq_NonEvident_find_spec_num\t{DiNovoTryLys_NonEvident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryLys_denovoSeq_find_spe_num\t{DiNovoTryLys_evident_find_spec_num_denovo + DiNovoTryLys_NonEvident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)

    DiNovoTry_evident_spec_num_denovo = 0
    DiNovoTry_evident_find_spec_num_denovo = 0
    DiNovoTry_NonEvident_spec_num_denovo = 0
    DiNovoTry_NonEvident_find_spec_num_denovo = 0
    DiNovoTry_denovoSeq_to_info_dict = {}
    for index, row in DiNovoTry_title_to_denovo_df.iterrows():
        if index % 1000 == 0:
            print(f"\r", index, "/", len(DiNovoTry_title_to_denovo_df), end="")
        title, denovo_seq, ifEvident, ifFind = row["title"], row["denovoSeq"], row["ifEvident_denovo"], row["ifFind_denovo"]
        ifFindDecoy = row["ifFindDecoy_denovo"]
        if ifEvident:
            DiNovoTry_evident_spec_num_denovo += 1
            if ifFind and not ifFindDecoy:
                DiNovoTry_evident_find_spec_num_denovo += 1
        else:
            DiNovoTry_NonEvident_spec_num_denovo += 1
            if ifFind and not ifFindDecoy:
                DiNovoTry_NonEvident_find_spec_num_denovo += 1
        if denovo_seq == denovo_seq:
            if denovo_seq not in DiNovoTry_denovoSeq_to_info_dict:
                DiNovoTry_denovoSeq_to_info_dict[denovo_seq] = [ifEvident, ifFind, ifFindDecoy,
                                                                    [(title, ifEvident, ifFind, ifFindDecoy)]]
            else:
                ifEvident_new = DiNovoTry_denovoSeq_to_info_dict[denovo_seq][0] | ifEvident
                ifFind_new = DiNovoTry_denovoSeq_to_info_dict[denovo_seq][1] | ifFind
                ifFindDecoy_new = DiNovoTry_denovoSeq_to_info_dict[denovo_seq][2] | ifFindDecoy
                DiNovoTry_denovoSeq_to_info_dict[denovo_seq] = [ifEvident_new, ifFind_new, ifFindDecoy_new,
                                                                    DiNovoTry_denovoSeq_to_info_dict[denovo_seq][
                                                                        3] + [
                                                                        (title, ifEvident, ifFind, ifFindDecoy)]]

    DiNovoTry_total_spec_num = len(DiNovoTry_title_to_denovo_df)
    string = f"DiNovoTry_total_spec_num\t{DiNovoTry_total_spec_num}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTry_denovoSeq_evident_spec_num\t{DiNovoTry_evident_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTry_denovoSeq_evident_find_spec_num\t{DiNovoTry_evident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTry_denovoSeq_NonEvident_spec_num\t{DiNovoTry_NonEvident_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTry_denovoSeq_NonEvident_find_spec_num\t{DiNovoTry_NonEvident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTry_denovoSeq_find_spe_num\t{DiNovoTry_evident_find_spec_num_denovo + DiNovoTry_NonEvident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)

    DiNovoLys_evident_spec_num_denovo = 0
    DiNovoLys_evident_find_spec_num_denovo = 0
    DiNovoLys_NonEvident_spec_num_denovo = 0
    DiNovoLys_NonEvident_find_spec_num_denovo = 0
    DiNovoLys_denovoSeq_to_info_dict = {}
    for index, row in DiNovoLys_title_to_denovo_df.iterrows():
        if index % 1000 == 0:
            print(f"\r", index, "/", len(DiNovoLys_title_to_denovo_df), end="")
        title, denovo_seq, ifEvident, ifFind = row["title"], row["denovoSeq"], row["ifEvident_denovo"], row[
            "ifFind_denovo"]
        ifFindDecoy = row["ifFindDecoy_denovo"]
        if ifEvident:
            DiNovoLys_evident_spec_num_denovo += 1
            if ifFind and not ifFindDecoy:
                DiNovoLys_evident_find_spec_num_denovo += 1
        else:
            DiNovoLys_NonEvident_spec_num_denovo += 1
            if ifFind and not ifFindDecoy:
                DiNovoLys_NonEvident_find_spec_num_denovo += 1
        if denovo_seq == denovo_seq:
            if denovo_seq not in DiNovoLys_denovoSeq_to_info_dict:
                DiNovoLys_denovoSeq_to_info_dict[denovo_seq] = [ifEvident, ifFind, ifFindDecoy,
                                                                    [(title, ifEvident, ifFind, ifFindDecoy)]]
            else:
                ifEvident_new = DiNovoLys_denovoSeq_to_info_dict[denovo_seq][0] | ifEvident
                ifFind_new = DiNovoLys_denovoSeq_to_info_dict[denovo_seq][1] | ifFind
                ifFindDecoy_new = DiNovoLys_denovoSeq_to_info_dict[denovo_seq][2] | ifFindDecoy
                DiNovoLys_denovoSeq_to_info_dict[denovo_seq] = [ifEvident_new, ifFind_new, ifFindDecoy_new,
                                                                    DiNovoLys_denovoSeq_to_info_dict[denovo_seq][
                                                                        3] + [
                                                                        (title, ifEvident, ifFind, ifFindDecoy)]]

    DiNovoLys_total_spec_num = len(DiNovoLys_title_to_denovo_df)
    string = f"DiNovoLys_total_spec_num\t{DiNovoLys_total_spec_num}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLys_denovoSeq_evident_spec_num\t{DiNovoLys_evident_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLys_denovoSeq_evident_find_spec_num\t{DiNovoLys_evident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLys_denovoSeq_NonEvident_spec_num\t{DiNovoLys_NonEvident_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLys_denovoSeq_NonEvident_find_spec_num\t{DiNovoLys_NonEvident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLys_denovoSeq_find_spe_num\t{DiNovoLys_evident_find_spec_num_denovo + DiNovoLys_NonEvident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    
    #Best的结果
    DiNovoTryBest_evident_spec_num_denovo = 0
    DiNovoTryBest_evident_find_spec_num_denovo = 0
    DiNovoTryBest_NonEvident_spec_num_denovo = 0
    DiNovoTryBest_NonEvident_find_spec_num_denovo = 0
    DiNovoTryBest_denovoSeq_to_info_dict = {}
    for index, row in DiNovoTryBest_title_to_denovo_df.iterrows():
        if index % 1000 == 0:
            print(f"\r", index, "/", len(DiNovoTryBest_title_to_denovo_df), end="")
        title, denovo_seq, ifEvident, ifFind = row["title"], row["denovoSeq"], row["ifEvident_denovo"], row["ifFind_denovo"]
        match_type_class = row["match_type_class"]
        if ifEvident:
            DiNovoTryBest_evident_spec_num_denovo += 1
            if ifFind :
                DiNovoTryBest_evident_find_spec_num_denovo += 1
        else:
            DiNovoTryBest_NonEvident_spec_num_denovo += 1
            if ifFind:
                DiNovoTryBest_NonEvident_find_spec_num_denovo += 1
        if denovo_seq == denovo_seq:
            if denovo_seq not in DiNovoTryBest_denovoSeq_to_info_dict:
                DiNovoTryBest_denovoSeq_to_info_dict[denovo_seq] = [ifEvident, ifFind,
                                                                [(title, ifEvident, ifFind, match_type_class)]]
            else:
                ifEvident_new = DiNovoTryBest_denovoSeq_to_info_dict[denovo_seq][0] | ifEvident
                ifFind_new = DiNovoTryBest_denovoSeq_to_info_dict[denovo_seq][1] | ifFind
                DiNovoTryBest_denovoSeq_to_info_dict[denovo_seq] = [ifEvident_new, ifFind_new,
                                                                DiNovoTryBest_denovoSeq_to_info_dict[denovo_seq][2] + [(title, ifEvident, ifFind, match_type_class)]]
    DiNovoTryBest_denovoSeq_info_fw = open(DiNovo_res_folder + f"\\DiNovoTryBest_denovoSeq_to_ifEvidentFind[{massFilter1}DaU{massFilter12}Da].res", "w")
    header = "denovoSeq_DiNovoTryBest\tifEvident_DiNovoTryBest\tifFind_DiNovoTryBest\ttitles_DiNovoTryBest\n"
    DiNovoTryBest_denovoSeq_info_fw.write(header)
    for key in DiNovoTryBest_denovoSeq_to_info_dict:
        string = f"{key}\t{DiNovoTryBest_denovoSeq_to_info_dict[key][0]}\t{DiNovoTryBest_denovoSeq_to_info_dict[key][1]}\t{DiNovoTryBest_denovoSeq_to_info_dict[key][2]}\n"
        DiNovoTryBest_denovoSeq_info_fw.write(string)
    DiNovoTryBest_denovoSeq_info_fw.flush()
    DiNovoTryBest_denovoSeq_info_fw.close()
    
    DiNovoTryBest_total_spec_num = len(DiNovoTryBest_title_to_denovo_df)
    string = f"DiNovoTryBest_total_spec_num\t{DiNovoTryBest_total_spec_num}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryBest_denovoSeq_evident_spec_num\t{DiNovoTryBest_evident_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryBest_denovoSeq_evident_find_spec_num\t{DiNovoTryBest_evident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryBest_denovoSeq_NonEvident_spec_num\t{DiNovoTryBest_NonEvident_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryBest_denovoSeq_NonEvident_find_spec_num\t{DiNovoTryBest_NonEvident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryBest_denovoSeq_find_spe_num\t{DiNovoTryBest_evident_find_spec_num_denovo + DiNovoTryBest_NonEvident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)

    DiNovoLysBest_evident_spec_num_denovo = 0
    DiNovoLysBest_evident_find_spec_num_denovo = 0
    DiNovoLysBest_NonEvident_spec_num_denovo = 0
    DiNovoLysBest_NonEvident_find_spec_num_denovo = 0
    DiNovoLysBest_denovoSeq_to_info_dict = {}
    for index, row in DiNovoLysBest_title_to_denovo_df.iterrows():
        if index % 1000 == 0:
            print(f"\r", index, "/", len(DiNovoLysBest_title_to_denovo_df), end="")
        title, denovo_seq, ifEvident, ifFind = row["title"], row["denovoSeq"], row["ifEvident_denovo"], row["ifFind_denovo"]
        match_type_class = row["match_type_class"]
        if ifEvident:
            DiNovoLysBest_evident_spec_num_denovo += 1
            if ifFind :
                DiNovoLysBest_evident_find_spec_num_denovo += 1
        else:
            DiNovoLysBest_NonEvident_spec_num_denovo += 1
            if ifFind :
                DiNovoLysBest_NonEvident_find_spec_num_denovo += 1
        if denovo_seq == denovo_seq:
            if denovo_seq not in DiNovoLysBest_denovoSeq_to_info_dict:
                DiNovoLysBest_denovoSeq_to_info_dict[denovo_seq] = [ifEvident, ifFind,
                                                                    [(title, ifEvident, ifFind,match_type_class)]]
            else:
                ifEvident_new = DiNovoLysBest_denovoSeq_to_info_dict[denovo_seq][0] | ifEvident
                ifFind_new = DiNovoLysBest_denovoSeq_to_info_dict[denovo_seq][1] | ifFind
                DiNovoLysBest_denovoSeq_to_info_dict[denovo_seq] = [ifEvident_new, ifFind_new,
                                                                    DiNovoLysBest_denovoSeq_to_info_dict[denovo_seq][
                                                                        2] + [
                                                                        (title, ifEvident, ifFind, match_type_class)]]
    DiNovoLysBest_denovoSeq_info_fw = open(
        DiNovo_res_folder + f"\\DiNovoLysBest_denovoSeq_to_ifEvidentFind[{massFilter2}DaU{massFilter12}Da].res", "w")
    header = "denovoSeq_DiNovoLysBest\tifEvident_DiNovoLysBest\tifFind_DiNovoLysBest\ttitles_DiNovoLysBest\n"
    DiNovoLysBest_denovoSeq_info_fw.write(header)
    for key in DiNovoLysBest_denovoSeq_to_info_dict:
        string = f"{key}\t{DiNovoLysBest_denovoSeq_to_info_dict[key][0]}\t{DiNovoLysBest_denovoSeq_to_info_dict[key][1]}\t{DiNovoLysBest_denovoSeq_to_info_dict[key][2]}\n"
        DiNovoLysBest_denovoSeq_info_fw.write(string)
    DiNovoLysBest_denovoSeq_info_fw.flush()
    DiNovoLysBest_denovoSeq_info_fw.close()

    DiNovoLysBest_total_spec_num = len(DiNovoLysBest_title_to_denovo_df)
    string = f"DiNovoLysBest_total_spec_num\t{DiNovoLysBest_total_spec_num}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLysBest_denovoSeq_evident_spec_num\t{DiNovoLysBest_evident_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLysBest_denovoSeq_evident_find_spec_num\t{DiNovoLysBest_evident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLysBest_denovoSeq_NonEvident_spec_num\t{DiNovoLysBest_NonEvident_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLysBest_denovoSeq_NonEvident_find_spec_num\t{DiNovoLysBest_NonEvident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLysBest_denovoSeq_find_spe_num\t{DiNovoLysBest_evident_find_spec_num_denovo + DiNovoLysBest_NonEvident_find_spec_num_denovo}\n"
    intsection_info_fout.write(string)

    #intsection of peptide
    DiNovoTry_denovoSeq_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTry_denovoSeq_to_ifEvidentFind.res", sep="\t")
    DiNovoTry_denovoSeq_df = DiNovoTry_denovoSeq_df[DiNovoTry_denovoSeq_df["Pep_mass"] >= massFilter1]
    DiNovoLys_denovoSeq_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoLys_denovoSeq_to_ifEvidentFind.res", sep="\t")
    DiNovoLys_denovoSeq_df = DiNovoLys_denovoSeq_df[DiNovoLys_denovoSeq_df["Pep_mass"] >= massFilter2]
    DiNovoTryLys_denovoSeq_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTryLys_denovoSeq_to_ifEvidentFind.res", sep="\t")
    DiNovoTryLys_denovoSeq_df = DiNovoTryLys_denovoSeq_df[DiNovoTryLys_denovoSeq_df["Pep_mass"] >= massFilter12]
    DiNovoTryBest_denovoSeq_df = pd.read_csv(DiNovo_res_folder + f"\\DiNovoTryBest_denovoSeq_to_ifEvidentFind[{massFilter1}DaU{massFilter12}Da].res", sep="\t")
    DiNovoLysBest_denovoSeq_df = pd.read_csv(DiNovo_res_folder + f"\\DiNovoLysBest_denovoSeq_to_ifEvidentFind[{massFilter2}DaU{massFilter12}Da].res", sep="\t")

    DiNovoTry_denovoSeq_Find_set = set(DiNovoTry_denovoSeq_df[(DiNovoTry_denovoSeq_df["ifFind_DiNovoTry"] == True) & (DiNovoTry_denovoSeq_df["ifFindDecoy_DiNovoTry"] == False)]["denovoSeq_DiNovoTry"])
    DiNovoTry_denovoSeq_Find_Evident_set = set(DiNovoTry_denovoSeq_df[(DiNovoTry_denovoSeq_df["ifFind_DiNovoTry"] == True) & (DiNovoTry_denovoSeq_df["ifFindDecoy_DiNovoTry"] == False) & (DiNovoTry_denovoSeq_df["ifEvident_DiNovoTry"] == True)]["denovoSeq_DiNovoTry"])
    DiNovoLys_denovoSeq_Find_set = set(DiNovoLys_denovoSeq_df[(DiNovoLys_denovoSeq_df["ifFind_DiNovoLys"] == True) & (DiNovoLys_denovoSeq_df["ifFindDecoy_DiNovoLys"] == False)]["denovoSeq_DiNovoLys"])
    DiNovoLys_denovoSeq_Find_Evident_set = set(DiNovoLys_denovoSeq_df[(DiNovoLys_denovoSeq_df["ifFind_DiNovoLys"] == True) & (DiNovoLys_denovoSeq_df["ifFindDecoy_DiNovoLys"] == False) & (DiNovoLys_denovoSeq_df["ifEvident_DiNovoLys"] == True)]["denovoSeq_DiNovoLys"])
    DiNovoTryLys_denovoSeq_Find_set = set(DiNovoTryLys_denovoSeq_df[(DiNovoTryLys_denovoSeq_df["ifFind_DiNovoTryLys"] == True) & (DiNovoTryLys_denovoSeq_df["TDTag_DiNovoTryLys"] == "T")]["denovoSeq_DiNovoTryLys"])
    DiNovoTryLys_denovoSeq_Find_Evident_set = set(DiNovoTryLys_denovoSeq_df[(DiNovoTryLys_denovoSeq_df["ifFind_DiNovoTryLys"] == True) & (DiNovoTryLys_denovoSeq_df["TDTag_DiNovoTryLys"] == "T") & (DiNovoTryLys_denovoSeq_df["ifEvident_DiNovoTryLys"] == True)]["denovoSeq_DiNovoTryLys"])
    DiNovoTryBest_denovoSeq_Find_set = set(DiNovoTryBest_denovoSeq_df[(DiNovoTryBest_denovoSeq_df["ifFind_DiNovoTryBest"] == True)]["denovoSeq_DiNovoTryBest"])
    DiNovoTryBest_denovoSeq_Find_Evident_set = set(DiNovoTryBest_denovoSeq_df[(DiNovoTryBest_denovoSeq_df["ifFind_DiNovoTryBest"] == True) & (DiNovoTryBest_denovoSeq_df["ifEvident_DiNovoTryBest"] == True)]["denovoSeq_DiNovoTryBest"])
    DiNovoLysBest_denovoSeq_Find_set = set(DiNovoLysBest_denovoSeq_df[(DiNovoLysBest_denovoSeq_df["ifFind_DiNovoLysBest"] == True)]["denovoSeq_DiNovoLysBest"])
    DiNovoLysBest_denovoSeq_Find_Evident_set = set(DiNovoLysBest_denovoSeq_df[(DiNovoLysBest_denovoSeq_df["ifFind_DiNovoLysBest"] == True) & (DiNovoLysBest_denovoSeq_df["ifEvident_DiNovoLysBest"] == True)]["denovoSeq_DiNovoLysBest"])
    DiNovoBest_denovoSeq_Find_set = DiNovoTryBest_denovoSeq_Find_set | DiNovoLysBest_denovoSeq_Find_set
    DiNovoBest_denovoSeq_Find_Evident_set = DiNovoTryBest_denovoSeq_Find_Evident_set | DiNovoLysBest_denovoSeq_Find_Evident_set
    string = f"DiNovoTry denovoSeq Find num: {len(DiNovoTry_denovoSeq_Find_set)}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTry dneovoSeq Find Evident num: {len(DiNovoTry_denovoSeq_Find_Evident_set)}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLys denovoSeq Find num: {len(DiNovoLys_denovoSeq_Find_set)}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLys denovoSeq Find Evident num: {len(DiNovoLys_denovoSeq_Find_Evident_set)}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryLys denovoSeq Find num: {len(DiNovoTryLys_denovoSeq_Find_set)}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryLys denovoSeq Find Evident num: {len(DiNovoTryLys_denovoSeq_Find_Evident_set)}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryBest denovoSeq Find num: {len(DiNovoTryBest_denovoSeq_Find_set)}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoTryBest denovoSeq Find Evident num: {len(DiNovoTryBest_denovoSeq_Find_Evident_set)}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLysBest denovoSeq Find num: {len(DiNovoLysBest_denovoSeq_Find_set)}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoLysBest denovoSeq Find Evident num: {len(DiNovoLysBest_denovoSeq_Find_Evident_set)}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoBest denovoSeq Find num: {len(DiNovoBest_denovoSeq_Find_set)}\n"
    intsection_info_fout.write(string)
    string = f"DiNovoBest denovoSeq Find Evident num: {len(DiNovoBest_denovoSeq_Find_Evident_set)}\n"
    intsection_info_fout.write(string)
    
    ##########################################################################################################################
    #谱图水平
    DiNovoTry_total_spec_set = set(DiNovoTryFind_title_to_denovo_df["title"])
    DiNovoTry_Evident_spec_set = set(DiNovoTryFind_title_to_denovo_df[DiNovoTryFind_title_to_denovo_df["ifEvident_denovo"] == True]["title"])
    DiNovoLys_total_spec_set = set(DiNovoLysFind_title_to_denovo_df["title"])
    DiNovoLys_Evident_spec_set = set(DiNovoLysFind_title_to_denovo_df[DiNovoLysFind_title_to_denovo_df["ifEvident_denovo"] == True]["title"])
    DiNovoTryLys_total_Tryspec_set = set(DiNovoTryLysFind_title_to_denovo_df["A_title"])
    DiNovoTryLys_Evident_Tryspec_set = set(DiNovoTryLysFind_title_to_denovo_df[DiNovoTryLysFind_title_to_denovo_df["ifEvident_denovo"] == True]["A_title"])
    DiNovoTryLys_total_Lysspec_set = set(DiNovoTryLysFind_title_to_denovo_df["B_title"])
    DiNovoTryLys_Evident_Lysspec_set = set(DiNovoTryLysFind_title_to_denovo_df[DiNovoTryLysFind_title_to_denovo_df["ifEvident_denovo"] == True]["B_title"])
    DiNovo_total_Tryspec_set = DiNovoTryLys_total_Tryspec_set | DiNovoTry_total_spec_set
    DiNovo_Evident_Tryspec_set = DiNovoTryLys_Evident_Tryspec_set | DiNovoTry_Evident_spec_set
    DiNovo_total_Lysspec_set = DiNovoTryLys_total_Lysspec_set | DiNovoLys_total_spec_set
    DiNovo_Evident_Lysspec_set = DiNovoTryLys_Evident_Lysspec_set | DiNovoLys_Evident_spec_set
    DiNovo_total_spec_set = DiNovo_total_Tryspec_set | DiNovo_total_Lysspec_set
    DiNovo_Evident_spec_set = DiNovo_Evident_Tryspec_set | DiNovo_Evident_Lysspec_set

    DiNovoTryBest_total_spec_set = set(DiNovoTryBest_title_to_denovo_df["title"])
    DiNovoTryBest_Evident_spec_set = set(DiNovoTryBest_title_to_denovo_df[DiNovoTryBest_title_to_denovo_df["ifEvident_denovo"] == True]["title"])
    DiNovoLysBest_total_spec_set = set(DiNovoLysBest_title_to_denovo_df["title"])
    DiNovoLysBest_Evident_spec_set = set(DiNovoLysBest_title_to_denovo_df[DiNovoLysBest_title_to_denovo_df["ifEvident_denovo"] == True]["title"])
    DiNovoBest_total_spec_set = DiNovoTryBest_total_spec_set | DiNovoLysBest_total_spec_set
    DiNovoBest_Evident_spec_set = DiNovoTryBest_Evident_spec_set | DiNovoLysBest_Evident_spec_set

    intsection_info_fout.write(f"DiNovoTry total spec num: {len(DiNovoTry_total_spec_set)}\n")
    intsection_info_fout.write(f"DiNovoTry Evident spec num: {len(DiNovoTry_Evident_spec_set)}\n")
    intsection_info_fout.write(f"DiNovoLys total spec num: {len(DiNovoLys_total_spec_set)}\n")
    intsection_info_fout.write(f"DiNovoLys Evident spec num: {len(DiNovoLys_Evident_spec_set)}\n")
    intsection_info_fout.write(f"DiNovoTryLys total Tryspec num: {len(DiNovoTryLys_total_Tryspec_set)}\n")
    intsection_info_fout.write(f"DiNovoTryLys Evident Tryspec num: {len(DiNovoTryLys_Evident_Tryspec_set)}\n")
    intsection_info_fout.write(f"DiNovoTryLys total Lysspec num: {len(DiNovoTryLys_total_Lysspec_set)}\n")
    intsection_info_fout.write(f"DiNovoTryLys Evident Lysspec num: {len(DiNovoTryLys_Evident_Lysspec_set)}\n")
    intsection_info_fout.write(f"DiNovo total Tryspec num: {len(DiNovo_total_Tryspec_set)}\n")
    intsection_info_fout.write(f"DiNovo Evident Tryspec num: {len(DiNovo_Evident_Tryspec_set)}\n")
    intsection_info_fout.write(f"DiNovo total Lysspec num: {len(DiNovo_total_Lysspec_set)}\n")
    intsection_info_fout.write(f"DiNovo Evident Lysspec num: {len(DiNovo_Evident_Lysspec_set)}\n")
    intsection_info_fout.write(f"DiNovo total spec num: {len(DiNovo_total_spec_set)}\n")
    intsection_info_fout.write(f"DiNovo Evident spec num: {len(DiNovo_Evident_spec_set)}\n")

    intsection_info_fout.write(f"DiNovoTryBest total spec num: {len(DiNovoTryBest_total_spec_set)}\n")
    intsection_info_fout.write(f"DiNovoTryBest Evident spec num: {len(DiNovoTryBest_Evident_spec_set)}\n")
    intsection_info_fout.write(f"DiNovoLysBest total spec num: {len(DiNovoLysBest_total_spec_set)}\n")
    intsection_info_fout.write(f"DiNovoLysBest Evident spec num: {len(DiNovoLysBest_Evident_spec_set)}\n")
    intsection_info_fout.write(f"DiNovoBest total spec num: {len(DiNovoBest_total_spec_set)}\n")
    intsection_info_fout.write(f"DiNovoBest Evident spec num: {len(DiNovoBest_Evident_spec_set)}\n")


    #######################################################################
    # 与DiNovo交集
    # 抛弃数据库搜库结果，统计denovoSeq的肽段数量
    # DiNovoTryLys_DenovoRSeq_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTryLys_denovoSeq_to_ifEvidentFind[{massFilter}Da].res", sep="\t")
    # DiNovoTryLys_DenovoRSeq_df["ifDiNovoTryLys"] = True
    # Na_df = DiNovoTryLys_DenovoRSeq_df[DiNovoTryLys_DenovoRSeq_df['denovoSeq_DiNovoTryLys'].isna()]
    # assert len(Na_df) == 1 or len(Na_df) == 0
    # DiNovoTryLys_DenovoRSeq_df = DiNovoTryLys_DenovoRSeq_df[DiNovoTryLys_DenovoRSeq_df['denovoSeq_DiNovoTryLys'].notna()]
    # DiNovoTryLys_DenovoRSeq_df['Rseq_try'] = DiNovoTryLys_DenovoRSeq_df.apply(lambda x: transfer_Rseq_to_trySeq(x['denovoSeq_DiNovoTryLys']), axis=1)
    # DiNovoTryLys_DenovoRSeq_df['Rseq_lys'] = DiNovoTryLys_DenovoRSeq_df.apply(lambda x: transfer_Rseq_to_lysSeq(x['denovoSeq_DiNovoTryLys']), axis=1)
    #
    # DiNovoTry_DenovoSeq_df = pd.read_csv(DiNovo_res_folder + "\\ALL_DiNovoTry_denovoSeq_to_ifEvidentFind[{massFilter}Da].res", sep="\t")
    # DiNovoTry_DenovoSeq_df["denovo_seqNOKR"] = DiNovoTry_DenovoSeq_df["denovoSeq_DiNovoTry"].apply(remove_try_KR_function)
    # DiNovoTry_DenovoSeq_df["ifDiNovoTry"] = True
    #
    # DiNovoLys_DenovoSeq_df = pd.read_csv(DiNovo_res_folder + "\\ALL_DiNovoLys_denovoSeq_to_ifEvidentFind[{massFilter}Da].res", sep="\t")
    # DiNovoLys_DenovoSeq_df["denovo_seqNOKR"] = DiNovoLys_DenovoSeq_df["denovoSeq_DiNovoLys"].apply(remove_lys_KR_function)
    # DiNovoLys_DenovoSeq_df["ifDiNovoLys"] = True
    #
    # DiNovo_intsection_DenovoSeq_df = pd.merge(DiNovoTry_DenovoSeq_df, DiNovoLys_DenovoSeq_df, on="denovo_seqNOKR",how="outer")
    # DiNovo_intsection_DenovoSeq_df['denovo_Rseq'] = DiNovo_intsection_DenovoSeq_df.apply(
    #     lambda x: generate_targetRseq_function(x['denovoSeq_DiNovoTry'], x['denovoSeq_DiNovoLys']), axis=1)
    # DiNovo_intsection_DenovoSeq_df["tag"] = DiNovo_intsection_DenovoSeq_df.apply(
    #     lambda x: 2 if x['ifEvident_DiNovoTry'] != x['ifEvident_DiNovoTry'] else (
    #         1 if x['ifEvident_DiNovoLys'] != x['ifEvident_DiNovoLys'] else 3), axis=1)
    #
    # print("The length of intsection_DenovoSeq_df before merge DiNovo: ", len(DiNovo_intsection_DenovoSeq_df))
    # length1 = len(DiNovo_intsection_DenovoSeq_df)
    # DiNovoTryLys_DenovoRSeq_df = DiNovoTryLys_DenovoRSeq_df.rename(
    #     columns={"denovoSeq_DiNovoTryLys": "denovo_Rseq"})
    # DiNovo_intsection_DenovoSeq_df = pd.merge(DiNovo_intsection_DenovoSeq_df, DiNovoTryLys_DenovoRSeq_df,
    #                                               on="denovo_Rseq", how="outer")
    # DiNovo_intsection_DenovoSeq_df["index"] = DiNovo_intsection_DenovoSeq_df.index
    # print("The length of intsection_DenovoSeq_df after merge DiNovo: ", len(DiNovo_intsection_DenovoSeq_df))
    # length2 = len(DiNovo_intsection_DenovoSeq_df)
    # # 去除三者都不能Find的行
    # DiNovo_intsection_DenovoSeq_df.to_csv(
    #     DiNovo_res_folder + "\\intsection_DenovoSeq_to_ifEvidentFind[DiNovoNei][{massFilter}Da].res", sep="\t", index=False)
    # DiNovo_intsection_DenovoSeq_df = DiNovo_intsection_DenovoSeq_df.loc[
    #     ((DiNovo_intsection_DenovoSeq_df["ifFind_DiNovoTry"] == True) & (
    #                 DiNovo_intsection_DenovoSeq_df["ifFindDecoy_DiNovoTry"] == False))
    #     | ((DiNovo_intsection_DenovoSeq_df["ifFind_DiNovoLys"] == True) & (
    #                 DiNovo_intsection_DenovoSeq_df["ifFindDecoy_DiNovoLys"] == False))
    #     | ((DiNovo_intsection_DenovoSeq_df["ifFind_DiNovoTryLys"] == True) & (
    #                 DiNovo_intsection_DenovoSeq_df["ifFindDecoy_DiNovoTryLys"] == False))]
    # print("The length of intsection_DenovoSeq_df after remove all not find: ", len(DiNovo_intsection_DenovoSeq_df))
    # length3 = len(DiNovo_intsection_DenovoSeq_df)
    #
    # if not os.path.exists(
    #         DiNovo_res_folder + "\\intsection_DenovoSeq_to_ifEvidentFind_mini[DiNovoNei][{massFilter}Da].res"):
    #     DiNovo_intsection_DenovoSeq_df["New_Rseq"] = DiNovo_intsection_DenovoSeq_df.apply(
    #         lambda x: find_new_Rseq(x, DiNovo_intsection_DenovoSeq_df), axis=1)
    #     mini_df = DiNovo_intsection_DenovoSeq_df.loc[(DiNovo_intsection_DenovoSeq_df["New_Rseq"] != '')].copy()
    #     for index, row in mini_df.iterrows():
    #         if index % 1000 == 0:
    #             print(f"\r", index, "/", len(mini_df), end="")
    #         new_row = row.copy()
    #         if row["Rseq_try"] != row["Rseq_try"]:
    #             DiNovo_intsection_DenovoSeq_df.drop(
    #                 DiNovo_intsection_DenovoSeq_df[
    #                     DiNovo_intsection_DenovoSeq_df["index"] == row["index"]].index,
    #                 inplace=True)
    #
    #         for Rseq_temp in new_row["New_Rseq"].split(','):
    #             new_new_row = new_row.copy()
    #             new_new_row["denovo_Rseq"] = Rseq_temp
    #             DiNovo_intsection_DenovoSeq_df = pd.concat(
    #                 [DiNovo_intsection_DenovoSeq_df, new_new_row.to_frame().T], ignore_index=True)
    #
    #     # 把ifMatch_try和ifMatch_lys中的nan替换为False
    #     DiNovo_intsection_DenovoSeq_df["ifEvident_DiNovoTry"].fillna(False, inplace=True)
    #     DiNovo_intsection_DenovoSeq_df['ifFind_DiNovoTry'].fillna(False, inplace=True)
    #     DiNovo_intsection_DenovoSeq_df["ifEvident_DiNovoLys"].fillna(False, inplace=True)
    #     DiNovo_intsection_DenovoSeq_df['ifFind_DiNovoLys'].fillna(False, inplace=True)
    #     DiNovo_intsection_DenovoSeq_df["ifEvident_DiNovoTryLys"].fillna(False, inplace=True)
    #     DiNovo_intsection_DenovoSeq_df['ifFind_DiNovoTryLys'].fillna(False, inplace=True)
    #     DiNovo_intsection_DenovoSeq_df["ifDiNovoTry"].fillna(False, inplace=True)
    #     DiNovo_intsection_DenovoSeq_df["ifDiNovoLys"].fillna(False, inplace=True)
    #     DiNovo_intsection_DenovoSeq_df["ifDiNovoTryLys"].fillna(False, inplace=True)
    #
    #     # 保存到文件
    #     DiNovo_intsection_DenovoSeq_df.to_csv(DiNovo_res_folder + "\\intsection_DenovoSeq_to_ifEvidentFind_mini[DiNovoNei][{massFilter}Da].res", sep="\t", index=False)
    # else:
    #     DiNovo_intsection_DenovoSeq_df = pd.read_csv(
    #         DiNovo_res_folder + "\\intsection_DenovoSeq_to_ifEvidentFind_mini[DiNovoNei][{massFilter}Da].res", sep="\t")
    #
    # # 利用上面得到的数据画肽段水平和谱图水平的韦恩图
    # set1 = set(DiNovo_intsection_DenovoSeq_df.loc[
    #                (DiNovo_intsection_DenovoSeq_df["ifFind_DiNovoTry"] == True) & (
    #                            DiNovo_intsection_DenovoSeq_df["ifFindDecoy_DiNovoTry"] == False)][
    #                "denovo_Rseq"])
    # set2 = set(DiNovo_intsection_DenovoSeq_df.loc[
    #                (DiNovo_intsection_DenovoSeq_df["ifFind_DiNovoLys"] == True) & (
    #                            DiNovo_intsection_DenovoSeq_df["ifFindDecoy_DiNovoLys"] == False)][
    #                "denovo_Rseq"])
    # set3 = set(DiNovo_intsection_DenovoSeq_df.loc[
    #                (DiNovo_intsection_DenovoSeq_df["ifFind_DiNovoTryLys"] == True) & (
    #                            DiNovo_intsection_DenovoSeq_df["ifFindDecoy_DiNovoTryLys"] == False)][
    #                "denovo_Rseq"])
    # venn3([set1, set2, set3], (fig_name1, fig_name2, fig_name1 + '&' + fig_name2))
    # fig_path = DiNovo_res_folder + "\\ALL_intsection_denovoRSeq[DiNovoNei].png"
    # plt.savefig(fig_path, dpi=300)
    # plt.close()
    #
    # set1_num = len(set1)
    # set2_num = len(set2)
    # set3_num = len(set3)
    # all_num = len(set1 | set2 | set3)
    # string = f"Venn_DiNovoTry_DenovoRSeq_num[DiNovoNei]: {set1_num}\n"
    # intsection_info_fout.write(string)
    # string = f"Venn_DiNovoLys_DenovoRSeq_num[DiNovoNei]: {set2_num}\n"
    # intsection_info_fout.write(string)
    # string = f"Venn_DiNovoTryLys_DenovoRSeq_num[DiNovoNei]: {set3_num}\n"
    # intsection_info_fout.write(string)
    # string = f"Venn_all_DenovoRSeq_num[DiNovoNei]: {all_num}\n"
    # intsection_info_fout.write(string)
    #
    # set1 = set(
    #     DiNovo_intsection_DenovoSeq_df.loc[(DiNovo_intsection_DenovoSeq_df["ifFind_DiNovoTry"] == True) & (
    #             DiNovo_intsection_DenovoSeq_df["ifEvident_DiNovoTry"] == True) & (
    #                                                    DiNovo_intsection_DenovoSeq_df[
    #                                                        "ifFindDecoy_DiNovoTry"] == False)]["denovo_Rseq"])
    # set2 = set(
    #     DiNovo_intsection_DenovoSeq_df.loc[(DiNovo_intsection_DenovoSeq_df["ifFind_DiNovoLys"] == True) & (
    #             DiNovo_intsection_DenovoSeq_df["ifEvident_DiNovoLys"] == True) & (
    #                                                    DiNovo_intsection_DenovoSeq_df[
    #                                                        "ifFindDecoy_DiNovoLys"] == False)]["denovo_Rseq"])
    # set3 = set(DiNovo_intsection_DenovoSeq_df.loc[
    #                (DiNovo_intsection_DenovoSeq_df["ifFind_DiNovoTryLys"] == True) & (
    #                        DiNovo_intsection_DenovoSeq_df["ifEvident_DiNovoTryLys"] == True) & (
    #                        DiNovo_intsection_DenovoSeq_df["ifFindDecoy_DiNovoTryLys"] == False)]["denovo_Rseq"])
    # venn3([set1, set2, set3], (fig_name1, fig_name2, fig_name1 + '&' + fig_name2))
    # fig_path = DiNovo_res_folder + "\\intsection_EvidentDenovoRSeq[DiNovoNei].png"
    # plt.savefig(fig_path, dpi=300)
    # plt.close()
    #
    # set1_num = len(set1)
    # set2_num = len(set2)
    # set3_num = len(set3)
    # all_num = len(set1 | set2 | set3)
    # string = f"Venn_DiNovoTry_Evident_DenovoRSeq_num[DiNovoNei]: {set1_num}\n"
    # intsection_info_fout.write(string)
    # string = f"Venn_DiNovoLys_Evident_DenovoRSeq_num[DiNovoNei]: {set2_num}\n"
    # intsection_info_fout.write(string)
    # string = f"Venn_DiNovoTryLys_Evident_DenovoRSeq_num[DiNovoNei]: {set3_num}\n"
    # intsection_info_fout.write(string)
    # string = f"Venn_all_Evident_DenovoRSeq_num[DiNovoNei]: {all_num}\n"
    # intsection_info_fout.write(string)
    #
    # # 得到在set1/2/3但是不在另外两集合中的peptide
    # evident_peptide_try_only_set = set1 - set2 - set3
    # evident_peptide_lys_only_set = set2 - set1 - set3
    # evident_peptide_DiNovo_only_set = set3 - set1 - set2
    # evident_peptide_try_length_list = []
    # for targetSeq in evident_peptide_try_only_set:
    #     evident_peptide_try_length_list.append(len(targetSeq))
    # evident_peptide_lys_length_list = []
    # for targetSeq in evident_peptide_lys_only_set:
    #     evident_peptide_lys_length_list.append(len(targetSeq))
    # evident_peptide_DiNovo_length_list = []
    # for targetSeq in evident_peptide_DiNovo_only_set:
    #     evident_peptide_DiNovo_length_list.append(len(targetSeq))
    #
    # string = f"evident Rpeptide found num in trypsin-only[DiNovoNei]\t{len(evident_peptide_try_length_list)}\n"
    # print("evident Rpeptide found num in trypsin-only[DiNovoNei]: ", len(evident_peptide_try_length_list))
    # intsection_info_fout.write(string)
    #
    # string = f"ave_length_trypsin[DiNovoNei]\t{np.mean(evident_peptide_try_length_list)}\n"
    # print("ave_length_trypsin[DiNovoNei]: ", np.mean(evident_peptide_try_length_list))
    # intsection_info_fout.write(string)
    #
    # string = f"evident Rpeptide found num in lysargiNase-only[DiNovoNei]\t{len(evident_peptide_lys_length_list)}\n"
    # print("evident Rpeptide found num in lysargiNase-only[DiNovoNei]: ", len(evident_peptide_lys_length_list))
    # intsection_info_fout.write(string)
    #
    # string = f"ave_length_lysargiNase[DiNovoNei]\t{np.mean(evident_peptide_lys_length_list)}\n"
    # print("ave_length_lysargiNase[DiNovoNei]: ", np.mean(evident_peptide_lys_length_list))
    # intsection_info_fout.write(string)
    #
    # string = f"evident Rpeptide found num in DiNovoNei-only[DiNovoNei]\t{len(evident_peptide_DiNovo_length_list)}\n"
    # print("evident Rpeptide found num in DiNovoNei-only[DiNovoNei]: ",
    #       len(evident_peptide_DiNovo_length_list))
    # intsection_info_fout.write(string)
    #
    # string = f"ave_length_DiNovoNei[DiNovoNei]\t{np.mean(evident_peptide_DiNovo_length_list)}\n"
    # print("ave_length_DiNovoNei[DiNovoNei]: ", np.mean(evident_peptide_DiNovo_length_list))
    # intsection_info_fout.write(string)
    #
    # plt.figure(figsize=(10, 5))
    # plt.hist(evident_peptide_try_length_list, bins=20, alpha=0.5, color='r', label=fig_name1, density=True)
    # plt.hist(evident_peptide_lys_length_list, bins=20, alpha=0.5, color='g', label=fig_name2, density=True)
    # plt.hist(evident_peptide_DiNovo_length_list, bins=20, alpha=0.5, color='b', label='DiNovo', density=True)
    # sns.kdeplot(evident_peptide_try_length_list, color='r')
    # sns.kdeplot(evident_peptide_lys_length_list, color='g')
    # sns.kdeplot(evident_peptide_DiNovo_length_list, color='b')
    # # 添加图例
    # plt.legend(loc='upper right')
    # # 显示图形
    # plt.savefig(DiNovo_res_folder + f"\\ALL_evident_Rpeptides_length_distribution_found_Density[DiNovoNei].png",
    #             dpi=300)
    # plt.close()
    #
    # plt.figure(figsize=(10, 5))
    # plt.hist(evident_peptide_try_length_list, bins=20, alpha=0.5, color='r', label=fig_name1, density=False)
    # plt.hist(evident_peptide_lys_length_list, bins=20, alpha=0.5, color='g', label=fig_name2, density=False)
    # plt.hist(evident_peptide_DiNovo_length_list, bins=20, alpha=0.5, color='b', label='DiNovo', density=False)
    # sns.kdeplot(evident_peptide_try_length_list, color='r')
    # sns.kdeplot(evident_peptide_lys_length_list, color='g')
    # sns.kdeplot(evident_peptide_DiNovo_length_list, color='b')
    # # 添加图例
    # plt.legend(loc='upper right')
    # # 显示图形
    # plt.savefig(DiNovo_res_folder + f"\\ALL_evident_Rpeptides_length_distribution_found_count[DiNovoNei].png",
    #             dpi=300)
    # plt.close()




