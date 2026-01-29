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
    generate_DiNovoPairs_to_matchtype_dict,merge_target_and_decoy_fasta_seq,generate_Best_title_to_denovo_df,build_singleSpectra_location_function,\
    read_seqs_from_DiNovoSingleRes

#CasaNovo/PEAKS/PointNovo/pNovo3
topk_peaks = config.topk_peaks
DiNovo_tool = "MirrorNovo"#DiNovo_MirrorNovo/DiNovo_pNovoM2/MirrorNovo/pNovoM2
DiNovo_merge = False
fasta_file = config.fasta_file
DiNovo_single_folder = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MirrorNovo_output\single"
DiNovoTry_file = DiNovo_single_folder + r"\try\Yeast_trypsin.txt.beamsearch[23charge]"
DiNovoLys_file = DiNovo_single_folder + r"\lys\Yeast_lysargiNase.txt.beamsearch[23charge]"
DiNovo_res_folder = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MirrorNovo_output\mirror\20240525\23charge"
DiNovo_res_file = DiNovo_res_folder + "\\MirrorNovo.res[23charge]"
DiNovoPairs_file = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\spectraPairs\20240525\[DiNovo]SpectralPairs[23charge].res"
try_mgf = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\mgf\try\23charge"
lys_mgf = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\mgf\lys\23charge"
fig_name1 = "Try"
fig_name2 = "Lys"
mapped_version = 2
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
modifications_MirrorNovo = config.modifications_MirrorNovo
aa_to_mass_dict = config.aa_to_mass_dict
atom_mass = config.atom_mass

def generate_try_lys_seq_from_RseqAndMatchtype(Rseq,match_type = None):
    flag = False
    if type(Rseq) == str:
        flag = True
        Rseq = transfer_str_to_list_seq(Rseq)

    if match_type == "A1:K-K" or match_type == "A2:R-R" or match_type == "B: R-K" or match_type == "C: K-R":
        try_seq = Rseq[1:]
        lys_seq = Rseq[:-1]
    elif match_type == "F: X-K" or match_type == "G: X-R":
        try_seq = Rseq[1:]
        lys_seq = Rseq
    elif match_type == "D: K-X" or match_type == "E: R-X":
        try_seq = Rseq
        lys_seq = Rseq[:-1]
    else:
        if match_type == None:
            if Rseq[0] == 'K' or Rseq[0] == 'R':
                if Rseq[-1] == 'K' or Rseq[-1] == 'R':
                    try_seq = Rseq[1:]
                    lys_seq = Rseq[:-1]
                else:
                    try_seq = Rseq[1:]
                    lys_seq = Rseq
            else:
                if Rseq[-1] == 'K' or Rseq[-1] == 'R':
                    try_seq = Rseq
                    lys_seq = Rseq[:-1]
                else:
                    try_seq = Rseq
                    lys_seq = Rseq
        else:
            print(Rseq,match_type)
            assert False
    if flag:
        try_seq = ''.join(try_seq)
        lys_seq = ''.join(lys_seq)

    return try_seq,lys_seq


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


def read_seqs_from_pNovoM2SingleRes(location, fi, title, DiNovo_merge):

    #只选择top1结果
    fi.seek(location)
    line = fi.readline().strip()
    title_temp = line.split("\t")[1].split("@")[0]
    assert title == title_temp,f"title:{title}, title_temp:{title_temp}"
    line = fi.readline()
    if "@" in line:
        seq = ['']
        score = 0.0
    else:
        line = line.strip().split("\t")
        seq = line[0]
        if DiNovo_merge:
            score = float(line[2])
        else:
            score = float(line[1])
        seq = list(seq)
        seq = transfer_mod_function("pnovom", seq)

    return seq, score


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
    try_spectrum_location_dict, try_suffix_name, try_filepathAndOrderNum_to_title_dict, try_filepathAndScan_to_title_dict = build_mgf_location_function(try_mgf + "\\")
    lys_spectrum_location_dict, lys_suffix_name, lys_filepathAndOrderNum_to_title_dict, lys_filepathAndScan_to_title_dict = build_mgf_location_function(lys_mgf + "\\")
    DiNovoTry_title_to_location = build_singleSpectra_location_function(DiNovoTry_file, DiNovo_tool)
    DiNovoLys_title_to_location = build_singleSpectra_location_function(DiNovoLys_file, DiNovo_tool)

    if DiNovo_tool == "pNovoM2":
        DiNovoTryLys_file = DiNovo_res_folder + "\\DiNovoTryLys_title_to_denovo_info.res"
        if not os.path.exists(DiNovoTryLys_file):
            DiNovoTryLys_fout = open(DiNovoTryLys_file, "a+")
            DiNovoTryLys_fout.write(
                "title\tdenovoSeq\tscore_software\tsite_tag_denovo\tsite_peaknum_denovo\tsite_score_denovo\tpep_scoremean_denovo\tpep_scoresum_denovo\tifEvident_denovo\tifFind_denovo\tmatch_type\tSite_tag_denovo_A\tsite_peaknum_denovo_A\tsite_score_denovo_A\tpep_scoremean_denovo_A\tpep_scoresum_denovo_A\tSite_tag_denovo_B\tsite_peaknum_denovo_B\tsite_score_denovo_B\tpep_scoremean_denovo_B\tpep_scoresum_denovo_B\n")
            print("reading DiNovo Res file...")
            print("reading ", DiNovo_res_file)
            psm_list = []
            if DiNovo_merge == True or DiNovo_merge == False:
                print("reading ", DiNovo_res_file)
                with open(DiNovo_res_file, "r") as fin:
                    while True:
                        line = fin.readline().strip()
                        if len(line) == 0:
                            break
                        line = line.split("\t")
                        assert line[0].startswith("S")
                        titlesPair = line[1].strip()
                        try_title = titlesPair.split("@")[0]
                        lys_title = titlesPair.split("@")[1]
                        candidate_num = int(line[-1].strip())
                        if DiNovo_merge:
                            match_type = line[-2].strip()
                        else:
                            match_type = line[-3].strip()
                        if candidate_num != 0:
                            seq_line = fin.readline().strip().split("\t")
                            top1_Rseq = seq_line[0]
                            top1_Rseq = transfer_mod_function('pnovom', list(top1_Rseq))
                            try_denovo_seq, lys_denovo_seq = generate_try_lys_seq_from_RseqAndMatchtype(top1_Rseq,match_type)
                            if DiNovo_merge:
                                score = seq_line[2]
                            else:
                                score = seq_line[1]
                        else:
                            top1_Rseq = ['']
                            try_denovo_seq = ['']
                            lys_denovo_seq = ['']
                        psm_list.append((titlesPair, try_title, lys_title, top1_Rseq, try_denovo_seq, lys_denovo_seq, match_type, score))
                        if candidate_num > 1:
                            for k in range(candidate_num - 1):
                                fin.readline()
                try_res_dict, lys_res_dict = parallel_get_confident_info_mirror(psm_list,
                                                                                try_spectrum_location_dict,
                                                                                try_suffix_name, try_mgf,
                                                                                lys_spectrum_location_dict,
                                                                                lys_suffix_name, lys_mgf)
                for i in range(len(psm_list)):
                    title = psm_list[i][0]
                    try_title = psm_list[i][1]
                    lys_title = psm_list[i][2]
                    denovoRseq = psm_list[i][3]
                    try_denovoSeq = psm_list[i][4]
                    lys_denovoSeq = psm_list[i][5]
                    match_type = psm_list[i][6]
                    score = psm_list[i][7]

                    denovo_illegal = True
                    if denovoRseq != ['']:
                        denovo_illegal = False
                        try_denovoSeq_str = "".join(try_denovoSeq)
                        lys_denovoSeq_str = "".join(lys_denovoSeq)
                        denovoRSeq_str = "".join(denovoRseq)
                        try_denovo_psm = try_title + "@" + try_denovoSeq_str
                        lys_denovo_psm = lys_title + "@" + lys_denovoSeq_str
                        merge_info_dict = merge_try_and_lys_info(try_res_dict[try_denovo_psm],
                                                                 lys_res_dict[lys_denovo_psm])

                        # output_data
                        ##try
                        try_info_dict = try_res_dict[try_denovo_psm]
                        try_site_tag = try_info_dict["site_tag"]
                        try_site_peaknum = try_info_dict["site_peaknum"]
                        try_site_score = try_info_dict["site_score"]
                        try_pepscoremean = try_info_dict["pepscore_mean"]
                        try_pepscoresum = try_info_dict["pepscore_sum"]
                        ##lys
                        lys_info_dict = lys_res_dict[lys_denovo_psm]
                        lys_site_tag = lys_info_dict["site_tag"]
                        lys_site_peaknum = lys_info_dict["site_peaknum"]
                        lys_site_score = lys_info_dict["site_score"]
                        lys_pepscoremean = lys_info_dict["pepscore_mean"]
                        lys_pepscoresum = lys_info_dict["pepscore_sum"]
                        ##merge
                        site_tag = merge_info_dict["site_tag"]
                        site_peaknum = merge_info_dict["site_peaknum"]
                        site_score = merge_info_dict["site_score"]
                        pep_scoremean = merge_info_dict["pep_scoremean"]
                        pep_scoresum = merge_info_dict["pep_scoresum"]
                        site_intensitytemp = merge_info_dict["site_intensitytemp_A"]
                        # 判断是否top200
                        sort_intensity_list = sorted(try_info_dict["intensity_list"])
                        rank = len(sort_intensity_list) - np.array(
                            [bisect.bisect_left(sort_intensity_list, i) for i in site_intensitytemp]) + 1
                        site_tag = np.array(
                            [1 if tag == 1 and rank[i] <= topk_peaks else 0 for i, tag in enumerate(site_tag)])
                        miss_num = sum(site_tag == 0)
                        coverage = 1 - miss_num / len(site_tag)
                    else:
                        coverage = 0
                        score = 0.0
                        site_tag = ''
                        site_peaknum = ''
                        site_score = ''
                        pep_scoremean = ''
                        pep_scoresum = ''
                        ifEvident = ''
                        try_site_tag = ''
                        try_site_peaknum = ''
                        try_site_score = ''
                        try_pepscoremean = ''
                        try_pepscoresum = ''
                        lys_site_tag = ''
                        lys_site_peaknum = ''
                        lys_site_score = ''
                        lys_pepscoremean = ''
                        lys_pepscoresum = ''
                    ifEvident = if_evident_function(coverage)
                    DiNovoTryLys_fout.write(
                        f"{title}\t{''.join(denovoRseq)}\t{score}\t{','.join(list(map(str, site_tag)))}\t{','.join(list(map(str, site_peaknum)))}\t{','.join(list(map(str, site_score)))}\t{str(pep_scoremean)}\t{str(pep_scoresum)}\t{ifEvident}\t{False}\t{match_type}\t"
                        f"{','.join(list(map(str, try_site_tag)))}\t{','.join(list(map(str, try_site_peaknum)))}\t{','.join(list(map(str, try_site_score)))}\t{str(try_pepscoremean)}\t{str(try_pepscoresum)}\t"
                        f"{','.join(list(map(str, lys_site_tag)))}\t{','.join(list(map(str, lys_site_peaknum)))}\t{','.join(list(map(str, lys_site_score)))}\t{str(lys_pepscoremean)}\t{str(lys_pepscoresum)}\n")
                DiNovoTryLys_fout.close()
                DiNovoTryLys_title_to_denovo_df = pd.read_csv(DiNovoTryLys_file, sep="\t")
            else:
                assert False
        else:
            print("loading DiNovoTryLys_title_to_denovo_info.res file...")
            DiNovoTryLys_title_to_denovo_df = pd.read_csv(DiNovoTryLys_file, sep="\t")
            DiNovoTryLys_title_to_denovo_df = DiNovoTryLys_title_to_denovo_df.astype({
                'site_tag_denovo': 'str',
                'site_peaknum_denovo': 'str',
                'site_score_denovo': 'str',
            })
        #删除没有结果的谱图对
        DiNovoTryLys_title_to_denovo_df = DiNovoTryLys_title_to_denovo_df[(DiNovoTryLys_title_to_denovo_df["denovoSeq"] == DiNovoTryLys_title_to_denovo_df["denovoSeq"]) & (DiNovoTryLys_title_to_denovo_df["denovoSeq"] != '')]
        if "MirrorFinderScore" not in DiNovoTryLys_title_to_denovo_df.columns:
            DiNovoTryLys_title_to_denovo_df["MirrorFinderScore"] = DiNovoTryLys_title_to_denovo_df.apply(lambda row: titlesPair_to_location_dict[row["title"]][2], axis=1)
            DiNovoTryLys_title_to_denovo_df.to_csv(DiNovoTryLys_file, sep="\t", index=False)
        DiNovoTryLys_title_to_denovo_df[["A_title", "B_title"]] = DiNovoTryLys_title_to_denovo_df["title"].str.split("@", n=1, expand=True)

        if not os.path.exists(DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res"):
            DiNovoTry_fout = open(DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res", "a+")
            DiNovoTry_fout.write("title\tdenovoSeq\tscore_software\tsite_tag_denovo\tsite_peaknum_denovo\tsite_score_denovo\tpep_scoremean_denovo\tpep_scoresum_denovo\tifEvident_denovo\tifFind_denovo\n")
            DiNovoTry_fin = open(DiNovoTry_file, 'r')
            print("reading DiNovoTry mgf files...")
            denovo_psm_list = []
            db_psm_list = []
            for filename in os.listdir(try_mgf):
                if filename.endswith(".mgf"):
                    file = try_mgf + f"\\{filename}"
                    print(f"reading {file}...")
                    with open(file, 'r') as f:
                        while True:
                            line = f.readline()
                            if len(line) == 0:
                                break
                            if line.startswith("TITLE="):
                                title = line.split('=')[1].rstrip()
                                if title in DiNovoTry_title_to_location.keys():
                                    denovo_seq,score = read_seqs_from_pNovoM2SingleRes(DiNovoTry_title_to_location[title],DiNovoTry_fin,title,DiNovo_merge)
                                else:
                                    denovo_seq = ['']
                                    score = 0.0
                                denovo_psm_list.append((title, denovo_seq, score))
            denovo_res_dict = parallel_get_confident_info(denovo_psm_list,
                                                          try_spectrum_location_dict,
                                                          try_suffix_name, try_mgf)
            for i in range(len(denovo_psm_list)):
                title = denovo_psm_list[i][0]
                score = denovo_psm_list[i][2]
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
                     site_intensitytemp_denovo) = denovo_res_dict[denovo_psm]["site_tag"], \
                        denovo_res_dict[denovo_psm]["site_peaknum"], denovo_res_dict[denovo_psm][
                        "site_score"], denovo_res_dict[denovo_psm]["pepscore_mean"], \
                        denovo_res_dict[denovo_psm]["pepscore_sum"], denovo_res_dict[denovo_psm][
                        "intensity_list"], denovo_res_dict[denovo_psm]["site_intensitytemp"]
                    sort_intensity_list_denovo = sorted(intensity_list_denovo)
                    rank = len(sort_intensity_list_denovo) - np.array(
                        [bisect.bisect_left(sort_intensity_list_denovo, i) for i in
                         site_intensitytemp_denovo]) + 1
                    site_tag_denovo = np.array([1 if tag == 1 and rank[i] <= topk_peaks else 0 for i, tag in
                                                enumerate(site_tag_denovo)])
                    miss_num = np.sum(site_tag_denovo == 0)
                    coverage = 1 - miss_num / len(site_tag_denovo)
                    ifEvident_denovo = if_evident_function(coverage)

                if denovo_illegal:
                    denovoSeq_str = ''
                    site_tag_denovo = ''
                    site_peaknum_denovo = ''
                    site_score_denovo = ''
                    pep_scoremean_denovo = ''
                    pep_scoresum_denovo = ''
                    ifEvident_denovo = ''
                DiNovoTry_fout.write(
                    f"{title}\t{denovoSeq_str}\t{score}\t{''.join(list(map(str, site_tag_denovo)))}\t{','.join(list(map(str, site_peaknum_denovo)))}\t{','.join(list(map(str, site_score_denovo)))}\t{pep_scoremean_denovo}\t{pep_scoresum_denovo}\t{ifEvident_denovo}\t{False}\n")
            DiNovoTry_fout.close()
            DiNovoTry_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res", sep="\t")
        else:
            print("loading DiNovoTry_title_to_denovo_info.res file...")
            DiNovoTry_title_to_denovo_df = pd.read_csv(
                DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res", sep="\t")

        if not os.path.exists(DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res"):
            DiNovoLys_fout = open(DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res", "a+")
            DiNovoLys_fout.write(
                "title\tdenovoSeq\tscore_software\tsite_tag_denovo\tsite_peaknum_denovo\tsite_score_denovo\tpep_scoremean_denovo\tpep_scoresum_denovo\tifEvident_denovo\tifFind_denovo\n")
            DiNovoLys_fin = open(DiNovoLys_file, 'r')
            print("reading DiNovoLys mgf files...")
            denovo_psm_list = []
            db_psm_list = []
            for filename in os.listdir(lys_mgf):
                if filename.endswith(".mgf"):
                    file = lys_mgf + f"\\{filename}"
                    print(f"reading {file}...")
                    with open(file, 'r') as f:
                        while True:
                            line = f.readline()
                            if len(line) == 0:
                                break
                            if line.startswith("TITLE="):
                                title = line.split('=')[1].rstrip()
                                if title in DiNovoLys_title_to_location.keys():
                                    denovo_seq, score = read_seqs_from_pNovoM2SingleRes(
                                        DiNovoLys_title_to_location[title], DiNovoLys_fin, title, DiNovo_merge)
                                else:
                                    denovo_seq = ['']
                                    score = 0.0
                                denovo_psm_list.append((title, denovo_seq, score))
            denovo_res_dict = parallel_get_confident_info(denovo_psm_list,
                                                          lys_spectrum_location_dict,
                                                          lys_suffix_name, lys_mgf)
            for i in range(len(denovo_psm_list)):
                title = denovo_psm_list[i][0]
                score = denovo_psm_list[i][2]
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
                     site_intensitytemp_denovo) = denovo_res_dict[denovo_psm]["site_tag"], \
                        denovo_res_dict[denovo_psm]["site_peaknum"], denovo_res_dict[denovo_psm][
                        "site_score"], denovo_res_dict[denovo_psm]["pepscore_mean"], \
                        denovo_res_dict[denovo_psm]["pepscore_sum"], denovo_res_dict[denovo_psm][
                        "intensity_list"], denovo_res_dict[denovo_psm]["site_intensitytemp"]
                    sort_intensity_list_denovo = sorted(intensity_list_denovo)
                    rank = len(sort_intensity_list_denovo) - np.array(
                        [bisect.bisect_left(sort_intensity_list_denovo, i) for i in
                         site_intensitytemp_denovo]) + 1
                    site_tag_denovo = np.array([1 if tag == 1 and rank[i] <= topk_peaks else 0 for i, tag in
                                                enumerate(site_tag_denovo)])
                    miss_num = np.sum(site_tag_denovo == 0)
                    coverage = 1 - miss_num / len(site_tag_denovo)
                    ifEvident_denovo = if_evident_function(coverage)

                if denovo_illegal:
                    denovoSeq_str = ''
                    site_tag_denovo = ''
                    site_peaknum_denovo = ''
                    site_score_denovo = ''
                    pep_scoremean_denovo = ''
                    pep_scoresum_denovo = ''
                    ifEvident_denovo = ''
                DiNovoLys_fout.write(
                    f"{title}\t{denovoSeq_str}\t{score}\t{''.join(list(map(str, site_tag_denovo)))}\t{','.join(list(map(str, site_peaknum_denovo)))}\t{','.join(list(map(str, site_score_denovo)))}\t{pep_scoremean_denovo}\t{pep_scoresum_denovo}\t{ifEvident_denovo}\t{False}\n")
            DiNovoLys_fout.close()
            DiNovoLys_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res",
                                                       sep="\t")
        else:
            print("loading DiNovoLys_title_to_denovo_info.res file...")
            DiNovoLys_title_to_denovo_df = pd.read_csv(
                DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res", sep="\t")

    ###################################################################################################################################################################################################################################
    elif DiNovo_tool == "MirrorNovo":
        DiNovoTryLys_file = DiNovo_res_folder + "\\DiNovoTryLys_title_to_denovo_info.res"
        if not os.path.exists(DiNovoTryLys_file):
            DiNovoTryLys_fout = open(DiNovoTryLys_file, "a+")
            DiNovoTryLys_fout.write("title\tdenovoSeq\tscore_software\tsite_tag_denovo\tsite_peaknum_denovo\tsite_score_denovo\tpep_scoremean_denovo\tpep_scoresum_denovo\tifEvident_denovo\tifFind_denovo\tmatch_type\tSite_tag_denovo_A\tsite_peaknum_denovo_A\tsite_score_denovo_A\tpep_scoremean_denovo_A\tpep_scoresum_denovo_A\tSite_tag_denovo_B\tsite_peaknum_denovo_B\tsite_score_denovo_B\tpep_scoremean_denovo_B\tpep_scoresum_denovo_B\n")
            print("reading DiNovo Res file...")
            print("reading ", DiNovo_res_file)
            psm_list = []
            with open(DiNovo_res_file, "r") as fin:
                line = fin.readline()
                line = fin.readline()
                count = 0
                empty_tag = False
                while True:
                    if count % 1000 == 0:
                        print(f"\r", f"count: {count} ", end="")
                    line = fin.readline().rstrip()
                    if len(line) == 0:
                        if empty_tag == False:
                            empty_tag = True
                            continue
                        else:
                            break
                    if line.startswith("\t"):
                        line = line.strip().split("\t")
                        rank = int(line[0])

                        if rank == 1:
                            top1_tag = True
                        else:
                            continue

                        try_denovo_seq = list(line[2])
                        score = float(line[4])
                        modification = line[3]
                        modification = modification.split(';')
                        assert modification[-1] == ''
                        modification = modification[:-1]
                        for item in modification:
                            site = int(item.split(',')[0])
                            modification_type = item.split(',')[1]
                            if modification_type not in modifications_MirrorNovo:
                                print(modification_type)
                                assert False
                            index = modifications_MirrorNovo.index(modification_type)
                            aa = modifications_in_aa_dict_keys[index]
                            try_denovo_seq[site - 1] = aa
                        Rseq, lys_denovo_seq = generate_Rseq_lys_seq_from_try_seq(try_denovo_seq, match_type)
                        psm_list.append(
                            (titlesPair, try_title, lys_title, Rseq, try_denovo_seq, lys_denovo_seq, match_type, score))
                    else:
                        line = line.strip().split("\t")
                        empty_tag = False
                        try_title = line[0]
                        lys_title = line[1]
                        titlesPair = try_title + "@" + lys_title
                        match_type = titlesPair_to_location_dict[titlesPair][1]
                        top1_tag = False
                        count += 1
            try_res_dict, lys_res_dict = parallel_get_confident_info_mirror(psm_list,
                                                                            try_spectrum_location_dict,
                                                                            try_suffix_name, try_mgf,
                                                                            lys_spectrum_location_dict,
                                                                            lys_suffix_name, lys_mgf)
            for i in range(len(psm_list)):
                title = psm_list[i][0]
                try_title = psm_list[i][1]
                lys_title = psm_list[i][2]
                denovoRseq = psm_list[i][3]
                try_denovoSeq = psm_list[i][4]
                lys_denovoSeq = psm_list[i][5]
                match_type = psm_list[i][6]
                score = psm_list[i][7]

                denovo_illegal = True
                if denovoRseq != ['']:
                    denovo_illegal = False
                    try_denovoSeq_str = "".join(try_denovoSeq)
                    lys_denovoSeq_str = "".join(lys_denovoSeq)
                    denovoRSeq_str = "".join(denovoRseq)
                    try_denovo_psm = try_title + "@" + try_denovoSeq_str
                    lys_denovo_psm = lys_title + "@" + lys_denovoSeq_str
                    merge_info_dict = merge_try_and_lys_info(try_res_dict[try_denovo_psm],
                                                             lys_res_dict[lys_denovo_psm])

                    # output_data
                    ##try
                    try_info_dict = try_res_dict[try_denovo_psm]
                    try_site_tag = try_info_dict["site_tag"]
                    try_site_peaknum = try_info_dict["site_peaknum"]
                    try_site_score = try_info_dict["site_score"]
                    try_pepscoremean = try_info_dict["pepscore_mean"]
                    try_pepscoresum = try_info_dict["pepscore_sum"]
                    ##lys
                    lys_info_dict = lys_res_dict[lys_denovo_psm]
                    lys_site_tag = lys_info_dict["site_tag"]
                    lys_site_peaknum = lys_info_dict["site_peaknum"]
                    lys_site_score = lys_info_dict["site_score"]
                    lys_pepscoremean = lys_info_dict["pepscore_mean"]
                    lys_pepscoresum = lys_info_dict["pepscore_sum"]
                    ##merge
                    site_tag = merge_info_dict["site_tag"]
                    site_peaknum = merge_info_dict["site_peaknum"]
                    site_score = merge_info_dict["site_score"]
                    pep_scoremean = merge_info_dict["pep_scoremean"]
                    pep_scoresum = merge_info_dict["pep_scoresum"]
                    site_intensitytemp = merge_info_dict["site_intensitytemp_A"]
                    # 判断是否top200
                    sort_intensity_list = sorted(try_info_dict["intensity_list"])
                    rank = len(sort_intensity_list) - np.array(
                        [bisect.bisect_left(sort_intensity_list, i) for i in site_intensitytemp]) + 1
                    site_tag = np.array(
                        [1 if tag == 1 and rank[i] <= topk_peaks else 0 for i, tag in enumerate(site_tag)])
                    miss_num = sum(site_tag == 0)
                    coverage = 1 - miss_num / len(site_tag)
                else:
                    coverage = 0
                ifEvident = if_evident_function(coverage)
                DiNovoTryLys_fout.write(
                    f"{title}\t{''.join(denovoRseq)}\t{score}\t{','.join(list(map(str, site_tag)))}\t{','.join(list(map(str, site_peaknum)))}\t{','.join(list(map(str, site_score)))}\t{str(pep_scoremean)}\t{str(pep_scoresum)}\t{ifEvident}\t{False}\t{match_type}\t"
                    f"{','.join(list(map(str, try_site_tag)))}\t{','.join(list(map(str, try_site_peaknum)))}\t{','.join(list(map(str, try_site_score)))}\t{str(try_pepscoremean)}\t{str(try_pepscoresum)}\t"
                    f"{','.join(list(map(str, lys_site_tag)))}\t{','.join(list(map(str, lys_site_peaknum)))}\t{','.join(list(map(str, lys_site_score)))}\t{str(lys_pepscoremean)}\t{str(lys_pepscoresum)}\n")
            DiNovoTryLys_fout.close()
            DiNovoTryLys_title_to_denovo_df = pd.read_csv(DiNovoTryLys_file, sep="\t")
        else:
            print("loading DiNovo targetRseq file...")
            DiNovoTryLys_title_to_denovo_df = pd.read_csv(DiNovoTryLys_file, sep="\t")
        if "MirrorFinderScore" not in DiNovoTryLys_title_to_denovo_df.columns:
            DiNovoTryLys_title_to_denovo_df["MirrorFinderScore"] = DiNovoTryLys_title_to_denovo_df.apply(lambda row: titlesPair_to_location_dict[row["title"]][2], axis=1)
            DiNovoTryLys_title_to_denovo_df.to_csv(DiNovoTryLys_file, sep="\t", index=False)
        DiNovoTryLys_title_to_denovo_df[["A_title", "B_title"]] = DiNovoTryLys_title_to_denovo_df["title"].str.split("@", n=1, expand=True)

        if not os.path.exists(DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res"):
            DiNovoTry_fout = open(DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res", "a+")
            DiNovoTry_fout.write("title\tdenovoSeq\tscore_software\tsite_tag_denovo\tsite_peaknum_denovo\tsite_score_denovo\tpep_scoremean_denovo\tpep_scoresum_denovo\tifEvident_denovo\tifFind_denovo\n")
            DiNovoTry_fin = open(DiNovoTry_file, 'r')
            print("reading DiNovoTry mgf files...")
            denovo_psm_list = []
            db_psm_list = []
            for filename in os.listdir(try_mgf):
                if filename.endswith(".mgf"):
                    file = try_mgf + f"\\{filename}"
                    print(f"reading {file}...")
                    with open(file, 'r') as f:
                        while True:
                            line = f.readline()
                            if len(line) == 0:
                                break
                            if line.startswith("TITLE="):
                                title = line.split('=')[1].rstrip()
                                if title in DiNovoTry_title_to_location.keys():
                                    denovo_seq,score = read_seqs_from_DiNovoSingleRes(DiNovoTry_title_to_location[title],DiNovoTry_fin,title,DiNovo_tool)
                                else:
                                    denovo_seq = ['']
                                    score = 0.0
                                denovo_psm_list.append((title, denovo_seq, score))
            denovo_res_dict = parallel_get_confident_info(denovo_psm_list,
                                                          try_spectrum_location_dict,
                                                          try_suffix_name, try_mgf)
            for i in range(len(denovo_psm_list)):
                title = denovo_psm_list[i][0]
                score = denovo_psm_list[i][2]
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
                     site_intensitytemp_denovo) = denovo_res_dict[denovo_psm]["site_tag"], \
                        denovo_res_dict[denovo_psm]["site_peaknum"], denovo_res_dict[denovo_psm][
                        "site_score"], denovo_res_dict[denovo_psm]["pepscore_mean"], \
                        denovo_res_dict[denovo_psm]["pepscore_sum"], denovo_res_dict[denovo_psm][
                        "intensity_list"], denovo_res_dict[denovo_psm]["site_intensitytemp"]
                    sort_intensity_list_denovo = sorted(intensity_list_denovo)
                    rank = len(sort_intensity_list_denovo) - np.array(
                        [bisect.bisect_left(sort_intensity_list_denovo, i) for i in
                         site_intensitytemp_denovo]) + 1
                    site_tag_denovo = np.array([1 if tag == 1 and rank[i] <= topk_peaks else 0 for i, tag in
                                                enumerate(site_tag_denovo)])
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
                DiNovoTry_fout.write(
                    f"{title}\t{denovoSeq_str}\t{score}\t{''.join(list(map(str, site_tag_denovo)))}\t{','.join(list(map(str, site_peaknum_denovo)))}\t{','.join(list(map(str, site_score_denovo)))}\t{pep_scoremean_denovo}\t{pep_scoresum_denovo}\t{ifEvident_denovo}\t{False}\n")
            DiNovoTry_fout.close()
            DiNovoTry_title_to_denovo_df = pd.read_csv(
                DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res", sep="\t")
        else:
            print("loading DiNovoTry_title_to_denovo_info.res file...")
            DiNovoTry_title_to_denovo_df = pd.read_csv(
                DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res", sep="\t")
        
        if not os.path.exists(DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res"):
            DiNovoLys_fout = open(DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res", "a+")
            DiNovoLys_fout.write("title\tdenovoSeq\tscore_software\tsite_tag_denovo\tsite_peaknum_denovo\tsite_score_denovo\tpep_scoremean_denovo\tpep_scoresum_denovo\tifEvident_denovo\tifFind_denovo\n")
            DiNovoLys_fin = open(DiNovoLys_file, 'r')
            print("reading DiNovoLys mgf files...")
            denovo_psm_list = []
            db_psm_list = []
            for filename in os.listdir(lys_mgf):
                if filename.endswith(".mgf"):
                    file = lys_mgf + f"\\{filename}"
                    print(f"reading {file}...")
                    with open(file, 'r') as f:
                        while True:
                            line = f.readline()
                            if len(line) == 0:
                                break
                            if line.startswith("TITLE="):
                                title = line.split('=')[1].rstrip()
                                if title in DiNovoLys_title_to_location.keys():
                                    denovo_seq, score = read_seqs_from_DiNovoSingleRes(
                                        DiNovoLys_title_to_location[title], DiNovoLys_fin, title,DiNovo_tool)
                                else:
                                    denovo_seq = ['']
                                    score = 0.0
                                denovo_psm_list.append((title, denovo_seq, score))
            denovo_res_dict = parallel_get_confident_info(denovo_psm_list,
                                                          lys_spectrum_location_dict,
                                                          lys_suffix_name, lys_mgf)
            for i in range(len(denovo_psm_list)):
                title = denovo_psm_list[i][0]
                score = denovo_psm_list[i][2]
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
                     site_intensitytemp_denovo) = denovo_res_dict[denovo_psm]["site_tag"], \
                        denovo_res_dict[denovo_psm]["site_peaknum"], denovo_res_dict[denovo_psm][
                        "site_score"], denovo_res_dict[denovo_psm]["pepscore_mean"], \
                        denovo_res_dict[denovo_psm]["pepscore_sum"], denovo_res_dict[denovo_psm][
                        "intensity_list"], denovo_res_dict[denovo_psm]["site_intensitytemp"]
                    sort_intensity_list_denovo = sorted(intensity_list_denovo)
                    rank = len(sort_intensity_list_denovo) - np.array(
                        [bisect.bisect_left(sort_intensity_list_denovo, i) for i in
                         site_intensitytemp_denovo]) + 1
                    site_tag_denovo = np.array([1 if tag == 1 and rank[i] <= topk_peaks else 0 for i, tag in
                                                enumerate(site_tag_denovo)])
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
                DiNovoLys_fout.write(
                    f"{title}\t{denovoSeq_str}\t{score}\t{''.join(list(map(str, site_tag_denovo)))}\t{','.join(list(map(str, site_peaknum_denovo)))}\t{','.join(list(map(str, site_score_denovo)))}\t{pep_scoremean_denovo}\t{pep_scoresum_denovo}\t{ifEvident_denovo}\t{False}\n")
            DiNovoLys_fout.close()
            DiNovoLys_title_to_denovo_df = pd.read_csv(
                DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res", sep="\t")
        else:
            print("loading DiNovoLys_title_to_denovo_info.res file...")
            DiNovoLys_title_to_denovo_df = pd.read_csv(
                DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res", sep="\t")
    
    if not os.path.exists(DiNovo_res_folder + "\\DiNovoTryLys_fasta_coverage_denovo_info.res"):
        DiNovoTryLys_title_to_denovo_df["match_type"] = DiNovoTryLys_title_to_denovo_df.apply(
            lambda x: titlesPair_to_location_dict[x["title"]][1], axis=1)
        seq_matchtype_list = []
        for index, row in DiNovoTryLys_title_to_denovo_df.iterrows():
            if row["denovoSeq"] == row["denovoSeq"] and row["denovoSeq"] != "":
                seq_matchtype_list.append(row["denovoSeq"] + "@" + row["match_type"])
        seq_matchtype_set = set(seq_matchtype_list)
        seq_matchtype_list = []
        for seq_matchtype in seq_matchtype_set:
            seq_matchtype_split = seq_matchtype.split("@")
            seq_matchtype_list.append((seq_matchtype_split[0], seq_matchtype_split[1]))
        top1_res_dict = mapped_function_mirror(seq_matchtype_list, fasta_seq_targetAnddecoy, fasta_seq_first_4aa_index_targetAnddecoy,
                                               mapped_version=mapped_version)
        DiNovoTryLys_title_to_denovo_df["denovoSeqID"] = DiNovoTryLys_title_to_denovo_df.apply(
            lambda row: row["denovoSeq"] + "@" + row["match_type"] if (row["denovoSeq"] == row["denovoSeq"] and row["denovoSeq"] != "") else "", axis=1)
        DiNovoTryLys_title_to_denovo_df["FindLocation_denovo"] = DiNovoTryLys_title_to_denovo_df[
            "denovoSeqID"].apply(lambda x: top1_res_dict[x][2] if x != "" and x == x else "")
        DiNovoTryLys_title_to_denovo_df["FindLocationDecoy_denovo"] = DiNovoTryLys_title_to_denovo_df[
            "denovoSeqID"].apply(lambda x: top1_res_dict[x][3] if x != "" and x == x else "")
        DiNovoTryLys_title_to_denovo_df["ifFind_denovo"] = DiNovoTryLys_title_to_denovo_df["denovoSeqID"].apply(lambda x: top1_res_dict[x][1] if x != "" and x == x else False)
        DiNovoTryLys_title_to_denovo_df["TDTag_denovo"] = DiNovoTryLys_title_to_denovo_df["denovoSeqID"].apply(lambda x: top1_res_dict[x][4] if x != "" and x == x else False)
        seq_to_unique_dict, seq_to_split_dict, DiNovoTryLys_fasta_coverage_denovo_dict, DiNovoTryLys_denovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_mirror( fasta_seq, DiNovoTryLys_title_to_denovo_df , False, 0.0)
        DiNovoTryLys_title_to_denovo_df["Seq_Unique_denovo"] = DiNovoTryLys_title_to_denovo_df[
            "denovoSeqID"].apply(lambda x: seq_to_unique_dict[x] if x != "" and x == x else "")
        DiNovoTryLys_title_to_denovo_df["Seq_split_denovo"] = DiNovoTryLys_title_to_denovo_df[
            "denovoSeqID"].apply(lambda x: seq_to_split_dict[x] if x != "" and x == x else "")
        DiNovoTryLys_title_to_denovo_df.to_csv(
            DiNovo_res_folder + "\\DiNovoTryLys_title_to_denovo_info.res", sep="\t", index=False)
        write_dict_to_csv(DiNovoTryLys_fasta_coverage_denovo_dict,
                          DiNovo_res_folder + "\\DiNovoTryLys_fasta_coverage_denovo_info.res", fasta_seq,
                          None, DiNovoTryLys_denovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTryLys_fasta_coverage_denovo_dict, DiNovoTryLys_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + "\\DiNovoTryLys_fasta_coverage_denovo_info.res")

    if not os.path.exists(DiNovo_res_folder + "\\DiNovoTryLys_fasta_coverage_Evidentdenovo_info.res"):
        _, _, DiNovoTryLys_fasta_coverage_Evidentdenovo_dict, DiNovoTryLys_Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_mirror(
            fasta_seq,
            DiNovoTryLys_title_to_denovo_df,
        True, 0.0)
        write_dict_to_csv(DiNovoTryLys_fasta_coverage_Evidentdenovo_dict,
                          DiNovo_res_folder + "\\DiNovoTryLys_fasta_coverage_Evidentdenovo_info.res",
                          fasta_seq, None, DiNovoTryLys_Evidentdenovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTryLys_fasta_coverage_Evidentdenovo_dict, DiNovoTryLys_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + "\\DiNovoTryLys_fasta_coverage_Evidentdenovo_info.res")

    if not os.path.exists(DiNovo_res_folder + "\\DiNovoTry_fasta_coverage_denovo_info.res"):
        seq_str_set = set(DiNovoTry_title_to_denovo_df["denovoSeq"])
        top1_res_dict = mapped_function_single(seq_str_set, fasta_seq, fasta_seq_first_4aa_index)
        DiNovoTry_title_to_denovo_df["ifFind_denovo"] = DiNovoTry_title_to_denovo_df[
            "denovoSeq"].apply(
            lambda x: top1_res_dict[x][1] if x == x else False)
        DiNovoTry_title_to_denovo_df["FindLocation_denovo"] = DiNovoTry_title_to_denovo_df[
            "denovoSeq"].apply(
            lambda x: top1_res_dict[x][3] if x == x else "")
        top1_res_dict_decoy = mapped_function_single(seq_str_set, fasta_seq_decoy, fasta_seq_first_4aa_index_decoy)
        DiNovoTry_title_to_denovo_df["ifFindDecoy_denovo"] = DiNovoTry_title_to_denovo_df[
            "denovoSeq"].apply(
            lambda x: top1_res_dict_decoy[x][2] if x == x else False)
        DiNovoTry_title_to_denovo_df["FindLocationDecoy_denovo"] = DiNovoTry_title_to_denovo_df["denovoSeq"].apply(
            lambda x: top1_res_dict_decoy[x][4] if x == x else "")
        seq_to_unique_dict, DiNovoTry_fasta_coverage_denovo_dict, DiNovoTry_denovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoTry_title_to_denovo_df,
            "denovo", False, 0.0)
        DiNovoTry_title_to_denovo_df["Seq_Unique_denovo"] = DiNovoTry_title_to_denovo_df[
            "denovoSeq"].apply(
            lambda x: seq_to_unique_dict[x] if x == x else "")
        DiNovoTry_title_to_denovo_df.to_csv(
            DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res", sep="\t",
            index=False)
        write_dict_to_csv(DiNovoTry_fasta_coverage_denovo_dict,
                          DiNovo_res_folder + "\\DiNovoTry_fasta_coverage_denovo_info.res", fasta_seq, None,
                          DiNovoTry_denovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTry_fasta_coverage_denovo_dict, DiNovoTry_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + "\\DiNovoTry_fasta_coverage_denovo_info.res")

    if not os.path.exists(DiNovo_res_folder + "\\DiNovoTry_fasta_coverage_Evidentdenovo_info.res"):
        _, DiNovoTry_fasta_coverage_Evidentdenovo_dict, DiNovoTry_Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoTry_title_to_denovo_df,
            "denovo", True, 0.0)
        write_dict_to_csv(DiNovoTry_fasta_coverage_Evidentdenovo_dict,
                          DiNovo_res_folder + "\\DiNovoTry_fasta_coverage_Evidentdenovo_info.res",
                          fasta_seq, None, DiNovoTry_Evidentdenovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTry_fasta_coverage_Evidentdenovo_dict, DiNovoTry_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + "\\DiNovoTry_fasta_coverage_Evidentdenovo_info.res")

    if not os.path.exists(DiNovo_res_folder + "\\DiNovoLys_fasta_coverage_denovo_info.res"):
        seq_str_set = set(DiNovoLys_title_to_denovo_df["denovoSeq"])
        top1_res_dict = mapped_function_single(seq_str_set, fasta_seq, fasta_seq_first_4aa_index)
        DiNovoLys_title_to_denovo_df["ifFind_denovo"] = DiNovoLys_title_to_denovo_df[
            "denovoSeq"].apply(
            lambda x: top1_res_dict[x][1] if x == x else False)
        DiNovoLys_title_to_denovo_df["FindLocation_denovo"] = DiNovoLys_title_to_denovo_df[
            "denovoSeq"].apply(
            lambda x: top1_res_dict[x][3] if x == x else "")
        top1_res_dict_decoy = mapped_function_single(seq_str_set, fasta_seq_decoy, fasta_seq_first_4aa_index_decoy)
        DiNovoLys_title_to_denovo_df["ifFindDecoy_denovo"] = DiNovoLys_title_to_denovo_df[
            "denovoSeq"].apply(
            lambda x: top1_res_dict_decoy[x][2] if x == x else False)
        DiNovoLys_title_to_denovo_df["FindLocationDecoy_denovo"] = \
        DiNovoLys_title_to_denovo_df["denovoSeq"].apply(
            lambda x: top1_res_dict_decoy[x][4] if x == x else "")
        seq_to_unique_dict, DiNovoLys_fasta_coverage_denovo_dict, DiNovoLys_denovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoLys_title_to_denovo_df,
            "denovo", False, 0.0)
        DiNovoLys_title_to_denovo_df["Seq_Unique_denovo"] = DiNovoLys_title_to_denovo_df[
            "denovoSeq"].apply(
            lambda x: seq_to_unique_dict[x] if x == x else "")
        DiNovoLys_title_to_denovo_df.to_csv(
            DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res", sep="\t",
            index=False)
        write_dict_to_csv(DiNovoLys_fasta_coverage_denovo_dict,
                          DiNovo_res_folder + "\\DiNovoLys_fasta_coverage_denovo_info.res", fasta_seq, None,
                          DiNovoLys_denovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoLys_fasta_coverage_denovo_dict, DiNovoLys_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + "\\DiNovoLys_fasta_coverage_denovo_info.res")

    if not os.path.exists(DiNovo_res_folder + "\\DiNovoLys_fasta_coverage_Evidentdenovo_info.res"):
        _, DiNovoLys_fasta_coverage_Evidentdenovo_dict, DiNovoLys_Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoLys_title_to_denovo_df,
            "denovo", True, 0.0)
        write_dict_to_csv(DiNovoLys_fasta_coverage_Evidentdenovo_dict,
                          DiNovo_res_folder + "\\DiNovoLys_fasta_coverage_Evidentdenovo_info.res",
                          fasta_seq, None, DiNovoLys_Evidentdenovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoLys_fasta_coverage_Evidentdenovo_dict, DiNovoLys_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + "\\DiNovoLys_fasta_coverage_Evidentdenovo_info.res")

    #生成TryBest和LysBest的title_to_denovo_info以及fasta_coverage
    intsection_info_fout = open(DiNovo_res_folder + "\\info.res", "w")
    print("原始镜像谱共计",len(DiNovoTryLys_title_to_denovo_df),"对")
    intsection_info_fout.write("原始镜像谱共计" + str(len(DiNovoTryLys_title_to_denovo_df)) + "对\n")
    DiNovoTryLysFind_title_to_denovo_df = DiNovoTryLys_title_to_denovo_df[(DiNovoTryLys_title_to_denovo_df["ifFind_denovo"] == True) & (DiNovoTryLys_title_to_denovo_df["TDTag_denovo"] == "T")].copy()
    print("镜像谱回贴后共计",len(DiNovoTryLysFind_title_to_denovo_df),"对")
    intsection_info_fout.write("镜像谱回贴后共计" + str(len(DiNovoTryLysFind_title_to_denovo_df)) + "对\n")
    DiNovoTryLysFind_title_to_denovo_df.loc[:,"A_title"] = DiNovoTryLysFind_title_to_denovo_df["title"].apply(lambda title: title.split("@")[0])
    DiNovoTryLysFind_title_to_denovo_df.loc[:, "B_title"] = DiNovoTryLysFind_title_to_denovo_df["title"].apply(lambda title: title.split("@")[1])
    DiNovoTryLysFind_title_to_denovo_df.loc[:,"A_denovoSeq"] = DiNovoTryLysFind_title_to_denovo_df.apply(lambda row: generate_try_lys_seq_from_RseqAndMatchtype(row["denovoSeq"],row["match_type"])[0],axis=1)
    DiNovoTryLysFind_title_to_denovo_df.loc[:,"B_denovoSeq"] = DiNovoTryLysFind_title_to_denovo_df.apply(lambda row: generate_try_lys_seq_from_RseqAndMatchtype(row["denovoSeq"],row["match_type"])[1],axis=1)
    DiNovoTryFind_title_to_denovo_df = DiNovoTry_title_to_denovo_df[(DiNovoTry_title_to_denovo_df["ifFind_denovo"] == True) & (DiNovoTry_title_to_denovo_df["ifFindDecoy_denovo"] == False)]
    DiNovoLysFind_title_to_denovo_df = DiNovoLys_title_to_denovo_df[(DiNovoLys_title_to_denovo_df["ifFind_denovo"] == True) & (DiNovoLys_title_to_denovo_df["ifFindDecoy_denovo"] == False)]
    #过滤有矛盾的结果
    DiNovoTryFind_title_to_resRow = generate_title_to_res_dict(DiNovoTryFind_title_to_denovo_df)
    DiNovoLysFind_title_to_resRow = generate_title_to_res_dict(DiNovoLysFind_title_to_denovo_df)
    DiNovoTryLysFindFilter_title_to_denovo_df = filter_conflict_result(DiNovoTryLysFind_title_to_denovo_df,DiNovoTryFind_title_to_resRow,DiNovoLysFind_title_to_resRow)
    DiNovoTryLysFindFilter_title_to_denovo_df.to_csv(DiNovo_res_folder + "\\DiNovoTryLysFindFilter_title_to_denovo_info.res", sep="\t", index=False)
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
    if not os.path.exists(DiNovo_res_folder + "\\DiNovoTryBest_title_to_denovo_info.res"):
        DiNovoTryFindFilter_title_to_BestRow, DiNovoLysFindFilter_title_to_BestRow = get_maxscore_info(DiNovoTryLysFindFilter_title_to_denovo_df)
        A_titles = set(DiNovoTryFind_title_to_denovo_df["title"]) | set(DiNovoTryFindFilter_title_to_BestRow.keys())
        B_titles = set(DiNovoLysFind_title_to_denovo_df["title"]) | set(DiNovoLysFindFilter_title_to_BestRow.keys())
        (DiNovoTryBest_title_to_denovo_df,
         A_count1, A_count2, A_count21, A_count22, A_count3, A_count31, A_count32) = generate_Best_title_to_denovo_df(A_titles, DiNovoTryFind_title_to_denovo_df, DiNovoTryFindFilter_title_to_BestRow, "A")
        (DiNovoLysBest_title_to_denovo_df,
         B_count1, B_count2, B_count21, B_count22, B_count3, B_count31, B_count32) = generate_Best_title_to_denovo_df(B_titles, DiNovoLysFind_title_to_denovo_df, DiNovoLysFindFilter_title_to_BestRow, "B")
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
        DiNovoTryBest_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTryBest_title_to_denovo_info.res", sep="\t")
        DiNovoLysBest_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoLysBest_title_to_denovo_info.res", sep="\t")

    if not os.path.exists(DiNovo_res_folder + "\\DiNovoTryBest_fasta_coverage_denovo_info.res"):
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
        DiNovoTryBest_title_to_denovo_df.to_csv(DiNovo_res_folder + "\\DiNovoTryBest_title_to_denovo_info.res",index=False, sep ="\t")
        write_dict_to_csv(DiNovoTryBest_fasta_coverage_denovo_dict,
                          DiNovo_res_folder + "\\DiNovoTryBest_fasta_coverage_denovo_info.res",
                          fasta_seq, None, DiNovoTryBest_denovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTryBest_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTryBest_title_to_denovo_info.res", sep="\t")
        DiNovoTryBest_fasta_coverage_denovo_dict, DiNovoTryBest_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + "\\DiNovoTryBest_fasta_coverage_denovo_info.res")

    if not os.path.exists(DiNovo_res_folder + "\\DiNovoTryBest_fasta_coverage_Evidentdenovo_info.res"):
        _, DiNovoTryBest_fasta_coverage_Evidentdenovo_dict, DiNovoTryBest_Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoTryBest_title_to_denovo_df,
            "denovo", True, 0.0)
        write_dict_to_csv(DiNovoTryBest_fasta_coverage_Evidentdenovo_dict,
                          DiNovo_res_folder + "\\DiNovoTryBest_fasta_coverage_Evidentdenovo_info.res",
                          fasta_seq, None, DiNovoTryBest_Evidentdenovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoTryBest_fasta_coverage_Evidentdenovo_dict, DiNovoTryBest_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + "\\DiNovoTryBest_fasta_coverage_Evidentdenovo_info.res")

    if not os.path.exists(DiNovo_res_folder + "\\DiNovoLysBest_fasta_coverage_denovo_info.res"):
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
        DiNovoLysBest_title_to_denovo_df.to_csv(DiNovo_res_folder + "\\DiNovoLysBest_title_to_denovo_info.res",index=False, sep ="\t")
        write_dict_to_csv(DiNovoLysBest_fasta_coverage_denovo_dict,
                          DiNovo_res_folder + "\\DiNovoLysBest_fasta_coverage_denovo_info.res",
                          fasta_seq, None, DiNovoLysBest_denovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoLysBest_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoLysBest_title_to_denovo_info.res", sep="\t")
        DiNovoLysBest_fasta_coverage_denovo_dict, DiNovoLysBest_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + "\\DiNovoLysBest_fasta_coverage_denovo_info.res")

    if not os.path.exists(DiNovo_res_folder + "\\DiNovoLysBest_fasta_coverage_Evidentdenovo_info.res"):
        _, DiNovoLysBest_fasta_coverage_Evidentdenovo_dict, DiNovoLysBest_Evidentdenovo_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq,
            DiNovoLysBest_title_to_denovo_df,
            "denovo", True, 0.0)
        write_dict_to_csv(DiNovoLysBest_fasta_coverage_Evidentdenovo_dict,
                          DiNovo_res_folder + "\\DiNovoLysBest_fasta_coverage_Evidentdenovo_info.res",
                          fasta_seq, None, DiNovoLysBest_Evidentdenovo_proteinName_to_uniqueSeq_dict)
    else:
        DiNovoLysBest_fasta_coverage_Evidentdenovo_dict, DiNovoLysBest_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            DiNovo_res_folder + "\\DiNovoLysBest_fasta_coverage_Evidentdenovo_info.res")

    # 把DiNovo镜像谱和单谱合并在一起
    plot_aa_intsection_venn3(DiNovoTry_fasta_coverage_denovo_dict, DiNovoLys_fasta_coverage_denovo_dict,
                             DiNovoTryLys_fasta_coverage_denovo_dict,
                             DiNovo_res_folder + "\\intsection_denovo_AA[DiNovoNei].png", fig_name1,
                             fig_name2, fig_name1 + "@" + fig_name2)
    DiNovo_fasta_coverage_denovo_dict = merge_fasta_coverage_dict(
        [DiNovoTry_fasta_coverage_denovo_dict, DiNovoLys_fasta_coverage_denovo_dict,
         DiNovoTryLys_fasta_coverage_denovo_dict])
    DiNovo_denovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
        [DiNovoTry_denovo_proteinName_to_uniqueSeq_dict, DiNovoLys_denovo_proteinName_to_uniqueSeq_dict,
         DiNovoTryLys_denovo_proteinName_to_uniqueSeq_dict])
    write_dict_to_csv(DiNovo_fasta_coverage_denovo_dict,
                      DiNovo_res_folder + "\\Union_fasta_coverage_denovo_info[DiNovoNei].res", fasta_seq,
                      None, DiNovo_denovo_proteinName_to_uniqueSeq_dict)

    plot_aa_intsection_venn3(DiNovoTry_fasta_coverage_Evidentdenovo_dict,
                             DiNovoLys_fasta_coverage_Evidentdenovo_dict,
                             DiNovoTryLys_fasta_coverage_Evidentdenovo_dict,
                             DiNovo_res_folder + "\\intsection_Evidentdenovo_AA[DiNovoNei].png", fig_name1,
                             fig_name2, fig_name1 + "@" + fig_name2)
    DiNovo_fasta_coverage_Evidentdenovo_dict = merge_fasta_coverage_dict(
        [DiNovoTry_fasta_coverage_Evidentdenovo_dict, DiNovoLys_fasta_coverage_Evidentdenovo_dict,
         DiNovoTryLys_fasta_coverage_Evidentdenovo_dict])
    DiNovo_Evidentdenovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
        [DiNovoTry_Evidentdenovo_proteinName_to_uniqueSeq_dict,
         DiNovoLys_Evidentdenovo_proteinName_to_uniqueSeq_dict,
         DiNovoTryLys_Evidentdenovo_proteinName_to_uniqueSeq_dict])
    write_dict_to_csv(DiNovo_fasta_coverage_Evidentdenovo_dict,
                      DiNovo_res_folder + "\\Union_fasta_coverage_Evidentdenovo_info[DiNovoNei].res",
                      fasta_seq, None, DiNovo_Evidentdenovo_proteinName_to_uniqueSeq_dict)

    #TryBest和LysBest合并
    plot_aa_intsection_venn2(DiNovoTryBest_fasta_coverage_denovo_dict, DiNovoLysBest_fasta_coverage_denovo_dict,
                             DiNovo_res_folder + "\\intsection_denovo_AA[DiNovoNeiBest].png", fig_name1,
                             fig_name2)
    DiNovoBest_fasta_coverage_denovo_dict = merge_fasta_coverage_dict(
        [DiNovoTryBest_fasta_coverage_denovo_dict, DiNovoLysBest_fasta_coverage_denovo_dict])
    DiNovoBest_denovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
        [DiNovoTryBest_denovo_proteinName_to_uniqueSeq_dict, DiNovoLysBest_denovo_proteinName_to_uniqueSeq_dict])
    write_dict_to_csv(DiNovoBest_fasta_coverage_denovo_dict,
                      DiNovo_res_folder + "\\Union_fasta_coverage_denovo_info[DiNovoNeiBest].res", fasta_seq,
                      None, DiNovoBest_denovo_proteinName_to_uniqueSeq_dict)

    plot_aa_intsection_venn2(DiNovoTryBest_fasta_coverage_Evidentdenovo_dict,
                             DiNovoLysBest_fasta_coverage_Evidentdenovo_dict,
                             DiNovo_res_folder + "\\intsection_Evidentdenovo_AA[DiNovoNeiBest].png", fig_name1,
                             fig_name2)
    DiNovoBest_fasta_coverage_Evidentdenovo_dict = merge_fasta_coverage_dict(
        [DiNovoTryBest_fasta_coverage_Evidentdenovo_dict, DiNovoLysBest_fasta_coverage_Evidentdenovo_dict])
    DiNovoBest_Evidentdenovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
        [DiNovoTryBest_Evidentdenovo_proteinName_to_uniqueSeq_dict,
         DiNovoLysBest_Evidentdenovo_proteinName_to_uniqueSeq_dict])
    write_dict_to_csv(DiNovoBest_fasta_coverage_Evidentdenovo_dict,
                      DiNovo_res_folder + "\\Union_fasta_coverage_Evidentdenovo_info[DiNovoNeiBest].res",
                      fasta_seq, None, DiNovoBest_Evidentdenovo_proteinName_to_uniqueSeq_dict)

    #######################################################################
    #肽段
    DiNovoTryLys_evident_spec_num_denovo = 0
    DiNovoTryLys_NonEvident_spec_num_denovo = 0
    DiNovoTryLys_evident_find_spec_num_denovo = 0
    DiNovoTryLys_NonEvident_find_spec_num_denovo = 0
    DiNovoTryLys_denovoSeq_to_info_dict = {}
    DiNovoTryLys_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTryLys_title_to_denovo_info.res", sep="\t")
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
    DiNovoTryLys_denovoSeq_info_fw = open(
        DiNovo_res_folder + "\\DiNovoTryLys_denovoSeq_to_ifEvidentFind.res", "w")
    header = "denovoSeq_DiNovoTryLys\tifEvident_DiNovoTryLys\tifFind_DiNovoTryLys\tTDTag_DiNovoTryLys\ttitles_DiNovoTryLys\n"
    DiNovoTryLys_denovoSeq_info_fw.write(header)
    for key in DiNovoTryLys_denovoSeq_to_info_dict:
        string = f"{key}\t{DiNovoTryLys_denovoSeq_to_info_dict[key][0]}\t{DiNovoTryLys_denovoSeq_to_info_dict[key][1]}\t{DiNovoTryLys_denovoSeq_to_info_dict[key][2]}\t{DiNovoTryLys_denovoSeq_to_info_dict[key][3]}\n"
        DiNovoTryLys_denovoSeq_info_fw.write(string)
    DiNovoTryLys_denovoSeq_info_fw.flush()
    DiNovoTryLys_denovoSeq_info_fw.close()
    
    intsection_info_fout = open(DiNovo_res_folder + "\\info.res", "a+")
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
    DiNovoTry_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTry_title_to_denovo_info.res", sep="\t")
    for index, row in DiNovoTry_title_to_denovo_df.iterrows():
        if index % 1000 == 0:
            print(f"\r", index, "/", len(DiNovoTry_title_to_denovo_df), end="")
        title, denovo_seq, ifEvident, ifFind = row["title"], row["denovoSeq"], row["ifEvident_denovo"], row[
            "ifFind_denovo"]
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
    DiNovoTry_denovoSeq_info_fw = open(DiNovo_res_folder + "\\DiNovoTry_denovoSeq_to_ifEvidentFind.res","w")
    header = "denovoSeq_DiNovoTry\tifEvident_DiNovoTry\tifFind_DiNovoTry\tifFindDecoy_DiNovoTry\ttitles_DiNovoTry\n"
    DiNovoTry_denovoSeq_info_fw.write(header)
    for key in DiNovoTry_denovoSeq_to_info_dict:
        string = f"{key}\t{DiNovoTry_denovoSeq_to_info_dict[key][0]}\t{DiNovoTry_denovoSeq_to_info_dict[key][1]}\t{DiNovoTry_denovoSeq_to_info_dict[key][2]}\t{DiNovoTry_denovoSeq_to_info_dict[key][3]}\n"
        DiNovoTry_denovoSeq_info_fw.write(string)
    DiNovoTry_denovoSeq_info_fw.flush()
    DiNovoTry_denovoSeq_info_fw.close()

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
    DiNovoLys_title_to_denovo_df = pd.read_csv(
        DiNovo_res_folder + "\\DiNovoLys_title_to_denovo_info.res", sep="\t")
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
    DiNovoLys_denovoSeq_info_fw = open(DiNovo_res_folder + "\\DiNovoLys_denovoSeq_to_ifEvidentFind.res","w")
    header = "denovoSeq_DiNovoLys\tifEvident_DiNovoLys\tifFind_DiNovoLys\tifFindDecoy_DiNovoLys\ttitles_DiNovoLys\n"
    DiNovoLys_denovoSeq_info_fw.write(header)
    for key in DiNovoLys_denovoSeq_to_info_dict:
        string = f"{key}\t{DiNovoLys_denovoSeq_to_info_dict[key][0]}\t{DiNovoLys_denovoSeq_to_info_dict[key][1]}\t{DiNovoLys_denovoSeq_to_info_dict[key][2]}\t{DiNovoLys_denovoSeq_to_info_dict[key][3]}\n"
        DiNovoLys_denovoSeq_info_fw.write(string)
    DiNovoLys_denovoSeq_info_fw.flush()
    DiNovoLys_denovoSeq_info_fw.close()

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
    DiNovoTryBest_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTryBest_title_to_denovo_info.res", sep="\t")
    for index, row in DiNovoTryBest_title_to_denovo_df.iterrows():
        if index % 1000 == 0:
            print(f"\r", index, "/", len(DiNovoTryBest_title_to_denovo_df), end="")
        title, denovo_seq, ifEvident, ifFind = row["title"], row["denovoSeq"], row["ifEvident_denovo"], row["ifFind_denovo"]
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
                                                                [(title, ifEvident, ifFind)]]
            else:
                ifEvident_new = DiNovoTryBest_denovoSeq_to_info_dict[denovo_seq][0] | ifEvident
                ifFind_new = DiNovoTryBest_denovoSeq_to_info_dict[denovo_seq][1] | ifFind
                DiNovoTryBest_denovoSeq_to_info_dict[denovo_seq] = [ifEvident_new, ifFind_new,
                                                                DiNovoTryBest_denovoSeq_to_info_dict[denovo_seq][
                                                                    2] + [
                                                                    (title, ifEvident, ifFind)]]
    DiNovoTryBest_denovoSeq_info_fw = open(DiNovo_res_folder + "\\DiNovoTryBest_denovoSeq_to_ifEvidentFind.res", "w")
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
    DiNovoLysBest_title_to_denovo_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoLysBest_title_to_denovo_info.res",sep="\t")
    for index, row in DiNovoLysBest_title_to_denovo_df.iterrows():
        if index % 1000 == 0:
            print(f"\r", index, "/", len(DiNovoLysBest_title_to_denovo_df), end="")
        title, denovo_seq, ifEvident, ifFind = row["title"], row["denovoSeq"], row["ifEvident_denovo"], row["ifFind_denovo"]
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
                                                                    [(title, ifEvident, ifFind)]]
            else:
                ifEvident_new = DiNovoLysBest_denovoSeq_to_info_dict[denovo_seq][0] | ifEvident
                ifFind_new = DiNovoLysBest_denovoSeq_to_info_dict[denovo_seq][1] | ifFind
                DiNovoLysBest_denovoSeq_to_info_dict[denovo_seq] = [ifEvident_new, ifFind_new,
                                                                    DiNovoLysBest_denovoSeq_to_info_dict[denovo_seq][
                                                                        2] + [
                                                                        (title, ifEvident, ifFind)]]
    DiNovoLysBest_denovoSeq_info_fw = open(DiNovo_res_folder + "\\DiNovoLysBest_denovoSeq_to_ifEvidentFind.res", "w")
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
    DiNovoLys_denovoSeq_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoLys_denovoSeq_to_ifEvidentFind.res", sep="\t")
    DiNovoTryLys_denovoSeq_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTryLys_denovoSeq_to_ifEvidentFind.res", sep="\t")
    DiNovoTryBest_denovoSeq_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTryBest_denovoSeq_to_ifEvidentFind.res", sep="\t")
    DiNovoLysBest_denovoSeq_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoLysBest_denovoSeq_to_ifEvidentFind.res", sep="\t")

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
    # DiNovoTryLys_DenovoRSeq_df = pd.read_csv(DiNovo_res_folder + "\\DiNovoTryLys_denovoSeq_to_ifEvidentFind.res", sep="\t")
    # DiNovoTryLys_DenovoRSeq_df["ifDiNovoTryLys"] = True
    # Na_df = DiNovoTryLys_DenovoRSeq_df[DiNovoTryLys_DenovoRSeq_df['denovoSeq_DiNovoTryLys'].isna()]
    # assert len(Na_df) == 1 or len(Na_df) == 0
    # DiNovoTryLys_DenovoRSeq_df = DiNovoTryLys_DenovoRSeq_df[DiNovoTryLys_DenovoRSeq_df['denovoSeq_DiNovoTryLys'].notna()]
    # DiNovoTryLys_DenovoRSeq_df['Rseq_try'] = DiNovoTryLys_DenovoRSeq_df.apply(lambda x: transfer_Rseq_to_trySeq(x['denovoSeq_DiNovoTryLys']), axis=1)
    # DiNovoTryLys_DenovoRSeq_df['Rseq_lys'] = DiNovoTryLys_DenovoRSeq_df.apply(lambda x: transfer_Rseq_to_lysSeq(x['denovoSeq_DiNovoTryLys']), axis=1)
    #
    # DiNovoTry_DenovoSeq_df = pd.read_csv(DiNovo_res_folder + "\\ALL_DiNovoTry_denovoSeq_to_ifEvidentFind.res", sep="\t")
    # DiNovoTry_DenovoSeq_df["denovo_seqNOKR"] = DiNovoTry_DenovoSeq_df["denovoSeq_DiNovoTry"].apply(remove_try_KR_function)
    # DiNovoTry_DenovoSeq_df["ifDiNovoTry"] = True
    #
    # DiNovoLys_DenovoSeq_df = pd.read_csv(DiNovo_res_folder + "\\ALL_DiNovoLys_denovoSeq_to_ifEvidentFind.res", sep="\t")
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
    #     DiNovo_res_folder + "\\intsection_DenovoSeq_to_ifEvidentFind[DiNovoNei].res", sep="\t", index=False)
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
    #         DiNovo_res_folder + "\\intsection_DenovoSeq_to_ifEvidentFind_mini[DiNovoNei].res"):
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
    #     DiNovo_intsection_DenovoSeq_df.to_csv(DiNovo_res_folder + "\\intsection_DenovoSeq_to_ifEvidentFind_mini[DiNovoNei].res", sep="\t", index=False)
    # else:
    #     DiNovo_intsection_DenovoSeq_df = pd.read_csv(
    #         DiNovo_res_folder + "\\intsection_DenovoSeq_to_ifEvidentFind_mini[DiNovoNei].res", sep="\t")
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




