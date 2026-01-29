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
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import ast

peaks_suffix_name = "_HCDFT.mgf"

# 计算离子覆盖率时用到的参数
if_use_ppm = config.if_use_ppm
fragment_tolerance_ppm = config.fragment_tolerance_ppm
fragment_tolerance_da = config.fragment_tolerance_da
modifications = config.modifications
modifications_in_aa_dict_keys = config.modifications_in_aa_dict_keys
modifications_other  = config.modifications_other
modifications_other_in_aa_dict_keys  = config.modifications_other_in_aa_dict_keys
modifications_other_mass  = config.modifications_other_mass
modifications_DiNovo = config.modifications_DiNovo
aa_to_mass_dict = config.aa_to_mass_dict
atom_mass = config.atom_mass

def plot_spectra_venn2(try_spectra_set, lys_spectra_set, fig_path, fig_name1, fig_name2):
    plt.figure()
    venn2([try_spectra_set, lys_spectra_set], (fig_name1, fig_name2))
    plt.savefig(fig_path)

def analyse_pep_match_type_class(pep_set, pep_df, colunm_name, fig_path):
    pep_to_index = {}
    for index, row in pep_df.iterrows():
        pep_to_index[row[f"denovoSeq_{colunm_name}"]] = index

    match_type_class_list = []
    for pep in pep_set:
        if pep not in pep_to_index:
            print(pep)
            assert False
        else:
            assert pep == pep_df.loc[pep_to_index[pep]][f"denovoSeq_{colunm_name}"], f"pep {pep} not match {pep_df.loc[pep_to_index[pep]][f'denovoSeq_{colunm_name}']}"
            titles_info = ast.literal_eval(pep_df.loc[pep_to_index[pep]][f"titles_{colunm_name}"])
            for title_info in titles_info:
                match_type_class_list.append(title_info[3])

    #画图
    match_type_class_to_count = {}
    for match_type_class in match_type_class_list:
        if match_type_class not in match_type_class_to_count:
            match_type_class_to_count[match_type_class] = 1
        else:
            match_type_class_to_count[match_type_class] += 1
    plt.figure()
    plt.bar(match_type_class_to_count.keys(), match_type_class_to_count.values())
    plt.savefig(fig_path)

def plot_pep_venn2(try_pep_set, lys_pep_set, fig_path, fig_name1, fig_name2):
    plt.figure()
    venn2([try_pep_set, lys_pep_set], (fig_name1, fig_name2))
    plt.savefig(fig_path)
    plt.close()

    #去除修饰后的交并差
    try_pep_set_pure = set([i.replace("C(+57.02)", "C").replace("M(+15.99)", "M").replace("I","L") for i in try_pep_set])
    lys_pep_set_pure = set([i.replace("C(+57.02)", "C").replace("M(+15.99)", "M").replace("I","L") for i in lys_pep_set])
    plt.figure()
    venn2([try_pep_set_pure, lys_pep_set_pure], (fig_name1, fig_name2))
    plt.savefig(fig_path + "_no_mod.png")
    plt.close()

    # 所有鉴定到的肽段的长度箱形图
    try_pep_length = [len(i.replace("C(+57.02)", "C").replace("M(+15.99)", "M")) for i in (try_pep_set)]
    lys_pep_length = [len(i.replace("C(+57.02)", "C").replace("M(+15.99)", "M")) for i in (lys_pep_set)]
    data = [try_pep_length, lys_pep_length]  # 将两个数据集放入一个列表中
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 16})
    box = plt.boxplot(data, patch_artist=True)  # 使用patch_artist=True来填充箱型图的颜色
    # Set custom colors
    colors = ["orchid", "dodgerblue"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.ylabel("Peptide length",fontsize=16)
    plt.xticks(np.arange(1, 3), [fig_name1, fig_name2],fontsize=16)  # 修改xticks的范围和标签以匹配两个数据集
    plt.savefig(fig_path + "_lengthALL[boxplot].png")
    plt.close()

    #所有肽段的长度数量
    try_pep_length_to_count = {}
    lys_pep_length_to_count = {}
    for i in try_pep_length:
        if i not in try_pep_length_to_count:
            try_pep_length_to_count[i] = 1
        else:
            try_pep_length_to_count[i] += 1
    for i in lys_pep_length:
        if i not in lys_pep_length_to_count:
            lys_pep_length_to_count[i] = 1
        else:
            lys_pep_length_to_count[i] += 1
    pep_length_set = set(try_pep_length_to_count.keys()) | set(lys_pep_length_to_count.keys())
    for i in pep_length_set:
        if i not in try_pep_length_to_count:
            try_pep_length_to_count[i] = 0
        if i not in lys_pep_length_to_count:
            lys_pep_length_to_count[i] = 0
    # 设置柱子的宽度
    bar_width = 0.35
    # 确保两个字典的键是相同的
    assert try_pep_length_to_count.keys() == lys_pep_length_to_count.keys(), "Keys of the dictionaries do not match"
    # 设置柱子的位置，基于字典的键
    index = np.array(list(try_pep_length_to_count.keys()))
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 16})
    # 画try_pep的柱子，位置向左偏移半个柱子宽度
    plt.bar(index - bar_width / 2, try_pep_length_to_count.values(), bar_width, alpha=0.7, label=fig_name1)
    # 画lys_pep的柱子，位置向右偏移半个柱子宽度
    plt.bar(index + bar_width / 2, lys_pep_length_to_count.values(), bar_width, alpha=0.7, label=fig_name2)
    plt.xlabel("Peptide length", fontsize=16)
    plt.ylabel("Peptide count", fontsize=16)
    plt.xticks(index, index)  # 设置x轴标签为字典的键
    plt.legend()
    plt.savefig(fig_path + "_lengthALL[bar].png")
    plt.close()


    #各自单独鉴定到的肽段的长度箱形图
    try_pep_length = [len(i.replace("C(+57.02)", "C").replace("M(+15.99)", "M")) for i in (try_pep_set - lys_pep_set)]
    lys_pep_length = [len(i.replace("C(+57.02)", "C").replace("M(+15.99)", "M")) for i in (lys_pep_set - try_pep_set)]
    data = [try_pep_length, lys_pep_length]  # 将两个数据集放入一个列表中
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 16})
    box = plt.boxplot(data, patch_artist=True)  # 使用patch_artist=True来填充箱型图的颜色
    # Set custom colors
    colors = ["orchid", "dodgerblue"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.ylabel("Peptide length",fontsize=16)
    plt.xticks(np.arange(1, 3), [fig_name1, fig_name2],fontsize=16)  # 修改xticks的范围和标签以匹配两个数据集
    plt.savefig(fig_path + "_lengthSeperate[boxplot].png")
    plt.close()

    #不同肽段的数量
    try_pep_length_to_count = {}
    lys_pep_length_to_count = {}
    for i in try_pep_length:
        if i not in try_pep_length_to_count:
            try_pep_length_to_count[i] = 1
        else:
            try_pep_length_to_count[i] += 1
    for i in lys_pep_length:
        if i not in lys_pep_length_to_count:
            lys_pep_length_to_count[i] = 1
        else:
            lys_pep_length_to_count[i] += 1
    pep_length_set = set(try_pep_length_to_count.keys()) | set(lys_pep_length_to_count.keys())
    for i in pep_length_set:
        if i not in try_pep_length_to_count:
            try_pep_length_to_count[i] = 0
        if i not in lys_pep_length_to_count:
            lys_pep_length_to_count[i] = 0
    # 设置柱子的宽度
    bar_width = 0.35
    # 确保两个字典的键是相同的
    assert try_pep_length_to_count.keys() == lys_pep_length_to_count.keys(), "Keys of the dictionaries do not match"
    # 设置柱子的位置，基于字典的键
    index = np.array(list(try_pep_length_to_count.keys()))
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 16})
    # 画try_pep的柱子，位置向左偏移半个柱子宽度
    plt.bar(index - bar_width/2, try_pep_length_to_count.values(), bar_width, alpha=0.7, label=fig_name1)
    # 画lys_pep的柱子，位置向右偏移半个柱子宽度
    plt.bar(index + bar_width/2, lys_pep_length_to_count.values(), bar_width, alpha=0.7, label=fig_name2)
    plt.xlabel("Peptide length",fontsize=16)
    plt.ylabel("Peptide count",fontsize=16)
    plt.xticks(index, index)  # 设置x轴标签为字典的键
    plt.legend()
    plt.savefig(fig_path + "_lengthSeperate[bar].png")
    plt.close()

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

def get_ScanIndex_from_pFindTitle(row):
    title = row["title"]
    title_split = title.split(".")
    pre = title_split[0]
    assert title_split[1] == title_split[2]
    scan = int(title_split[1])
    charge = int(title_split[3])
    return pre + "." + str(scan) + "." + str(charge)

def get_ScanIndex_from_MSGFTitle(row):
    filename = row["#SpecFile"].split(".")[0]
    scan = int(row["ScanNum"])
    charge = int(row["Charge"])
    return filename + "." + str(scan) + "." + str(charge)

def get_ScanIndex_from_MSFraggerTitle(row):
    title = row["title"].strip()
    title_split = title.split(".")
    pre = title_split[0]
    assert title_split[1] == title_split[2]
    scan = int(title_split[1])
    charge = int(title_split[3])
    return pre + "." + str(scan) + "." + str(charge)

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

def merge_fasta_coverage_dict(fasta_coverage_dict_list):
    fasta_coverage_denovo_dict = {}
    for key in fasta_coverage_dict_list[0]:
        length = len(fasta_coverage_dict_list[0][key][0])
        fasta_coverage_denovo_dict[key] = [np.zeros(length,dtype=int), np.zeros(length, dtype=int),\
                                    np.zeros(length, dtype=float),0,0,0]# site_tag, site_score, unique peptide num, non-unique peptide num,peptide num,
    for fasta_coverage_dict in fasta_coverage_dict_list:
        for key in fasta_coverage_dict:
            fasta_coverage_denovo_dict[key][0] = np.maximum(fasta_coverage_denovo_dict[key][0], fasta_coverage_dict[key][0])
            fasta_coverage_denovo_dict[key][1] += fasta_coverage_dict[key][1]
            fasta_coverage_denovo_dict[key][2] = np.maximum(fasta_coverage_denovo_dict[key][1], fasta_coverage_dict[key][2])
            fasta_coverage_denovo_dict[key][3] += fasta_coverage_dict[key][3]
            fasta_coverage_denovo_dict[key][4] += fasta_coverage_dict[key][4]
            fasta_coverage_denovo_dict[key][5] += fasta_coverage_dict[key][5]
    return fasta_coverage_denovo_dict

def plot_aa_intsection_venn3(dict_try, dict_lys, dict_mirror, fig_path, fig_name1, fig_name2, fig_name3):
    default_colors = [
        # r, g, b
        [250, 128, 114, 0.4],
        [255, 165, 0, 0.4],
        [30, 144, 255, 0.5],
        [241, 90, 96, 0.4],
        [255, 117, 0, 0.3],
        [82, 82, 190, 0.2],
    ]
    default_colors = [
        [i[0] / 255.0, i[1] / 255.0, i[2] / 255.0, i[3]]
        for i in default_colors
    ]

    assert len(dict_try) == len(dict_lys) == len(dict_mirror)
    Abc = 0
    aBc = 0
    ABc = 0
    abC = 0
    AbC = 0
    aBC = 0
    ABC = 0
    for key in dict_try:
        try_coverage = dict_try[key][0]
        lys_coverage = dict_lys[key][0]
        DiNovo_coverage = dict_mirror[key][0]
        Abc += sum([1 for i in range(len(try_coverage)) if
                    try_coverage[i] != 0  and lys_coverage[i] == 0 and DiNovo_coverage[i] == 0])
        aBc += sum([1 for i in range(len(try_coverage)) if
                    try_coverage[i] == 0 and lys_coverage[i] != 0  and DiNovo_coverage[i] == 0])
        ABc += sum([1 for i in range(len(try_coverage)) if
                    try_coverage[i] != 0  and lys_coverage[i] != 0  and DiNovo_coverage[i] == 0])
        abC += sum([1 for i in range(len(try_coverage)) if
                    try_coverage[i] == 0 and lys_coverage[i] == 0 and DiNovo_coverage[i] != 0 ])
        AbC += sum([1 for i in range(len(try_coverage)) if
                    try_coverage[i] != 0  and lys_coverage[i] == 0 and DiNovo_coverage[i] != 0 ])
        aBC += sum([1 for i in range(len(try_coverage)) if
                    try_coverage[i] == 0 and lys_coverage[i] != 0  and DiNovo_coverage[i] != 0 ])
        ABC += sum([1 for i in range(len(try_coverage)) if
                    try_coverage[i] != 0  and lys_coverage[i] != 0  and DiNovo_coverage[i] != 0 ])
    print("Abc: ", Abc, "aBc: ", aBc, "ABc: ", ABc, "abC: ", abC, "AbC: ", AbC, "aBC: ", aBC, "ABC: ", ABC)
    # venn3(subsets=(Abc, aBc, ABc, abC, AbC, aBC, ABC), set_labels=(fig_name1, fig_name2, fig_name3))
    # plt.savefig(fig_path, dpi=300)
    # plt.close()
    scale_ABc = ABc
    for i in tqdm(range(100)):
        scale_ABc += 1000

        subsets = (Abc, aBc, scale_ABc, abC, AbC, aBC, ABC)
        venn_diagram = venn3(subsets=subsets, set_labels=(fig_name1, fig_name2, fig_name3),
                             set_colors=(default_colors[0][:3], default_colors[1][:3], default_colors[2][:3]),
                             alpha=0.7)
        # 获取当前的轴对象
        ax = plt.gca()
        # 获取标签对象并设置字体大小
        for text in ax.texts:
            text.set_fontsize(16)  # 设置字体大小
        # 获取每个区域的位置
        circles = venn_diagram.get_patch_by_id
        try:
            venn_diagram.get_label_by_id('110').set_text(f'{ABc}')
            venn_diagram.get_label_by_id('100').set_text(f'{Abc}')
            venn_diagram.get_label_by_id('010').set_text(f'{aBc}')
            venn_diagram.get_label_by_id('001').set_text(f'{abC}')
            venn_diagram.get_label_by_id('101').set_text(f'{AbC}')
            venn_diagram.get_label_by_id('011').set_text(f'{aBC}')
            venn_diagram.get_label_by_id('111').set_text(f'{ABC}')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            break
        except:
            # 清空已经画的图
            plt.clf()
            pass

def read_seqs_from_DiNovoSingleRes(location, fi, title, res_type, fasta_seq = None, index_first_4aa = None ):

    if res_type == "MirrorNovo":
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
                if seq.count("(") != seq.count("C(") + seq.count("M("):
                    return ([''], 0.0)
                else:
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

    elif res_type == "DiNovo_MirrorNovo" or res_type == "DiNovo_pNovoM2":
        fi.seek(location)
        line = fi.readline().strip().split("\t")
        assert title == line[0]
        line = fi.readline().split("\t")
        if line[0] != '' or line == ['']:
            # 说明当前谱图无结果
            return ([''], None)
        seq = line[2]
        mod = line[3]
        score = float(line[4])
        if mod.count("[") != mod.count("[C") + mod.count("[M"):
            return ([''], None)
        else:
            seq = seq.replace("I", "L")
            if len(mod) > 0:
                seq_list = list(seq)
                mod_list = mod.split(";")[:-1]
                for mod_item in mod_list:
                    mod_item_list = mod_item.split(",")
                    idx, mod_name = mod_item_list[0], mod_item_list[1]
                    idx = int(idx)
                    mod = modifications_in_aa_dict_keys[modifications_DiNovo.index(mod_name)]
                    seq_list[idx - 1] = mod
            else:
                seq_list = list(seq)

            return (seq_list, score)

# def read_seqs_from_DiNovoSingleRes(location, fi, title, fasta_seq = None, index_first_4aa = None ):
#
#     if fasta_seq == None:
#         #只选择top1且不含有意外修饰的结果
#         fi.seek(location)
#         line = fi.readline().strip()
#         title_temp = line.split("\t")[0]
#         assert title == title_temp,f"title:{title}, title_temp:{title_temp}"
#         line = fi.readline()
#         print(line)
#         line = fi.readline()
#         print(line)
#         while True:
#             if "END" in line:
#                 seq = ['']
#                 score = 0.0
#                 break
#             else:
#                 line = line.strip().split("\t")
#                 print(title,line)
#                 seq = line[2]
#                 score = float(line[3])
#                 seq = seq.replace("C(Carbamidomethylation)","C(+57.02)")
#                 seq = seq.replace("M(Oxidation)","M(+15.99)")
#                 seq = seq.replace("I","L")
#                 #确保没有其他意外修饰
#                 if seq.count("(") != seq.count("C(") + seq.count("M("):
#                     line = fi.readline()
#                     continue
#                 else:
#                     seq = seq.split(",")
#                     break
#     else:
#         #选择topn中第一条能够回贴到数据库的结果
#         fi.seek(location)
#         line = fi.readline().strip()
#         title_temp = line.split("\t")[0]
#         assert title == title_temp, f"title:{title}, title_temp:{title_temp}"
#         line = fi.readline()
#         line = fi.readline()
#         if "END" in line:
#             seq = ['']
#         else:
#             find_tag_this_spectrum = False
#             while "END" not in line:
#                 if find_tag_this_spectrum:
#                     break
#                 line = line.strip().split("\t")
#                 seq_ori = line[2]
#                 seq = line[2].replace(',','')
#                 seq = seq.replace("C(Carbamidomethylation)", "C")
#                 seq = seq.replace("M(Oxidation)", "M")
#                 seq = seq.replace("I", "L")
#                 assert "(" not in seq
#                 find_tag = find_location_in_fasta_single(seq, fasta_seq, index_first_4aa)
#                 if find_tag:
#                     seq = seq_ori.replace("C(Carbamidomethylation)", "C(+57.02)")
#                     seq = seq.replace("M(Oxidation)", "M(+15.99)")
#                     seq = seq.replace("I", "L")
#                     seq = seq.split(",")
#                     find_tag_this_spectrum = True
#                 else:
#                     seq = ['']
#                 line = fi.readline()
#
#     return seq, score

def build_singleSpectra_location_function(file_path,software):
    #pNovoM原始结果格式
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
            return result_location_dict #
    #MirrorNovo结果格式
    elif software == "MirrorNovo":#MirrorNovo
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

    elif software == "pNovoM2":
        location_file = os.path.join(file_path + "_location")
        if os.path.exists(location_file) == False:
            print("building location for ", file_path)
            result_location_dict = {}
            with open(file_path, "r") as fin:
                while True:
                    location = fin.tell()
                    line = fin.readline().strip()
                    if len(line) == 0:
                        break
                    assert "@" in line
                    candidate_num = int(line.split("\t")[-1])
                    title = line.split("\t")[1].split("@")[0]
                    result_location_dict[title] = location
                    for i in range(candidate_num):
                        line = fin.readline()
            location_file = file_path + "_location"
            with open(location_file, 'wb') as fw:
                pickle.dump(result_location_dict, fw)
        else:
            result_location_dict = {}
            with open(location_file, 'rb') as fr:
                result_location_dict = pickle.load(fr)
        return result_location_dict

    elif software == "DiNovo_pNovoM2" or software == "DiNovo_MirrorNovo":
        location_file = os.path.join(file_path + "_location")
        if os.path.exists(location_file) == False:
            print("building location for ", file_path)
            result_location_dict = {}
            with open(file_path, "r") as f:
                line = f.readline()
                line = f.readline()
                while True:
                    location = f.tell()
                    line = f.readline()
                    if len(line.strip()) == 0:
                        break
                    if not line.startswith("\t"):
                        # match_type = line.split("\t")[2].strip()
                        title = line.strip()
                        result_location_dict[title] = location

            location_file = file_path + "_location"
            with open(location_file, 'wb') as fw:
                pickle.dump(result_location_dict, fw)
        else:
            result_location_dict = {}
            with open(location_file, 'rb') as fr:
                result_location_dict = pickle.load(fr)
        return result_location_dict



def plot_aa_intsection_venn2(dict_1, dict_2, fig_path, fig_name1, fig_name2):
    assert len(dict_1) == len(dict_2)
    Ab = 0
    aB = 0
    AB = 0
    for key in dict_1:
        dict_1_coverage = dict_1[key][0]
        dict_2_coverage = dict_2[key][0]
        Ab += sum([1 for i in range(len(dict_1_coverage)) if dict_1_coverage[i] == 1 and dict_2_coverage[i] == 0])
        aB += sum([1 for i in range(len(dict_1_coverage)) if dict_1_coverage[i] == 0 and dict_2_coverage[i] == 1])
        AB += sum([1 for i in range(len(dict_1_coverage)) if dict_1_coverage[i] == 1 and dict_2_coverage[i] == 1])
    print("Ab: ", Ab, "aB: ", aB, "AB: ", AB)
    venn2(subsets=(Ab, aB, AB), set_labels=(fig_name1, fig_name2))
    plt.savefig(fig_path, dpi=300)
    plt.close()

def generate_DiNovoPairs_to_matchtype_dict(file):
    titlesPair_to_location_dict = {}
    with open(file,"r") as fr:
        line = fr.readline()
        header_list = line.split("\t")
        for i,item in enumerate(header_list):
            if item == "A_TITLE" or item == "TRY_TITLE":
                try_title_header_index = i
            elif item == "B_TITLE" or item == "LYS_TITLE":
                lys_title_header_index = i
            elif item == "MIRROR_ANNO":
                match_type_header_index = i
            elif item == "TARGET_P-VALUE":
                target_score_header_index = i
        location = fr.tell()
        line = fr.readline().strip()
        while True:
            line = line.split("\t")
            try_title = line[try_title_header_index]
            lys_title = line[lys_title_header_index]
            match_type = line[match_type_header_index]
            target_score = line[target_score_header_index]
            titlesPair_to_location_dict[ try_title + "@" + lys_title] = (location,match_type,target_score)
            location = fr.tell()
            line = fr.readline().strip()
            if len(line) == 0:
                break
    return try_title_header_index, lys_title_header_index, match_type_header_index, titlesPair_to_location_dict

def plot_protein_intsection_venn2(dict_1, dict_2, fig_path, fig_name1, fig_name2, proteinName2UniqueSeq_dict1, proteinName2UniqueSeq_dict2, uniquePepCount_threshold):
    assert len(dict_1) == len(dict_2)
    dict1_all_protein = []
    dict2_all_protein = []

    dict1_protein = []
    dict2_protein = []
    UniqueSeq_count_dict1 = []
    UniqueSeq_count_dict2 = []
    dict1_coverage_count = {}
    dict2_coverage_count = {}
    for key in dict_1:
        if key not in proteinName2UniqueSeq_dict1:
            dict_1_UniqueSeqCount = 0
            UniqueSeq_count_dict1.append(dict_1_UniqueSeqCount)
        else:
            dict_1_UniqueSeq = proteinName2UniqueSeq_dict1[key]
            dict_1_UniqueSeqCount = len(dict_1_UniqueSeq)
            UniqueSeq_count_dict1.append(dict_1_UniqueSeqCount)
        if dict_1_UniqueSeqCount >= uniquePepCount_threshold:
            dict1_protein.append(key)
        if sum(dict_1[key][0]) > 0:
            dict1_all_protein.append(key)
            dict1_coverage_count[key] = sum(dict_1[key][0])

        if key not in proteinName2UniqueSeq_dict2:
            dict_2_UniqueSeqCount = 0
            UniqueSeq_count_dict2.append(dict_2_UniqueSeqCount)
        else:
            dict_2_UniqueSeq = proteinName2UniqueSeq_dict2[key]
            dict_2_UniqueSeqCount = len(dict_2_UniqueSeq)
            UniqueSeq_count_dict2.append(dict_2_UniqueSeqCount)
        if dict_2_UniqueSeqCount >= uniquePepCount_threshold:
            dict2_protein.append(key)
        if sum(dict_2[key][0]) > 0:
            dict2_all_protein.append(key)
            dict2_coverage_count[key] = sum(dict_2[key][0])

    # plot protein num vs unique pep num
    unique_pep_num = [i for i in range(31)]
    unique_pepnum_per_protein_dict1 = np.array(UniqueSeq_count_dict1)
    protein_num_dict1 = [np.sum(unique_pepnum_per_protein_dict1 > 0)]
    unique_pepnum_per_protein_dict2 = np.array(UniqueSeq_count_dict2)
    protein_num_dict2 = [np.sum(unique_pepnum_per_protein_dict2 > 0)]
    for i in unique_pep_num:
        if i == 0:
            continue
        this_protein_num = np.sum(unique_pepnum_per_protein_dict1 > i)
        protein_num_dict1.append(this_protein_num)
        this_protein_num = np.sum(unique_pepnum_per_protein_dict2 > i)
        protein_num_dict2.append(this_protein_num)
    plt.figure(figsize=(6, 8))
    plt.plot(unique_pep_num, protein_num_dict1)
    plt.plot(unique_pep_num, protein_num_dict2)
    plt.legend([fig_name1, fig_name2])
    plt.xlabel("Unique Peptide Count")
    plt.ylabel("Protein Count")
    plt.title("Line Graph of Protein Count")
    plt.savefig(fig_path + f"[LineGraphOfProteinFromPep].png", dpi=300)
    plt.show()

    # plot venn2 of dict1_protein and dict2_protein
    Ab = len(set(dict1_protein) - set(dict2_protein))
    aB = len(set(dict2_protein) - set(dict1_protein))
    AB = len(set(dict1_protein) & set(dict2_protein))
    print("Ab: ", Ab, "aB: ", aB, "AB: ", AB)
    venn2(subsets=(Ab, aB, AB), set_labels=(fig_name1, fig_name2))
    plt.savefig(fig_path + f"[Venn2OfProteinFromPep][{uniquePepCount_threshold}].png", dpi=300)
    plt.close()

    # plot venn2 of dict1_all_protein and dict2_all_protein
    Ab = len(set(dict1_all_protein) - set(dict2_all_protein))
    aB = len(set(dict2_all_protein) - set(dict1_all_protein))
    AB = len(set(dict1_all_protein) & set(dict2_all_protein))
    print("Ab: ", Ab, "aB: ", aB, "AB: ", AB)
    venn2(subsets=(Ab, aB, AB), set_labels=(fig_name1, fig_name2))
    plt.savefig(fig_path, dpi=300)
    plt.close()

    #boxplot
    only_dict1_coverage_count = [dict1_coverage_count[key] for key in dict1_coverage_count if key not in dict2_coverage_count]
    only_dict2_coverage_count = [dict2_coverage_count[key] for key in dict2_coverage_count if key not in dict1_coverage_count]
    plt.figure(figsize=(6, 8))
    plt.boxplot([only_dict1_coverage_count, only_dict2_coverage_count], patch_artist=True)
    plt.ylabel("Coverage AA number")
    plt.title("Boxplot of Coverage AA number")
    plt.xticks(np.arange(1, 3), [fig_name1, fig_name2])  # 修改xticks的范围和标签以匹配两个数据集
    plt.savefig(fig_path + f"[BoxplotOfCoverageAA].png", dpi=300)


def calculate_mass(seq,if_delete_first_aa = False):
    if if_delete_first_aa == True:
        assert False

    if seq != seq:
        print("find a peptide with mass 0 -------------- Na pep")
        return 0
    # Calculate the total mass
    total_mass = 0
    i = 0
    while i < len(seq):
        if i == 0 :
            if seq[i] == 'B' and seq[i:i+9] == 'B(+42.01)':
                return 0
                # first_aa = seq[i+9]
                # total_mass += aa_to_mass_dict[first_aa] + modifications_other_mass[0]
                # i += 10
                # continue

        if seq[i] == 'C' and seq[i:i+9] == 'C(+57.02)':
            total_mass += aa_to_mass_dict['C(+57.02)']
            i += 9
        elif seq[i] == 'M' and seq[i:i+9] == 'M(+15.99)':
            total_mass += aa_to_mass_dict['M(+15.99)']
            i += 9
        else:
            try:
                total_mass += aa_to_mass_dict[seq[i]]
                i += 1
            except:
                print(seq)
                print(seq[i])
                assert False
    if if_delete_first_aa:
        if seq[0] == "K":
            total_mass -= aa_to_mass_dict["K"]
        if seq[0] == "R":
            total_mass -= aa_to_mass_dict["R"]

    return total_mass

def get_mirrorSeq_mass(Rseq, match_Type):
    try_seq, _ = generate_try_lys_seq_from_RseqAndMatchtype(Rseq, match_Type)
    return calculate_mass(try_seq, if_delete_first_aa=False)


def generate_A_B_format_of_info(row, enzyme):
    match_type = row["match_type"]
    site_tag = row["site_tag_denovo"]
    site_peaknum = row["site_peaknum_denovo"]
    site_score = row["site_score_denovo"]
    if match_type == "A1:K-K" or match_type == "A2:R-R" or match_type == "B: R-K" or match_type == "C: K-R":
        if enzyme == "A":
            try_site_tag = ",".join(site_tag.split(",")[1:])
            try_site_peaknum = ",".join(site_peaknum.split(",")[1:])
            try_site_score = ",".join(site_score.split(",")[1:])
        if enzyme == "B":
            lys_site_tag = ",".join(site_tag.split(",")[:-1])
            lys_site_peaknum = ",".join(site_peaknum.split(",")[:-1])
            lys_site_score = ",".join(site_score.split(",")[:-1])
    elif match_type == "F: X-K" or match_type == "G: X-R":
        if enzyme == "A":
            try_site_tag = ",".join(site_tag.split(",")[1:])
            try_site_peaknum = ",".join(site_peaknum.split(",")[1:])
            try_site_score = ",".join(site_score.split(",")[1:])
        if enzyme == "B":
            lys_site_tag = site_tag
            lys_site_peaknum = site_peaknum
            lys_site_score = site_score
    elif match_type == "D: K-X" or match_type == "E: R-X":
        if enzyme == "A":
            try_site_tag = site_tag
            try_site_peaknum = site_peaknum
            try_site_score = site_score
        if enzyme == "B":
            lys_site_tag = ",".join(site_tag.split(",")[:-1])
            lys_site_peaknum = ",".join(site_peaknum.split(",")[:-1])
            lys_site_score = ",".join(site_score.split(",")[:-1])
    else:
        assert False

    if enzyme == "A":
        return try_site_tag, try_site_peaknum, try_site_score
    elif enzyme == "B":
        return lys_site_tag, lys_site_peaknum, lys_site_score
    else:
        assert False


def generate_Best_title_to_denovo_df(A_titles, A_df, A_title_to_BestRow, enzyme):
    '''
    首先选择单谱测序结果，以单谱测序结果为主
    :param df:
    :param title_to_BestRow:
    :return:
    '''
    # 创建一个和A_df相同列名的df
    df = pd.DataFrame(columns=A_df.columns)
    df["title"] = list(A_titles)
    df["source"] = "Single"
    A_count1 = 0
    A_count2 = 0
    A_count21 = 0
    A_count22 = 0
    A_count3 = 0
    A_count31 = 0
    A_count32 = 0
    A_df_title_dict = {}
    for index, row in A_df.iterrows():
        A_df_title_dict[row["title"]] = row
    for index, row in tqdm(df.iterrows()):
        if row["title"] in A_title_to_BestRow:
            best_row = A_title_to_BestRow[row["title"]]
            # 有镜像谱结果
            if row["title"] not in A_df_title_dict:
                # 无单谱
                df.loc[index, ["denovoSeq", "score_software", "pep_scoremean_denovo", "pep_scoresum_denovo",
                               "ifEvident_denovo", "ifFind_denovo", "source", "match_type_class", "source_title", "ifTCLN", "ifMirror"]] = [
                    best_row[f"{enzyme}_denovoSeq"], best_row["score_software"], best_row["pep_scoremean_denovo"],
                    best_row["pep_scoresum_denovo"], best_row["ifEvident_denovo"], best_row["ifFind_denovo"], "Mirror",
                    best_row["match_type_class"], best_row["title"], False, True
                ]
                df.loc[index, ["site_tag_denovo", "site_peaknum_denovo",
                               "site_score_denovo"]] = generate_A_B_format_of_info(best_row, enzyme)
                A_count1 += 1
            else:
                # 有镜像普有单谱
                SingleRow = A_df_title_dict[row["title"]]
                A_count2 += 1
                if SingleRow["denovoSeq"] == best_row[f"{enzyme}_denovoSeq"]:
                    df.loc[index, ["denovoSeq", "score_software", "pep_scoremean_denovo", "pep_scoresum_denovo",
                                   "ifEvident_denovo", "ifFind_denovo", "source", "match_type_class", "source_title", "ifTCLN", "ifMirror"]] = [
                        best_row[f"{enzyme}_denovoSeq"], best_row["score_software"], best_row["pep_scoremean_denovo"],
                        best_row["pep_scoresum_denovo"], best_row["ifEvident_denovo"], best_row["ifFind_denovo"],
                        "Mirror", best_row["match_type_class"], best_row["title"], True, True
                    ]
                    df.loc[index, ["site_tag_denovo", "site_peaknum_denovo",
                                   "site_score_denovo"]] = generate_A_B_format_of_info(best_row, enzyme)
                    A_count21 += 1
                else:
                    df.loc[index, ["denovoSeq", "score_software", "site_tag_denovo", "site_peaknum_denovo",
                                   "site_score_denovo", "pep_scoremean_denovo", "pep_scoresum_denovo",
                                   "ifEvident_denovo", "ifFind_denovo", "source", "match_type_class", "source_title", "ifTCLN", "ifMirror"]] = [
                        SingleRow["denovoSeq"], SingleRow["score_software"], SingleRow["site_tag_denovo"],
                        SingleRow["site_peaknum_denovo"], SingleRow["site_score_denovo"],
                        SingleRow["pep_scoremean_denovo"], SingleRow["pep_scoresum_denovo"],
                        SingleRow["ifEvident_denovo"], SingleRow["ifFind_denovo"], "Single",99, SingleRow["title"], True, False
                    ]
                    A_count22 += 1
        else:
            # 没有镜像谱结果
            A_count3 += 1
            if row["title"] in A_df_title_dict:
                # 有单谱无镜像普,采用单谱的结果
                SingleRow = A_df_title_dict[row["title"]]
                df.loc[index, ["denovoSeq", "score_software", "site_tag_denovo", "site_peaknum_denovo",
                               "site_score_denovo", "pep_scoremean_denovo", "pep_scoresum_denovo", "ifEvident_denovo",
                               "ifFind_denovo", "source", "match_type_class", "source_title", "ifTCLN", "ifMirror"]] = [
                    SingleRow["denovoSeq"], SingleRow["score_software"], SingleRow["site_tag_denovo"],
                    SingleRow["site_peaknum_denovo"], SingleRow["site_score_denovo"], SingleRow["pep_scoremean_denovo"],
                    SingleRow["pep_scoresum_denovo"], SingleRow["ifEvident_denovo"], SingleRow["ifFind_denovo"],
                    "Single", 99, SingleRow["title"], True, False
                ]
                A_count31 += 1
            else:
                # 无镜像谱也无单谱
                A_count32 += 1

    return (df, A_count1, A_count2, A_count21, A_count22, A_count3, A_count31, A_count32)


def calculate_mass_mirror(row, titles_to_matchtype_dict, ifpNovoM=False):
    if ifpNovoM:
        RSeq = row["denovoSeq_pNovoMTryLys"]
        titles = ast.literal_eval(row["titles_pNovoMTryLys"])
    else:
        RSeq = row["denovoSeq_DiNovoTryLys"]
        titles = ast.literal_eval(row["titles_DiNovoTryLys"])
    max_mass = 0.0
    for title in titles:
        match_type = titles_to_matchtype_dict[title[0]][1]
        mass = get_mirrorSeq_mass(RSeq, match_type)
        if mass > max_mass:
            max_mass = mass
    return max_mass

def calculate_mass_mirror_from_specdf(row, titles_to_matchtype_dict):
    RSeq = row["denovoSeq"]
    title = row["title"]
    match_type = titles_to_matchtype_dict[title][1]
    mass = get_mirrorSeq_mass(RSeq, match_type)
    return mass

def parse_spectrum_ion(f):  # 删掉超过maxMZ的部分
    mz_list = []
    intensity_list = []
    line = f.readline()
    if line[:5] == "SCANS":
        line = f.readline()
    while not "END IONS" in line:
        mz, intensity = re.split(' |\r|\n', line)[:2]
        mz_float = float(mz)
        intensity_float = float(intensity)
        mz_list.append(mz_float)
        intensity_list.append(intensity_float)
        line = f.readline()
    return mz_list, intensity_list

def read_one_spec(title, spectype):
    if type(spectype) == list:
        spectrum_location_dict = spectype[0]
        suffix_name = spectype[1]
        mgf_path = spectype[2]
    else:
        assert False

    sub_title = title.split('.')[0]
    filename = sub_title + '_' + suffix_name
    file_path = os.path.join(mgf_path, filename)
    with open(file_path, 'r') as fi:
        if title not in spectrum_location_dict[filename + '_location'].keys():
            print(title,filename)
            assert False
        location = spectrum_location_dict[filename + '_location'][title]
        fi.seek(location)
        line = fi.readline()
        assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
        line = fi.readline()
        assert "TITLE=" in line, "Error: wrong input TITLE="
        assert title in line
        line = fi.readline()
        assert "CHARGE=" in line, "Error: wrong input CHARGE="
        charge_int = int(re.split('[=+]', line)[1])
        line = fi.readline()
        assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
        line = fi.readline()
        assert "PEPMASS=" in line, "Error: wrong input PEPMASS="
        mz_list, intensity_list = parse_spectrum_ion(fi)

    return mz_list, intensity_list, charge_int

def local_intensity_mask(mz,local_sliding_window):
    right_boundary = np.reshape(mz + local_sliding_window, (-1, 1))
    left_boundary = np.reshape(mz - local_sliding_window, (-1, 1))
    mask = np.logical_and(right_boundary > mz, left_boundary < mz)
    return mask

def local_reletive_intensityCal(local_intensity_array, this_intensity):
    return this_intensity / local_intensity_array.max()

def ion_cover_rate_function(try_sequence, try_mz_list, try_intensity_list, try_charge, fragment_tolerance_ppm):
    try_length = len(try_sequence)
    intensity_array = np.array(try_intensity_list)
    mz_array = np.array(try_mz_list)
    mask = local_intensity_mask(mz_array, 50)
    # try_rank_list = list(np.argsort(np.argsort(try_intensity_list))[::-1] + 1) #+1是为了避免rank为0

    ###################################################################################################
    # trypsin
    try:
        try_peptide_residue_mass = [aa_to_mass_dict[i] for i in try_sequence]
    except:
        print(try_sequence)
        assert False
    try_peptide_residue_cumsummass = np.cumsum(try_peptide_residue_mass)
    # if try_sequence == ['E', 'F', 'D', 'L', 'L', 'K']:
    #     print(1,try_peptide_residue_cumsummass)

    # b+
    try_theoretical_single_charge_b_ion = try_peptide_residue_cumsummass + atom_mass['PROTON']
    try_theoretical_single_charge_b_ion_tag = np.zeros_like(try_theoretical_single_charge_b_ion)
    try_theoretical_single_charge_b_ion_intensitytemp = np.zeros_like(try_theoretical_single_charge_b_ion) #intensity
    try_theoretical_single_charge_b_ion_intensity = np.zeros_like(try_theoretical_single_charge_b_ion) #实际是score
    for i, theoretical_mass in enumerate(try_theoretical_single_charge_b_ion):
        delta_mass = (np.array(try_mz_list) - theoretical_mass) / theoretical_mass * 1000000
        min_index = np.argmin(np.abs(delta_mass))
        min_delta_mass = delta_mass[min_index]
        if min_delta_mass >= -fragment_tolerance_ppm and min_delta_mass <= fragment_tolerance_ppm:
            try_theoretical_single_charge_b_ion_tag[i] = 1
            this_mask = mask[min_index]
            local_intensity = intensity_array[this_mask]
            this_intensity = intensity_array[min_index]
            try_theoretical_single_charge_b_ion_intensitytemp[i] = try_intensity_list[min_index]
            try_theoretical_single_charge_b_ion_intensity[i] = local_reletive_intensityCal(local_intensity,
                                                                                           this_intensity)
    # if try_sequence == ['E', 'F', 'D', 'L', 'L', 'K']:
    #     print(2,try_theoretical_single_charge_b_ion_intensitytemp)

    # b2+
    if try_charge > 2:
        try_theoretical_two_charge_b_ion = (try_peptide_residue_cumsummass + 2 * atom_mass['PROTON']) / 2.0
        try_theoretical_two_charge_b_ion_tag = np.zeros_like(try_theoretical_two_charge_b_ion)
        try_theoretical_two_charge_b_ion_intensitytemp = np.zeros_like(try_theoretical_two_charge_b_ion)
        try_theoretical_two_charge_b_ion_intensity = np.zeros_like(try_theoretical_two_charge_b_ion)
        for i, theoretical_mass in enumerate(try_theoretical_two_charge_b_ion):
            delta_mass = (np.array(try_mz_list) - theoretical_mass) / theoretical_mass * 1000000
            min_index = np.argmin(np.abs(delta_mass))
            min_delta_mass = delta_mass[min_index]
            if min_delta_mass >= -fragment_tolerance_ppm and min_delta_mass <= fragment_tolerance_ppm:
                try_theoretical_two_charge_b_ion_tag[i] = 1
                this_mask = mask[min_index]
                local_intensity = intensity_array[this_mask]
                this_intensity = intensity_array[min_index]
                try_theoretical_two_charge_b_ion_intensitytemp[i] = try_intensity_list[min_index]
                try_theoretical_two_charge_b_ion_intensity[i] = local_reletive_intensityCal(local_intensity,this_intensity)
        # if try_sequence == ['E', 'F', 'D', 'L', 'L', 'K']:
        #     print(3,try_theoretical_two_charge_b_ion_intensitytemp)

    try_peptide_residue_mass_reverse = try_peptide_residue_mass[::-1]
    try_peptide_residue_cumsummass_reverse = np.cumsum(try_peptide_residue_mass_reverse)

    # y+
    try_theoretical_single_charge_y_ion = try_peptide_residue_cumsummass_reverse + atom_mass['H2O'] + atom_mass[
        'PROTON']
    try_theoretical_single_charge_y_ion_tag = np.zeros_like(try_theoretical_single_charge_y_ion)
    try_theoretical_single_charge_y_ion_intensitytemp = np.zeros_like(try_theoretical_single_charge_y_ion)
    try_theoretical_single_charge_y_ion_intensity = np.zeros_like(try_theoretical_single_charge_y_ion)
    for i, theoretical_mass in enumerate(try_theoretical_single_charge_y_ion):
        delta_mass = (np.array(try_mz_list) - theoretical_mass) / theoretical_mass * 1000000
        min_index = np.argmin(np.abs(delta_mass))
        min_delta_mass = delta_mass[min_index]
        if min_delta_mass >= -fragment_tolerance_ppm and min_delta_mass <= fragment_tolerance_ppm:
            try_theoretical_single_charge_y_ion_tag[i] = 1
            this_mask = mask[min_index]
            local_intensity = intensity_array[this_mask]
            this_intensity = intensity_array[min_index]
            try_theoretical_single_charge_y_ion_intensitytemp[i] = try_intensity_list[min_index]
            try_theoretical_single_charge_y_ion_intensity[i] = local_reletive_intensityCal(local_intensity, this_intensity)
    # if try_sequence == ['E', 'F', 'D', 'L', 'L', 'K']:
    #     print(4,try_theoretical_single_charge_y_ion_intensitytemp)

    # y2+
    if try_charge > 2:
        try_theoretical_two_charge_y_ion = (try_peptide_residue_cumsummass_reverse + atom_mass['H2O'] + 2 * atom_mass[
            'PROTON']) / 2.0
        try_theoretical_two_charge_y_ion_tag = np.zeros_like(try_theoretical_two_charge_y_ion)
        try_theoretical_two_charge_y_ion_intensitytemp = np.zeros_like(try_theoretical_two_charge_y_ion)
        try_theoretical_two_charge_y_ion_intensity = np.zeros_like(try_theoretical_two_charge_y_ion)
        for i, theoretical_mass in enumerate(try_theoretical_two_charge_y_ion):
            delta_mass = (np.array(try_mz_list) - theoretical_mass) / theoretical_mass * 1000000
            min_index = np.argmin(np.abs(delta_mass))
            min_delta_mass = delta_mass[min_index]
            if min_delta_mass >= -fragment_tolerance_ppm and min_delta_mass <= fragment_tolerance_ppm:
                try_theoretical_two_charge_y_ion_tag[i] = 1
                this_mask = mask[min_index]
                local_intensity = intensity_array[this_mask]
                this_intensity = intensity_array[min_index]
                try_theoretical_two_charge_y_ion_intensitytemp[i] = try_intensity_list[min_index]
                try_theoretical_two_charge_y_ion_intensity[i] = local_reletive_intensityCal(local_intensity, this_intensity)
        # if try_sequence == ['E', 'F', 'D', 'L', 'L', 'K']:
        #     print(5, try_theoretical_two_charge_y_ion_intensitytemp)
    ##########################################################################################
    # return results

    try_theoretical_single_charge_b_ion_tag = try_theoretical_single_charge_b_ion_tag[:-1]
    try_theoretical_single_charge_b_ion_intensitytemp = try_theoretical_single_charge_b_ion_intensitytemp[:-1]
    try_theoretical_single_charge_b_ion_intensity = try_theoretical_single_charge_b_ion_intensity[:-1]
    try_theoretical_single_charge_y_ion_tag = try_theoretical_single_charge_y_ion_tag[:-1][::-1]
    try_theoretical_single_charge_y_ion_intensitytemp = try_theoretical_single_charge_y_ion_intensitytemp[:-1][::-1]
    try_theoretical_single_charge_y_ion_intensity = try_theoretical_single_charge_y_ion_intensity[:-1][::-1]
    if try_charge > 2:
        try_theoretical_two_charge_b_ion_tag = try_theoretical_two_charge_b_ion_tag[:-1]
        try_theoretical_two_charge_b_ion_intensitytemp = try_theoretical_two_charge_b_ion_intensitytemp[:-1]
        try_theoretical_two_charge_b_ion_intensity = try_theoretical_two_charge_b_ion_intensity[:-1]
        try_theoretical_two_charge_y_ion_tag = try_theoretical_two_charge_y_ion_tag[:-1][::-1]
        try_theoretical_two_charge_y_ion_intensitytemp = try_theoretical_two_charge_y_ion_intensitytemp[:-1][::-1]
        try_theoretical_two_charge_y_ion_intensity = try_theoretical_two_charge_y_ion_intensity[:-1][::-1]

    iontype = {
        "b+": True,
        "b2+": False,
        "y+": True,
        "y2+": False
    }
    if try_charge > 2:
        iontype["b2+"] = True
        iontype["y2+"] = True

    for i, ion in enumerate(iontype):
        if i == 0:
            if iontype[ion] == True and ion == "b+":
                try_tag_array = try_theoretical_single_charge_b_ion_tag
                try_intensitytemp_array = try_theoretical_single_charge_b_ion_intensitytemp
                try_intensity_array = try_theoretical_single_charge_b_ion_intensity
            else:
                assert False
        else:
            if ion == 'b2+' and iontype['b2+'] == True:
                try_tag_array = np.row_stack((try_tag_array, try_theoretical_two_charge_b_ion_tag))
                try_intensitytemp_array = np.row_stack((try_intensitytemp_array, try_theoretical_two_charge_b_ion_intensitytemp))
                try_intensity_array = np.row_stack((try_intensity_array, try_theoretical_two_charge_b_ion_intensity))
                continue
            if ion == 'y+' and iontype['y+'] == True:
                try_tag_array = np.row_stack((try_tag_array, try_theoretical_single_charge_y_ion_tag))
                try_intensitytemp_array = np.row_stack((try_intensitytemp_array, try_theoretical_single_charge_y_ion_intensitytemp))
                try_intensity_array = np.row_stack((try_intensity_array, try_theoretical_single_charge_y_ion_intensity))
                continue
            if ion == 'y2+' and iontype['y2+'] == True:
                try_tag_array = np.row_stack((try_tag_array, try_theoretical_two_charge_y_ion_tag))
                try_intensitytemp_array = np.row_stack((try_intensitytemp_array, try_theoretical_two_charge_y_ion_intensitytemp))
                try_intensity_array = np.row_stack((try_intensity_array, try_theoretical_two_charge_y_ion_intensity))
                continue

    ion_num = 0
    for i, ion in enumerate(iontype):
        if iontype[ion] == True:
            ion_num += 1
    assert try_tag_array.shape == (ion_num, try_length - 1), 'shape error!'
    assert try_intensitytemp_array.shape == (ion_num, try_length - 1), 'shape error!'
    assert try_intensity_array.shape == (ion_num, try_length - 1), 'shape error!'

    # try_site_tag = np.zeros(try_length - 1)
    # for i in range(try_length - 1):
    #     if 1 in try_tag_array[:, i]:
    #         try_site_tag[i] = 1
    try_site_tag = np.sum(try_tag_array, axis=0)

    try_site_intensitytemp = np.sum(try_intensitytemp_array, axis=0)

    # try_site_intensity = np.zeros(try_length - 1)
    # for i in range(try_length - 1):
    #     try_site_intensity[i] = np.sum(try_intensity_array[:, i])
    try_site_intensity = np.sum(try_intensity_array, axis=0)

    try_site_tag = np.array(try_site_tag, dtype=int)
    try_site_score = np.round(np.array(try_site_intensity, dtype=float), 2)
    try_pep_scoremean = np.round(np.mean(try_site_score), 2)
    try_pep_scoresum = np.round(np.sum(try_site_score), 2)

    return try_site_tag, try_site_intensity, try_pep_scoremean, try_pep_scoresum, try_site_intensitytemp


def build_fasta_dict_function(fasta_file):
    fasta_seq = {}
    with open(fasta_file, 'r') as f:
        line = f.readline()
        while True:
            if len(line) == 0:
                break
            if line.startswith(">"):
                key = re.split(r'[\t ]+', line[1:])[0].strip()
                # key = line[1:].strip()
                fasta_seq[key] = ''
            else:
                line = line.replace("I", "L")
                fasta_seq[key] += line.strip()
            line = f.readline()

    return fasta_seq

index_num = 4
def build_index_dict_function(fasta_seq_dict):
    Index_first_3aa = {}
    for key in fasta_seq_dict:
        value = fasta_seq_dict[key]
        for i in range(len(value) - (index_num-1)):
            three_aa = value[i:i + index_num]
            if three_aa not in Index_first_3aa:
                Index_first_3aa[three_aa] = []
            Index_first_3aa[three_aa].append((key, i))
    return Index_first_3aa

def process_antibody(fasta_seq):
    decoy_count = 0
    target_count = 0
    new_fasta_seq_TD = {}
    new_fasta_seq_T = {}
    new_fasta_seq_D = {}
    for key in fasta_seq:
        if key[:3] != "PXL":
            new_fasta_seq_TD["decoy_" + key] = fasta_seq[key]
            new_fasta_seq_D["decoy_" + key] = fasta_seq[key]
            decoy_count += len(fasta_seq[key])
        else:
            new_fasta_seq_TD[key] = fasta_seq[key]
            new_fasta_seq_T[key] = fasta_seq[key]
            target_count += len(fasta_seq[key])
    return decoy_count, target_count, new_fasta_seq_TD, new_fasta_seq_T, new_fasta_seq_D


def generate_reverse_fasta(fasta_seq):
    fasta_seq_reverse = {}
    for key in fasta_seq:
        fasta_seq_reverse["decoy_" + key] = fasta_seq[key][::-1]

    return fasta_seq_reverse

def merge_target_and_decoy_fasta_seq(target_fasta_seq, decoy_fasta_seq):
    merged_fasta_seq = {}
    for key in target_fasta_seq:
        merged_fasta_seq[key] = target_fasta_seq[key]
    for key in decoy_fasta_seq:
        merged_fasta_seq[key] = decoy_fasta_seq[key]

    return merged_fasta_seq

def build_mgf_location_function(mgf_folder, ifgraphnovo=False):
    '''
    如果是graphnovo的话，需要的mgf是有scan号的
    '''

    # 读取mgf文件，创建location字典
    path_list = os.listdir(mgf_folder)
    path_list.sort()  # 对读取的路径进行排序
    flag = False
    for filename in path_list:
        if not ifgraphnovo:
            if filename.endswith('.mgf'):
                suffix_name = filename.split('_')[-1]
        else:
            if filename.endswith('.mgf'):
                suffix_name = "_".join(filename.split('_')[-2:])
        if filename.endswith('_location'):
            flag = True
            break

    if flag == False:
        spectrum_location_dict = {}
        filepathAndOrderNum_to_title = {}
        if ifgraphnovo:
            filepathAndScan_to_title = {}
        else:
            filepathAndScan_to_title = None

        for filename in path_list:  # 挨个读取mgf文件

            if not filename.endswith(".mgf"):
                continue
            print(filename)

            mgf_file = os.path.join(mgf_folder + filename)
            prefixname = filename.rstrip(f"{peaks_suffix_name}")
            if os.path.isdir(mgf_file):
                continue

            spectrum_location_dict_Temp = {}
            line = True
            count = 0
            orderNum = 0
            with open(mgf_file, 'r') as f:
                while line:
                    current_location = f.tell()
                    line = f.readline()
                    count += 1
                    if len(line) == 0:
                        print('ends with empty line :', count)
                        break
                    if line[0] == 'B' and line[:5] == "BEGIN":
                        spectrum_location = current_location
                        orderNum += 1
                    elif line[0] == 'T' and line[:5] == "TITLE":
                        title = re.split('[=\n]', line)[1]
                        spectrum_location_dict_Temp[title] = spectrum_location
                        filepathAndOrderNum_to_title[(prefixname, orderNum)] = title
                    elif line[0] == 'S' and line[:5] == "SCANS":
                        if ifgraphnovo:
                            scan_num = int(re.split('[=\n]', line)[1])
                            filepathAndScan_to_title[(prefixname, scan_num)] = title

            spectrum_location_dict[filename + '_location'] = spectrum_location_dict_Temp
            spectrum_location_file = mgf_file + "_location"
            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(spectrum_location_dict_Temp, fw)

        with open(mgf_folder + 'filepathAndOrderNum_to_title', 'wb') as fw:
            pickle.dump(filepathAndOrderNum_to_title, fw)

        if ifgraphnovo:
            with open(mgf_folder + 'filepathAndScan_to_title', 'wb') as fw:
                pickle.dump(filepathAndScan_to_title, fw)

    else:
        spectrum_location_dict = {}
        for filename in path_list:
            if filename.endswith('_location'):
                spectrum_location_file = os.path.join(mgf_folder + filename)
                with open(spectrum_location_file, 'rb') as fr:
                    spectrum_location_dict_Temp = pickle.load(fr)
                spectrum_location_dict[filename] = spectrum_location_dict_Temp

        filepathAndOrderNum_to_title_file = mgf_folder + 'filepathAndOrderNum_to_title'
        with open(filepathAndOrderNum_to_title_file, 'rb') as fr:
            filepathAndOrderNum_to_title = pickle.load(fr)

        if ifgraphnovo:
            filepathAndScan_to_title_file = mgf_folder + 'filepathAndScan_to_title'
            with open(filepathAndScan_to_title_file, 'rb') as fr:
                filepathAndScan_to_title = pickle.load(fr)
        else:
            filepathAndScan_to_title = None

    return spectrum_location_dict, suffix_name, filepathAndOrderNum_to_title, filepathAndScan_to_title

def transfer_str_to_list_seq(seq_str):
    if "(" not in seq_str:
        return list(seq_str)
    else:
        #如果存在(，则把(前一个字符直到)的内容当做一个元素
        seq_list = []
        i = 0
        while i < len(seq_str):
            if seq_str[i] == "(":
                j = i
                while seq_str[j] != ")":
                    j += 1
                seq_list[-1] = seq_str[i-1:j+1]
                i = j+1
            else:
                seq_list.append(seq_str[i])
                i += 1
    return seq_list

def get_confident_info(args):
    title, seq_list, spec_spectrum_location_dict, suffix_name, mgf_path = args
    if seq_list == [''] or seq_list == None:
        return {
            "PSM": title + "@",
            "site_tag": "",
            "site_peaknum": "",
            "site_score": "",
            "pepscore_mean": "",
            "pepscore_sum": ""
        }

    mz_list, intensity_list, try_charge = read_one_spec(title, [spec_spectrum_location_dict, suffix_name, mgf_path])
    site_all_tag, site_score, pepscore_mean, pepscore_sum, site_intensitytemp = ion_cover_rate_function( seq_list, mz_list, intensity_list, try_charge,20)
    site_tag = np.array([1 if peaknum != 0 else 0 for peaknum in site_all_tag])

    return {
        "PSM": title + "@" + "".join(seq_list),
        "site_tag": site_tag,
        "site_peaknum": site_all_tag,
        "site_score": site_score,
        "pepscore_mean": pepscore_mean,
        "pepscore_sum": pepscore_sum,
        "intensity_list": intensity_list,
        "site_intensitytemp": site_intensitytemp
    }

def parallel_get_confident_info(psm_list, mgf_location_dict, suffix_name, mgf_path):
    print("Start to calculate coverage info")
    with multiprocessing.Pool(12) as p:
        res_list = list(tqdm(p.imap(get_confident_info, [(psm[0], psm[1], mgf_location_dict, suffix_name, mgf_path) for psm in psm_list], chunksize=100000), total=len(psm_list)))
    assert len(res_list) == len(psm_list)

    res_dict = {}
    for res_item in res_list:
        res_dict[res_item["PSM"]] = res_item
    return res_dict

def if_evident_function(max_match_coverage):
    if max_match_coverage == 1:
        return True
    else:
        return False

def mapped_function_single(Seq_str_set, fasta_seq_dict, first_index_4aa):
    processing_workers = 24
    with multiprocessing.Pool(processing_workers) as p:
        res_dict_list = list(tqdm(
            p.imap(find_location_in_fasta_single, [(seq, fasta_seq_dict, first_index_4aa) for seq in Seq_str_set],
                   chunksize=1000000), total=len(Seq_str_set)))
        res_dict = {seq_dict["seq"]: [seq_dict["pure_seq"], seq_dict["target_find"],seq_dict["decoy_find"], seq_dict["target_location"], seq_dict["decoy_location"]] for seq_dict in
                    res_dict_list}  # seq: [find, location, if_unique]
    return res_dict

def generate_db_title_to_seq_dict(df,software="pFind"):
    if software == "pFind":
        db_title_to_seq_dict = {}
        for index,row in df.iterrows():
            title = row["title"]
            seq = row["dbSeq"]
            db_title_to_seq_dict[title] = seq
        return db_title_to_seq_dict
    elif software == "MSFragger" or software == "MSGF":
        db_title_to_seq_dict = {}
        for index,row in df.iterrows():
            title = row["ScanIndex"]
            seq = row["dbSeq"]
            if title not in db_title_to_seq_dict:
                db_title_to_seq_dict[title] = seq
            else:
                assert False
                db_title_to_seq_dict[title].add(seq)
        return db_title_to_seq_dict

def generate_dinovoBest_title_to_seq_dict(df,software="pFind"):
    if software == "pFind":
        dinovoBest_title_to_seq_dict = {}
        for inedx,row in df.iterrows():
            title = row["title"]
            seq = row["denovoSeq"]
            dinovoBest_title_to_seq_dict[title] = seq
        return dinovoBest_title_to_seq_dict
    elif software == "MSFragger" or software == "MSGF":
        dinovoBest_title_to_seq_dict = {}
        for inedx,row in df.iterrows():
            title = row["ScanIndex"]
            seq = row["denovoSeq"]
            if row["ifFind_denovo"]:
                if title not in dinovoBest_title_to_seq_dict:
                    dinovoBest_title_to_seq_dict[title] = {seq}
                else:
                    dinovoBest_title_to_seq_dict[title].add(seq)
        return dinovoBest_title_to_seq_dict

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{val} ({pct:.1f}%)'
    return my_format

def analyse_intsection_of_db_and_denovo(title_to_seq_dict1, title_to_seq_dict2, spec_titles_set, fig_path, software_tool = "pFind"):
    if software_tool == "pFind":
        count_same = 0
        count_diff = 0
        #画饼图
        for spec_title in spec_titles_set:
            try:
                seq1 = title_to_seq_dict1[spec_title]
                seq1 = seq1.replace("I", "L").replace("C(+57.02)", "C").replace("M(+15.99)", "M")
                seq2 = title_to_seq_dict2[spec_title]
                seq2 = seq2.replace("I", "L").replace("C(+57.02)", "C").replace("M(+15.99)", "M")
                if seq1 == seq2:
                    count_same += 1
                else:
                    count_diff += 1
            except:
                print(spec_title)
                assert False
        plt.figure(figsize=(6, 6))
        labels = ['Same', 'Different']
        sizes = [count_same, count_diff]
        explode = (0, 0.1)
        plt.pie(sizes, explode=explode, labels=labels, autopct=autopct_format(sizes), shadow=False, startangle=90)
        plt.title("Pie Graph of Same and Diff")
        plt.savefig(fig_path, dpi=300)
        plt.show()

    elif software_tool == "MSFragger":
        count_same = 0
        count_diff = 0
        #画饼图
        for spec_title in spec_titles_set:
            try:
                seq1 = title_to_seq_dict1[spec_title]
                seq2_set = title_to_seq_dict2[spec_title]
                if seq1 in seq2_set:
                    count_same += 1
                else:
                    count_diff += 1
                    print(spec_title)
            except:
                print(spec_title)
                assert False
        plt.figure(figsize=(6, 6))
        labels = ['Same', 'Different']
        sizes = [count_same, count_diff]
        explode = (0, 0.1)
        plt.pie(sizes, explode=explode, labels=labels, autopct=autopct_format(sizes), shadow=False, startangle=90)
        plt.title("Pie Graph of Same and Diff")
        plt.savefig(fig_path, dpi=300)
        plt.show()

def find_location_in_fasta_single(args):
    denovo_seq, fasta_seq_dict, index_first_3aa = args

    if denovo_seq != denovo_seq or denovo_seq is None or denovo_seq == "" or denovo_seq == "None":
        return {"seq": denovo_seq, "pure_seq": denovo_seq, "target_find": False, "decoy_find": False, "target_location": "", "decoy_location": "", "TD_tag": "NoMatch"}
    pure_seq = denovo_seq.replace("B(+42.01)", "")
    pure_seq = pure_seq.replace("I", "L")
    pure_seq = pure_seq.replace("C(+57.02)", "C")
    pure_seq = pure_seq.replace("M(+15.99)", "M")

    # if pure_seq == "KLNLL":
    #     print(denovo_seq,pure_seq)

    if_find = False
    target_seq_key_to_location = []
    decoy_seq_key_to_location= []
    TD_tag = {"target":False, "decoy":False}
    if len(pure_seq) <= index_num:
        for seq_3aa in index_first_3aa.keys():
            if seq_3aa[:len(pure_seq)] == pure_seq:
                for key, i in index_first_3aa[seq_3aa]:
                    if fasta_seq_dict[key][i:i + len(pure_seq)] == pure_seq:
                        if_find = True
                        if key.startswith("decoy"):
                            TD_tag["decoy"] = True
                            decoy_seq_key_to_location.append((key, i, i + len(pure_seq)))
                        else:
                            TD_tag["target"] = True
                            target_seq_key_to_location.append((key, i, i + len(pure_seq)))
    else:
        if pure_seq[:index_num] in index_first_3aa:
            for key, i in index_first_3aa[pure_seq[:index_num]]:
                if fasta_seq_dict[key][i:i + len(pure_seq)] == pure_seq:
                    if_find = True
                    if key.startswith("decoy"):
                        TD_tag["decoy"] = True
                        decoy_seq_key_to_location.append((key, i, i + len(pure_seq)))
                    else:
                        TD_tag["target"] = True
                        target_seq_key_to_location.append((key, i, i + len(pure_seq)))
                        
    if not if_find:
        assert TD_tag["target"] == False and TD_tag["decoy"] == False
        return {"seq": denovo_seq, "pure_seq": pure_seq, "target_find": False, "decoy_find": False, "target_location": "", "decoy_location": "", "TD_tag": "NoMatch"}

    target_seq_key_to_location_str = ""
    for item in target_seq_key_to_location:
        target_seq_key_to_location_str += item[0] + "$" + str(item[1]) + '$' + str(item[2]) + "&"
    target_seq_key_to_location_str = target_seq_key_to_location_str[:-1]

    decoy_seq_key_to_location_str = ""
    for item in decoy_seq_key_to_location:
        decoy_seq_key_to_location_str += item[0] + "$" + str(item[1]) + '$' + str(item[2]) + "&"
    decoy_seq_key_to_location_str = decoy_seq_key_to_location_str[:-1]

    # if key.startswith("tr|A0A140N4R8") and pure_seq == "KLNLL":
    #     print(seq_key_to_location_str)

    TD_tag_temp = ""
    if TD_tag["target"] == True and TD_tag["decoy"] == True:
        TD_tag_temp = "TD"
        target_find_temp = True
        decoy_find_temp = True
    elif TD_tag["target"] == True and TD_tag["decoy"] == False:
        TD_tag_temp = "T"
        target_find_temp = True
        decoy_find_temp = False
    elif TD_tag["target"] == False and TD_tag["decoy"] == True:
        TD_tag_temp = "D"
        target_find_temp = False
        decoy_find_temp = True
    else:
        assert False
        TD_tag_temp = "NoMatch"
        target_find_temp = False
        decoy_find_temp = False
    return {"seq": denovo_seq, "pure_seq": pure_seq, "target_find": target_find_temp,"decoy_find": decoy_find_temp,  "target_location": target_seq_key_to_location_str,"decoy_location": decoy_seq_key_to_location_str, "TD_tag": TD_tag_temp}

def get_fasta_coverage_np_single(fasta_seq_dict, df, suffix, Evident_select = False, mass_threshold = 0.0):

    #seq_to_if_unique_dict
    seq_to_if_unique_dict = {}
    proteinName_to_uniqueSeq = {}

    # init
    coverage_fasta_dict = {}
    for key in fasta_seq_dict:
        seq = fasta_seq_dict[key]
        length = len(seq)
        coverage_fasta_dict[key] = [np.zeros(length, dtype=int), np.zeros(length, dtype=int), np.zeros(length, dtype=float), 0, 0, 0]  # site_tag, site_score, unique peptide num, non-unique peptide num,peptide num,

    for index, row in df.iterrows():

            seq = row[f"{suffix}Seq"]
            if seq != seq or seq == "":
                seq_to_if_unique_dict[seq] = False
                continue

            try:
                # ifFind, location, site_score = row[f"ifFind_{suffix}"], row[f"FindLocation_{suffix}"], row[f"site_score_{suffix}"]
                ifFind, location = row[f"ifFind_{suffix}"], row[f"FindLocation_{suffix}"]
                if suffix == "denovo":
                    ifFindDecoy = row[f"ifFindDecoy_{suffix}"]
                # site_score = np.array(site_score.split(","),dtype=float)
            except:
                assert False
            if Evident_select and (mass_threshold != 0.0):
                pep_mass, ifEvident = row[f"Pep_mass_{suffix}"], row[f"ifEvident_{suffix}"]
                if pep_mass < mass_threshold:
                    continue
                if not ifEvident:
                    continue
            else:
                if (not Evident_select) and (mass_threshold != 0.0):
                    pep_mass = row[f"Pep_mass_{suffix}"]
                    if pep_mass < mass_threshold:
                        continue
                else:
                    if Evident_select and (mass_threshold == 0.0):
                        ifEvident = row[f"ifEvident_{suffix}"]
                        if not ifEvident:
                            continue
                    else:
                        pass
            if suffix == "denovo":
                FindFinal  = ifFind and not ifFindDecoy
            else:
                FindFinal = ifFind
            if FindFinal:
                location_split = location.split("&")
                protein_name = {}
                for item in location_split:
                    location = item.split("$")
                    # if location[0].startswith("CON_MYG_HORSE|SWISS-PROT:P68082"):
                    #     print(row["title"],seq, location, coverage_fasta_dict[location[0]][0])
                    #     print(coverage_fasta_dict[location[0]][0][int(location[1]):int(location[2])])
                    #     print(fasta_seq_dict[location[0]][int(location[1]):int(location[2])])
                    coverage_fasta_dict[location[0]][0][int(location[1]):int(location[2])] = 1
                    coverage_fasta_dict[location[0]][1][int(location[1]):int(location[2])] += 1
                    # coverage_fasta_dict[location[0]][2][int(location[1]):int(location[2])] = np.maximum(coverage_fasta_dict[location[0]][1][int(location[1]):int(location[2])], site_score)
                    protein_name[location[0]] = 1
                if len(protein_name) == 1:
                    coverage_fasta_dict[list(protein_name.keys())[0]][3] += 1
                    coverage_fasta_dict[list(protein_name.keys())[0]][5] += 1
                    seq_to_if_unique_dict[seq] = True
                    this_protein_name = list(protein_name.keys())[0]
                    if this_protein_name not in proteinName_to_uniqueSeq:
                        proteinName_to_uniqueSeq[this_protein_name] = [seq]
                    else:
                        proteinName_to_uniqueSeq[this_protein_name].append(seq)
                else:
                    for protein in protein_name:
                        coverage_fasta_dict[protein][4] += 1
                        coverage_fasta_dict[protein][5] += 1
                    seq_to_if_unique_dict[seq] = False
            else:
                seq_to_if_unique_dict[seq] = False

    #去重
    for key in proteinName_to_uniqueSeq:
        proteinName_to_uniqueSeq[key] = list(set(proteinName_to_uniqueSeq[key]))

    return seq_to_if_unique_dict, coverage_fasta_dict, proteinName_to_uniqueSeq


def write_dict_to_csv(fasta_coverage_denovo_dict, output_path, fasta_seq_dict, select_protein_list, proteinName_to_uniqueSeq_dict):
    # 输出info信息
    all_full_coverage_protein_num = 0
    all_coverage_protein_num = 0
    all_total_aa_num = 0
    all_coverage_aa_num = 0
    all_coverage_per_protein = []
    with open(output_path,"w") as fw:
        fw.write("Protein\tCoverage_AA_num\tTotal_AA_num\tCoverage_rate\tPep_unique_num\tPep_Nonunique_num\tPep_total_num\tSeq_tag\tSeqNum_tag\tSeq_site_score\tSeq\tPep_unique_num_pepLevel\tPep_unique\n")
        for key in fasta_coverage_denovo_dict:
            if select_protein_list is not None and key not in select_protein_list:
                continue
            coverage_AA_num = np.sum(fasta_coverage_denovo_dict[key][0] != 0)
            total_AA_num = len(fasta_coverage_denovo_dict[key][0])
            coverage_rate = coverage_AA_num / total_AA_num
            Pep_unique_num = fasta_coverage_denovo_dict[key][3]#这个实际上是谱图数量
            if key in proteinName_to_uniqueSeq_dict:
                Pep_unique = "@".join(proteinName_to_uniqueSeq_dict[key])
                Pep_unique_num_PepLevel = len(proteinName_to_uniqueSeq_dict[key])
            else:
                Pep_unique = "None"
                Pep_unique_num_PepLevel = 0
            Pep_NonUnique_num = fasta_coverage_denovo_dict[key][4]
            Pep_total_num = fasta_coverage_denovo_dict[key][5]
            tag_str = ""
            seq_str = ""
            NumTag_str = ",".join(map(str, list(fasta_coverage_denovo_dict[key][1])))
            score_str = ",".join(map(str, list(np.round(fasta_coverage_denovo_dict[key][2],2))))
            for i in range(len(fasta_coverage_denovo_dict[key][0])):
                tag_str += str(int(fasta_coverage_denovo_dict[key][0][i]))
                if fasta_coverage_denovo_dict[key][0][i] != 0:
                    seq_str += fasta_seq_dict[key][i].lower()
                else:
                    seq_str += fasta_seq_dict[key][i]
            fw.write(f"{key}\t{coverage_AA_num}\t{total_AA_num}\t{coverage_rate}\t{Pep_unique_num}\t{Pep_NonUnique_num}\t{Pep_total_num}\t{tag_str}\t{NumTag_str}\t{score_str}\t{seq_str}\t{Pep_unique_num_PepLevel}\t{Pep_unique}\n")

            if coverage_AA_num == len(fasta_coverage_denovo_dict[key][0]):
                all_full_coverage_protein_num += 1
            if coverage_AA_num > 0:
                all_coverage_protein_num += 1
            all_total_aa_num += total_AA_num
            all_coverage_aa_num += coverage_AA_num
            all_coverage_per_protein.append(coverage_AA_num / total_AA_num)

        fw.write("\n")
        fw.write(f"protein_num: {len(fasta_coverage_denovo_dict.keys())}\n")
        fw.write(f"full_coverage_protein_num: {all_full_coverage_protein_num}\n")
        fw.write(f"coverage_protein_num: {all_coverage_protein_num}\n")
        fw.write(f"full_coverage_protein_rate: {all_full_coverage_protein_num / len(fasta_coverage_denovo_dict.keys())}\n")
        fw.write(f"coverage_protein_rate: {all_coverage_protein_num / len(fasta_coverage_denovo_dict.keys())}\n")
        fw.write(f"total_aa_num: {all_total_aa_num}\n")
        fw.write(f"coverage_aa_num: {all_coverage_aa_num}\n")
        fw.write(f"coverage_rate: {all_coverage_aa_num / all_total_aa_num}\n")
        fw.write(f"ave_coverage_per_protein: {np.mean(all_coverage_per_protein)}\n")
        fw.write(f"min_coverage_per_protein: {min(all_coverage_per_protein)}\n")
        fw.write(f"max_coverage_per_protein: {max(all_coverage_per_protein)}\n")

def read_dict_from_csv(input_path):
    fasta_coverage_denovo_dict = {}
    key_to_uniqueSeq_dict = {}
    with open(input_path, "r") as f:
        line = f.readline().strip().split("\t")
        protein_name_index = line.index("Protein")
        coverage_aa_num_index = line.index("Coverage_AA_num")
        pep_unique_num_index = line.index("Pep_unique_num")
        pep_unique_index = line.index("Pep_unique")
        pep_unique_num_PepLevel = line.index("Pep_unique_num_pepLevel")
        pep_Nonunique_num_index = line.index("Pep_Nonunique_num")
        pep_total_num_index = line.index("Pep_total_num")
        seq_tag_index = line.index("Seq_tag")
        seqNum_tag_index = line.index("SeqNum_tag")
        seq_site_score_index = line.index("Seq_site_score")
        seq_index = line.index("Seq")
        for line in f:
            line = line.strip().split("\t")
            if len(line) <= 1:
                break
            protein_name = line[protein_name_index]
            sum_aa = int(float(line[coverage_aa_num_index]))
            seq = line[seq_index]
            tag = np.array([1 if char.islower() else 0 for i, char in enumerate(seq)]).reshape(-1)
            NumTag_np = np.array([line[seqNum_tag_index].split(",")], dtype=int).reshape(-1)
            score_np = np.array([line[seq_site_score_index].split(",")], dtype=float).reshape(-1)
            pep_unique_num = int(line[pep_unique_num_index])
            pep_Nonunique_num = int(line[pep_Nonunique_num_index])
            pep_total_num = int(line[pep_total_num_index])

            fasta_coverage_denovo_dict[protein_name] = [tag, NumTag_np, score_np, pep_unique_num, pep_Nonunique_num, pep_total_num]

            if line[pep_unique_index] != "None":
                uniqueSeq = line[pep_unique_index].split("@")
                key_to_uniqueSeq_dict[protein_name] = uniqueSeq
            assert sum_aa == np.sum(tag)

    return fasta_coverage_denovo_dict, key_to_uniqueSeq_dict
