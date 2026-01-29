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
from common_function import build_fasta_dict_function, build_index_dict_function, calculate_mass,generate_DiNovoPairs_to_matchtype_dict,\
    calculate_mass_mirror

fasta_file = config.fasta_file
denovoSeq_folder = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MirrorNovo_output\mirror\20240525\23charge"
DiNovoPairs_file = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\spectraPairs\20240525\[DiNovo]SpectralPairs[23charge].res"
fig_name1 = "Try"
fig_name2 = "Lys"
estimated_error_rate = 1

if __name__ == "__main__":

    fasta_seq = build_fasta_dict_function(fasta_file)
    Index_first_3aa = build_index_dict_function(fasta_seq)
    (try_title_header_index,
     lys_title_header_index,
     match_type_header_index,
     titlesPair_to_location_dict) = generate_DiNovoPairs_to_matchtype_dict(DiNovoPairs_file)

    seq_to_fastaLocation_dict = {}
    data_dict_for_plt_usingPepmass = {}
    data_dict_for_plt_usingPepmass_qvalue = {}
    data_dict_for_plt_usingPepmass_countOnlyTarget = {}
    data_dict_for_plt_usingPepmass_countOnlyDecoy = {}
    pepmass_list = [i for i in range(2001)]
    print("pepmass_list: ", pepmass_list)
    # intsection_targetSeq_info_fout = open(try_mgf + "\\intsection_targetSeq_info.res", "a+")

    info_fw = open(denovoSeq_folder + "\\massFilterFDR_info.res", "w")
    # 抛弃数据库搜库结果，统计denovoSeq的肽段数量
    DiNovoTryLys_DenovoRSeq_df = pd.read_csv(denovoSeq_folder + "\\DiNovoTryLys_denovoSeq_to_ifEvidentFind.res", sep="\t")
    DiNovoTryLys_DenovoRSeq_df["ifDiNovoTryLys"] = True
    DiNovoTryLys_DenovoRSeq_df["Pep_mass"] = DiNovoTryLys_DenovoRSeq_df.apply(lambda row: calculate_mass_mirror(row, titlesPair_to_location_dict), axis=1)
    DiNovoTryLys_DenovoRSeq_df.to_csv(denovoSeq_folder + "\\DiNovoTryLys_denovoSeq_to_ifEvidentFind.res", sep="\t",index=False)  # 保存到文件
    print("start")

    data_list = []
    data_qvalue_list = []
    data_countTarget_list = []
    data_countDecoy_list = []
    for i in pepmass_list:
        DiNovoTryLys_Findtarget_df = DiNovoTryLys_DenovoRSeq_df.loc[
            (DiNovoTryLys_DenovoRSeq_df["ifFind_DiNovoTryLys"] == True) & (DiNovoTryLys_DenovoRSeq_df["TDTag_DiNovoTryLys"] == "T") & (DiNovoTryLys_DenovoRSeq_df["Pep_mass"] >= i)]
        DiNovoTryLys_Finddecoy_df = DiNovoTryLys_DenovoRSeq_df.loc[
            (DiNovoTryLys_DenovoRSeq_df["ifFind_DiNovoTryLys"] == True) & (DiNovoTryLys_DenovoRSeq_df["TDTag_DiNovoTryLys"] == "D") & (DiNovoTryLys_DenovoRSeq_df["Pep_mass"] >= i)]
        set1 = set(DiNovoTryLys_Findtarget_df["denovoSeq_DiNovoTryLys"])
        set2 = set(DiNovoTryLys_Finddecoy_df["denovoSeq_DiNovoTryLys"])

        only_set1_len = len(set1 - set2)
        only_set2_len = len(set2 - set1)
        error = (only_set2_len) / (only_set1_len + 1) * 100
        data_list.append(error)
        data_qvalue_list.append(min(data_list))
        data_countTarget_list.append(only_set1_len)
        data_countDecoy_list.append(only_set2_len)
    data_dict_for_plt_usingPepmass[f"{fig_name1}&{fig_name2}(DiNovo)"] = data_list
    data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}&{fig_name2}(DiNovo)"] = data_qvalue_list
    data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name1}&{fig_name2}(DiNovo)"] = data_countTarget_list
    data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name1}&{fig_name2}(DiNovo)"] = data_countDecoy_list
    error_rate = np.array(data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}&{fig_name2}(DiNovo)"])
    delta = error_rate - estimated_error_rate
    delta = np.abs(delta)
    min_index = np.argmin(delta)
    mass_threshold = pepmass_list[min_index]
    info_fw.write(f"{fig_name1}&{fig_name2}(DiNovo)\t{mass_threshold}\n")

    data_list = []
    data_qvalue_list = []
    data_countTarget_list = []
    data_countDecoy_list = []
    for i in pepmass_list:
        DiNovoTryLys_Findtarget_df = DiNovoTryLys_DenovoRSeq_df.loc[
            (DiNovoTryLys_DenovoRSeq_df["ifFind_DiNovoTryLys"] == True) & (DiNovoTryLys_DenovoRSeq_df["TDTag_DiNovoTryLys"] == "T") &
            (DiNovoTryLys_DenovoRSeq_df["ifEvident_DiNovoTryLys"] == True) &
            (DiNovoTryLys_DenovoRSeq_df["Pep_mass"] >= i)]
        DiNovoTryLys_Finddecoy_df = DiNovoTryLys_DenovoRSeq_df.loc[
            (DiNovoTryLys_DenovoRSeq_df["ifFind_DiNovoTryLys"] == True) & (DiNovoTryLys_DenovoRSeq_df["TDTag_DiNovoTryLys"] == "D") &
            (DiNovoTryLys_DenovoRSeq_df["ifEvident_DiNovoTryLys"] == True) &
            (DiNovoTryLys_DenovoRSeq_df["Pep_mass"] >= i)]
        set1 = set(DiNovoTryLys_Findtarget_df["denovoSeq_DiNovoTryLys"])
        set2 = set(DiNovoTryLys_Finddecoy_df["denovoSeq_DiNovoTryLys"])

        only_set1_len = len(set1 - set2)
        only_set2_len = len(set2 - set1)
        intsection_set_len = len(set1 & set2)
        error = (only_set2_len) / (only_set1_len + 1) * 100
        data_list.append(error)
        data_qvalue_list.append(min(data_list))
        data_countTarget_list.append(only_set1_len)
        data_countDecoy_list.append(only_set2_len)
    data_dict_for_plt_usingPepmass[f"{fig_name1}&{fig_name2}(Confident_DiNovo)"] = data_list
    data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}&{fig_name2}(Confident_DiNovo)"] = data_qvalue_list
    data_dict_for_plt_usingPepmass_countOnlyTarget[
        f"{fig_name1}&{fig_name2}(Confident_DiNovo)"] = data_countTarget_list
    data_dict_for_plt_usingPepmass_countOnlyDecoy[
        f"{fig_name1}&{fig_name2}(Confident_DiNovo)"] = data_countDecoy_list
    error_rate = np.array(data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}&{fig_name2}(Confident_DiNovo)"])
    delta = error_rate - estimated_error_rate
    delta = np.abs(delta)
    min_index = np.argmin(delta)
    mass_threshold = pepmass_list[min_index]
    info_fw.write(f"{fig_name1}&{fig_name2}(Confident_DiNovo)\t{mass_threshold}\n")

    DiNovoTry_DenovoRSeq_df = pd.read_csv(denovoSeq_folder + "\\DiNovoTry_denovoSeq_to_ifEvidentFind.res", sep="\t")
    DiNovoTry_DenovoRSeq_df["ifDiNovoTry"] = True
    DiNovoTry_DenovoRSeq_df["Pep_mass"] = DiNovoTry_DenovoRSeq_df["denovoSeq_DiNovoTry"].apply(lambda seq: calculate_mass(seq, if_delete_first_aa=False))
    DiNovoTry_DenovoRSeq_df.to_csv(denovoSeq_folder + "\\DiNovoTry_denovoSeq_to_ifEvidentFind.res",sep="\t", index=False)  # 保存到文件

    data_list = []
    data_qvalue_list = []
    data_countTarget_list = []
    data_countDecoy_list = []
    for i in pepmass_list:
        DiNovoTry_Findtarget_df = DiNovoTry_DenovoRSeq_df.loc[
            (DiNovoTry_DenovoRSeq_df["ifFind_DiNovoTry"] == True) & (
                        DiNovoTry_DenovoRSeq_df["Pep_mass"] >= i)]
        DiNovoTry_Finddecoy_df = DiNovoTry_DenovoRSeq_df.loc[
            (DiNovoTry_DenovoRSeq_df["ifFindDecoy_DiNovoTry"] == True) & (
                    DiNovoTry_DenovoRSeq_df["Pep_mass"] >= i)]
        set1 = set(DiNovoTry_Findtarget_df["denovoSeq_DiNovoTry"])
        set2 = set(DiNovoTry_Finddecoy_df["denovoSeq_DiNovoTry"])

        only_set1_len = len(set1 - set2)
        only_set2_len = len(set2 - set1)
        intsection_set_len = len(set1 & set2)
        error = (only_set2_len) / (only_set1_len + 1) * 100
        data_list.append(error)
        data_qvalue_list.append(min(data_list))
        data_countTarget_list.append(only_set1_len)
        data_countDecoy_list.append(only_set2_len)
    data_dict_for_plt_usingPepmass[f"{fig_name1}(DiNovo)"] = data_list
    data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}(DiNovo)"] = data_qvalue_list
    data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name1}(DiNovo)"] = data_countTarget_list
    data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name1}(DiNovo)"] = data_countDecoy_list
    error_rate = np.array(data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}(DiNovo)"])
    delta = error_rate - estimated_error_rate
    delta = np.abs(delta)
    min_index = np.argmin(delta)
    mass_threshold = pepmass_list[min_index]
    info_fw.write(f"{fig_name1}(DiNovo)\t{mass_threshold}\n")

    data_list = []
    data_qvalue_list = []
    data_countTarget_list = []
    data_countDecoy_list = []
    for i in pepmass_list:
        DiNovoTry_Findtarget_df = DiNovoTry_DenovoRSeq_df.loc[
            (DiNovoTry_DenovoRSeq_df["ifFind_DiNovoTry"] == True) & (
                    DiNovoTry_DenovoRSeq_df["ifEvident_DiNovoTry"] == True) & (
                    DiNovoTry_DenovoRSeq_df["Pep_mass"] >= i)]
        DiNovoTry_Finddecoy_df = DiNovoTry_DenovoRSeq_df.loc[
            (DiNovoTry_DenovoRSeq_df["ifFindDecoy_DiNovoTry"] == True) & (
                    DiNovoTry_DenovoRSeq_df["ifEvident_DiNovoTry"] == True) & (
                    DiNovoTry_DenovoRSeq_df["Pep_mass"] >= i)]
        set1 = set(DiNovoTry_Findtarget_df["denovoSeq_DiNovoTry"])
        set2 = set(DiNovoTry_Finddecoy_df["denovoSeq_DiNovoTry"])

        only_set1_len = len(set1 - set2)
        only_set2_len = len(set2 - set1)
        intsection_set_len = len(set1 & set2)
        error = (only_set2_len) / (only_set1_len + 1) * 100
        data_list.append(error)
        data_qvalue_list.append(min(data_list))
        data_countTarget_list.append(only_set1_len)
        data_countDecoy_list.append(only_set2_len)
    data_dict_for_plt_usingPepmass[f"{fig_name1}(Confident_DiNovo)"] = data_list
    data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}(Confident_DiNovo)"] = data_qvalue_list
    data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name1}(Confident_DiNovo)"] = data_countTarget_list
    data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name1}(Confident_DiNovo)"] = data_countDecoy_list
    error_rate = np.array(data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}(Confident_DiNovo)"])
    delta = error_rate - estimated_error_rate
    delta = np.abs(delta)
    min_index = np.argmin(delta)
    mass_threshold = pepmass_list[min_index]
    info_fw.write(f"{fig_name1}(Confident_DiNovo)\t{mass_threshold}\n")

    DiNovoLys_DenovoRSeq_df = pd.read_csv(denovoSeq_folder + "\\DiNovoLys_denovoSeq_to_ifEvidentFind.res", sep="\t")
    DiNovoLys_DenovoRSeq_df["ifDiNovoLys"] = True
    DiNovoLys_DenovoRSeq_df["Pep_mass"] = DiNovoLys_DenovoRSeq_df["denovoSeq_DiNovoLys"].apply(lambda seq: calculate_mass(seq, if_delete_first_aa=False))
    DiNovoLys_DenovoRSeq_df.to_csv(denovoSeq_folder + "\\DiNovoLys_denovoSeq_to_ifEvidentFind.res", sep="\t", index=False)  # 保存到文件


    data_list = []
    data_qvalue_list = []
    data_countTarget_list = []
    data_countDecoy_list = []
    for i in pepmass_list:
        DiNovoLys_Findtarget_df = DiNovoLys_DenovoRSeq_df.loc[
            (DiNovoLys_DenovoRSeq_df["ifFind_DiNovoLys"] == True) & (
                        DiNovoLys_DenovoRSeq_df["Pep_mass"] >= i)]
        DiNovoLys_Finddecoy_df = DiNovoLys_DenovoRSeq_df.loc[
            (DiNovoLys_DenovoRSeq_df["ifFindDecoy_DiNovoLys"] == True) & (
                    DiNovoLys_DenovoRSeq_df["Pep_mass"] >= i)]
        set1 = set(DiNovoLys_Findtarget_df["denovoSeq_DiNovoLys"])
        set2 = set(DiNovoLys_Finddecoy_df["denovoSeq_DiNovoLys"])

        only_set1_len = len(set1 - set2)
        only_set2_len = len(set2 - set1)
        intsection_set_len = len(set1 & set2)
        error = (only_set2_len) / (only_set1_len + 1) * 100
        data_list.append(error)
        data_qvalue_list.append(min(data_list))
        data_countTarget_list.append(only_set1_len)
        data_countDecoy_list.append(only_set2_len)
    data_dict_for_plt_usingPepmass[f"{fig_name2}(DiNovo)"] = data_list
    data_dict_for_plt_usingPepmass_qvalue[f"{fig_name2}(DiNovo)"] = data_qvalue_list
    data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name2}(DiNovo)"] = data_countTarget_list
    data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name2}(DiNovo)"] = data_countDecoy_list
    error_rate = np.array(data_dict_for_plt_usingPepmass_qvalue[f"{fig_name2}(DiNovo)"])
    delta = error_rate - estimated_error_rate
    delta = np.abs(delta)
    min_index = np.argmin(delta)
    mass_threshold = pepmass_list[min_index]
    info_fw.write(f"{fig_name2}(DiNovo)\t{mass_threshold}\n")

    data_list = []
    data_qvalue_list = []
    data_countTarget_list = []
    data_countDecoy_list = []
    for i in pepmass_list:
        DiNovoLys_Findtarget_df = DiNovoLys_DenovoRSeq_df.loc[
            (DiNovoLys_DenovoRSeq_df["ifFind_DiNovoLys"] == True) & (
                    DiNovoLys_DenovoRSeq_df["ifEvident_DiNovoLys"] == True) & (
                    DiNovoLys_DenovoRSeq_df["Pep_mass"] >= i)]
        DiNovoLys_Finddecoy_df = DiNovoLys_DenovoRSeq_df.loc[
            (DiNovoLys_DenovoRSeq_df["ifFindDecoy_DiNovoLys"] == True) & (
                    DiNovoLys_DenovoRSeq_df["ifEvident_DiNovoLys"] == True) & (
                    DiNovoLys_DenovoRSeq_df["Pep_mass"] >= i)]
        set1 = set(DiNovoLys_Findtarget_df["denovoSeq_DiNovoLys"])
        set2 = set(DiNovoLys_Finddecoy_df["denovoSeq_DiNovoLys"])

        only_set1_len = len(set1 - set2)
        only_set2_len = len(set2 - set1)
        intsection_set_len = len(set1 & set2)
        error = (only_set2_len) / (only_set1_len + 1) * 100
        data_list.append(error)
        data_qvalue_list.append(min(data_list))
        data_countTarget_list.append(only_set1_len)
        data_countDecoy_list.append(only_set2_len)
    data_dict_for_plt_usingPepmass[f"{fig_name2}(Confident_DiNovo)"] = data_list
    data_dict_for_plt_usingPepmass_qvalue[f"{fig_name2}(Confident_DiNovo)"] = data_qvalue_list
    data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name2}(Confident_DiNovo)"] = data_countTarget_list
    data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name2}(Confident_DiNovo)"] = data_countDecoy_list
    error_rate = np.array(data_dict_for_plt_usingPepmass_qvalue[f"{fig_name2}(Confident_DiNovo)"])
    delta = error_rate - estimated_error_rate
    delta = np.abs(delta)
    min_index = np.argmin(delta)
    mass_threshold = pepmass_list[min_index]
    info_fw.write(f"{fig_name2}(Confident_DiNovo)\t{mass_threshold}\n")

    # 画图
    error_csv_using_mass = denovoSeq_folder + "\\error_rate_using_mass.csv"
    f_mass = open(error_csv_using_mass, "w")
    error_csv_using_mass_qvalue = denovoSeq_folder + "\\error_rate_using_mass_qvalue.csv"
    f_mass_qvalue = open(error_csv_using_mass_qvalue, "w")
    error_csv_using_mass_target = denovoSeq_folder + "\\error_rate_using_mass_targetCount.csv"
    f_mass_target = open(error_csv_using_mass_target, "w")
    error_csv_using_mass_decoy = denovoSeq_folder + "\\error_rate_using_mass_decoyCount.csv"
    f_mass_decoy = open(error_csv_using_mass_decoy, "w")

    # 文件的header是0~30
    header = ",".join([str(i) for i in pepmass_list])
    header = "lines," + header + "\n"
    f_mass.write(header)
    f_mass_qvalue.write(header)
    f_mass_target.write(header)
    f_mass_decoy.write(header)

    # 创建一个新的图形
    plt.figure()
    # 对于字典中的每一条线，绘制一条折线
    for line_name, y_values in data_dict_for_plt_usingPepmass_qvalue.items():
        f_mass.write(line_name + "," + ",".join([str(i) for i in data_dict_for_plt_usingPepmass[line_name]]) + "\n")
        f_mass_qvalue.write(line_name + "," + ",".join([str(i) for i in y_values]) + "\n")
        f_mass_target.write(line_name + "," + ",".join(
            [str(i) for i in data_dict_for_plt_usingPepmass_countOnlyTarget[line_name]]) + "\n")
        f_mass_decoy.write(line_name + "," + ",".join(
            [str(i) for i in data_dict_for_plt_usingPepmass_countOnlyDecoy[line_name]]) + "\n")
        if "Confident" in line_name:
            continue
        x_values = pepmass_list  # 创建一个长度与y_values相同的x值列表
        plt.plot(x_values, y_values, label=line_name)
    # 添加图例
    plt.legend()
    # 显示并保存图形
    plt.savefig(denovoSeq_folder + "\\WithoutConfident_error_rate_using_mass.png", dpi=300)
    plt.show()

    # 包含evident的图
    plt.figure()
    # 对于字典中的每一条线，绘制一条折线
    for line_name, y_values in data_dict_for_plt_usingPepmass_qvalue.items():
        x_values = pepmass_list  # 创建一个长度与y_values相同的x值列表
        plt.plot(x_values, y_values, label=line_name)
    # 添加图例
    plt.legend()
    # 显示并保存图形
    plt.savefig(denovoSeq_folder + "\\WithConfident_error_rate_using_mass.png", dpi=300)
    plt.show()

    # 生成MirrorNovo全部的曲线
    DiNovo_target = []
    DiNovo_decoy = []
    DiNovo = []
    DiNovo_confident_target = []
    DiNovo_confident_decoy = []
    DiNovo_confident = []
    DiNovoTry_target = data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name1}(DiNovo)"]
    DiNovoTry_decoy = data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name1}(DiNovo)"]
    DiNovoTry_confident_target = data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name1}(Confident_DiNovo)"]
    DiNovoTry_confident_decoy = data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name1}(Confident_DiNovo)"]
    DiNovoLys_target = data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name2}(DiNovo)"]
    DiNovoLys_decoy = data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name2}(DiNovo)"]
    DiNovoLys_confident_target = data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name2}(Confident_DiNovo)"]
    DiNovoLys_confident_decoy = data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name2}(Confident_DiNovo)"]
    DiNovoTryLys_target = data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name1}&{fig_name2}(DiNovo)"]
    DiNovoTryLys_decoy = data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name1}&{fig_name2}(DiNovo)"]
    DiNovoTryLys_confident_target = data_dict_for_plt_usingPepmass_countOnlyTarget[
        f"{fig_name1}&{fig_name2}(Confident_DiNovo)"]
    DiNovoTryLys_confident_decoy = data_dict_for_plt_usingPepmass_countOnlyDecoy[
        f"{fig_name1}&{fig_name2}(Confident_DiNovo)"]
    for i in range(len(DiNovoTry_target)):
        DiNovo_target.append(DiNovoTry_target[i] + DiNovoLys_target[i] + DiNovoTryLys_target[i])
        DiNovo_decoy.append(DiNovoTry_decoy[i] + DiNovoLys_decoy[i] + DiNovoTryLys_decoy[i])
        DiNovo_confident_target.append(
            DiNovoTry_confident_target[i] + DiNovoLys_confident_target[i] + DiNovoTryLys_confident_target[
                i])
        DiNovo_confident_decoy.append(
            DiNovoTry_confident_decoy[i] + DiNovoLys_confident_decoy[i] + DiNovoTryLys_confident_decoy[i])

    for i in range(len(DiNovo_target)):
        if DiNovo_target[i] != 0:
            rate = DiNovo_decoy[i] / DiNovo_target[i] * 100
        else:
            rate = 0
        DiNovo.append(rate)

        if DiNovo_confident_target[i] != 0:
            rate = DiNovo_confident_decoy[i] / DiNovo_confident_target[i] * 100
        else:
            rate = 0
        DiNovo_confident.append(rate)

    # 写入文件
    f_mass.write("DiNovo" + "," + ",".join([str(i) for i in DiNovo]) + "\n")
    f_mass_target.write("DiNovo" + "," + ",".join([str(i) for i in DiNovo_target]) + "\n")
    f_mass_decoy.write("DiNovo" + "," + ",".join([str(i) for i in DiNovo_decoy]) + "\n")
    f_mass.write("Confident_DiNovo" + "," + ",".join([str(i) for i in DiNovo_confident]) + "\n")
    f_mass_target.write("Confident_DiNovo" + "," + ",".join([str(i) for i in DiNovo_confident_target]) + "\n")
    f_mass_decoy.write("Confident_DiNovo" + "," + ",".join([str(i) for i in DiNovo_confident_decoy]) + "\n")

    f_mass.close()
    f_mass_target.close()
    f_mass_decoy.close()

    





