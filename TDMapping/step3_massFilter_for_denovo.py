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
from common_function import build_fasta_dict_function, build_index_dict_function, calculate_mass

fasta_file = config.fasta_file
output_folder_list = [
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pnovo_output\lysC\result",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\peaks_deepnovo_output\lysC\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pointnovo_output\lysC\23charge\test",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\9species_model\lysC\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\massiveKB_model\lysC\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\GraphNovo_output\lysC",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\PrimeNovo_output\lysC",
    r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\MirrorNovo_output\single\lysC",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pnovo_output\LysN\result",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\peaks_deepnovo_output\LysN\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pointnovo_output\LysN\23charge\test",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\9species_model\LysN\23charge",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\massiveKB_model\LysN\23charge"
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\GraphNovo_output\lysN",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\PrimeNovo_output\lysN",
    r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\MirrorNovo_output\single\lysN",
]
# fig_name1_list = ["LysC","LysC","LysC","LysC","LysC","lysN","lysN","lysN","lysN","lysN"]
# singleSpec_tool_list = ["pNovo3","PEAKS","PointNovo","CasaNovo","CasaNovo","pNovo3","PEAKS","PointNovo","CasaNovo","CasaNovo"]
fig_name1_list = ["LysC","lysN"]
singleSpec_tool_list = ["GCNovo","GCNovo"]
estimated_error_rate = 1 #估计1%FDR

if __name__ == "__main__":

    for i in range(len(output_folder_list)):
        print(f"{singleSpec_tool_list[i]}")

        output_folder = output_folder_list[i]
        fig_name1 = fig_name1_list[i]
        singleSpec_tool = singleSpec_tool_list[i]

        fasta_seq = build_fasta_dict_function(fasta_file)
        Index_first_3aa = build_index_dict_function(fasta_seq)

        seq_to_fastaLocation_dict = {}
        data_dict_for_plt_usingPepmass = {}
        data_dict_for_plt_usingPepmass_qvalue = {}
        data_dict_for_plt_usingPepmass_countOnlyTarget = {}
        data_dict_for_plt_usingPepmass_countOnlyDecoy = {}
        pepmass_list = [i for i in range(2001)]
        print("pepmass_list: ", pepmass_list)
        # intsection_targetSeq_info_fout = open(try_mgf + "\\intsection_targetSeq_info.res", "a+")

        info_fw = open(output_folder + "\\massFilterFDR_info.res", "w")
        # 抛弃数据库搜库结果，统计denovoSeq的肽段数量
        DenovoSeq_df = pd.read_csv(output_folder + "\\denovoSeq_to_ifEvidentFind.res", sep="\t")
        DenovoSeq_df["ifTry"] = True
        DenovoSeq_df["Pep_mass"] = DenovoSeq_df["denovoSeq"].apply(lambda seq: calculate_mass(seq, if_delete_first_aa=False))
        DenovoSeq_df.to_csv(output_folder + "\\denovoSeq_to_ifEvidentFind.res", sep="\t", index=False)  # 保存到文件


        data_list = []
        data_qvalue_list = []
        data_countTarget_list = []
        data_countDecoy_list = []
        for i in pepmass_list:
            Findtarget_df = DenovoSeq_df.loc[(DenovoSeq_df["ifFind"] == True) & (DenovoSeq_df["Pep_mass"] >= i)]
            Finddecoy_df = DenovoSeq_df.loc[(DenovoSeq_df["ifFindDecoy"] == True) & (DenovoSeq_df["Pep_mass"] >= i)]
            set1 = set(Findtarget_df["denovoSeq"])
            set2 = set(Finddecoy_df["denovoSeq"])
            only_set1_len = len(set1 - set2)
            only_set2_len = len(set2 - set1)
            intsection_set_len = len(set1 & set2)
            error = (only_set2_len + 1) / max(only_set1_len , 1) * 100
            data_list.append(error)
            data_qvalue_list.append(min(data_list))
            data_countTarget_list.append(only_set1_len)
            data_countDecoy_list.append(only_set2_len)
        data_dict_for_plt_usingPepmass[f"{fig_name1}({singleSpec_tool})"] = data_list
        data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}({singleSpec_tool})"] = data_qvalue_list
        data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name1}({singleSpec_tool})"] = data_countTarget_list
        data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name1}({singleSpec_tool})"] = data_countDecoy_list
        error_rate = np.array(data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}({singleSpec_tool})"])
        delta = error_rate - estimated_error_rate
        delta = np.abs(delta)
        min_index = np.argmin(delta)
        mass_threshold = pepmass_list[min_index]
        info_fw.write(f"{fig_name1}({singleSpec_tool})\t{mass_threshold}\n")

        data_list = []
        data_qvalue_list = []
        data_countTarget_list = []
        data_countDecoy_list = []
        for i in pepmass_list:
            Findtarget_df = DenovoSeq_df.loc[
                (DenovoSeq_df["ifFind"] == True) & (DenovoSeq_df["Pep_mass"] >= i) & (DenovoSeq_df["ifEvident"] == True) ]
            Finddecoy_df = DenovoSeq_df.loc[
                (DenovoSeq_df["ifFindDecoy"] == True) & (DenovoSeq_df["Pep_mass"] >= i) & (DenovoSeq_df["ifEvident"] == True)]
            set1 = set(Findtarget_df["denovoSeq"])
            set2 = set(Finddecoy_df["denovoSeq"])
            only_set1_len = len(set1 - set2)
            only_set2_len = len(set2 - set1)
            intsection_set_len = len(set1 & set2)
            error = (only_set2_len + 1) / max(only_set1_len, 1) * 100
            data_list.append(error)
            data_qvalue_list.append(min(data_list))
            data_countTarget_list.append(only_set1_len)
            data_countDecoy_list.append(only_set2_len)
        data_dict_for_plt_usingPepmass[f"{fig_name1}(Confident_{singleSpec_tool})"] = data_list
        data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}(Confident_{singleSpec_tool})"] = data_qvalue_list
        data_dict_for_plt_usingPepmass_countOnlyTarget[f"{fig_name1}(Confident_{singleSpec_tool})"] = data_countTarget_list
        data_dict_for_plt_usingPepmass_countOnlyDecoy[f"{fig_name1}(Confident_{singleSpec_tool})"] = data_countDecoy_list
        error_rate = np.array(data_dict_for_plt_usingPepmass_qvalue[f"{fig_name1}(Confident_{singleSpec_tool})"])
        delta = error_rate - estimated_error_rate
        delta = np.abs(delta)
        min_index = np.argmin(delta)
        mass_threshold = pepmass_list[min_index]
        info_fw.write(f"{fig_name1}(Confident_{singleSpec_tool})\t{mass_threshold}\n")

        # 画图
        error_csv_using_mass = output_folder + "\\error_rate_using_mass.csv"
        f_mass = open(error_csv_using_mass, "w")
        error_csv_using_mass_qvalue = output_folder + "\\error_rate_using_mass_qvalue.csv"
        f_mass_qvalue = open(error_csv_using_mass_qvalue, "w")
        error_csv_using_mass_target = output_folder + "\\error_rate_using_mass_targetCount.csv"
        f_mass_target = open(error_csv_using_mass_target, "w")
        error_csv_using_mass_decoy = output_folder + "\\error_rate_using_mass_decoyCount.csv"
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
        plt.savefig(output_folder + "\\WithoutConfident_error_rate_using_mass.png", dpi=300)
        plt.show()

        #
        plt.figure()
        # 对于字典中的每一条线，绘制一条折线
        for line_name, y_values in data_dict_for_plt_usingPepmass_qvalue.items():
            x_values = pepmass_list  # 创建一个长度与y_values相同的x值列表
            plt.plot(x_values, y_values, label=line_name)
        # 添加图例
        plt.legend()
        # 显示并保存图形
        plt.savefig(output_folder + "\\WithConfident_error_rate_using_mass.png", dpi=300)
        plt.show()





