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
    mapped_function_single, get_fasta_coverage_np_single, write_dict_to_csv, read_dict_from_csv, get_ScanIndex_from_MSGFTitle

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

topk_peaks = config.topk_peaks
fasta_file = config.fasta_file
software_tool = "MSGF" #pFind,MSFragger,MSGF
# mgf = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\mgf\try\23charge"
peaks_suffix_name = common_function.peaks_suffix_name
# pfind_file = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\MSFragger\lysN\psm[23charge].tsv"#pFind-Filtered_trypsin[extract][v0315].spectra"  # r"\pFind-Filtered_trypsin.spectra"#pFind-Filtered_LysC[extract][v0315]_trypsin
pfind_file = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MSGF\try\MSGF_res[23charge].tsv"

def find_target_sequence(Title, pfind_location_dict, pfind_path, software_tool="pFind"):
    if software_tool == "pFind":
        with open(pfind_path, 'r') as result_f:
            if Title not in pfind_location_dict.keys():
                return None
            else:
                location = pfind_location_dict[Title]
                result_f.seek(location)
                line = str(result_f.readline()).split()
                assert line[0] == Title
                try_sequence = line[5]
                try_sequence_ori = try_sequence.replace('I', 'L')
                try_sequence = list(try_sequence_ori)
                modification = line[10]
                Nterm_tag = False
                if len(modification) != 1:
                    modification = modification.split(';')
                    assert modification[-1] == ''
                    modification = modification[:-1]
                    for item in modification:
                        site = int(item.split(',')[0])
                        if site == 0:  # 很奇怪，N端修饰时，site是0而不是1，但是其他修饰site都是index+1
                            #不考虑N端修饰了，去掉
                            return None

                            modification_type = item.split(',')[1]
                            if modification_type not in modifications_other:
                                return None
                            index = modifications_other.index(modification_type)
                            aa_Nterm = modifications_other_in_aa_dict_keys[index]
                            Nterm_tag = True
                        else:
                            modification_type = item.split(',')[1]
                            if modification_type not in modifications:
                                return None
                            index = modifications.index(modification_type)
                            aa = modifications_in_aa_dict_keys[index]
                            try_sequence[site - 1] = aa
                if Nterm_tag:
                    try_sequence = [aa_Nterm] + try_sequence
                    # print(Title,try_sequence_ori,try_sequence)

                if "C" in try_sequence:
                    return None

                return try_sequence

def build_pfind_location_function(pfind_path):
    # 读取txt文件，创建result的location字典
    location_file = os.path.join(pfind_path + "_location")

    if os.path.exists(location_file) == False:

        result_location_dict = {}
        line = True
        count = 0
        with open(pfind_path, 'r') as f:
            header = f.readline()
            while line:
                current_location = f.tell()
                line = str(f.readline()).strip().split("\t")
                count += 1
                if len(line) <= 1:
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

def get_MSFraggerSeq(row):
    #这里的C是固定修饰的C，M[147]为M(+15.99)
    seq_mod = row[3]
    if seq_mod == "" or seq_mod != seq_mod:  # 没有修饰的肽段
        seq_mod = row[2]
        if "C" in seq_mod:
            print(row)
            assert False
    else:
        seq_mod = seq_mod.replace("C", "C(+57.02)")  # C是固定修饰的C
        seq_mod = seq_mod.replace("M[147]", "M(+15.99)")  # M[147]为M(+15.99)
    seq_mod = seq_mod.replace("I","L")
    return seq_mod

def get_MSGFSeq(row):
    seq = row[9]
    assert seq[1] == "." and seq[-2] == "."
    seq = seq[2:-2]
    seq = seq.replace("I", "L")  # MSGF中I和L是一样的
    seq = seq.replace("M+15.995", "M(+15.99)")  # MSGF中M+15.995是M(+15.99)
    seq = seq.replace("C+57.021", "C(+57.02)")  # MSGF中C+57.021是C(+57.02)
    return seq

if __name__ == "__main__":

    fasta_seq = build_fasta_dict_function(fasta_file)
    fasta_seq_first_4aa_index = build_index_dict_function(fasta_seq)
    fasta_seq_decoy = generate_reverse_fasta(fasta_seq)
    fasta_seq_first_4aa_index_decoy = build_index_dict_function(fasta_seq_decoy)

    #MSGF需要先过滤一下，每个谱图只选择一个结果+卡肽段水平1%FDR，同时把不同文件的df合并在一起
    if software_tool == "MSGF":
        if not os.path.exists(pfind_file):
            print("filtering MSGF results ...")
            all_df = []
            for filename in os.listdir(os.path.dirname(pfind_file)):
                if filename.endswith("[23charge].tsv"):
                    file = os.path.dirname(pfind_file) + f"\\{filename}"
                    print(f"reading {file}...")
                    df = pd.read_csv(file, sep="\t")
                    df["ScanIndex"] = df.apply(lambda x: get_ScanIndex_from_MSGFTitle(x), axis=1)
                    df = df[df["PepQValue"] <= 0.01]
                    #相同ScanIndex只保留PepQvalue最小的结果
                    df = df.sort_values(by=["ScanIndex", "PepQValue"]).drop_duplicates(subset=["ScanIndex"], keep="first")
                    all_df.append(df)
            all_df = pd.concat(all_df, axis=0)
            all_df.to_csv(pfind_file, sep="\t", index=False)

    if software_tool == "pFind":
        pfind_location_dict = build_pfind_location_function(pfind_file)
        spectrum_location_dict, suffix_name, filepathAndOrderNum_to_title_dict = build_mgf_location_function(mgf + "\\")
    else:
        spectrum_location_dict, suffix_name, filepathAndOrderNum_to_title_dict = None, None, None

    if software_tool == "pFind": #需要用谱图计算离子覆盖率
        if not os.path.exists(os.path.dirname(pfind_file) + "\\title_to_db_info.res"):
            fout = open(os.path.dirname(pfind_file) + "\\title_to_db_info.res", "a+")
            fout.write("title\tdbSeq\tsite_tag_db\tsite_peaknum_db\tsite_score_db\tpep_scoremean_db\tpep_scoresum_db\tifEvident_db\tifFind_db\n")
            dbsearch_fin = open(pfind_file, 'r')
            print("reading mgf files ...")
            db_psm_list = []
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
                                db_seq = find_target_sequence(title, pfind_location_dict, pfind_file)
                                db_psm_list.append((title, db_seq))
            db_res_dict = parallel_get_confident_info(db_psm_list, spectrum_location_dict, suffix_name, mgf)
            for i in range(len(db_psm_list)):
                title = db_psm_list[i][0]
                db_illegal = True
                if db_psm_list[i][1] != None:
                    dbSeq = db_psm_list[i][1]
                    dbSeq_str = "".join(db_psm_list[i][1])
                    db_psm = title + "@" + dbSeq_str
                    db_illegal = False
                if not db_illegal:
                    (site_tag_db,
                     site_peaknum_db,
                     site_score_db,
                     pep_scoremean_db,
                     pep_scoresum_db,
                     intensity_list_db,
                     site_intensitytemp_db) = db_res_dict[db_psm]["site_tag"], db_res_dict[db_psm]["site_peaknum"], \
                    db_res_dict[db_psm]["site_score"], db_res_dict[db_psm]["pepscore_mean"], db_res_dict[db_psm][
                        "pepscore_sum"], db_res_dict[db_psm]["intensity_list"], db_res_dict[db_psm]["site_intensitytemp"]
                    sort_intensity_list_db = sorted(intensity_list_db)
                    rank = len(sort_intensity_list_db) - np.array(
                        [bisect.bisect_left(sort_intensity_list_db, i) for i in site_intensitytemp_db]) + 1
                    site_tag_db = np.array(
                        [1 if tag == 1 and rank[i] <= topk_peaks else 0 for i, tag in enumerate(site_tag_db)])
                    miss_num = np.sum(site_tag_db == 0)
                    coverage = 1 - miss_num / len(site_tag_db)
                    ifEvident_db = if_evident_function(coverage)

                if db_illegal:
                    dbSeq_str = ''
                    site_tag_db = ''
                    site_peaknum_db = ''
                    site_score_db = ''
                    pep_scoremean_db = ''
                    pep_scoresum_db = ''
                    ifEvident_db = ''
                try:
                    fout.write(f"{title}\t{dbSeq_str}\t{''.join(list(map(str, site_tag_db)))}\t{','.join(list(map(str, site_peaknum_db)))}\t{','.join(list(map(str, site_score_db)))}\t{pep_scoremean_db}\t{pep_scoresum_db}\t{ifEvident_db}\t{False}\n")
                except:
                    print(site_tag_db)
                    print(''.join(list(map(str, site_tag_db))))
                    print(site_score_db)
                    print(''.join(list(map(str, site_score_db))))
                    assert False
            fout.close()
            title_to_db_df = pd.read_csv(os.path.dirname(pfind_file) + "\\title_to_db_info.res", sep="\t")
        else:
            print("loading title_to_db_info.res file...")
            title_to_db_df = pd.read_csv(os.path.dirname(pfind_file) + "\\title_to_db_info.res", sep="\t")
    elif software_tool == "MSFragger":
        if not os.path.exists(os.path.dirname(pfind_file) + "\\title_to_db_info.res"):
            fout = open(os.path.dirname(pfind_file) + "\\title_to_db_info.res", "a+")
            fout.write("title\tdbSeq\tifFind_db\n")
            dbsearch_fin = open(pfind_file, 'r')
            print("reading mgf files ...")
            db_psm_list = []
            print(f"reading {pfind_file}...")
            with open(pfind_file, 'r') as f:
                lines = f.readlines()
                assert lines[0].startswith("Spectrum")
                lines = lines[1:]
                for line in tqdm(lines):
                    line = line.strip().split("\t")
                    title = line[0]
                    db_seq = get_MSFraggerSeq(line)
                    db_psm_list.append((title, db_seq))
            for i in range(len(db_psm_list)):
                title = db_psm_list[i][0]
                dbSeq_str = db_psm_list[i][1]
                fout.write(f"{title}\t{dbSeq_str}\t{False}\n")
            fout.close()
            title_to_db_df = pd.read_csv(os.path.dirname(pfind_file) + "\\title_to_db_info.res", sep="\t")
        else:
            print("loading title_to_db_info.res file...")
            title_to_db_df = pd.read_csv(os.path.dirname(pfind_file) + "\\title_to_db_info.res", sep="\t")
    elif software_tool == "MSGF":
        if not os.path.exists(os.path.dirname(pfind_file) + "\\title_to_db_info.res"):
            fout = open(os.path.dirname(pfind_file) + "\\title_to_db_info.res", "a+")
            fout.write("title\tdbSeq\tifFind_db\n")
            dbsearch_fin = open(pfind_file, 'r')
            print("reading mgf files ...")
            db_psm_list = []
            print(f"reading {pfind_file}...")
            with open(pfind_file, 'r') as f:
                lines = f.readlines()
                assert lines[0].startswith("#SpecFile")
                lines = lines[1:]
                for line in tqdm(lines):
                    line = line.strip().split("\t")
                    title = line[-1] #ScanIndex
                    db_seq = get_MSGFSeq(line)
                    db_psm_list.append((title, db_seq))
            for i in range(len(db_psm_list)):
                title = db_psm_list[i][0]
                dbSeq_str = db_psm_list[i][1]
                fout.write(f"{title}\t{dbSeq_str}\t{False}\n")
            fout.close()
            title_to_db_df = pd.read_csv(os.path.dirname(pfind_file) + "\\title_to_db_info.res", sep="\t")
        else:
            print("loading title_to_db_info.res file...")
            title_to_db_df = pd.read_csv(os.path.dirname(pfind_file) + "\\title_to_db_info.res", sep="\t")


    if not os.path.exists(os.path.dirname(pfind_file) + "\\fasta_coverage_db_info.res"):
        seq_str_set = set(title_to_db_df["dbSeq"])
        top1_res_dict = mapped_function_single(seq_str_set, fasta_seq, fasta_seq_first_4aa_index)
        title_to_db_df["ifFind_db"] = title_to_db_df["dbSeq"].apply(
            lambda x: top1_res_dict[x][1] if x == x else False)
        title_to_db_df["FindLocation_db"] = title_to_db_df["dbSeq"].apply(
            lambda x: top1_res_dict[x][3] if x == x else "")
        top1_res_dict_decoy = mapped_function_single(seq_str_set, fasta_seq_decoy, fasta_seq_first_4aa_index_decoy)
        title_to_db_df["ifFindDecoy_db"] = title_to_db_df["dbSeq"].apply(
            lambda x: top1_res_dict_decoy[x][2] if x == x else False)
        title_to_db_df["FindLocationDecoy_db"] = title_to_db_df["dbSeq"].apply(
            lambda x: top1_res_dict_decoy[x][4] if x == x else "")
        seq_to_unique_dict, fasta_coverage_db_dict, db_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq, title_to_db_df, "db", False, 0.0)
        title_to_db_df["Seq_Unique_db"] = title_to_db_df["dbSeq"].apply(
            lambda x: seq_to_unique_dict[x] if x == x else "")
        title_to_db_df.to_csv(os.path.dirname(pfind_file) + "\\title_to_db_info.res", sep="\t", index=False)
        write_dict_to_csv(fasta_coverage_db_dict, os.path.dirname(pfind_file) + "\\fasta_coverage_db_info.res",
                          fasta_seq, None, db_proteinName_to_uniqueSeq_dict)
    else:
        fasta_coverage_db_dict, db_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            os.path.dirname(pfind_file) + "\\fasta_coverage_db_info.res")

    if software_tool == "pFind":
        if not os.path.exists(os.path.dirname(pfind_file) + "\\fasta_coverage_Evidentdb_info.res"):
            _, fasta_coverage_Evidentdb_dict, Evidentdb_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
                fasta_seq, title_to_db_df, "db", True, 0.0)
            write_dict_to_csv(fasta_coverage_Evidentdb_dict,
                              os.path.dirname(pfind_file) + "\\fasta_coverage_Evidentdb_info.res", fasta_seq, None,
                              Evidentdb_proteinName_to_uniqueSeq_dict)
        else:
            fasta_coverage_Evidentdb_dict, Evidentdb_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                os.path.dirname(pfind_file) + "\\fasta_coverage_Evidentdb_info.res")
    else:
        pass
    
    ##############################################################################################
    #肽段
    if software_tool == "pFind":
        evident_spec_num_db = 0
        evident_find_spec_num_db = 0
        NonEvident_spec_num_db = 0
        NonEvident_find_spec_num_db = 0
        dbSeq_to_info_dict = {}
        title_to_db_df = pd.read_csv(os.path.dirname(pfind_file) + "\\title_to_db_info.res", sep="\t")
        for index, row in title_to_db_df.iterrows():
            if index % 1000 == 0:
                print(f"\r", index, "/", len(title_to_db_df), end="")
            title, db_seq, ifEvident, ifFind = row["title"], row["dbSeq"], row["ifEvident_db"], row["ifFind_db"]
            if ifEvident:
                evident_spec_num_db += 1
                if ifFind:
                    evident_find_spec_num_db += 1
            else:
                NonEvident_spec_num_db += 1
                if ifFind:
                    NonEvident_find_spec_num_db += 1
            if db_seq == db_seq:  # 不为Na
                if db_seq not in dbSeq_to_info_dict:
                    dbSeq_to_info_dict[db_seq] = [ifEvident, ifFind, [(title, ifEvident, ifFind)]]
                else:
                    ifEvident_new = dbSeq_to_info_dict[db_seq][0] | ifEvident
                    ifFind_new = dbSeq_to_info_dict[db_seq][1] | ifFind
                    dbSeq_to_info_dict[db_seq] = [ifEvident_new, ifFind_new,
                                                      dbSeq_to_info_dict[db_seq][2] + [
                                                          (title, ifEvident, ifFind)]]

        dbSeq_info_fw = open(os.path.dirname(pfind_file) + "\\dbSeq_to_ifEvidentFind.res", "w")
        header = "dbSeq\tifEvident\tifFind\ttitles\n"
        dbSeq_info_fw.write(header)
        for key in dbSeq_to_info_dict:
            string = f"{key}\t{dbSeq_to_info_dict[key][0]}\t{dbSeq_to_info_dict[key][1]}\t{dbSeq_to_info_dict[key][2]}\n"
            dbSeq_info_fw.write(string)
        dbSeq_info_fw.flush()
        dbSeq_info_fw.close()

        intsection_info_fout = open(os.path.dirname(pfind_file) + "\\info.res", "a+")
        total_spec_num = len(title_to_db_df)
        string = f"total_spec_num\t{len(title_to_db_df)}\n"
        intsection_info_fout.write(string)
        string = f"total_find_spec_num\t{title_to_db_df[title_to_db_df['ifFind_db'] == True].shape[0]}\n"
        intsection_info_fout.write(string)
        string = f"dbSeq_evident_spec_num\t{evident_spec_num_db}\n"
        intsection_info_fout.write(string)
        string = f"dbSeq_evident_find_spec_num\t{evident_find_spec_num_db}\n"
        intsection_info_fout.write(string)
        string = f"dbSeq_NonEvident_spec_num\t{NonEvident_spec_num_db}\n"
        intsection_info_fout.write(string)
        string = f"dbSeq_NonEvident_find_spec_num\t{NonEvident_find_spec_num_db}\n"
        intsection_info_fout.write(string)
        string = f"dbSeq_find_spec_num\t{evident_find_spec_num_db + NonEvident_find_spec_num_db}\n"
        intsection_info_fout.write(string)
        assert evident_find_spec_num_db + NonEvident_find_spec_num_db == title_to_db_df[title_to_db_df['ifFind_db'] == True].shape[0], f"{evident_find_spec_num_db} + {NonEvident_find_spec_num_db} != {title_to_db_df[title_to_db_df['ifFind_db'] == True].shape[0]}"

        dbSeq_df = pd.read_csv(os.path.dirname(pfind_file) + "\\dbSeq_to_ifEvidentFind.res", sep="\t")
        dbSeq_find_set = set(dbSeq_df[dbSeq_df["ifFind"] == True]["dbSeq"])
        dbSeq_find_evident_set = set(dbSeq_df[(dbSeq_df["ifFind"] == True) & (dbSeq_df["ifEvident"] == True)]["dbSeq"])
        string = f"dbSeq_find_num\t{len(dbSeq_find_set)}\n"
        intsection_info_fout.write(string)
        string = f"dbSeq_find_evident_num\t{len(dbSeq_find_evident_set)}\n"
        intsection_info_fout.write(string)
    elif software_tool == "MSFragger" or software_tool == "MSGF":
        dbSeq_to_info_dict = {}
        title_to_db_df = pd.read_csv(os.path.dirname(pfind_file) + "\\title_to_db_info.res", sep="\t")
        for index, row in title_to_db_df.iterrows():
            if index % 1000 == 0:
                print(f"\r", index, "/", len(title_to_db_df), end="")
            title, db_seq, ifFind = row["title"], row["dbSeq"], row["ifFind_db"]
            if db_seq == db_seq:  # 不为Na
                if db_seq not in dbSeq_to_info_dict:
                    dbSeq_to_info_dict[db_seq] = [ifFind, [(title, ifFind)]]
                else:
                    ifFind_new = dbSeq_to_info_dict[db_seq][0] | ifFind
                    dbSeq_to_info_dict[db_seq] = [ifFind_new,
                                                  dbSeq_to_info_dict[db_seq][1] + [
                                                      (title, ifFind)]]

        dbSeq_info_fw = open(os.path.dirname(pfind_file) + "\\dbSeq_to_ifEvidentFind.res", "w")
        header = "dbSeq\tifFind\ttitles\n"
        dbSeq_info_fw.write(header)
        for key in dbSeq_to_info_dict:
            string = f"{key}\t{dbSeq_to_info_dict[key][0]}\t{dbSeq_to_info_dict[key][1]}\n"
            dbSeq_info_fw.write(string)
        dbSeq_info_fw.flush()
        dbSeq_info_fw.close()

        intsection_info_fout = open(os.path.dirname(pfind_file) + "\\info.res", "a+")
        total_spec_num = len(title_to_db_df)
        string = f"total_spec_num\t{len(title_to_db_df)}\n"
        intsection_info_fout.write(string)
        string = f"total_find_spec_num\t{title_to_db_df[title_to_db_df['ifFind_db'] == True].shape[0]}\n"
        intsection_info_fout.write(string)

        dbSeq_df = pd.read_csv(os.path.dirname(pfind_file) + "\\dbSeq_to_ifEvidentFind.res", sep="\t")
        dbSeq_find_set = set(dbSeq_df[dbSeq_df["ifFind"] == True]["dbSeq"])
        string = f"dbSeq_find_num\t{len(dbSeq_find_set)}\n"
        intsection_info_fout.write(string)

    ################################################################
    #谱图
    if software_tool == "pFind":
        db_find_spec_set = set(title_to_db_df[title_to_db_df["ifFind_db"] == True]["title"])
        db_find_evident_spec_set = set(title_to_db_df[(title_to_db_df["ifFind_db"] == True) & (title_to_db_df["ifEvident_db"] == True)]["title"])
        string = f"db_find_spec_num\t{len(db_find_spec_set)}\n"
        intsection_info_fout.write(string)
        string = f"db_find_evident_spec_num\t{len(db_find_evident_spec_set)}\n"
        intsection_info_fout.write(string)
        intsection_info_fout.flush()
    elif software_tool == "MSFragger" or software_tool == "MSGF":
        db_find_spec_set = set(title_to_db_df[title_to_db_df["ifFind_db"] == True]["title"])
        string = f"db_find_spec_num\t{len(db_find_spec_set)}\n"
        intsection_info_fout.write(string)
        intsection_info_fout.flush()




