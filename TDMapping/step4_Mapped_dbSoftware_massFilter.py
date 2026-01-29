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
    mapped_function_single, get_fasta_coverage_np_single, write_dict_to_csv, read_dict_from_csv,calculate_mass

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
massFilter = 0
software_tool = "MSGF"
pfind_file = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MSGF\try\MSGF_res[23charge].tsv"#pFind-Filtered_trypsin[extract][v0315].spectra"  # r"\pFind-Filtered_trypsin.spectra"#pFind-Filtered_LysC[extract][v0315]_trypsin


def find_target_sequence(Title, pfind_location_dict, pfind_path):
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


if __name__ == "__main__":
    fasta_seq = build_fasta_dict_function(fasta_file)
    fasta_seq_first_4aa_index = build_index_dict_function(fasta_seq)
    fasta_seq_decoy = generate_reverse_fasta(fasta_seq)
    fasta_seq_first_4aa_index_decoy = build_index_dict_function(fasta_seq_decoy)

    if not os.path.exists(os.path.dirname(pfind_file) + "\\title_to_db_info.res"):
        assert False
    else:
        print(f"loading title_to_db_info[{massFilter}Da].res file...")
        title_to_db_df = pd.read_csv(os.path.dirname(pfind_file) + "\\title_to_db_info.res", sep="\t")
    if "Pep_mass_db" not in title_to_db_df.columns:
        title_to_db_df["Pep_mass_db"] = title_to_db_df["dbSeq"].apply(lambda x: calculate_mass(x, if_delete_first_aa=False))

    if not os.path.exists(os.path.dirname(pfind_file) + f"\\fasta_coverage_db_info[{massFilter}Da].res"):
        seq_to_unique_dict, fasta_coverage_db_dict, db_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
            fasta_seq, title_to_db_df, "db", False, massFilter)
        write_dict_to_csv(fasta_coverage_db_dict, os.path.dirname(pfind_file) + f"\\fasta_coverage_db_info[{massFilter}Da].res",
                          fasta_seq, None, db_proteinName_to_uniqueSeq_dict)
    else:
        fasta_coverage_db_dict, db_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            os.path.dirname(pfind_file) + f"\\fasta_coverage_db_info[{massFilter}Da].res")

    if software_tool == "pFind":
        if not os.path.exists(os.path.dirname(pfind_file) + f"\\fasta_coverage_Evidentdb_info[{massFilter}Da].res"):
            _, fasta_coverage_Evidentdb_dict, Evidentdb_proteinName_to_uniqueSeq_dict = get_fasta_coverage_np_single(
                fasta_seq, title_to_db_df, "db", True, massFilter)
            write_dict_to_csv(fasta_coverage_Evidentdb_dict,
                              os.path.dirname(pfind_file) + f"\\fasta_coverage_Evidentdb_info[{massFilter}Da].res", fasta_seq, None,
                              Evidentdb_proteinName_to_uniqueSeq_dict)
        else:
            fasta_coverage_Evidentdb_dict, Evidentdb_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                os.path.dirname(pfind_file) + f"\\fasta_coverage_Evidentdb_info[{massFilter}Da].res")
    
    ##############################################################################################
    #肽段
    if software_tool == "pFind":
        evident_spec_num_db = 0
        evident_find_spec_num_db = 0
        NonEvident_spec_num_db = 0
        NonEvident_find_spec_num_db = 0
        dbSeq_to_info_dict = {}
        for index, row in title_to_db_df.iterrows():
            if index % 1000 == 0:
                print(f"\r", index, "/", len(title_to_db_df), end="")
            title, db_seq, ifEvident, ifFind = row["title"], row["dbSeq"], row["ifEvident_db"], row["ifFind_db"]
            Pep_mass = row["Pep_mass_db"]
            if Pep_mass < massFilter:
                continue
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
        intsection_info_fout = open(os.path.dirname(pfind_file) + f"\\info[{massFilter}Da].res", "a+")
        total_spec_num = len(title_to_db_df)
        string = f"total_spec_num\t{len(title_to_db_df)}\n"
        intsection_info_fout.write(string)
        string = f"total_find_spec_num\t{title_to_db_df[(title_to_db_df['ifFind_db'] == True) & (title_to_db_df['Pep_mass_db'] >= massFilter)].shape[0]}\n"
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
        assert evident_find_spec_num_db + NonEvident_find_spec_num_db == title_to_db_df[(title_to_db_df['ifFind_db'] == True) & (title_to_db_df["Pep_mass_db"] >= massFilter)].shape[0], f"{evident_find_spec_num_db} + {NonEvident_find_spec_num_db} != {title_to_db_df[(title_to_db_df['ifFind_db'] == True) & (title_to_db_df['Pep_mass_db'] >= massFilter)].shape[0]}"

        dbSeq_df = pd.read_csv(os.path.dirname(pfind_file) + "\\dbSeq_to_ifEvidentFind.res", sep="\t")
        if "Pep_mass" not in dbSeq_df.columns:
            dbSeq_df["Pep_mass"] = dbSeq_df["dbSeq"].apply(lambda x: calculate_mass(x, if_delete_first_aa=False))
        dbSeq_find_set = set(dbSeq_df[(dbSeq_df["ifFind"] == True) & (dbSeq_df["Pep_mass"] >= massFilter)]["dbSeq"])
        dbSeq_find_evident_set = set(dbSeq_df[(dbSeq_df["ifFind"] == True) & (dbSeq_df["ifEvident"] == True) & (
                    dbSeq_df["Pep_mass"] >= massFilter)]["dbSeq"])
        string = f"dbSeq_find_num\t{len(dbSeq_find_set)}\n"
        intsection_info_fout.write(string)
        string = f"dbSeq_find_evident_num\t{len(dbSeq_find_evident_set)}\n"
        intsection_info_fout.write(string)

    elif software_tool == "MSFragger" or software_tool == "MSGF":
        dbSeq_to_info_dict = {}
        for index, row in title_to_db_df.iterrows():
            if index % 1000 == 0:
                print(f"\r", index, "/", len(title_to_db_df), end="")
            title, db_seq, ifFind = row["title"], row["dbSeq"], row["ifFind_db"]
            Pep_mass = row["Pep_mass_db"]
            if Pep_mass < massFilter:
                continue
            if db_seq == db_seq:  # 不为Na
                if db_seq not in dbSeq_to_info_dict:
                    dbSeq_to_info_dict[db_seq] = [ifFind, [(title, ifFind)]]
                else:
                    ifFind_new = dbSeq_to_info_dict[db_seq][0] | ifFind
                    dbSeq_to_info_dict[db_seq] = [ifFind_new,
                                                  dbSeq_to_info_dict[db_seq][1] + [(title, ifFind)]]
        intsection_info_fout = open(os.path.dirname(pfind_file) + f"\\info[{massFilter}Da].res", "a+")
        total_spec_num = len(title_to_db_df)
        string = f"total_spec_num\t{len(title_to_db_df)}\n"
        intsection_info_fout.write(string)
        string = f"total_find_spec_num\t{title_to_db_df[(title_to_db_df['ifFind_db'] == True) & (title_to_db_df['Pep_mass_db'] >= massFilter)].shape[0]}\n"
        intsection_info_fout.write(string)

        dbSeq_df = pd.read_csv(os.path.dirname(pfind_file) + "\\dbSeq_to_ifEvidentFind.res", sep="\t")
        if "Pep_mass" not in dbSeq_df.columns:
            dbSeq_df["Pep_mass"] = dbSeq_df["dbSeq"].apply(lambda x: calculate_mass(x, if_delete_first_aa=False))
        dbSeq_find_set = set(dbSeq_df[(dbSeq_df["ifFind"] == True) & (dbSeq_df["Pep_mass"] >= massFilter)]["dbSeq"])
        string = f"dbSeq_find_num\t{len(dbSeq_find_set)}\n"
        intsection_info_fout.write(string)

    ################################################################
    #谱图
    if software_tool == "pFind":
        db_find_spec_set = set(title_to_db_df[(title_to_db_df["ifFind_db"] == True) & (dbSeq_df["Pep_mass"] >= massFilter)]["title"])
        db_find_evident_spec_set = set(title_to_db_df[(title_to_db_df["ifFind_db"] == True) & (title_to_db_df["ifEvident_db"] == True) & (dbSeq_df["Pep_mass"] >= massFilter)]["title"])
        string = f"db_find_spec_num\t{len(db_find_spec_set)}\n"
        intsection_info_fout.write(string)
        string = f"db_find_evident_spec_num\t{len(db_find_evident_spec_set)}\n"
        intsection_info_fout.write(string)
        intsection_info_fout.flush()
    elif software_tool == "MSFragger" or software_tool == "MSGF":
        db_find_spec_set = set(title_to_db_df[(title_to_db_df["ifFind_db"] == True) & (dbSeq_df["Pep_mass"] >= massFilter)]["title"])
        string = f"db_find_spec_num\t{len(db_find_spec_set)}\n"
        intsection_info_fout.write(string)
        intsection_info_fout.flush()




