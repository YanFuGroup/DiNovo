import os
import config
import pandas as pd
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
from common_function import build_fasta_dict_function, build_index_dict_function, generate_reverse_fasta, build_mgf_location_function, transfer_str_to_list_seq, if_evident_function, parallel_get_confident_info,\
    mapped_function_single, get_fasta_coverage_np_single, write_dict_to_csv, read_dict_from_csv, get_confident_info,\
    plot_aa_intsection_venn3, plot_aa_intsection_venn2, merge_fasta_coverage_dict, merge_proteinName_to_uniqueSeq,\
    generate_try_lys_seq_from_RseqAndMatchtype, transfer_mod_function, find_location_in_fasta_single, find_location_in_fasta_single,\
    plot_pep_venn2,calculate_mass

fasta_file = config.fasta_file
software_tool = "MSGF"
pfind_folder = r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\MSGF"
try_pfind_file = pfind_folder + r"\lysC\MSGF_res[23charge].tsv"#pFind-Filtered_trypsin[extract][v0315].spectra"  # r"\pFind-Filtered_trypsin.spectra"#pFind-Filtered_LysC[extract][v0315]_trypsin
lys_pfind_file = pfind_folder + r"\lysN\MSGF_res[23charge].tsv"#pFind-Filtered_lysargiNase[extract][v0315].spectra"  # r"\pFind-Filtered_lysargiNase.spectra"#pFind-Filtered_LysN[extract][v0315]_lysargiNase
fig_name1 = "LysC"
fig_name2 = "LysN"
massFilter1 = 0
massFilter2 = 0

if __name__ == "__main__":
    #input
    # ==============================================================================
    fasta_seq = build_fasta_dict_function(fasta_file)

    if not os.path.exists(os.path.dirname(try_pfind_file) + f"\\fasta_coverage_db_info[{massFilter1}Da].res"):
        assert False
    else:
        try_fasta_coverage_db_dict, try_db_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            os.path.dirname(try_pfind_file)+ f"\\fasta_coverage_db_info[{massFilter1}Da].res")

    if software_tool == "pFind":
        if not os.path.exists(os.path.dirname(try_pfind_file) + f"\\fasta_coverage_Evidentdb_info[{massFilter1}Da].res"):
            assert False
        else:
            try_fasta_coverage_Evidentdb_dict, try_Evidentdb_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                os.path.dirname(try_pfind_file) + f"\\fasta_coverage_Evidentdb_info[{massFilter1}Da].res")

    if not os.path.exists(os.path.dirname(lys_pfind_file) + f"\\fasta_coverage_db_info[{massFilter2}Da].res"):
        assert False
    else:
        lys_fasta_coverage_db_dict, lys_db_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
            os.path.dirname(lys_pfind_file)+ f"\\fasta_coverage_db_info[{massFilter2}Da].res")

    if software_tool == "pFind":
        if not os.path.exists(os.path.dirname(lys_pfind_file) + f"\\fasta_coverage_Evidentdb_info[{massFilter2}Da].res"):
            assert False
        else:
            lys_fasta_coverage_Evidentdb_dict, lys_Evidentdb_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                os.path.dirname(lys_pfind_file) + f"\\fasta_coverage_Evidentdb_info[{massFilter2}Da].res")
    
    # ==============================================================================
    #analyse
    Union_fasta_coverage_db_dict = merge_fasta_coverage_dict([try_fasta_coverage_db_dict, lys_fasta_coverage_db_dict])
    Union_db_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
        [try_db_proteinName_to_uniqueSeq_dict, lys_db_proteinName_to_uniqueSeq_dict])
    write_dict_to_csv(Union_fasta_coverage_db_dict, pfind_folder + f"\\Union_fasta_coverage_db_info[{massFilter1}DaU{massFilter2}Da].res", fasta_seq, None,
                      Union_db_proteinName_to_uniqueSeq_dict)
    if software_tool == "pFind":
        Union_fasta_coverage_Evidentdb_dict = merge_fasta_coverage_dict(
            [try_fasta_coverage_Evidentdb_dict, lys_fasta_coverage_Evidentdb_dict])
        Union_Evidentdb_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
            [try_Evidentdb_proteinName_to_uniqueSeq_dict, lys_Evidentdb_proteinName_to_uniqueSeq_dict])
        write_dict_to_csv(Union_fasta_coverage_Evidentdb_dict, pfind_folder + f"\\Union_fasta_coverage_Evidentdb_info[{massFilter1}DaU{massFilter2}Da].res",
                          fasta_seq, None, Union_Evidentdb_proteinName_to_uniqueSeq_dict)

    #肽段和谱图水平
    try_dbSeq_df = pd.read_csv(os.path.dirname(try_pfind_file) + f"\\dbSeq_to_ifEvidentFind.res",sep="\t")
    if "Pep_mass" not in try_dbSeq_df.columns:
        try_dbSeq_df["Pep_mass"] = try_dbSeq_df["dbSeq"].apply(lambda x: calculate_mass(x, if_delete_first_aa=False))
    try_dbSeq_find_set = set(try_dbSeq_df[(try_dbSeq_df["ifFind"] == True) & (try_dbSeq_df["Pep_mass"] >= massFilter1)]["dbSeq"])

    lys_dbSeq_df = pd.read_csv(os.path.dirname(lys_pfind_file) + f"\\dbSeq_to_ifEvidentFind.res",sep="\t")
    if "Pep_mass" not in lys_dbSeq_df.columns:
        lys_dbSeq_df["Pep_mass"] = lys_dbSeq_df["dbSeq"].apply(lambda x: calculate_mass(x, if_delete_first_aa=False))
    lys_dbSeq_find_set = set(lys_dbSeq_df[(lys_dbSeq_df["ifFind"] == True) & (lys_dbSeq_df["Pep_mass"] >= massFilter2)]["dbSeq"])

    plot_pep_venn2(try_dbSeq_find_set, lys_dbSeq_find_set, pfind_folder + f"\\dbNei_pep_intsection_find[{massFilter1}DaU{massFilter2}Da].png", fig_name1, fig_name2)
    if software_tool == "pFind":
        try_dbSeq_find_Evident_set = set(try_dbSeq_df[(try_dbSeq_df["ifEvident"] == True) & (try_dbSeq_df["ifFind"] == True) & (try_dbSeq_df["Pep_mass"] >= massFilter1)]["dbSeq"])
        lys_dbSeq_find_Evident_set = set(lys_dbSeq_df[(lys_dbSeq_df["ifEvident"] == True) & (lys_dbSeq_df["ifFind"] == True) & (lys_dbSeq_df["Pep_mass"] >= massFilter2)]["dbSeq"])
        plot_pep_venn2(try_dbSeq_find_Evident_set, lys_dbSeq_find_Evident_set, pfind_folder + f"\\dbNei_pep_intsection_find_Evident[{massFilter1}DaU{massFilter2}Da].png", fig_name1, fig_name2)

    if software_tool == "pFind":
        info_fw = open(pfind_folder + f"\\pep_info[{massFilter1}DaU{massFilter2}Da].res", 'w')
        info_fw.write("try_find_seq_num=" + str(len(try_dbSeq_find_set)) + "\n")
        info_fw.write("lys_find_seq_num=" + str(len(lys_dbSeq_find_set)) + "\n")
        info_fw.write("try_find_Evident_seq_num=" + str(len(try_dbSeq_find_Evident_set)) + "\n")
        info_fw.write("lys_find_Evident_seq_num=" + str(len(lys_dbSeq_find_Evident_set)) + "\n")
        info_fw.write("Sum_trylys_find_seq_num=" + str(len(try_dbSeq_find_set) + len(lys_dbSeq_find_set)) + "\n")
        info_fw.write("Sum_trylys_find_Evident_seq_num=" + str(len(try_dbSeq_find_Evident_set) + len(lys_dbSeq_find_Evident_set)) + "\n")
        info_fw.write("tryUlys_find_seq_num=" + str(len(try_dbSeq_find_set | lys_dbSeq_find_set)) + "\n")
        info_fw.write("tryUlys_find_Evident_seq_num=" + str(len(try_dbSeq_find_Evident_set | lys_dbSeq_find_Evident_set)) + "\n")
        info_fw.write("try&lys_find_seq_num=" + str(len(try_dbSeq_find_set & lys_dbSeq_find_set)) + "\n")
        info_fw.write("try&lys_find_Evident_seq_num=" + str(len(try_dbSeq_find_Evident_set & lys_dbSeq_find_Evident_set)) + "\n")
    elif software_tool == "MSFragger" or software_tool == "MSGF":
        info_fw = open(pfind_folder + f"\\pep_info[{massFilter1}DaU{massFilter2}Da].res", 'w')
        info_fw.write("try_find_seq_num=" + str(len(try_dbSeq_find_set)) + "\n")
        info_fw.write("lys_find_seq_num=" + str(len(lys_dbSeq_find_set)) + "\n")
        info_fw.write("Sum_trylys_find_seq_num=" + str(len(try_dbSeq_find_set) + len(lys_dbSeq_find_set)) + "\n")
        info_fw.write("tryUlys_find_seq_num=" + str(len(try_dbSeq_find_set | lys_dbSeq_find_set)) + "\n")
        info_fw.write("try&lys_find_seq_num=" + str(len(try_dbSeq_find_set & lys_dbSeq_find_set)) + "\n")

    #谱图水平
    if software_tool == "pFind":
        try_title_to_db_df = pd.read_csv(os.path.dirname(try_pfind_file) + "\\title_to_db_info.res",sep="\t")
        if "Pep_mass_db" not in try_title_to_db_df.columns:
            try_title_to_db_df["Pep_mass_db"] = try_title_to_db_df["dbSeq"].apply(lambda x: calculate_mass(x, if_delete_first_aa=False))
        try_db_find_spec_set = set(try_title_to_db_df[(try_title_to_db_df["ifFind_db"] == True) & (try_title_to_db_df["Pep_mass_db"] >= massFilter1)]["title"])
        try_db_find_Evident_spec_set = set(try_title_to_db_df[(try_title_to_db_df["ifEvident_db"] == True) & (try_title_to_db_df["ifFind_db"] == True) & (try_title_to_db_df["Pep_mass_db"] >= massFilter1)]["title"])
        lys_title_to_db_df = pd.read_csv(os.path.dirname(lys_pfind_file) + "\\title_to_db_info.res",sep="\t")
        if "Pep_mass_db" not in lys_title_to_db_df.columns:
            lys_title_to_db_df["Pep_mass_db"] = lys_title_to_db_df["dbSeq"].apply(lambda x: calculate_mass(x, if_delete_first_aa=False))
        lys_db_find_spec_set = set(lys_title_to_db_df[(lys_title_to_db_df["ifFind_db"] == True) & (lys_title_to_db_df["Pep_mass_db"] >= massFilter2)]["title"])
        lys_db_find_Evident_spec_set = set(lys_title_to_db_df[(lys_title_to_db_df["ifEvident_db"] == True) & (lys_title_to_db_df["ifFind_db"] == True) & (lys_title_to_db_df["Pep_mass_db"] >= massFilter2)]["title"])

        trylysSum_db_find_spec_set = try_db_find_spec_set | lys_db_find_spec_set
        trylysSum_db_find_Evident_spec_set = try_db_find_Evident_spec_set | lys_db_find_Evident_spec_set
        info_fw.write("try_find_spec_num=" + str(len(try_db_find_spec_set)) + "\n")
        info_fw.write("try_find_Evident_spec_num=" + str(len(try_db_find_Evident_spec_set)) + "\n")
        info_fw.write("lys_find_spec_num=" + str(len(lys_db_find_spec_set)) + "\n")
        info_fw.write("lys_find_Evident_spec_num=" + str(len(lys_db_find_Evident_spec_set)) + "\n")
        info_fw.write("Sum_trylys_find_spec_num=" + str(len(trylysSum_db_find_spec_set)) + "\n")
        info_fw.write("Sum_trylys_find_Evident_spec_num=" + str(len(trylysSum_db_find_Evident_spec_set)) + "\n")
    else:
        try_title_to_db_df = pd.read_csv(os.path.dirname(try_pfind_file) + "\\title_to_db_info.res",sep="\t")
        if "Pep_mass_db" not in try_title_to_db_df.columns:
            try_title_to_db_df["Pep_mass_db"] = try_title_to_db_df["dbSeq"].apply(lambda x: calculate_mass(x, if_delete_first_aa=False))
        try_db_find_spec_set = set(try_title_to_db_df[(try_title_to_db_df["ifFind_db"] == True) & (try_title_to_db_df["Pep_mass_db"] >= massFilter1)]["title"])
        lys_title_to_db_df = pd.read_csv(os.path.dirname(lys_pfind_file) + "\\title_to_db_info.res",sep="\t")
        if "Pep_mass_db" not in lys_title_to_db_df.columns:
            lys_title_to_db_df["Pep_mass_db"] = lys_title_to_db_df["dbSeq"].apply(lambda x: calculate_mass(x, if_delete_first_aa=False))
        lys_db_find_spec_set = set(lys_title_to_db_df[(lys_title_to_db_df["ifFind_db"] == True) & (lys_title_to_db_df["Pep_mass_db"] >= massFilter2)]["title"])

        trylysSum_db_find_spec_set = try_db_find_spec_set | lys_db_find_spec_set
        info_fw.write("try_find_spec_num=" + str(len(try_db_find_spec_set)) + "\n")
        info_fw.write("lys_find_spec_num=" + str(len(lys_db_find_spec_set)) + "\n")
        info_fw.write("Sum_trylys_find_spec_num=" + str(len(trylysSum_db_find_spec_set)) + "\n")






