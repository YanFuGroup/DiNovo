import os
import config
import pandas as pd
from common_function import build_fasta_dict_function, build_index_dict_function, generate_reverse_fasta, build_mgf_location_function, transfer_str_to_list_seq, if_evident_function, parallel_get_confident_info,\
    mapped_function_single, get_fasta_coverage_np_single, write_dict_to_csv, read_dict_from_csv, get_confident_info,\
    plot_aa_intsection_venn3, plot_aa_intsection_venn2, merge_fasta_coverage_dict, merge_proteinName_to_uniqueSeq,\
    generate_try_lys_seq_from_RseqAndMatchtype, transfer_mod_function, find_location_in_fasta_single, find_location_in_fasta_single,\
    plot_pep_venn2,calculate_mass

fasta_file = config.fasta_file
fig_name1 = "Try"#"LysC"
fig_name2 = "Lys"#"LysN"
# massFilter1_list = [746,795,810,770,770]
# massFilter2_list = [777,822,857,775,775]
massFilter1_list = [748]
massFilter2_list = [764]
denovo_folder_list = [
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pnovo_output",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\peaks_deepnovo_output",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pointnovo_output",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\casanovo_output\9species_model",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\casanovo_output\massiveKB_model"
    #  r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\PrimeNovo_output",
    r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_combine_7peak_10ppm_v0.8\MirrorNovo_output\single"
]
try_denovo_folder_list = [
    denovo_folder_list[0] + r"\try",
    # denovo_folder_list[1] + r"\try\23charge",
    # denovo_folder_list[2] + r"\lysC\23charge",
    # denovo_folder_list[3] + r"\lysC\23charge",
    # denovo_folder_list[4] + r"\lysC\23charge"
]
lys_denovo_folder_list = [
    denovo_folder_list[0] + r"\lys",
    # denovo_folder_list[1] + r"\lys\23charge",
    # denovo_folder_list[2] + r"\lysN\23charge",
    # denovo_folder_list[3] + r"\lysN\23charge",
    # denovo_folder_list[4] + r"\lysN\23charge"
]

if __name__ == "__main__":

    for i in range(len(massFilter1_list)):

        massFilter1 = massFilter1_list[i]
        massFilter2 = massFilter2_list[i]
        denovo_folder = denovo_folder_list[i]
        try_denovo_folder = try_denovo_folder_list[i]
        lys_denovo_folder = lys_denovo_folder_list[i]

        #input
        # ==============================================================================
        fasta_seq = build_fasta_dict_function(fasta_file)

        if not os.path.exists(try_denovo_folder + f"\\fasta_coverage_denovo_info[{massFilter1}Da].res"):
            assert False
        else:
            try_fasta_coverage_denovo_dict, try_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                try_denovo_folder+ f"\\fasta_coverage_denovo_info[{massFilter1}Da].res")

        if not os.path.exists(try_denovo_folder + f"\\fasta_coverage_Evidentdenovo_info[{massFilter1}Da].res"):
            assert False
        else:
            try_fasta_coverage_Evidentdenovo_dict, try_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                try_denovo_folder + f"\\fasta_coverage_Evidentdenovo_info[{massFilter1}Da].res")

        if not os.path.exists(lys_denovo_folder + f"\\fasta_coverage_denovo_info[{massFilter2}Da].res"):
            assert False
        else:
            lys_fasta_coverage_denovo_dict, lys_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                lys_denovo_folder+ f"\\fasta_coverage_denovo_info[{massFilter2}Da].res")

        if not os.path.exists(lys_denovo_folder + f"\\fasta_coverage_Evidentdenovo_info[{massFilter2}Da].res"):
            assert False
        else:
            lys_fasta_coverage_Evidentdenovo_dict, lys_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                lys_denovo_folder + f"\\fasta_coverage_Evidentdenovo_info[{massFilter2}Da].res")

        # ==============================================================================
        #analyse
        Union_fasta_coverage_denovo_dict = merge_fasta_coverage_dict([try_fasta_coverage_denovo_dict, lys_fasta_coverage_denovo_dict])
        Union_denovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
            [try_denovo_proteinName_to_uniqueSeq_dict, lys_denovo_proteinName_to_uniqueSeq_dict])
        write_dict_to_csv(Union_fasta_coverage_denovo_dict, denovo_folder + f"\\Union_fasta_coverage_denovo_info[{massFilter1}DaU{massFilter2}Da].res", fasta_seq, None,
                          Union_denovo_proteinName_to_uniqueSeq_dict)
        Union_fasta_coverage_Evidentdenovo_dict = merge_fasta_coverage_dict(
            [try_fasta_coverage_Evidentdenovo_dict, lys_fasta_coverage_Evidentdenovo_dict])
        Union_Evidentdenovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
            [try_Evidentdenovo_proteinName_to_uniqueSeq_dict, lys_Evidentdenovo_proteinName_to_uniqueSeq_dict])
        write_dict_to_csv(Union_fasta_coverage_Evidentdenovo_dict, denovo_folder + f"\\Union_fasta_coverage_Evidentdenovo_info[{massFilter1}DaU{massFilter2}Da].res",
                          fasta_seq, None, Union_Evidentdenovo_proteinName_to_uniqueSeq_dict)

        #肽段
        # ==============================================================================
        try_denovoSeq_df = pd.read_csv(try_denovo_folder + "\\denovoSeq_to_ifEvidentFind.res",sep="\t")
        try_denovoSeq_find_set = set(try_denovoSeq_df[(try_denovoSeq_df["ifFind"] == True) & (try_denovoSeq_df["ifFindDecoy"] == False) & (try_denovoSeq_df["Pep_mass"] >= massFilter1)]["denovoSeq"])
        try_denovoSeq_find_Evident_set = set(try_denovoSeq_df[(try_denovoSeq_df["ifEvident"] == True) & (try_denovoSeq_df["ifFind"] == True) & (try_denovoSeq_df["ifFindDecoy"] == False) & (try_denovoSeq_df["Pep_mass"] >= massFilter1)]["denovoSeq"])
        lys_denovoSeq_df = pd.read_csv(lys_denovo_folder + "\\denovoSeq_to_ifEvidentFind.res",sep="\t")
        lys_denovoSeq_find_set = set(lys_denovoSeq_df[(lys_denovoSeq_df["ifFind"] == True) & (lys_denovoSeq_df["ifFindDecoy"] == False) & (lys_denovoSeq_df["Pep_mass"] >= massFilter2)]["denovoSeq"])
        lys_denovoSeq_find_Evident_set = set(lys_denovoSeq_df[(lys_denovoSeq_df["ifEvident"] == True) & (lys_denovoSeq_df["ifFind"] == True) & (lys_denovoSeq_df["ifFindDecoy"] == False) & (lys_denovoSeq_df["Pep_mass"] >= massFilter2)]["denovoSeq"])
        plot_pep_venn2(try_denovoSeq_find_set, lys_denovoSeq_find_set, denovo_folder + f"\\denovoNei_pep_intsection_find[{massFilter1}DaU{massFilter2}Da].png", fig_name1, fig_name2)
        plot_pep_venn2(try_denovoSeq_find_Evident_set, lys_denovoSeq_find_Evident_set, denovo_folder + f"\\denovoNei_pep_intsection_find_Evident[{massFilter1}DaU{massFilter2}Da].png", fig_name1, fig_name2)

        # print(denovo_folder)
        # try_denovoSeq_df = try_denovoSeq_df[try_denovoSeq_df["Pep_mass"] >= massFilter1]
        # lys_denovoSeq_df = lys_denovoSeq_df[lys_denovoSeq_df["Pep_mass"] >= massFilter2]
        # try_denovoSeq_find_set = set(try_denovoSeq_df[(try_denovoSeq_df["ifFind"] == True) & (try_denovoSeq_df["ifFindDecoy"] == False)]["denovoSeq"])
        # lys_denovoSeq_find_set = set(lys_denovoSeq_df[(lys_denovoSeq_df["ifFind"] == True) & (lys_denovoSeq_df["ifFindDecoy"] == False)]["denovoSeq"])
        # print(len(try_denovoSeq_find_set),len(lys_denovoSeq_find_set))
        # input()

        info_fw = open(denovo_folder + f"\\pep_info[{massFilter1}DaU{massFilter2}Da].res", 'w')
        info_fw.write("try_find_seq_num=" + str(len(try_denovoSeq_find_set)) + "\n")
        info_fw.write("lys_find_seq_num=" + str(len(lys_denovoSeq_find_set)) + "\n")
        info_fw.write("try_find_Evident_seq_num=" + str(len(try_denovoSeq_find_Evident_set)) + "\n")
        info_fw.write("lys_find_Evident_seq_num=" + str(len(lys_denovoSeq_find_Evident_set)) + "\n")
        info_fw.write("Sum_trylys_find_seq_num=" + str(len(try_denovoSeq_find_set) + len(lys_denovoSeq_find_set)) + "\n")
        info_fw.write("Sum_trylys_find_Evident_seq_num=" + str(len(try_denovoSeq_find_Evident_set) + len(lys_denovoSeq_find_Evident_set)) + "\n")
        info_fw.write("tryUlys_find_seq_num=" + str(len(try_denovoSeq_find_set | lys_denovoSeq_find_set)) + "\n")
        info_fw.write("tryUlys_find_Evident_seq_num=" + str(len(try_denovoSeq_find_Evident_set | lys_denovoSeq_find_Evident_set)) + "\n")
        info_fw.write("try&lys_find_seq_num=" + str(len(try_denovoSeq_find_set & lys_denovoSeq_find_set)) + "\n")
        info_fw.write("try&lys_find_Evident_seq_num=" + str(len(try_denovoSeq_find_Evident_set & lys_denovoSeq_find_Evident_set)) + "\n")

        #谱图水平
        try_title_to_denovo_df = pd.read_csv(try_denovo_folder + f"\\title_to_denovo_info.res",sep="\t")
        if "Pep_mass_denovo" not in try_title_to_denovo_df.columns:
            try_title_to_denovo_df["Pep_mass_denovo"] = try_title_to_denovo_df.apply(lambda x: calculate_mass(x["denovoSeq"],if_delete_first_aa=False),axis=1)
        try_denovo_find_spec_set = set(try_title_to_denovo_df[(try_title_to_denovo_df["ifFind_denovo"] == True) & (try_title_to_denovo_df["ifFindDecoy_denovo"] == False) & (try_title_to_denovo_df["Pep_mass_denovo"] >= massFilter1)]["title"])
        try_denovo_find_Evident_spec_set = set(try_title_to_denovo_df[(try_title_to_denovo_df["ifEvident_denovo"] == True) & (try_title_to_denovo_df["ifFind_denovo"] == True) & (try_title_to_denovo_df["ifFindDecoy_denovo"] == False) & (try_title_to_denovo_df["Pep_mass_denovo"] >= massFilter1)]["title"])
        lys_title_to_denovo_df = pd.read_csv(lys_denovo_folder + "\\title_to_denovo_info.res",sep="\t")
        if "Pep_mass_denovo" not in lys_title_to_denovo_df.columns:
            lys_title_to_denovo_df["Pep_mass_denovo"] = lys_title_to_denovo_df.apply(lambda x: calculate_mass(x["denovoSeq"],if_delete_first_aa=False),axis=1)
        lys_denovo_find_spec_set = set(lys_title_to_denovo_df[(lys_title_to_denovo_df["ifFind_denovo"] == True) & (lys_title_to_denovo_df["ifFindDecoy_denovo"] == False) & (lys_title_to_denovo_df["Pep_mass_denovo"] >= massFilter2)]["title"])
        lys_denovo_find_Evident_spec_set = set(lys_title_to_denovo_df[(lys_title_to_denovo_df["ifEvident_denovo"] == True) & (lys_title_to_denovo_df["ifFind_denovo"] == True) & (lys_title_to_denovo_df["ifFindDecoy_denovo"] == False) & (lys_title_to_denovo_df["Pep_mass_denovo"] >= massFilter2)]["title"])
        trylysSum_denovo_find_spec_set = try_denovo_find_spec_set | lys_denovo_find_spec_set
        trylysSum_denovo_find_Evident_spec_set = try_denovo_find_Evident_spec_set | lys_denovo_find_Evident_spec_set
        info_fw.write("try_find_spec_num=" + str(len(try_denovo_find_spec_set)) + "\n")
        info_fw.write("try_find_Evident_spec_num=" + str(len(try_denovo_find_Evident_spec_set)) + "\n")
        info_fw.write("lys_find_spec_num=" + str(len(lys_denovo_find_spec_set)) + "\n")
        info_fw.write("lys_find_Evident_spec_num=" + str(len(lys_denovo_find_Evident_spec_set)) + "\n")
        info_fw.write("Sum_trylys_find_spec_num=" + str(len(trylysSum_denovo_find_spec_set)) + "\n")
        info_fw.write("Sum_trylys_find_Evident_spec_num=" + str(len(trylysSum_denovo_find_Evident_spec_set)) + "\n")


