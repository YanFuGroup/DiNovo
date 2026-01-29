import os
import config
import pandas as pd
from common_function import build_fasta_dict_function, build_index_dict_function, generate_reverse_fasta, build_mgf_location_function, transfer_str_to_list_seq, if_evident_function, parallel_get_confident_info,\
    mapped_function_single, get_fasta_coverage_np_single, write_dict_to_csv, read_dict_from_csv, get_confident_info,\
    plot_aa_intsection_venn3, plot_aa_intsection_venn2, merge_fasta_coverage_dict, merge_proteinName_to_uniqueSeq,\
    generate_try_lys_seq_from_RseqAndMatchtype, transfer_mod_function, find_location_in_fasta_single, find_location_in_fasta_single,\
    plot_pep_venn2

fasta_file = config.fasta_file
fig_name1 = "LysC"
fig_name2 = "LysN"
denovo_folder_list = [
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pnovo_output",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\peaks_deepnovo_output",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\pointnovo_output",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\9species_model",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\casanovo_output\massiveKB_model",
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\GraphNovo_output"
    # r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\PrimeNovo_output"
    r"H:\Fugroup\pNovoR_svn\total_unify_data\new_data_conbine_and_single_protease\20240329unify_data\ALL\ALL_Yeast_lysClysN_7peak_10ppm_v0.8\MirrorNovo_output\single"
]
try_denovo_folder_list = [
    # denovo_folder_list[0] + r"\lysC\result",
    # denovo_folder_list[1] + r"\lysC\23charge",
    denovo_folder_list[0] + r"\lysC",
    # denovo_folder_list[3] + r"\lysC\23charge",
    # denovo_folder_list[4] + r"\lysC\23charge"
]
lys_denovo_folder_list = [
    # denovo_folder_list[0] + r"\lysN\result",
    # denovo_folder_list[1] + r"\lysN\23charge",
    denovo_folder_list[0] + r"\lysN",
    # denovo_folder_list[3] + r"\lysN\23charge",
    # denovo_folder_list[4] + r"\lysN\23charge"
]


if __name__ == "__main__":

    for i in range(len(try_denovo_folder_list)):
        try_denovo_folder = try_denovo_folder_list[i]
        lys_denovo_folder = lys_denovo_folder_list[i]
        denovo_folder = denovo_folder_list[i]
        print("preprocessing:", i)

        #input
        # ==============================================================================
        fasta_seq = build_fasta_dict_function(fasta_file)

        if not os.path.exists(try_denovo_folder + "\\fasta_coverage_denovo_info.res"):
            assert False
        else:
            try_fasta_coverage_denovo_dict, try_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                try_denovo_folder+ "\\fasta_coverage_denovo_info.res")

        if not os.path.exists(try_denovo_folder + "\\fasta_coverage_Evidentdenovo_info.res"):
            assert False
        else:
            try_fasta_coverage_Evidentdenovo_dict, try_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                try_denovo_folder + "\\fasta_coverage_Evidentdenovo_info.res")

        if not os.path.exists(lys_denovo_folder + "\\fasta_coverage_denovo_info.res"):
            assert False
        else:
            lys_fasta_coverage_denovo_dict, lys_denovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                lys_denovo_folder+ "\\fasta_coverage_denovo_info.res")

        if not os.path.exists(lys_denovo_folder + "\\fasta_coverage_Evidentdenovo_info.res"):
            assert False
        else:
            lys_fasta_coverage_Evidentdenovo_dict, lys_Evidentdenovo_proteinName_to_uniqueSeq_dict = read_dict_from_csv(
                lys_denovo_folder + "\\fasta_coverage_Evidentdenovo_info.res")

        # ==============================================================================
        #analyse
        Union_fasta_coverage_denovo_dict = merge_fasta_coverage_dict([try_fasta_coverage_denovo_dict, lys_fasta_coverage_denovo_dict])
        Union_denovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
            [try_denovo_proteinName_to_uniqueSeq_dict, lys_denovo_proteinName_to_uniqueSeq_dict])
        write_dict_to_csv(Union_fasta_coverage_denovo_dict, denovo_folder + "\\Union_fasta_coverage_denovo_info.res", fasta_seq, None,
                          Union_denovo_proteinName_to_uniqueSeq_dict)
        Union_fasta_coverage_Evidentdenovo_dict = merge_fasta_coverage_dict(
            [try_fasta_coverage_Evidentdenovo_dict, lys_fasta_coverage_Evidentdenovo_dict])
        Union_Evidentdenovo_proteinName_to_uniqueSeq_dict = merge_proteinName_to_uniqueSeq(
            [try_Evidentdenovo_proteinName_to_uniqueSeq_dict, lys_Evidentdenovo_proteinName_to_uniqueSeq_dict])
        write_dict_to_csv(Union_fasta_coverage_Evidentdenovo_dict, denovo_folder + "\\Union_fasta_coverage_Evidentdenovo_info.res",
                          fasta_seq, None, Union_Evidentdenovo_proteinName_to_uniqueSeq_dict)

        #肽段
        # ==============================================================================
        try_denovoSeq_df = pd.read_csv(try_denovo_folder + "\\denovoSeq_to_ifEvidentFind.res",sep="\t")
        try_denovoSeq_find_set = set(try_denovoSeq_df[(try_denovoSeq_df["ifFind"] == True) & (try_denovoSeq_df["ifFindDecoy"] == False)]["denovoSeq"])
        try_denovoSeq_find_Evident_set = set(try_denovoSeq_df[(try_denovoSeq_df["ifEvident"] == True) & (try_denovoSeq_df["ifFind"] == True) & (try_denovoSeq_df["ifFindDecoy"] == False)]["denovoSeq"])
        lys_denovoSeq_df = pd.read_csv(lys_denovo_folder + "\\denovoSeq_to_ifEvidentFind.res",sep="\t")
        lys_denovoSeq_find_set = set(lys_denovoSeq_df[(lys_denovoSeq_df["ifFind"] == True) & (lys_denovoSeq_df["ifFindDecoy"] == False)]["denovoSeq"])
        lys_denovoSeq_find_Evident_set = set(lys_denovoSeq_df[(lys_denovoSeq_df["ifEvident"] == True) & (lys_denovoSeq_df["ifFind"] == True) & (lys_denovoSeq_df["ifFindDecoy"] == False)]["denovoSeq"])
        plot_pep_venn2(try_denovoSeq_find_set, lys_denovoSeq_find_set, denovo_folder + "\\denovoNei_pep_intsection_find.png", fig_name1, fig_name2)
        plot_pep_venn2(try_denovoSeq_find_Evident_set, lys_denovoSeq_find_Evident_set, denovo_folder + "\\denovoNei_pep_intsection_find_Evident.png", fig_name1, fig_name2)

        info_fw = open(denovo_folder + "\\pep_info.res", 'w')
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
        try_title_to_denovo_df = pd.read_csv(try_denovo_folder + "\\title_to_denovo_info.res",sep="\t")
        try_denovo_find_spec_set = set(try_title_to_denovo_df[(try_title_to_denovo_df["ifFind_denovo"] == True) & (try_title_to_denovo_df["ifFindDecoy_denovo"] == False)]["title"])
        try_denovo_find_Evident_spec_set = set(try_title_to_denovo_df[(try_title_to_denovo_df["ifEvident_denovo"] == True) & (try_title_to_denovo_df["ifFind_denovo"] == True) & (try_title_to_denovo_df["ifFindDecoy_denovo"] == False)]["title"])
        lys_title_to_denovo_df = pd.read_csv(lys_denovo_folder + "\\title_to_denovo_info.res",sep="\t")
        lys_denovo_find_spec_set = set(lys_title_to_denovo_df[(lys_title_to_denovo_df["ifFind_denovo"] == True) & (lys_title_to_denovo_df["ifFindDecoy_denovo"] == False)]["title"])
        lys_denovo_find_Evident_spec_set = set(lys_title_to_denovo_df[(lys_title_to_denovo_df["ifEvident_denovo"] == True) & (lys_title_to_denovo_df["ifFind_denovo"] == True) & (lys_title_to_denovo_df["ifFindDecoy_denovo"] == False)]["title"])

        trylysSum_denovo_find_spec_set = try_denovo_find_spec_set | lys_denovo_find_spec_set
        trylysSum_denovo_find_Evident_spec_set = try_denovo_find_Evident_spec_set | lys_denovo_find_Evident_spec_set
        info_fw.write("try_find_spec_num=" + str(len(try_denovo_find_spec_set)) + "\n")
        info_fw.write("try_find_Evident_spec_num=" + str(len(try_denovo_find_Evident_spec_set)) + "\n")
        info_fw.write("lys_find_spec_num=" + str(len(lys_denovo_find_spec_set)) + "\n")
        info_fw.write("lys_find_Evident_spec_num=" + str(len(lys_denovo_find_Evident_spec_set)) + "\n")
        info_fw.write("Sum_trylys_find_spec_num=" + str(len(trylysSum_denovo_find_spec_set)) + "\n")
        info_fw.write("Sum_trylys_find_Evident_spec_num=" + str(len(trylysSum_denovo_find_Evident_spec_set)) + "\n")

