import os
import multiprocessing
import numpy as np
from tqdm import tqdm
import bisect
import time
from MSLogging import logToUser, logGetError
from MSSystem import IO_NAME_FILE_PNOVOM_FINAL, IO_NAME_FILE_GCNOVO_FINAL
from MSSystem import IO_NAME_FILE_PNOVOM_SINGLE_A_FINAL, IO_NAME_FILE_PNOVOM_SINGLE_B_FINAL
from MSSystem import IO_NAME_FILE_GCNOVO_SINGLE_A_FINAL, IO_NAME_FILE_GCNOVO_SINGLE_B_FINAL

modifications = ['Carbamidomethyl[C]',
                 'Oxidation[M]',
                 'Deamidated[N]',
                 'Deamidated[Q]',
                 'NeuCodeK602[K]',
                 'NeuCodeR004[R]']

modifications_in_aa_dict_keys = ['C(+57.02)' , 'M(+15.99)','N(+.98)', 'Q(+.98)','K(+8.0142)','R(+3.9881)']

pure_aa_list = ['C', 'M', 'N', 'Q', 'K', 'R']

index_num = 4  # 用于快速查找的索引长度
fragment_tolerance_ppm = 20  # 碎片离子误差
topk_peaks = 200  # 高可靠肽段的定义————匹配信号峰必须强度为top200

aa_to_mass_dict = {
    'A': 71.037114,
    'R': 156.101111,
    'R(+3.9881)': 160.08921,  # // 156.10111 + 3.9881, R + NeuCode(R)
    'N': 114.042927,
    'N(+.98)': 115.02695,  # Deamidation
    'D': 115.026943,
    'C(+57.02)': 160.03065,  # Carbamidomethylation
    'E': 129.042593,
    'Q': 128.058578,
    'Q(+.98)': 129.0426,  # Deamidation
    'G': 57.021464,
    'H': 137.058912,
    'I': 113.084064,
    'L': 113.084064,
    'K': 128.094963,
    'K(+8.0142)': 136.10916,  # //128.09496 + 8.0142, K + NeuCode(K)
    'M': 131.040485,
    'M(+15.99)': 147.0354,  # Oxidation
    'F': 147.068414,
    'P': 97.052764,
    'S': 87.032028,
    'T': 101.047679,
    'W': 186.079313,
    'Y': 163.063329,
    'V': 99.068414}

atom_mass = {
    'H': 1.0078,
    'PROTON': 1.00727646677,
    'H2O': 18.010565,
    'NH3': 17.02655,
    'CO': 27.99492,
    'CHO': 29.00272,
    'Isotope': 1.00335
}

def reverse_fasta(fasta_seq):
    reverse_fasta_seq_dict = {}
    for key in fasta_seq:
        fasta_seq_reverse = fasta_seq[key][::-1]
        reverse_fasta_seq_dict[key] = fasta_seq_reverse
    return reverse_fasta_seq_dict

def read_mgf_to_generate_spectrum(title, mgf_file, spectrum_location):
    mz_list = []
    intensity_list = []
    charge = 1
    with open(mgf_file, 'r') as f:
        f.seek(spectrum_location)
        line = f.readline()
        assert line == "BEGIN IONS\n"
        line = f.readline()
        while line:
            if line.startswith("TITLE="):
                assert title == line.split('=')[1].strip()
            elif line.startswith("CHARGE="):
                charge = int(line.split('=')[1].strip()[:-1])
            elif line.startswith("PEPMASS="):
                pass
            elif line == "END IONS\n":
                break
            else:
                # 如果line是以数字开头，则是mz和intensity
                if line[0].isdigit():
                    line = line.split()
                    mz_list.append(float(line[0]))
                    intensity_list.append(float(line[1]))
            line = f.readline()

    return Spectrum(title, charge, mz_list, intensity_list)

def cal_ion_coverage_for_singleSpec_from_Rseq(args):
    #输入的seq是list
    title, mgf_file, mgf_location, matchtype_and_sequence_list = args

    spectrum = read_mgf_to_generate_spectrum(title, mgf_file, mgf_location)

    sitetag_and_siteintensity_list = []
    for matchtype_and_sequence in matchtype_and_sequence_list:
        match_type, sequence = matchtype_and_sequence
        site_tag, site_intensity = spectrum.calculate_ion_coverage(sequence)
        sitetag_and_siteintensity_list.append((site_tag,site_intensity))

    sitetag_and_siteintensity_list.append(spectrum.intensity_list)
    sitetag_and_siteintensity_list.append(title)
    return sitetag_and_siteintensity_list

def cal_ion_coverage_for_singleSpec(args):
    #输入的seq是str
    title, mgf_file, mgf_location, sequence_list = args

    spectrum = read_mgf_to_generate_spectrum(title, mgf_file, mgf_location)
    sitetag_and_siteintensity_list = []
    for sequence in sequence_list:
        #print(match_type,sequence)
        #sequence = transfer_DiNovo_seq_to_SeqModSeq(sequence)
        sequence = transfer_str_to_list_seq(sequence)
        site_tag, site_intensity = spectrum.calculate_ion_coverage(sequence)
        sitetag_and_siteintensity_list.append((site_tag,site_intensity))

    sitetag_and_siteintensity_list.append(spectrum.intensity_list)
    sitetag_and_siteintensity_list.append(title)
    return sitetag_and_siteintensity_list

class Spectrum:
    def __init__(self, title, charge, mz_list, intensity_list):
        self.title = title
        self.charge = charge
        self.mz_list = mz_list
        self.intensity_list = intensity_list

    def calculate_ion_coverage(self, try_sequence):
        try_length = len(try_sequence)
        # try_rank_list = list(np.argsort(np.argsort(try_intensity_list))[::-1] + 1) #+1是为了避免rank为0
        ###################################################################################################
        # trypsin
        try:
            try_peptide_residue_mass = [aa_to_mass_dict[i] for i in try_sequence]
        except:
            print(try_sequence)
            assert False
        try_peptide_residue_cumsummass = np.cumsum(try_peptide_residue_mass)

        # b+
        try_theoretical_single_charge_b_ion = try_peptide_residue_cumsummass + atom_mass['PROTON']
        try_theoretical_single_charge_b_ion_tag = np.zeros_like(try_theoretical_single_charge_b_ion)
        try_theoretical_single_charge_b_ion_intensity = np.zeros_like(try_theoretical_single_charge_b_ion)
        for i, theoretical_mass in enumerate(try_theoretical_single_charge_b_ion):
            delta_mass = (np.array(self.mz_list) - theoretical_mass) / theoretical_mass * 1000000
            min_index = np.argmin(np.abs(delta_mass))
            min_delta_mass = delta_mass[min_index]
            if min_delta_mass >= -fragment_tolerance_ppm and min_delta_mass <= fragment_tolerance_ppm:
                try_theoretical_single_charge_b_ion_tag[i] = 1
                try_theoretical_single_charge_b_ion_intensity[i] = self.intensity_list[min_index]

        # b2+
        if self.charge > 2:
            try_theoretical_two_charge_b_ion = (try_peptide_residue_cumsummass + 2 * atom_mass['PROTON']) / 2.0
            try_theoretical_two_charge_b_ion_tag = np.zeros_like(try_theoretical_two_charge_b_ion)
            try_theoretical_two_charge_b_ion_intensity = np.zeros_like(try_theoretical_two_charge_b_ion)
            for i, theoretical_mass in enumerate(try_theoretical_two_charge_b_ion):
                delta_mass = (np.array(self.mz_list) - theoretical_mass) / theoretical_mass * 1000000
                min_index = np.argmin(np.abs(delta_mass))
                min_delta_mass = delta_mass[min_index]
                if min_delta_mass >= -fragment_tolerance_ppm and min_delta_mass <= fragment_tolerance_ppm:
                    try_theoretical_two_charge_b_ion_tag[i] = 1
                    try_theoretical_two_charge_b_ion_intensity[i] = self.intensity_list[min_index]

        try_peptide_residue_mass_reverse = try_peptide_residue_mass[::-1]
        try_peptide_residue_cumsummass_reverse = np.cumsum(try_peptide_residue_mass_reverse)

        # y+
        try_theoretical_single_charge_y_ion = try_peptide_residue_cumsummass_reverse + atom_mass['H2O'] + atom_mass[
            'PROTON']
        try_theoretical_single_charge_y_ion_tag = np.zeros_like(try_theoretical_single_charge_y_ion)
        try_theoretical_single_charge_y_ion_intensity = np.zeros_like(try_theoretical_single_charge_y_ion)
        for i, theoretical_mass in enumerate(try_theoretical_single_charge_y_ion):
            delta_mass = (np.array(self.mz_list) - theoretical_mass) / theoretical_mass * 1000000
            min_index = np.argmin(np.abs(delta_mass))
            min_delta_mass = delta_mass[min_index]
            if min_delta_mass >= -fragment_tolerance_ppm and min_delta_mass <= fragment_tolerance_ppm:
                try_theoretical_single_charge_y_ion_tag[i] = 1
                try_theoretical_single_charge_y_ion_intensity[i] = self.intensity_list[min_index]

        # y2+
        if self.charge > 2:
            try_theoretical_two_charge_y_ion = (try_peptide_residue_cumsummass_reverse + atom_mass['H2O'] + 2 *
                                                atom_mass[
                                                    'PROTON']) / 2.0
            try_theoretical_two_charge_y_ion_tag = np.zeros_like(try_theoretical_two_charge_y_ion)
            try_theoretical_two_charge_y_ion_intensity = np.zeros_like(try_theoretical_two_charge_y_ion)
            for i, theoretical_mass in enumerate(try_theoretical_two_charge_y_ion):
                delta_mass = (np.array(self.mz_list) - theoretical_mass) / theoretical_mass * 1000000
                min_index = np.argmin(np.abs(delta_mass))
                min_delta_mass = delta_mass[min_index]
                if min_delta_mass >= -fragment_tolerance_ppm and min_delta_mass <= fragment_tolerance_ppm:
                    try_theoretical_two_charge_y_ion_tag[i] = 1
                    try_theoretical_two_charge_y_ion_intensity[i] = self.intensity_list[min_index]

        ##########################################################################################
        # return results

        try_theoretical_single_charge_b_ion_tag = try_theoretical_single_charge_b_ion_tag[:-1]
        try_theoretical_single_charge_b_ion_intensity = try_theoretical_single_charge_b_ion_intensity[:-1]
        try_theoretical_single_charge_y_ion_tag = try_theoretical_single_charge_y_ion_tag[:-1][::-1]
        try_theoretical_single_charge_y_ion_intensity = try_theoretical_single_charge_y_ion_intensity[:-1][::-1]
        if self.charge > 2:
            try_theoretical_two_charge_b_ion_tag = try_theoretical_two_charge_b_ion_tag[:-1]
            try_theoretical_two_charge_b_ion_intensity = try_theoretical_two_charge_b_ion_intensity[:-1]
            try_theoretical_two_charge_y_ion_tag = try_theoretical_two_charge_y_ion_tag[:-1][::-1]
            try_theoretical_two_charge_y_ion_intensity = try_theoretical_two_charge_y_ion_intensity[:-1][::-1]

        iontype = {
            "b+": True,
            "b2+": False,
            "y+": True,
            "y2+": False
        }
        if self.charge > 2:
            iontype["b2+"] = True
            iontype["y2+"] = True

        for i, ion in enumerate(iontype):
            if i == 0:
                if iontype[ion] == True and ion == "b+":
                    try_tag_array = try_theoretical_single_charge_b_ion_tag
                    try_intensity_array = try_theoretical_single_charge_b_ion_intensity
                else:
                    assert False
            else:
                if ion == 'b2+' and iontype['b2+'] == True:
                    try_tag_array = np.row_stack((try_tag_array, try_theoretical_two_charge_b_ion_tag))
                    try_intensity_array = np.row_stack(
                        (try_intensity_array, try_theoretical_two_charge_b_ion_intensity))
                    continue
                if ion == 'y+' and iontype['y+'] == True:
                    try_tag_array = np.row_stack((try_tag_array, try_theoretical_single_charge_y_ion_tag))
                    try_intensity_array = np.row_stack(
                        (try_intensity_array, try_theoretical_single_charge_y_ion_intensity))
                    continue
                if ion == 'y2+' and iontype['y2+'] == True:
                    try_tag_array = np.row_stack((try_tag_array, try_theoretical_two_charge_y_ion_tag))
                    try_intensity_array = np.row_stack(
                        (try_intensity_array, try_theoretical_two_charge_y_ion_intensity))
                    continue

        ion_num = 0
        for i, ion in enumerate(iontype):
            if iontype[ion] == True:
                ion_num += 1
        assert try_tag_array.shape == (ion_num, try_length - 1), 'shape error!'
        assert try_intensity_array.shape == (ion_num, try_length - 1), 'shape error!'

        try_site_tag = np.zeros(try_length - 1)
        for i in range(try_length - 1):
            if 1 in try_tag_array[:, i]:
                try_site_tag[i] = 1

        try_site_intensity = np.zeros(try_length - 1)
        for i in range(try_length - 1):
            try_site_intensity[i] = np.sum(try_intensity_array[:, i])

        return try_site_tag, try_site_intensity


def transfer_DiNovo_seq_to_PureSeq(seq):
    seq = seq.replace("I", "L").replace("J", "M")
    return seq

def transfer_DiNovo_seq_to_SeqModSeq(seq):
    seq = seq.replace("I", "L").replace("C", "C(+57.02)").replace("J", "M(+15.99)")
    return seq

class RSequence:
    def __init__(self, seq):
        self.seq = seq
        self.pure_seq = None#seq.replace("I", "L")#.replace("C(+57.02)", "C").replace("J", "M")
        self.matchtype_to_location = {}

    def add_matchtype(self,match_type):
        if match_type not in self.matchtype_to_location:
            self.matchtype_to_location[match_type] = []

    def add_location(self,match_type,location):
        self.matchtype_to_location[match_type].append(location)

    def __str__(self):
        return f"Sequence: {self.seq}, Match Type: {self.match_type}, Location: {self.location}"

    def __repr__(self):
        return self.__str__()

class Sequence:
    def __init__(self, seq):
        self.seq = seq
        self.pure_seq = None#seq.replace("I", "L").replace("C(+57.02)", "C").replace("J", "M")
        self.location = []

    def add_location(self,location):
        self.location.append(location)

    def __str__(self):
        return f"Sequence: {self.seq}, Location: {self.location}"

    def __repr__(self):
        return self.__str__()

def build_mgf_location_function(path_list):

    # 读取mgf文件，创建location字典
    # path_list = os.listdir(mgf_folder)
    # path_list.sort()  # 对读取的路径进行排序

    spectrum_location_dict = {}

    for file_path in path_list:  # 挨个读取mgf文件

        logToUser(f"Building location for: " + file_path)

        # line = True
        count = 0
        current_location = 0
        spectrum_location = 0
        with open(file_path, 'r') as f:
            # for line in f:
            while True:
                # current_location = f.tell()
                line = f.readline()
                count += 1

                if len(line) == 0:
                    print('ends with empty line : ' + str(count))
                    break

                if line[0] == "B" and line[:5] == "BEGIN":
                    spectrum_location = current_location

                elif line[0] == "T" and line[:5] == "TITLE":
                    spectrum_location_dict[line[6:-1]] = (file_path, spectrum_location)
                current_location += len(line) + 1

    return spectrum_location_dict # , suffix_name, filepathAndOrderNum_to_title


def build_fasta_dict_function(fasta_file):
    fasta_seq = {}
    with open(fasta_file, 'r') as f:
        line = f.readline()
        while True:
            if len(line) == 0:
                break
            if line.startswith(">"):
                key = line[1:].strip()
                fasta_seq[key] = ''
            else:
                line = line.replace("I", "L")
                fasta_seq[key] += line.strip()
            line = f.readline()

    return fasta_seq

def generate_try_lys_seq_from_RseqAndMatchtype(Rseq,match_type):
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
        print(match_type)
        assert False
    return try_seq,lys_seq

def find_location_in_fasta(seq_dict_class, fasta_seq_dict,index_first_3aa):

    seq_IL_Mod = seq_dict_class.pure_seq
    for match_type in seq_dict_class.matchtype_to_location:
        try_seq, lys_seq = generate_try_lys_seq_from_RseqAndMatchtype(seq_IL_Mod,match_type)
        try_location = []
        lys_location = []

        #try
        if_find = False
        if len(try_seq) <= index_num:
            for seq_3aa in index_first_3aa.keys():
                if seq_3aa[:len(try_seq)] == try_seq:
                    for key, i in index_first_3aa[seq_3aa]:
                        try_location.append((key,[(i,i+len(try_seq))]))
                        if_find = True
        else:
            if try_seq[:index_num] in index_first_3aa:
                for key, i in index_first_3aa[try_seq[:index_num]]:
                    if fasta_seq_dict[key][i:i+len(try_seq)] == try_seq:
                        try_location.append((key,[(i,i+len(try_seq))]))
                        if_find = True
        if not if_find:
            continue

        #lys
        if_find = False
        if len(lys_seq) <= index_num:
            for seq_3aa in index_first_3aa.keys():
                if seq_3aa[:len(lys_seq)] == lys_seq:
                    for key, i in index_first_3aa[seq_3aa]:
                        lys_location.append((key,[(i,i+len(lys_seq))]))
                        if_find = True
        else:
            if lys_seq[:index_num] in index_first_3aa:
                for key, i in index_first_3aa[lys_seq[:index_num]]:
                    if fasta_seq_dict[key][i:i+len(lys_seq)] == lys_seq:
                        lys_location.append((key,[(i,i+len(lys_seq))]))
                        if_find = True
        if if_find:
            seq_dict_class.add_location(match_type,(try_location,lys_location))

    return seq_dict_class

def find_location_in_fasta_single(seq_dict_class, fasta_seq_dict,index_first_3aa):

    seq_IL_Mod = seq_dict_class.pure_seq
    location = []

    #try
    if_find = False
    if len(seq_IL_Mod) <= index_num:
        for seq_3aa in index_first_3aa.keys():
            if seq_3aa[:len(seq_IL_Mod)] == seq_IL_Mod:
                for key, i in index_first_3aa[seq_3aa]:
                    location.append((key,[(i,i+len(seq_IL_Mod))]))
                    if_find = True
    else:
        if seq_IL_Mod[:index_num] in index_first_3aa:
            for key, i in index_first_3aa[seq_IL_Mod[:index_num]]:
                if fasta_seq_dict[key][i:i+len(seq_IL_Mod)] == seq_IL_Mod:
                    location.append((key,[(i,i+len(seq_IL_Mod))]))
                    if_find = True

    if if_find:
        seq_dict_class.add_location(location)

    return seq_dict_class


# def find_location_in_fasta(seq_dict_class, fasta_seq_dict):
#     seq_IL_Mod = seq_dict_class.seq_IL_Mod
#     for match_type in seq_dict_class.matchtype_to_location:
#         try_seq, lys_seq = generate_try_lys_seq_from_RseqAndMatchtype(seq_IL_Mod,match_type)
#         try_location = []
#         lys_location = []
#         if_find = False
#         for key in fasta_seq_dict:
#             if lys_seq in fasta_seq_dict[key]:
#                 site = [(i,i+len(lys_seq)) for i in range(len(fasta_seq_dict[key])) if fasta_seq_dict[key].startswith(lys_seq, i)]
#                 lys_location.append((key,site))
#                 if_find = True
#         if not if_find:
#             continue
#         else:
#             if_find = False
#             for key in fasta_seq_dict:
#                 if try_seq in fasta_seq_dict[key]:
#                     site = [(i,i+len(try_seq)) for i in range(len(fasta_seq_dict[key])) if fasta_seq_dict[key].startswith(try_seq, i)]
#                     try_location.append((key,site))
#                     if_find = True
#         if if_find:
#             seq_dict_class.add_location(match_type,(try_location,lys_location))
#
#     return seq_dict_class

def get_result(result):
    return result.get()

def worker(args):
    seq1_dict_class, fasta_seq_dict, index_first_3aa = args
    return find_location_in_fasta(seq1_dict_class, fasta_seq_dict, index_first_3aa)

def worker_single(args):
    seq1_dict_class, fasta_seq_dict, index_first_3aa = args
    return find_location_in_fasta_single(seq1_dict_class, fasta_seq_dict, index_first_3aa)

def parse_DiNovo_res_find_function(res_file_dict,fasta_seq_dict,processing_workers,res_type,if_confident):

    index_first_3aa = build_index_dict_function(fasta_seq_dict)
    seq1_dict = {}
    seq1_single_try_dict = {}
    seq1_single_lys_dict = {}
    fasta_coverage_dict1 = {}
    fasta_coverage_single_try_dict1 = {}
    fasta_coverage_single_lys_dict1 = {}
    Union_dict_list = []
    for key in fasta_seq_dict:
        value = fasta_seq_dict[key]
        length = len(value)
        fasta_coverage_dict1[key] = np.zeros(length)
        fasta_coverage_single_try_dict1[key] = np.zeros(length)
        fasta_coverage_single_lys_dict1[key] = np.zeros(length)

    mirror_file = res_file_dict[f"{res_type}_mirror_res_file"]
    if mirror_file != "":
        print(f"Parsing TOPN res from {mirror_file}")
        with open(mirror_file, "r") as f:
            lines = f.readlines()
            mirror_file_first_2lines = lines[:2]
            lines = lines[2:]
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    match_type = line.split("\t")[2].strip()
                else:
                    line = line.split("\t")
                    seq = line[2]
                    pure_seq = seq
                    mod = line[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx-1]=mod
                        seq = ''.join(seq_list)
                    if seq not in seq1_dict:
                        seq1_dict[seq] = RSequence(seq)
                        seq1_dict[seq].pure_seq = pure_seq
                    seq1_dict[seq].add_matchtype(match_type)

        with multiprocessing.Pool(processing_workers) as p:
            seq1_dict_list = list(tqdm(
                p.imap(worker, [(seq1_dict[seq], fasta_seq_dict, index_first_3aa) for seq in seq1_dict],
                       chunksize=1000000),total=len(seq1_dict)))
            seq1_dict = {seq_dict.seq: seq_dict for seq_dict in seq1_dict_list}
        for seq in seq1_dict:
            for match_type in seq1_dict[seq].matchtype_to_location:
                for try_location, lys_location in seq1_dict[seq].matchtype_to_location[match_type]:
                    for key, site in try_location:
                        for site_item in site:
                            fasta_coverage_dict1[key][site_item[0]:site_item[1]] = 1
                    for key, site in lys_location:
                        for site_item in site:
                            fasta_coverage_dict1[key][site_item[0]:site_item[1]] = 1
        Union_dict_list.append(fasta_coverage_dict1)

    single_A_res_file = res_file_dict[f"{res_type}_singleA_res_file"]
    if single_A_res_file != "":
        print(f"Parsing  TOPN res from {single_A_res_file}")
        with open(single_A_res_file, "r") as f:
            lines = f.readlines()
            single_A_res_file_first_2lines = lines[:2]
            lines = lines[2:]
            for line in tqdm(lines):
                if line.startswith("\t"):
                    line = line.split("\t")
                    seq = line[2]
                    pure_seq = seq
                    mod = line[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    if seq not in seq1_single_try_dict:
                        seq1_single_try_dict[seq] = Sequence(seq)
                        seq1_single_try_dict[seq].pure_seq = pure_seq

        with multiprocessing.Pool(processing_workers) as p:
            seq1_single_try_dict_list = list(tqdm(
                p.imap(worker_single, [(seq1_single_try_dict[seq], fasta_seq_dict, index_first_3aa) for seq in seq1_single_try_dict],
                       chunksize=1000000),total=len(seq1_single_try_dict)))
            seq1_single_try_dict = {seq_dict.seq: seq_dict for seq_dict in seq1_single_try_dict_list}

        for seq in seq1_single_try_dict:
            if len(seq1_single_try_dict[seq].location) >= 1:
                for location in seq1_single_try_dict[seq].location:
                    for key, site in location:
                        for site_item in site:
                            fasta_coverage_single_try_dict1[key][site_item[0]:site_item[1]] = 1
        Union_dict_list.append(fasta_coverage_single_try_dict1)


    single_B_res_file = res_file_dict[f"{res_type}_singleB_res_file"]
    if single_B_res_file != "":
        print(f"Parsing TOPN res from {single_B_res_file}")
        with open(single_B_res_file, "r") as f:
            lines = f.readlines()
            single_B_res_file_first_2lines = lines[:2]
            lines = lines[2:]
            for line in tqdm(lines):
                if line.startswith("\t"):
                    line = line.split("\t")
                    seq = line[2]
                    pure_seq = seq
                    mod = line[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    if seq not in seq1_single_lys_dict:
                        seq1_single_lys_dict[seq] = Sequence(seq)
                        seq1_single_lys_dict[seq].pure_seq = pure_seq

        with multiprocessing.Pool(processing_workers) as p:
            seq1_single_lys_dict_list = list(tqdm(
                p.imap(worker_single, [(seq1_single_lys_dict[seq], fasta_seq_dict, index_first_3aa) for seq in seq1_single_lys_dict],
                       chunksize=1000000),total=len(seq1_single_lys_dict)))
            seq1_single_lys_dict = {seq_dict.seq: seq_dict for seq_dict in seq1_single_lys_dict_list}

        for seq in seq1_single_lys_dict:
            if len(seq1_single_lys_dict[seq].location) >= 1:
                for location in seq1_single_lys_dict[seq].location:
                    for key, site in location:
                        for site_item in site:
                            fasta_coverage_single_lys_dict1[key][site_item[0]:site_item[1]] = 1
        Union_dict_list.append(fasta_coverage_single_lys_dict1)

    #把Union_dict_list里面的所有dict进行合并
    fasta_coverage_dictUnion = {}
    for key in fasta_seq_dict:
        fasta_coverage_dictUnion[key] = np.logical_or.reduce([fasta_coverage_dict[key] for fasta_coverage_dict in Union_dict_list])

    total_aa_num = 0
    match_aa_num = 0
    total_protein_num = 0
    match_protein_num = 0
    full_coverage_match_protein_num = 0
    coverage_list = []
    protein_topN_coverage_dict = {}
    if mirror_file != "":
        output_folder = os.path.dirname(mirror_file)
    elif single_A_res_file != "":
        output_folder = os.path.dirname(single_A_res_file)
    elif single_B_res_file != "":
        output_folder = os.path.dirname(single_B_res_file)
    else:
        assert False," The res file is empty!"
    print(f"Writing {os.path.join(output_folder, f'[{res_type}]TOPN_Database_Coverage.txt')}")
    with open(os.path.join(output_folder, f"[{res_type}]TOPN_Database_Coverage.txt"), "w") as f:
        f.write("\t".join(
            ["PROTEIN_INFO", "MAPPED_AA_NUM", "TOTAL_AA_NUM", "COVERAGE_RATE", "MAPPED_PROTEIN_SEQUENCE"]) + "\n")
        for key in tqdm(fasta_coverage_dictUnion):
            protein_seq = fasta_seq_dict[key]
            coverage_num = sum(fasta_coverage_dictUnion[key])
            total_num = len(fasta_coverage_dictUnion[key])
            coverage_rate = coverage_num / total_num
            seq_temp = ''.join([char.lower() if fasta_coverage_dictUnion[key][i] == 1 else char for i, char in enumerate(protein_seq)])
            f.write(f"{key}\t{coverage_num}\t{total_num}\t{coverage_rate}\t{seq_temp}\n")
            total_aa_num += total_num
            match_aa_num += coverage_num
            total_protein_num += 1
            if coverage_rate > 0:
                match_protein_num += 1
            if coverage_rate == 1:
                full_coverage_match_protein_num += 1
            coverage_list.append(coverage_rate)
            protein_topN_coverage_dict[key] = coverage_rate

###################################################################################################################3
    #top1 结果
    seq1_top1_dict = {}
    seq1_single_try_top1_dict = {}
    seq1_single_lys_top1_dict = {}
    fasta_coverage_dict1 = {}
    fasta_coverage_single_try_dict1 = {}
    fasta_coverage_single_lys_dict1 = {}
    Union_dict_list = []
    for key in fasta_seq_dict:
        value = fasta_seq_dict[key]
        length = len(value)
        fasta_coverage_dict1[key] = np.zeros(length)
        fasta_coverage_single_try_dict1[key] = np.zeros(length)
        fasta_coverage_single_lys_dict1[key] = np.zeros(length)

    if mirror_file != "":
        print(f"Parsing TOP1 res from {mirror_file}")
        with open(mirror_file, "r") as f:
            lines = f.readlines()[2:]
            tag_new_spec = False
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    match_type = line.split("\t")[2].strip()
                    tag_new_spec = True
                else:
                    if tag_new_spec:
                        line = line.split("\t")
                        seq = line[2]
                        pure_seq = seq
                        mod = line[3]
                        if len(mod) > 0:
                            seq_list = list(seq)
                            mod_list = mod.split(";")[:-1]
                            for mod_item in mod_list:
                                mod_item_list = mod_item.split(",")
                                idx, mod_name = mod_item_list[0], mod_item_list[1]
                                idx = int(idx)
                                mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                                seq_list[idx - 1] = mod
                            seq = ''.join(seq_list)
                        if seq not in seq1_top1_dict:
                            seq1_top1_dict[seq] = RSequence(seq)
                            seq1_top1_dict[seq].pure_seq = pure_seq
                        seq1_top1_dict[seq].add_matchtype(match_type)
                        tag_new_spec = False

        with multiprocessing.Pool(processing_workers) as p:
            seq1_top1_dict_list = list(tqdm(
                p.imap(worker, [(seq1_top1_dict[seq], fasta_seq_dict, index_first_3aa) for seq in seq1_top1_dict],
                       chunksize=1000000), total=len(seq1_top1_dict)))
            seq1_top1_dict = {seq_dict.seq: seq_dict for seq_dict in seq1_top1_dict_list}

        for seq in tqdm(seq1_top1_dict):
            for match_type in seq1_top1_dict[seq].matchtype_to_location:
                for try_location, lys_location in seq1_top1_dict[seq].matchtype_to_location[match_type]:
                    for key, site in try_location:
                        for site_item in site:
                            fasta_coverage_dict1[key][site_item[0]:site_item[1]] = 1
                    for key, site in lys_location:
                        for site_item in site:
                            fasta_coverage_dict1[key][site_item[0]:site_item[1]] = 1
        Union_dict_list.append(fasta_coverage_dict1)

    if single_A_res_file != "":
        print(f"Parsing TOP1 res from {single_A_res_file}")
        with open(single_A_res_file, "r") as f:
            lines = f.readlines()[2:]
            tag_new_spec = False
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    tag_new_spec = True
                else:
                    if tag_new_spec:
                        line = line.split("\t")
                        seq = line[2]
                        pure_seq = seq
                        mod = line[3]
                        if len(mod) > 0:
                            seq_list = list(seq)
                            mod_list = mod.split(";")[:-1]
                            for mod_item in mod_list:
                                mod_item_list = mod_item.split(",")
                                idx, mod_name = mod_item_list[0], mod_item_list[1]
                                idx = int(idx)
                                mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                                seq_list[idx - 1] = mod
                            seq = ''.join(seq_list)
                        if seq not in seq1_single_try_top1_dict:
                            seq1_single_try_top1_dict[seq] = Sequence(seq)
                            seq1_single_try_top1_dict[seq].pure_seq = pure_seq
                        tag_new_spec = False

        with multiprocessing.Pool(processing_workers) as p:
            seq1_single_try_top1_dict_list = list(tqdm(
                p.imap(worker_single, [(seq1_single_try_top1_dict[seq], fasta_seq_dict, index_first_3aa) for seq in seq1_single_try_top1_dict],
                       chunksize=1000000),total=len(seq1_single_try_top1_dict)))
            seq1_single_try_top1_dict = {seq_dict.seq: seq_dict for seq_dict in seq1_single_try_top1_dict_list}

        for seq in seq1_single_try_top1_dict:
            if len(seq1_single_try_top1_dict[seq].location) >= 1:
                for location in seq1_single_try_top1_dict[seq].location:
                    for key, site in location:
                        for site_item in site:
                            fasta_coverage_single_try_dict1[key][site_item[0]:site_item[1]] = 1
        Union_dict_list.append(fasta_coverage_single_try_dict1)

    if single_B_res_file != "":
        print(f"Parsing TOP1 res from {single_B_res_file}")
        with open(single_B_res_file, "r") as f:
            lines = f.readlines()[2:]
            tag_new_spec = False
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    tag_new_spec = True
                else:
                    if tag_new_spec:
                        line = line.split("\t")
                        seq = line[2]
                        pure_seq = seq
                        mod = line[3]
                        if len(mod) > 0:
                            seq_list = list(seq)
                            mod_list = mod.split(";")[:-1]
                            for mod_item in mod_list:
                                mod_item_list = mod_item.split(",")
                                idx, mod_name = mod_item_list[0], mod_item_list[1]
                                idx = int(idx)
                                mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                                seq_list[idx - 1] = mod
                            seq = ''.join(seq_list)
                        if seq not in seq1_single_lys_top1_dict:
                            seq1_single_lys_top1_dict[seq] = Sequence(seq)
                            seq1_single_lys_top1_dict[seq].pure_seq = pure_seq
                        tag_new_spec = False

        with multiprocessing.Pool(processing_workers) as p:
            seq1_single_lys_top1_dict_list = list(tqdm(
                p.imap(worker_single, [(seq1_single_lys_top1_dict[seq], fasta_seq_dict, index_first_3aa) for seq in seq1_single_lys_top1_dict],
                       chunksize=1000000),total=len(seq1_single_lys_top1_dict)))
            seq1_single_lys_top1_dict = {seq_dict.seq: seq_dict for seq_dict in seq1_single_lys_top1_dict_list}

        for seq in seq1_single_lys_top1_dict:
            if len(seq1_single_lys_top1_dict[seq].location) >= 1:
                for location in seq1_single_lys_top1_dict[seq].location:
                    for key, site in location:
                        for site_item in site:
                            fasta_coverage_single_lys_dict1[key][site_item[0]:site_item[1]] = 1
        Union_dict_list.append(fasta_coverage_single_lys_dict1)

    #把Union_dict_list里面的所有dict进行合并
    fasta_coverage_dictUnion = {}
    for key in fasta_seq_dict:
        fasta_coverage_dictUnion[key] = np.logical_or.reduce([fasta_coverage_dict[key] for fasta_coverage_dict in Union_dict_list])

    total_aa_num = 0
    match_aa_num = 0
    total_protein_num = 0
    match_protein_num = 0
    full_coverage_match_protein_num = 0
    coverage_list = []
    protein_top1_coverage_dict = {}
    print(f"Writing {os.path.join(output_folder, f'[{res_type}]TOP1_Database_Coverage.txt')}")
    with open(os.path.join(output_folder, f"[{res_type}]TOP1_Database_Coverage.txt"), "w") as f:
        f.write("\t".join(
            ["PROTEIN_INFO", "MAPPED_AA_NUM", "TOTAL_AA_NUM", "COVERAGE_RATE", "MAPPED_PROTEIN_SEQUENCE"]) + "\n")
        for key in tqdm(fasta_coverage_dictUnion):
            protein_seq = fasta_seq_dict[key]
            coverage_num = sum(fasta_coverage_dictUnion[key])
            total_num = len(fasta_coverage_dictUnion[key])
            coverage_rate = coverage_num / total_num
            seq_temp = ''.join(
                [char.lower() if fasta_coverage_dictUnion[key][i] == 1 else char for i, char in enumerate(protein_seq)])
            f.write(f"{key}\t{coverage_num}\t{total_num}\t{coverage_rate}\t{seq_temp}\n")
            total_aa_num += total_num
            match_aa_num += coverage_num
            total_protein_num += 1
            if coverage_rate > 0:
                match_protein_num += 1
            if coverage_rate == 1:
                full_coverage_match_protein_num += 1
            coverage_list.append(coverage_rate)
            protein_top1_coverage_dict[key] = coverage_rate
############################################################################################################
    #decoy
    fasta_seq_decoy_dict = reverse_fasta(fasta_seq_dict)
    decoy_index_first_3aa = build_index_dict_function(fasta_seq_decoy_dict)

    decoy_seq1_dict = {}
    decoy_seq1_single_try_dict = {}
    decoy_seq1_single_lys_dict = {}

    if mirror_file != "":
        print(f"Parsing decoy TOPN res from {mirror_file}")
        with open(mirror_file, "r") as f:
            lines = f.readlines()
            mirror_file_first_2lines = lines[:2]
            lines = lines[2:]
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    match_type = line.split("\t")[2].strip()
                else:
                    line = line.split("\t")
                    seq = line[2]
                    pure_seq = seq
                    mod = line[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    if seq not in decoy_seq1_dict:
                        decoy_seq1_dict[seq] = RSequence(seq)
                        decoy_seq1_dict[seq].pure_seq = pure_seq
                    decoy_seq1_dict[seq].add_matchtype(match_type)

        with multiprocessing.Pool(processing_workers) as p:
            decoy_seq1_dict_list = list(tqdm(
                p.imap(worker, [(decoy_seq1_dict[seq], fasta_seq_decoy_dict, decoy_index_first_3aa) for seq in decoy_seq1_dict],
                       chunksize=1000000),total=len(decoy_seq1_dict)))
            decoy_seq1_dict = {seq_dict.seq: seq_dict for seq_dict in decoy_seq1_dict_list}

    if single_A_res_file != "":
        print(f"Parsing decoy TOPN res from {single_A_res_file}")
        with open(single_A_res_file, "r") as f:
            lines = f.readlines()
            single_A_res_file_first_2lines = lines[:2]
            lines = lines[2:]
            for line in tqdm(lines):
                if line.startswith("\t"):
                    line = line.split("\t")
                    seq = line[2]
                    pure_seq = seq
                    mod = line[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    if seq not in decoy_seq1_single_try_dict:
                        decoy_seq1_single_try_dict[seq] = Sequence(seq)
                        decoy_seq1_single_try_dict[seq].pure_seq = pure_seq

        with multiprocessing.Pool(processing_workers) as p:
            decoy_seq1_single_try_dict_list = list(tqdm(
                p.imap(worker_single, [(decoy_seq1_single_try_dict[seq], fasta_seq_decoy_dict, decoy_index_first_3aa) for seq in decoy_seq1_single_try_dict],
                       chunksize=1000000),total=len(decoy_seq1_single_try_dict)))
            decoy_seq1_single_try_dict = {seq_dict.seq: seq_dict for seq_dict in decoy_seq1_single_try_dict_list}

    if single_B_res_file != "":
        print(f"Parsing decoy TOPN res from {single_B_res_file}")
        with open(single_B_res_file, "r") as f:
            lines = f.readlines()
            single_B_res_file_first_2lines = lines[:2]
            lines = lines[2:]
            for line in tqdm(lines):
                if line.startswith("\t"):
                    line = line.split("\t")
                    seq = line[2]
                    pure_seq = seq
                    mod = line[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    if seq not in decoy_seq1_single_lys_dict:
                        decoy_seq1_single_lys_dict[seq] = Sequence(seq)
                        decoy_seq1_single_lys_dict[seq].pure_seq = pure_seq

        with multiprocessing.Pool(processing_workers) as p:
            decoy_seq1_single_lys_dict_list = list(tqdm(
                p.imap(worker_single, [(decoy_seq1_single_lys_dict[seq], fasta_seq_decoy_dict, decoy_index_first_3aa) for seq in decoy_seq1_single_lys_dict],
                       chunksize=1000000),total=len(decoy_seq1_single_lys_dict)))
            decoy_seq1_single_lys_dict = {seq_dict.seq: seq_dict for seq_dict in decoy_seq1_single_lys_dict_list}

############################################################################################################
    ##写入文件
    if mirror_file != "":
        suffix = mirror_file.split(".")[-1]
        out_tmp = mirror_file.replace(f".{suffix}",f"[new].{suffix}")
        fw = open(out_tmp, "w")
        # write first 2 lines
        if if_confident:
            for line in mirror_file_first_2lines:
                fw.write(line)
        else:
            fw.write(mirror_file_first_2lines[0])
            fw.write(mirror_file_first_2lines[1].rstrip() + "\tDecoy\tTargetSuffix(Protein/Top1Coverage/TopNCoverage)\tTargetPrefix(Protein/Top1Coverage/TopNCoverage)\n")
        # write res
        print(f"Writing {out_tmp}")
        with open(mirror_file, "r") as f:
            lines = f.readlines()[2:]
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    match_type = line.split("\t")[2].strip()
                    fw.write(line)
                else:
                    line_split = line.split("\t")
                    seq = line_split[2]
                    mod = line_split[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    if seq in seq1_dict:
                        #判断是否匹配上decoy
                        seq_dict = decoy_seq1_dict[seq]
                        location_list = seq_dict.matchtype_to_location[match_type]
                        if len(location_list) == 1:
                            decoy = True
                        elif len(location_list) == 0:
                            decoy = False
                        else:
                            print(decoy)
                            assert False
                        #写入文件
                        seq_dict = seq1_dict[seq]
                        string = f"{decoy}\t"
                        location_list = seq_dict.matchtype_to_location[match_type]
                        if len(location_list) == 1:
                            try_location, lys_location = location_list[0]
                            for try_key,try_site_list in try_location:
                                try_key_write = try_key.split('\t')[0].split()[0]
                                string += f"({try_key_write}/{round(protein_top1_coverage_dict[try_key],2)}/{round(protein_topN_coverage_dict[try_key],2)})"
                            string += f"\t"
                            for lys_key,lys_site_list in lys_location:
                                lys_key_write = lys_key.split('\t')[0].split()[0]
                                string += f"({lys_key_write}/{round(protein_top1_coverage_dict[lys_key],2)}/{round(protein_topN_coverage_dict[lys_key],2)})"
                        elif len(location_list) == 0:
                            string += f"(None/0/0)\t(None/0/0)"
                        else:
                            assert False
                        fw.write(f"{line.rstrip()}\t{string}\n")
                    else:
                        print(seq)
                        #print(seq1_dict.keys())
                        assert False
        fw.flush()
        fw.close()

    if single_A_res_file != "":
        suffix = single_A_res_file.split(".")[-1]
        out_tmp = single_A_res_file.replace(f".{suffix}", f"[new].{suffix}")
        fw = open(out_tmp, "w")
        # write first 2 lines
        if if_confident:
            for line in single_A_res_file_first_2lines:
                fw.write(line)
        else:
            fw.write(single_A_res_file_first_2lines[0])
            fw.write(single_A_res_file_first_2lines[1].rstrip() + "\tDecoy\tTarget(Protein/Top1Coverage/TopNCoverage)\n")
        print(f"Writing {out_tmp}")
        with open(single_A_res_file, "r") as f:
            lines = f.readlines()[2:]
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    fw.write(line)
                else:
                    line_split = line.split("\t")
                    seq = line_split[2]
                    mod = line_split[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    if seq in seq1_single_try_dict:
                        #decoy
                        seq_dict = decoy_seq1_single_try_dict[seq]
                        location_list = seq_dict.location
                        if len(location_list) == 1:
                            decoy = True
                        elif len(location_list) == 0:
                            decoy = False
                        else:
                            assert False
                        #写入文件
                        seq_dict = seq1_single_try_dict[seq]
                        string = f"{decoy}\t"
                        location_list = seq_dict.location
                        if len(location_list) == 1:
                            for location in location_list:
                                for key,site_list in location:
                                    key_write = key.split('\t')[0].split()[0]
                                    string += f"({key_write}/{round(protein_top1_coverage_dict[key],2)}/{round(protein_topN_coverage_dict[key],2)})"
                        elif len(location_list) == 0:
                            string += f"(None/0/0)"
                        else:
                            assert False
                        fw.write(f"{line.rstrip()}\t{string}\n")
                    else:
                        print(seq)
                        assert False
        fw.flush()
        fw.close()

    if single_B_res_file != "":
        suffix = single_B_res_file.split(".")[-1]
        out_tmp = single_B_res_file.replace(f".{suffix}", f"[new].{suffix}")
        fw = open(out_tmp, "w")
        # write first 2 lines
        if if_confident:
            for line in single_B_res_file_first_2lines:
                fw.write(line)
        else:
            fw.write(single_B_res_file_first_2lines[0])
            fw.write(single_B_res_file_first_2lines[1].rstrip() + "\tDecoy\tTarget(Protein/Top1Coverage/TopNCoverage)\n")
        print(f"Writing {out_tmp}")
        with open(single_B_res_file, "r") as f:
            lines = f.readlines()[2:]
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    fw.write(line)
                else:
                    line_split = line.split("\t")
                    seq = line_split[2]
                    mod = line_split[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    if seq in seq1_single_lys_dict:
                        #decoy
                        seq_dict = decoy_seq1_single_lys_dict[seq]
                        location_list = seq_dict.location
                        if len(location_list) == 1:
                            decoy = True
                        elif len(location_list) == 0:
                            decoy = False
                        else:
                            assert False
                        #写入文件
                        seq_dict = seq1_single_lys_dict[seq]
                        string = f"{decoy}\t"
                        location_list = seq_dict.location
                        if len(location_list) == 1:
                            for location in location_list:
                                for key,site_list in location:
                                    key_write = key.split('\t')[0].split()[0]
                                    string += f"({key_write}/{round(protein_top1_coverage_dict[key],2)}/{round(protein_topN_coverage_dict[key],2)})"
                        elif len(location_list) == 0:
                            string += f"(None/0/0)"
                        else:
                            assert False
                        fw.write(f"{line.rstrip()}\t{string}\n")
                    else:
                        assert False
        fw.flush()
        fw.close()

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

def merge_site_tag_and_intensity(match_type,try_site_tag,try_site_intensity,lys_site_tag,lys_site_intensity,try_intensity_list):
    if match_type == "A1:K-K" or match_type == "A2:R-R" or match_type == "B: R-K" or match_type == "C: K-R":
        site_tag_from_lys = np.ones(len(try_site_tag))
        site_tag_from_lys[:-1] = lys_site_tag[1:]
        site_tag = np.array(
            [int(try_site_tag[i]) | int(site_tag_from_lys[i]) for i in
             range(len(try_site_tag))])
        try_site_intensity[:-1] += lys_site_intensity[1:]  # lys匹配上的谱峰和try匹配上的加起来
    elif match_type == "F: X-K" or match_type == "G: X-R":
        site_tag_from_lys = np.ones(len(try_site_tag))
        site_tag_from_lys[:] = lys_site_tag[1:]
        site_tag = np.array(
            [int(try_site_tag[i]) | int(site_tag_from_lys[i]) for i in
             range(len(try_site_tag))])
        try_site_intensity[:] += lys_site_intensity[1:]  # lys匹配上的谱峰和try匹配上的加起来
    elif match_type == "D: K-X" or match_type == "E: R-X":
        site_tag_from_lys = np.ones(len(try_site_tag))
        site_tag_from_lys[:-1] = lys_site_tag[:]
        site_tag = np.array(
            [int(try_site_tag[i]) | int(site_tag_from_lys[i]) for i in
             range(len(try_site_tag))])
        try_site_intensity[:-1] += lys_site_intensity[:]  # lys匹配上的谱峰和try匹配上的加起来
    else:
        print(match_type)
        assert False

    try_intensity_list.sort()
    rank = len(try_intensity_list) - np.array(
        [bisect.bisect_left(try_intensity_list, i) for i in try_site_intensity]) + 1
    site_tag = [1 if tag == 1 and rank[i] <= topk_peaks else 0 for i, tag in
                enumerate(site_tag)]
    if match_type == "F: X-K" or match_type == "G: X-R":
        pass
    else:
        site_tag[-1] = 1
    miss_num = len(site_tag) - sum(site_tag)
    coverage = sum(site_tag) / len(site_tag)

    return site_tag,miss_num,coverage

def process_title(args):
    title, spectrum_to_confident_dict, spectrum_to_Rseq_dict, spectrum_to_confident_dict_temp1, spectrum_to_confident_dict_temp2, mgf1_location = args

    try_title = title.split("@")[0]
    lys_title = title.split("@")[1]
    match_type = spectrum_to_Rseq_dict[title][0][0]
    confident_res_dict = {}
    for item in spectrum_to_confident_dict[title]:
        try_seq,lys_seq,seq_ori = item
        try_seq = ''.join(try_seq)
        lys_seq = ''.join(lys_seq)
        try_siteandintensity = spectrum_to_confident_dict_temp1[try_title][try_seq]
        lys_siteandintensity = spectrum_to_confident_dict_temp2[lys_title][lys_seq]
        try_site_tag, try_site_intensity = try_siteandintensity[0], try_siteandintensity[1]
        lys_site_tag, lys_site_intensity = lys_siteandintensity[0], lys_siteandintensity[1]
        try_spectrum = read_mgf_to_generate_spectrum(try_title,mgf1_location[try_title][0],mgf1_location[try_title][1])
        try_intensity_list = try_spectrum.intensity_list
        site_tag,miss_num,coverage = merge_site_tag_and_intensity(match_type,try_site_tag,try_site_intensity,lys_site_tag,lys_site_intensity,try_intensity_list)
        confident_res_dict[seq_ori] = (site_tag,miss_num,coverage)
    return {title: confident_res_dict}

def parse_DiNovo_res_Confident_function(res_file_dict,mgf1_location,mgf2_location,res_type, processing_workers):

    spectrum_to_Rseq_dict = {}
    mirror_res_file = res_file_dict[f"{res_type}_mirror_res_file"]
    if mirror_res_file != "":
        print(f"Calculating ion coverage from {mirror_res_file}...")
        with open(mirror_res_file, "r") as f:
            lines = f.readlines()
            mirror_first_2_lines = lines[:2]
            lines = lines[2:]
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    match_type = line.split("\t")[2].strip()
                    title = line.split("\t")[0] + "@" + line.split("\t")[1]
                else:
                    line = line.split("\t")
                    seq = line[2]
                    mod = line[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    if title not in spectrum_to_Rseq_dict:
                        spectrum_to_Rseq_dict[title] = [(match_type,seq)]
                    else:
                        spectrum_to_Rseq_dict[title].append((match_type, seq))
        # print(spectrum_to_Rseq_dict)
        # input()

        # 去除重复的match_type和seq组合
        for title in spectrum_to_Rseq_dict:
            temp = spectrum_to_Rseq_dict[title]
            temp = list(set(temp))
            spectrum_to_Rseq_dict[title] = temp

        #把spectrum_to_Rseq_dict中每一项分解成try和lys两个部分
        spectrum_to_Rseq_dict_temp1 = {}
        spectrum_to_Rseq_dict_temp2 = {}
        spectrum_to_confident_dict = {}
        for title in spectrum_to_Rseq_dict:
            try_title = title.split("@")[0]
            lys_title = title.split("@")[1]
            for match_type,seq_ori in spectrum_to_Rseq_dict[title]:
                #seq = transfer_DiNovo_seq_to_SeqModSeq(seq_ori)
                seq = transfer_str_to_list_seq(seq_ori)
                try_seq,lys_seq = generate_try_lys_seq_from_RseqAndMatchtype(seq,match_type)
                if try_title not in spectrum_to_Rseq_dict_temp1:
                    spectrum_to_Rseq_dict_temp1[try_title] = [(match_type,try_seq)]
                else:
                    spectrum_to_Rseq_dict_temp1[try_title].append((match_type,try_seq))
                if lys_title not in spectrum_to_Rseq_dict_temp2:
                    spectrum_to_Rseq_dict_temp2[lys_title] = [(match_type,lys_seq)]
                else:
                    spectrum_to_Rseq_dict_temp2[lys_title].append((match_type,lys_seq))
                if title not in spectrum_to_confident_dict:
                    spectrum_to_confident_dict[title] = [(try_seq,lys_seq,seq_ori)]
                else:
                    spectrum_to_confident_dict[title].append((try_seq,lys_seq,seq_ori))

        with multiprocessing.Pool(processing_workers) as p:
            res_list_list1 = list(tqdm(
                p.imap(cal_ion_coverage_for_singleSpec_from_Rseq, [(title, mgf1_location[title][0], mgf1_location[title][1], spectrum_to_Rseq_dict_temp1[title]) for title in spectrum_to_Rseq_dict_temp1]), total=len(spectrum_to_Rseq_dict_temp1)))
        assert len(res_list_list1) == len(spectrum_to_Rseq_dict_temp1)
        spectrum_to_confident_dict_temp1 = {}
        for res_list in res_list_list1:
            title = res_list[-1]
            intensity_list = res_list[-2]
            seq_to_siteandintensity = {}
            for i,item in enumerate(spectrum_to_Rseq_dict_temp1[title]):
                seq = ''.join(item[1])
                siteandintensity = res_list[i]
                seq_to_siteandintensity[seq] = siteandintensity
            spectrum_to_confident_dict_temp1[title] = (seq_to_siteandintensity,intensity_list)

        with multiprocessing.Pool(processing_workers) as p:
            res_list_list2 = list(tqdm(
                p.imap(cal_ion_coverage_for_singleSpec_from_Rseq, [(title, mgf2_location[title][0], mgf2_location[title][1], spectrum_to_Rseq_dict_temp2[title]) for title in spectrum_to_Rseq_dict_temp2]), total=len(spectrum_to_Rseq_dict_temp2)))
        assert len(res_list_list2) == len(spectrum_to_Rseq_dict_temp2)
        spectrum_to_confident_dict_temp2 = {}
        for res_list in res_list_list2:
            title = res_list[-1]
            intensity_list = res_list[-2]
            seq_to_siteandintensity = {}
            for i,item in enumerate(spectrum_to_Rseq_dict_temp2[title]):
                seq = ''.join(item[1])
                siteandintensity = res_list[i]
                seq_to_siteandintensity[seq] = siteandintensity
            spectrum_to_confident_dict_temp2[title] = (seq_to_siteandintensity,intensity_list)

        #遍历
        time1 = time.time()
        spectrum_to_confidentres_dict = {}
        for title in tqdm(spectrum_to_confident_dict):
            try_title = title.split("@")[0]
            lys_title = title.split("@")[1]
            try_intensity_list = spectrum_to_confident_dict_temp1[try_title][1]
            match_type = spectrum_to_Rseq_dict[title][0][0]
            confident_res_dict = {}
            for item in spectrum_to_confident_dict[title]:
                try_seq,lys_seq,seq_ori = item
                try_seq = ''.join(try_seq)
                lys_seq = ''.join(lys_seq)
                try_siteandintensity = spectrum_to_confident_dict_temp1[try_title][0][try_seq]
                lys_siteandintensity = spectrum_to_confident_dict_temp2[lys_title][0][lys_seq]
                try_site_tag, try_site_intensity = try_siteandintensity[0], try_siteandintensity[1]
                lys_site_tag, lys_site_intensity = lys_siteandintensity[0], lys_siteandintensity[1]
                #try_spectrum = read_mgf_to_generate_spectrum(try_title,mgf1_location[try_title][0],mgf1_location[try_title][1])
                #try_intensity_list = try_spectrum.intensity_list
                site_tag,miss_num,coverage = merge_site_tag_and_intensity(match_type,try_site_tag,try_site_intensity,lys_site_tag,lys_site_intensity,try_intensity_list)
                confident_res_dict[seq_ori] = (site_tag,miss_num,coverage)
            spectrum_to_confidentres_dict[title] = confident_res_dict
        time2 = time.time()

        #写入文件
        suffix = mirror_res_file.split(".")[-1]
        out_tmp = mirror_res_file.replace(f".{suffix}",f"[new1].{suffix}")
        print(f"Writing to {out_tmp}")
        fw = open(out_tmp, "w")
        # write first 2 lines
        for line in mirror_first_2_lines:
            fw.write(line)
        # write res
        with open(mirror_res_file, "r") as f:
            lines = f.readlines()[2:]
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    line_split = line.split("\t")
                    title = line_split[0] + "@" + line_split[1]
                    fw.write(line)
                else:
                    line_split = line.split("\t")
                    seq = line_split[2]
                    mod = line_split[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    confident_res = spectrum_to_confidentres_dict[title][seq]
                    site_tag,miss_num,coverage = confident_res
                    fw.write(f"{line.rstrip()}\t{''.join(list(map(str,site_tag)))}\t{str(miss_num)}\t{str(round(coverage,4))}\n")
        fw.flush()
        fw.close()

######################################################################################################################33
    #单谱
    try_spectrum_to_seq_dict = {}
    lys_spectrum_to_seq_dict = {}
    singleA_res_file = res_file_dict[f"{res_type}_singleA_res_file"]
    if singleA_res_file != "":
        print(f"Calculating ion coverage from {singleA_res_file}...")
        with open(singleA_res_file, "r") as f:
            lines = f.readlines()
            singleA_first_2_lines = lines[:2]
            lines = lines[2:]
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    title = line.split("\t")[0].strip()
                else:
                    line = line.split("\t")
                    seq = line[2]
                    mod = line[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    if title not in try_spectrum_to_seq_dict:
                        try_spectrum_to_seq_dict[title] = [seq]
                    else:
                        try_spectrum_to_seq_dict[title].append(seq)

        for title in try_spectrum_to_seq_dict:
            temp = try_spectrum_to_seq_dict[title]
            temp = list(set(temp))
            try_spectrum_to_seq_dict[title] = temp

        # 多进程调用cal_coverage函数计算离子覆盖率
        with multiprocessing.Pool(processing_workers) as p:
            res_list_list1 = list(tqdm(
                p.imap(cal_ion_coverage_for_singleSpec,
                       [(title, mgf1_location[title][0], mgf1_location[title][1], try_spectrum_to_seq_dict[title])
                        for title in try_spectrum_to_seq_dict]), total=len(try_spectrum_to_seq_dict)))
        assert len(res_list_list1) == len(try_spectrum_to_seq_dict)

        try_spectrum_to_confident_dict = {}
        for res_list in res_list_list1:
            title = res_list[-1]
            intensity_list = res_list[-2]
            seq_to_siteandintensity = {}
            for i, item in enumerate(try_spectrum_to_seq_dict[title]):
                seq = item
                siteandintensity = res_list[i]
                seq_to_siteandintensity[seq] = siteandintensity
            try_spectrum_to_confident_dict[title] = (seq_to_siteandintensity, intensity_list)


    singleB_res_file = res_file_dict[f"{res_type}_singleB_res_file"]
    if singleB_res_file != "":
        print(f"Calculating ion coverage from {singleB_res_file}...")
        with open(singleB_res_file, "r") as f:
            lines = f.readlines()
            singleB_first_2_lines = lines[:2]
            lines = lines[2:]
            for line in tqdm(lines):
                if not line.startswith("\t"):
                    title = line.split("\t")[0].strip()
                else:
                    line = line.split("\t")
                    seq = line[2]
                    mod = line[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    if title not in lys_spectrum_to_seq_dict:
                        lys_spectrum_to_seq_dict[title] = [seq]
                    else:
                        lys_spectrum_to_seq_dict[title].append(seq)

        for title in lys_spectrum_to_seq_dict:
            temp = lys_spectrum_to_seq_dict[title]
            temp = list(set(temp))
            lys_spectrum_to_seq_dict[title] = temp

        with multiprocessing.Pool(processing_workers) as p:
            res_list_list2 = list(tqdm(
                p.imap(cal_ion_coverage_for_singleSpec, [(title, mgf2_location[title][0], mgf2_location[title][1], lys_spectrum_to_seq_dict[title]) for title in lys_spectrum_to_seq_dict]), total=len(lys_spectrum_to_seq_dict)))
        assert len(res_list_list2) == len(lys_spectrum_to_seq_dict)

        lys_spectrum_to_confident_dict = {}
        for res_list in res_list_list2:
            title = res_list[-1]
            intensity_list = res_list[-2]
            seq_to_siteandintensity = {}
            for i,item in enumerate(lys_spectrum_to_seq_dict[title]):
                seq = item
                siteandintensity = res_list[i]
                seq_to_siteandintensity[seq] = siteandintensity
            lys_spectrum_to_confident_dict[title] = (seq_to_siteandintensity,intensity_list)

################################################################################################
    #写入文件
    if singleA_res_file != "":
        suffix = singleA_res_file.split(".")[-1]
        out_tmp = singleA_res_file.replace(f".{suffix}", f"[new1].{suffix}")
        print(f"Writing to {out_tmp}")
        fw = open(out_tmp, "w")
        # write first 2 lines
        for line in singleA_first_2_lines:
            fw.write(line)
        print(f"Writing {out_tmp}")
        with open(singleA_res_file, "r") as f:
            lines = f.readlines()[2:]
            for line in lines:
                if not line.startswith("\t"):
                    title = line.split("\t")[0].strip()
                    fw.write(line)
                else:
                    line_split = line.split("\t")
                    seq = line_split[2].strip()
                    mod = line_split[3].strip()
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    confident_res = try_spectrum_to_confident_dict[title][0][seq]
                    site_tag,_ = confident_res
                    site_tag = list(map(int,site_tag))
                    miss_num = len(site_tag) - sum(site_tag)
                    coverage = round(sum(site_tag) / len(site_tag),2)
                    fw.write(f"{line.rstrip()}\t{''.join(list(map(str,site_tag)))}\t{str(miss_num)}\t{str(round(coverage,4))}\n")
        fw.flush()
        fw.close()

    if singleB_res_file != "":
        suffix = singleB_res_file.split(".")[-1]
        out_tmp = singleB_res_file.replace(f".{suffix}", f"[new1].{suffix}")
        print(f"Writing to {out_tmp}")
        fw = open(out_tmp, "w")
        # write first 2 lines
        for line in singleB_first_2_lines:
            fw.write(line)
        print(f"Writing {out_tmp}")
        with open(singleB_res_file, "r") as f:
            lines = f.readlines()
            singleB_first_2_lines = lines[:2]
            lines = lines[2:]
            for line in lines:
                if not line.startswith("\t"):
                    title = line.split("\t")[0].strip()
                    fw.write(line)
                else:
                    line_split = line.split("\t")
                    seq = line_split[2]
                    mod = line_split[3]
                    if len(mod) > 0:
                        seq_list = list(seq)
                        mod_list = mod.split(";")[:-1]
                        for mod_item in mod_list:
                            mod_item_list = mod_item.split(",")
                            idx, mod_name = mod_item_list[0], mod_item_list[1]
                            idx = int(idx)
                            mod = modifications_in_aa_dict_keys[modifications.index(mod_name)]
                            seq_list[idx - 1] = mod
                        seq = ''.join(seq_list)
                    confident_res = lys_spectrum_to_confident_dict[title][0][seq]
                    site_tag,_ = confident_res
                    site_tag = list(map(int,site_tag))
                    miss_num = len(site_tag) - sum(site_tag)
                    coverage = round(sum(site_tag) / len(site_tag),2)
                    fw.write(f"{line.rstrip()}\t{''.join(list(map(str,site_tag)))}\t{str(miss_num)}\t{str(round(coverage,4))}\n")
        fw.flush()
        fw.close()

def merge_find_and_confident_function(res_file_dict,res_type):
    if res_file_dict[f"{res_type}_singleA_res_file"] != "":
        single_A_res_file_confident = res_file_dict[f"{res_type}_singleA_res_file"].replace(".txt","[new1].txt")
        with open(single_A_res_file_confident, "r") as f:
            single_A_res_file_confident_lines = f.readlines()
        single_A_res_file_find = res_file_dict[f"{res_type}_singleA_res_file"].replace(".txt","[new].txt")
        with open(single_A_res_file_find, "r") as f:
            single_A_res_file_find_lines = f.readlines()
        with open(single_A_res_file_find, "w") as f:
            for i in range(len(single_A_res_file_find_lines)):
                find_line = single_A_res_file_find_lines[i]
                if i == 0 or i == 1:
                    if i == 1:
                        find_line = find_line.rstrip() + "\tDecoy\tTarget(Protein/Top1Coverage/TopNCoverage)\tSiteTag\tMissNum\tCoverage\n"
                    f.write(find_line)
                else:
                    if not find_line.startswith("\t"):
                        f.write(find_line)
                    else:
                        try:
                            confident_line = single_A_res_file_confident_lines[i]
                        except:
                            print(i,find_line)
                            print(len(single_A_res_file_confident_lines))
                            print(single_A_res_file_confident_lines[i-1])
                            assert False
                        f.write(f"{find_line.rstrip()}\t" + confident_line.strip().split('\t')[-3] + "\t" + confident_line.strip().split('\t')[-2] + "\t" + confident_line.strip().split('\t')[-1] + "\n")

    if res_file_dict[f"{res_type}_singleB_res_file"] != "":
        single_B_res_file_confident = res_file_dict[f"{res_type}_singleB_res_file"].replace(".txt","[new1].txt")
        with open(single_B_res_file_confident, "r") as f:
            single_B_res_file_confident_lines = f.readlines()
        single_B_res_file_find = res_file_dict[f"{res_type}_singleB_res_file"].replace(".txt","[new].txt")
        with open(single_B_res_file_find, "r") as f:
            single_B_res_file_find_lines = f.readlines()
        with open(single_B_res_file_find, "w") as f:
            for i in range(len(single_B_res_file_find_lines)):
                find_line = single_B_res_file_find_lines[i]
                if i == 0 or i == 1:
                    if i == 1:
                        find_line = find_line.rstrip() + "\tDecoy\tTarget(Protein/Top1Coverage/TopNCoverage)\tSiteTag\tMissNum\tCoverage\n"
                    f.write(find_line)
                else:
                    if not find_line.startswith("\t"):
                        f.write(find_line)
                    else:
                        confident_line = single_B_res_file_confident_lines[i]
                        f.write(f"{find_line.rstrip()}\t" + confident_line.strip().split('\t')[-3] + "\t" + confident_line.strip().split('\t')[-2] + "\t" + confident_line.strip().split('\t')[-1] + "\n")

    if res_file_dict[f"{res_type}_mirror_res_file"] != "":
        mirror_res_file_confident = res_file_dict[f"{res_type}_mirror_res_file"].replace(".txt","[new1].txt")
        with open(mirror_res_file_confident, "r") as f:
            mirror_res_file_confident_lines = f.readlines()
        mirror_res_file_find = res_file_dict[f"{res_type}_mirror_res_file"].replace(".txt","[new].txt")
        with open(mirror_res_file_find, "r") as f:
            mirror_res_file_find_lines = f.readlines()
        with open(mirror_res_file_find, "w") as f:
            for i in range(len(mirror_res_file_find_lines)):
                find_line = mirror_res_file_find_lines[i]
                if i == 0 or i == 1:
                    if i == 1:
                        find_line = find_line.rstrip() + "\tDecoy\tTargetSuffix(Protein/Top1Coverage/TopNCoverage)\tTargetPrefix(Protein/Top1Coverage/TopNCoverage)\tSiteTag\tMissNum\tCoverage\n"
                    f.write(find_line)
                else:
                    if not mirror_res_file_find_lines[i].startswith("\t"):
                        f.write(find_line)
                    else:
                        confident_line = mirror_res_file_confident_lines[i]
                        f.write(f"{find_line.rstrip()}\t" + confident_line.strip().split('\t')[-3] + "\t" + confident_line.strip().split('\t')[-2] + "\t" + confident_line.strip().split('\t')[-1] + "\n")

def delete_and_rename_resfile(res_file_dict,res_type,if_delete,if_rename):
    # 删除多余文件
    if if_delete:
        if res_file_dict[f"{res_type}_singleA_res_file"] != "":
            singleA_res_file_confident = res_file_dict[f"{res_type}_singleA_res_file"].replace(".txt", "[new1].txt")
            os.remove(singleA_res_file_confident)
        if res_file_dict[f"{res_type}_singleB_res_file"] != "":
            singleB_res_file_confident = res_file_dict[f"{res_type}_singleB_res_file"].replace(".txt", "[new1].txt")
            os.remove(singleB_res_file_confident)
        if res_file_dict[f"{res_type}_mirror_res_file"] != "":
            mirror_res_file_confident = res_file_dict[f"{res_type}_mirror_res_file"].replace(".txt", "[new1].txt")
            os.remove(mirror_res_file_confident)

    if res_file_dict[f"{res_type}_singleA_res_file"] != "":
        singleA_res_file_find = res_file_dict[f"{res_type}_singleA_res_file"]
        os.remove(singleA_res_file_find)
    if res_file_dict[f"{res_type}_singleB_res_file"] != "":
        singleB_res_file_find = res_file_dict[f"{res_type}_singleB_res_file"]
        os.remove(singleB_res_file_find)
    if res_file_dict[f"{res_type}_mirror_res_file"] != "":
        mirror_res_file_find = res_file_dict[f"{res_type}_mirror_res_file"]
        os.remove(mirror_res_file_find)

    # rename
    if if_rename:
        if res_file_dict[f"{res_type}_singleA_res_file"] != "":
            singleA_res_file_find = res_file_dict[f"{res_type}_singleA_res_file"].replace(".txt", "[new].txt")
            os.rename(singleA_res_file_find, res_file_dict[f"{res_type}_singleA_res_file"])
        if res_file_dict[f"{res_type}_singleB_res_file"] != "":
            singleB_res_file_find = res_file_dict[f"{res_type}_singleB_res_file"].replace(".txt", "[new].txt")
            os.rename(singleB_res_file_find, res_file_dict[f"{res_type}_singleB_res_file"])
        if res_file_dict[f"{res_type}_mirror_res_file"] != "":
            mirror_res_file_find = res_file_dict[f"{res_type}_mirror_res_file"].replace(".txt", "[new].txt")
            os.rename(mirror_res_file_find, res_file_dict[f"{res_type}_mirror_res_file"])

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

def analyse(res_file_dict,mgf1_location,mgf2_location,if_pNovoM2,if_MirrorNovo,if_confident, fasta_seq_dict, processing_workers):
    if if_pNovoM2:
        if if_confident:
            parse_DiNovo_res_Confident_function(res_file_dict,mgf1_location,mgf2_location,"pNovoM2", processing_workers)
        parse_DiNovo_res_find_function(res_file_dict,fasta_seq_dict,processing_workers,"pNovoM2",if_confident)
        if if_confident:
            merge_find_and_confident_function(res_file_dict,"pNovoM2")
            delete_and_rename_resfile(res_file_dict,"pNovoM2",if_delete=True,if_rename=True)
        else:
            delete_and_rename_resfile(res_file_dict,"pNovoM2",if_delete=False,if_rename=True)

    if if_MirrorNovo:
        if if_confident:
            parse_DiNovo_res_Confident_function(res_file_dict,mgf1_location,mgf2_location,"MirrorNovo", processing_workers)
        parse_DiNovo_res_find_function(res_file_dict,fasta_seq_dict,processing_workers,"MirrorNovo",if_confident)
        if if_confident:
            merge_find_and_confident_function(res_file_dict,"MirrorNovo")
            delete_and_rename_resfile(res_file_dict,"MirrorNovo",if_delete=True,if_rename=True)
        else:
            delete_and_rename_resfile(res_file_dict,"MirrorNovo",if_delete=False,if_rename=True)

def get_res_type(folder):
    filename_list = os.listdir(folder)
    if_DiNovo = False
    if_MirrorNovo = False
    for filename in filename_list:
        if "DiNovo" in filename:
            if_DiNovo = True
        if "MirrorNovo" in filename:
            if_MirrorNovo = True

    return if_DiNovo,if_MirrorNovo

# if __name__ == "__main__":

class CFunctionMapping:

    def __init__(self, inputDP):
        self.dp = inputDP

    def map(self):

        # input

        # initial
        pNovoM2_mirror_res_file = ""
        pNovoM2_singleA_res_file = ""
        pNovoM2_singleB_res_file = ""

        MirrorNovo_mirror_res_file = ""
        MirrorNovo_singleA_res_file = ""
        MirrorNovo_singleB_res_file = ""
        # """
        # pNovoM2 are selected
        if self.dp.myCFG.D9_DE_NOVO_APPROACH > 1:
            pNovoM2_mirror_res_file = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_PNOVOM_FINAL
            # de novo single
            if self.dp.myCFG.D12_DE_NOVO_SINGLE_SPECTRUM == 1:
                pNovoM2_singleA_res_file = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_PNOVOM_SINGLE_A_FINAL
                pNovoM2_singleB_res_file = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_PNOVOM_SINGLE_B_FINAL
        
        # MirrorNovo are selected
        if self.dp.myCFG.D9_DE_NOVO_APPROACH % 2 == 1:
            # the sequencing result of MirrorNovo
            MirrorNovo_mirror_res_file = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_GCNOVO_FINAL
            # de novo single
            if self.dp.myCFG.D12_DE_NOVO_SINGLE_SPECTRUM == 1:
                MirrorNovo_singleA_res_file = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_GCNOVO_SINGLE_A_FINAL
                MirrorNovo_singleB_res_file = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_GCNOVO_SINGLE_B_FINAL
        
        # """
        # the sequencing result of pNovoM2
        # pNovoM2_singleA_res_file = r"E:\MyPythonWorkSpace\DiNovo\aatest_0904-neucode\[pNovoM2]SingleSpecSeq_A.txt"
        # pNovoM2_singleB_res_file = r"E:\MyPythonWorkSpace\DiNovo\aatest_0904-neucode\[pNovoM2]SingleSpecSeq_B.txt"
        # pNovoM2_mirror_res_file = r"E:\MyPythonWorkSpace\DiNovo\aatest_0904-neucode\[pNovoM2]MirrorSpecSeq.txt"

        # fasta file path
        # fasta_file = r"G:\ResearchAssistant@AMSS\DiNovoData\BL21-UP000002032_469008.fasta"
        fasta_file = self.dp.myCFG.V3_PATH_FASTA_FILE

        # mgf file list, 1 for A and 2 for B
        # mgf1 = ["G:\\ResearchAssistant@AMSS\\DiNovoData\\NeuCodePhageT7\\try\\HF02_20240423_XP_LYC_T7_Trypsin_HCDFT.mgf"]
        # mgf2 = ["G:\\ResearchAssistant@AMSS\\DiNovoData\\NeuCodePhageT7\\lys\\HF02_20240423_XP_LYC_T7_lysargiNase_HCDFT.mgf"]

        mgf1 = self.dp.LIST_PATH_MGF_TRY
        mgf2 = self.dp.LIST_PATH_MGF_LYS

        #多进程
        processing_workers = 20
        #是否计算每条肽段的覆盖率，是否高可靠肽段
        #计算离子覆盖率稍微有点耗时（即使使用了多进程）。如果只需要回帖结果，那么这里设置为False，仅回帖只需耗时几分钟
        if_calculate_ion_coverage = True

    ################################################################################################################

        #下面的不用动
        if_pNovoM2 = True
        if_MirrorNovo = True
        if pNovoM2_mirror_res_file == "" and pNovoM2_singleA_res_file == "" and pNovoM2_singleB_res_file == "":
            if_pNovoM2 = False
        if MirrorNovo_mirror_res_file == "" and MirrorNovo_singleA_res_file == "" and MirrorNovo_singleB_res_file == "":
            if_MirrorNovo = False
        if not if_pNovoM2 and not if_MirrorNovo:
            logGetError("[MappingError] Some files are missing, please check results of pNovoM2 and MirrorNovo...")
            assert False

        try:
            fasta_seq_dict = build_fasta_dict_function(fasta_file)
            mgf1_location = build_mgf_location_function(mgf1)
            mgf2_location = build_mgf_location_function(mgf2)
        except:
            logGetError("[MappingError] Some files are missing, please check fasta and mgf files...")
            assert False

        #把上面6个文件路径变成字典，键为变量名，值为变量值
        res_file_dict = {"pNovoM2_singleA_res_file":pNovoM2_singleA_res_file,"pNovoM2_singleB_res_file":pNovoM2_singleB_res_file,"pNovoM2_mirror_res_file":pNovoM2_mirror_res_file,"MirrorNovo_singleA_res_file":MirrorNovo_singleA_res_file,"MirrorNovo_singleB_res_file":MirrorNovo_singleB_res_file,"MirrorNovo_mirror_res_file":MirrorNovo_mirror_res_file}
        analyse(res_file_dict,mgf1_location,mgf2_location,if_pNovoM2,if_MirrorNovo,if_calculate_ion_coverage, fasta_seq_dict, processing_workers)

