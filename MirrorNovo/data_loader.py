import os
import math
import torch
from torch.utils.data import Dataset
from member import DDAFeature, DenovoData
import pickle
import csv
import re
import numpy as np


import config

mass_ID_np = config.mass_ID_np
GO_ID = config.GO_ID
EOS_ID = config.EOS_ID
mass_H2O = config.mass_H2O
mass_NH3 = config.mass_NH3
mass_H = config.mass_H
mass_CO = config.mass_CO
mass_proton = config.mass_proton
WINDOW_SIZE = config.WINDOW_SIZE
vocab_size = config.vocab_size


def parse_raw_sequence(raw_sequence: str):
    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
        if raw_sequence[index] == "(":
            if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":  # +57.021
                peptide[-1] = "C(Carbamidomethylation)"
                index += 8
            elif peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":  # +15.995
                peptide[-1] = 'M(Oxidation)'
                index += 8
            elif peptide[-1] == 'C' and raw_sequence[index] != "(":  # +15.995
                return False, peptide
            # elif peptide[-1] == 'N' and raw_sequence[index:index + 6] == "(+.98)":  # +0.984
            #     peptide[-1] = 'N(Deamidation)'
            #     index += 6
            # elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
            #     peptide[-1] = 'Q(Deamidation)'
            #     index += 6
            else:  # unknown modification
                # logger.warning(f"unknown modification in seq {raw_sequence}")
                return False, peptide
        else:
            peptide.append(raw_sequence[index])
            index += 1

    return True, peptide
# print(parse_raw_sequence('AIIISC(+57.02)TYIK'))

def to_tensor(data_dict: dict) -> dict:
    temp = [(k, torch.from_numpy(v)) for k, v in data_dict.items()]
    return dict(temp)


def pad_to_length(input_data: list, pad_token, max_length: int) -> list:
    assert len(input_data) <= max_length
    result = input_data[:]
    for i in range(max_length - len(result)):
        result.append(pad_token)
    return result


class GCNovoDenovoDataset(Dataset):
    def __init__(self, args, spectrum_path, mirror_spectrum_path, feature_path, transform=None):
        """
        read all feature information and store in memory,
        :param feature_filename:
        :param spectrum_filename:
        """
        print('start')
        # logger.info(f"input spectrum file: {spectrum_path}")
        # logger.info(f"input feature file: {feature_path}")
        self.args = args
        self.try_spectrum_filename = spectrum_path
        self.lys_spectrum_filename = mirror_spectrum_path
        self.input_spectrum_handle = None
        self.input_mirror_spectrum_handle = None
        self.feature_list = []
        self.try_spectrum_location_dict = self.load_MSData(spectrum_path)
        self.lys_spectrum_location_dict = self.load_MSData(mirror_spectrum_path)
        self.transform = transform
        # read spectrum location file

        # read feature file
        skipped_by_mass = 0
        skipped_by_ptm = 0
        skipped_by_length = 0
        skipped_by_file = 0
        with open(feature_path, 'r') as fr:
            reader = csv.reader(fr, delimiter='\t')
            header = next(reader)
            # feature_id_index = header.index(config.col_feature_id)
            # category_index = header.index(config.col_mirror_category)
            # mz_index = header.index(config.col_precursor_mz)
            # z_index = header.index(config.col_precursor_charge)
            # seq_index = header.index(config.col_raw_sequence)
            #
            # lys_mz_index = header.index(config.lys_col_precursor_mz)
            # lys_z_index = header.index(config.lys_col_precursor_charge)
            # lys_seq_index = header.index(config.lys_col_raw_sequence)
            TRY_TITLE  = header.index("TRY_TITLE")
            LYS_TITLE  = header.index("LYS_TITLE")
            TRY_CHARGE  = header.index("TRY_CHARGE")
            LYS_CHARGE  = header.index("LYS_CHARGE")
            TRY_PM  = header.index("TRY_PM")
            LYS_PM  = header.index("LYS_PM")
            
            # TRY_TITLE  = header.index("A_TITLE")
            # LYS_TITLE  = header.index("B_TITLE")
            # TRY_CHARGE  = header.index("A_CHARGE")
            # LYS_CHARGE  = header.index("B_CHARGE")
            # TRY_PM  = header.index("A_PM")
            # LYS_PM  = header.index("B_PM")
            
            DELTA_PM  = header.index("DELTA_PM")
            MIRROR_ANNO  = header.index("MIRROR_ANNO")
            for line in reader:
                mass = float(line[TRY_PM]) - config.mass_proton
                lys_mass = float(line[LYS_PM]) - config.mass_proton
                if not line[TRY_TITLE] in self.try_spectrum_location_dict.keys() and not line[LYS_TITLE] in self.lys_spectrum_location_dict.keys():
                    skipped_by_file += 1
                    print(f"{line[TRY_TITLE]}@{line[LYS_TITLE]} skipped by spectrum")
                    continue
                if mass > self.args.MZ_MAX or lys_mass>self.args.MZ_MAX:
                    skipped_by_mass += 1
                    print(f"{line[TRY_TITLE],mass} or {line[LYS_TITLE],lys_mass} skipped by mass")
                    continue
                new_feature = DDAFeature(feature_id=line[TRY_TITLE]+"@"+line[LYS_TITLE],
                                         category=config.MirrorFormat[line[MIRROR_ANNO]],
                                         mz=mass/int(line[TRY_CHARGE])+config.mass_proton,
                                         z=float(line[TRY_CHARGE]),
                                         peptide=None,
                                         mass=mass,
                                         lys_mz=lys_mass/int(line[LYS_CHARGE])+config.mass_proton,
                                         lys_z=float(line[LYS_CHARGE]),
                                         lys_peptide=None,
                                         lys_mass=lys_mass
                                         )
                self.feature_list.append(new_feature)
        print(f"read {len(self.feature_list)} features, {skipped_by_mass} skipped by mass, "
                    f"{skipped_by_ptm} skipped by unknown modification, {skipped_by_length} skipped by length"
                    f"{skipped_by_file} skipped by spectrum")

    def __len__(self):
        return len(self.feature_list)

    def close(self):
        self.input_spectrum_handle.close()

    def load_MSData(self, spectrum_path):
        spectrum_location_file = spectrum_path + '.location.pytorch.pkl'
        if os.path.exists(spectrum_location_file):
            print(f"read cached spectrum locations")
            with open(spectrum_location_file, 'rb') as fr:
                spectrum_location_dict = pickle.load(fr)
        else:
            print("build spectrum location from scratch")
            spectrum_location_dict = {}
            line = True
            with open(spectrum_path, 'r') as f:
                while line:
                    line = f.readline()
                    if "BEGIN IONS" in line:
                        spectrum_location = f.tell()
                    elif "TITLE=" in line:
                        title = re.split('[=\r\n]', line)[1]
                        spectrum_location_dict[title] = spectrum_location
            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(spectrum_location_dict, fw)
            print("build end")
        return spectrum_location_dict

    def _parse_spectrum_ion(self, input_spectrum_handle):
        mz_list = []
        intensity_list = []
        line = input_spectrum_handle.readline()
        while not "END IONS" in line:
            mz, intensity = re.split(' |\r|\n', line)[:2]
            mz_float = float(mz)
            intensity_float = float(intensity)
            # skip an ion if its mass > MZ_MAX
            if mz_float > self.args.MZ_MAX:
                line = input_spectrum_handle.readline()
                continue
            mz_list.append(mz_float)
            intensity_list.append(math.sqrt(intensity_float))
            line = input_spectrum_handle.readline()
        return mz_list, intensity_list

    def getmzandintensitylist(self, input_spectrum_handle):
        line = input_spectrum_handle.readline()
        # assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
        # line = input_spectrum_handle.readline()
        assert "TITLE=" in line, "Error: wrong input TITLE="
        line = input_spectrum_handle.readline()
        assert "CHARGE=" in line, "Error: wrong input CHARGE="
        line = input_spectrum_handle.readline()
        assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
        line = input_spectrum_handle.readline()
        assert "PEPMASS=" in line, "Error: wrong input PEPMASS="

        mz_list, intensity_list = self._parse_spectrum_ion(input_spectrum_handle)
        return mz_list, intensity_list

    def _get_theory_ions(self, peptide, mass):
        peptide_id_list = [config.vocab["L"] if x == "I" else config.vocab[x] for x in peptide]
        # peptide_id_list = [config.vocab[x] for x in peptide]
        forward_id_input = [config.GO_ID] + peptide_id_list
        forward_id_target = peptide_id_list + [config.EOS_ID]
        forward_ion_location_index_list = []
        prefix_mass = 0.
        for i, id in enumerate(forward_id_input):
            prefix_mass += config.mass_ID[id]
            ion_location = get_ion_index(mass, prefix_mass, forward_id_input[:i + 1], 0, args=self.args)
            forward_ion_location_index_list.append(ion_location)

        backward_id_input = [config.EOS_ID] + peptide_id_list[::-1]
        backward_id_target = peptide_id_list[::-1] + [config.GO_ID]
        backward_ion_location_index_list = []
        suffix_mass = 0
        for i, id in enumerate(backward_id_input):
            suffix_mass += config.mass_ID[id]
            ion_location = get_ion_index(mass, suffix_mass, backward_id_input[:i + 1], 1, args=self.args)
            backward_ion_location_index_list.append(ion_location)
        return forward_id_input, forward_id_target, forward_ion_location_index_list, backward_id_input, backward_id_target, backward_ion_location_index_list

    def _get_feature(self, feature: DDAFeature) -> DenovoData:
        try_filename, lys_filename = feature.feature_id.split("@")
        try_spectrum_location = self.try_spectrum_location_dict[try_filename]
        lys_spectrum_location = self.lys_spectrum_location_dict[lys_filename]
        self.input_spectrum_handle.seek(try_spectrum_location)
        self.input_mirror_spectrum_handle.seek(lys_spectrum_location)
        mz_list, intensity_list = self.getmzandintensitylist(self.input_spectrum_handle)
        lys_mz_list, lys_intensity_list = self.getmzandintensitylist(self.input_mirror_spectrum_handle)

        peak_location, peak_intensity = process_peaks(mz_list, intensity_list, feature.mass, self.args)
        lys_peak_location, lys_peak_intensity = process_peaks(lys_mz_list, lys_intensity_list, feature.lys_mass,
                                                              self.args)
        # print("mz list: ", mz_list[0], try_filename)
        # print("B mz list: ", lys_mz_list[0], lys_filename)
        return DenovoData(peak_location=peak_location,
                          peak_intensity=peak_intensity,
                          mirror_peak_location=lys_peak_location,
                          mirror_peak_intensity=lys_peak_intensity,
                          original_dda_feature=feature)
    def __getitem__(self, idx):
        if self.input_spectrum_handle is None:
            self.input_spectrum_handle = open(self.try_spectrum_filename, 'r')
        if self.input_mirror_spectrum_handle is None:
            self.input_mirror_spectrum_handle = open(self.lys_spectrum_filename, 'r')
        feature = self.feature_list[idx]
        return self._get_feature(feature)


def collate_func(train_data_list):
    """
    :param train_data_list: list of TrainData
    :return:
        peak_location: [batch, N]
        peak_intensity: [batch, N]
        forward_target_id: [batch, T]
        backward_target_id: [batch, T]
        forward_ion_index_list: [batch, T, 26, 8]
        backward_ion_index_list: [batch, T, 26, 8]
    """
    # sort data by seq length (decreasing order)
    train_data_list.sort(key=lambda x: len(x.forward_id_target), reverse=True)
    batch_max_seq_len = len(train_data_list[0].forward_id_target)
    ion_index_shape = train_data_list[0].forward_ion_location_index_list[0].shape

    peak_location = [x.peak_location for x in train_data_list]
    peak_location = np.stack(peak_location) # [batch_size, N]
    peak_location = torch.from_numpy(peak_location)

    peak_intensity = [x.peak_intensity for x in train_data_list]
    peak_intensity = np.stack(peak_intensity) # [batch_size, N]
    peak_intensity = torch.from_numpy(peak_intensity)

    lys_peak_location = [x.lys_peak_location for x in train_data_list]
    lys_peak_location = np.stack(lys_peak_location)  # [batch_size, N]
    lys_peak_location = torch.from_numpy(lys_peak_location)

    lys_peak_intensity = [x.lys_peak_intensity for x in train_data_list]
    lys_peak_intensity = np.stack(lys_peak_intensity)  # [batch_size, N]
    lys_peak_intensity = torch.from_numpy(lys_peak_intensity)

    batch_forward_ion_index = []
    lys_batch_forward_ion_index = []
    batch_forward_id_target = []
    batch_forward_id_input = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                               np.float32)
        lys_ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                                np.float32)
        forward_ion_index = np.stack(data.forward_ion_location_index_list)
        lys_forward_ion_index = np.stack(data.lys_forward_ion_location_index_list)
        ion_index[:forward_ion_index.shape[0], :, :] = forward_ion_index
        lys_ion_index[:lys_forward_ion_index.shape[0], :, :] = lys_forward_ion_index
        batch_forward_ion_index.append(ion_index)
        lys_batch_forward_ion_index.append(lys_ion_index)

        f_target = np.zeros((batch_max_seq_len,), np.int64)
        forward_target = np.array(data.forward_id_target, np.int64)
        f_target[:forward_target.shape[0]] = forward_target
        batch_forward_id_target.append(f_target)

        f_input = np.zeros((batch_max_seq_len,), np.int64)
        forward_input = np.array(data.forward_id_input, np.int64)
        f_input[:forward_input.shape[0]] = forward_input
        batch_forward_id_input.append(f_input)
    batch_forward_id_target = torch.from_numpy(np.stack(batch_forward_id_target))  # [batch_size, T]
    batch_forward_ion_index = torch.from_numpy(np.stack(batch_forward_ion_index))  # [batch, T, 26, 8]
    batch_forward_id_input = torch.from_numpy(np.stack(batch_forward_id_input))
    lys_batch_forward_ion_index = torch.from_numpy(np.stack(lys_batch_forward_ion_index))

    batch_backward_ion_index = []
    lys_batch_backward_ion_index = []
    batch_backward_id_target = []
    batch_backward_id_input = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                             np.float32)
        lys_ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                               np.float32)
        backward_ion_index = np.stack(data.backward_ion_location_index_list)
        ion_index[:backward_ion_index.shape[0], :, :] = backward_ion_index
        batch_backward_ion_index.append(ion_index)
        lys_backward_ion_index = np.stack(data.lys_backward_ion_location_index_list)
        lys_ion_index[:lys_backward_ion_index.shape[0], :, :] = lys_backward_ion_index
        lys_batch_backward_ion_index.append(lys_ion_index)

        b_target = np.zeros((batch_max_seq_len,), np.int64)
        backward_target = np.array(data.backward_id_target, np.int64)
        b_target[:backward_target.shape[0]] = backward_target
        batch_backward_id_target.append(b_target)

        b_input = np.zeros((batch_max_seq_len,), np.int64)
        backward_input = np.array(data.backward_id_input, np.int64)
        b_input[:backward_input.shape[0]] = backward_input
        batch_backward_id_input.append(b_input)
    batch_backward_id_target = torch.from_numpy(np.stack(batch_backward_id_target))  # [batch_size, T]
    batch_backward_ion_index = torch.from_numpy(np.stack(batch_backward_ion_index))  # [batch, T, 26, 8]
    batch_backward_id_input = torch.from_numpy(np.stack(batch_backward_id_input))
    lys_batch_backward_ion_index = torch.from_numpy(np.stack(lys_batch_backward_ion_index))

    # lys_batch_forward_ion_index = []
    # for data in train_data_list:
    #     ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
    #                          np.float32)
    #     forward_ion_index = np.stack(data.lys_forward_ion_location_index_list[:1] + data.lys_forward_ion_location_index_list[2:])
    #     ion_index[:forward_ion_index.shape[0], :, :] = forward_ion_index
    #     lys_batch_forward_ion_index.append(ion_index)
    #
    # lys_batch_forward_ion_index = torch.from_numpy(np.stack(lys_batch_forward_ion_index))  # [batch, T, 26, 8]
    #
    # lys_batch_backward_ion_index = []
    # for data in train_data_list:
    #     ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
    #                          np.float32)
    #     backward_ion_index = np.stack(data.lys_backward_ion_location_index_list[:-1])
    #     ion_index[:backward_ion_index.shape[0], :, :] = backward_ion_index
    #     lys_batch_backward_ion_index.append(ion_index)
    #
    # lys_batch_backward_ion_index = torch.from_numpy(np.stack(lys_batch_backward_ion_index))  # [batch, T, 26, 8]

    return (peak_location,
            peak_intensity,
            lys_peak_location,
            lys_peak_intensity,
            batch_forward_id_target,
            batch_backward_id_target,
            batch_forward_ion_index,
            batch_backward_ion_index,
            batch_forward_id_input,
            batch_backward_id_input,
            lys_batch_forward_ion_index,
            lys_batch_backward_ion_index
            )

# helper functions
def chunks(l, n: int):
    for i in range(0, len(l), n):
        yield l[i:i + n]
def process_peaks(spectrum_mz_list, spectrum_intensity_list, peptide_mass, args):

    charge = 1.0
    spectrum_intensity_max = np.max(spectrum_intensity_list)
    # charge 1 peptide location 1电荷肽的位置  C端终点
    spectrum_mz_list.append(peptide_mass + charge * mass_proton)
    spectrum_intensity_list.append(spectrum_intensity_max)

    # N-terminal, b-ion, peptide_mass_C
    # append N-terminal 1个电荷的位置 N端起点
    spectrum_mz_list.append(charge * mass_proton)
    spectrum_intensity_list.append(spectrum_intensity_max)

    # append peptide_mass_C C端失去水针对b离子的情况，肽不加水分子的位置 N端终点
    mass_C = mass_H2O
    peptide_mass_C = peptide_mass - mass_C
    spectrum_mz_list.append(peptide_mass_C + charge * mass_proton)
    spectrum_intensity_list.append(spectrum_intensity_max)

    # C-terminal, y-ion, peptide_mass_N
    # append C-terminal 针对y离子的情况,加入水分子点18的位置  C端起点
    mass_C = mass_H2O
    spectrum_mz_list.append(mass_C + charge * mass_proton)
    spectrum_intensity_list.append(spectrum_intensity_max)

    # sort before padding
    sort_indices = sorted(enumerate(spectrum_mz_list), key=lambda x:x[1])
    spectrum_mz_list = [index[1] for index in sort_indices]
    spectrum_intensity_list = [spectrum_intensity_list[index[0]] for index in sort_indices]

    pad_to_length(spectrum_mz_list, args.MAX_NUM_PEAK)
    pad_to_length(spectrum_intensity_list, args.MAX_NUM_PEAK)

    spectrum_mz = np.array(spectrum_mz_list, dtype=np.float32)
    spectrum_mz_location = np.ceil(spectrum_mz * config.spectrum_reso).astype(np.int32)

    neutral_mass = spectrum_mz - charge * mass_proton
    in_bound_mask = np.logical_and(neutral_mass > 0., neutral_mass < args.MZ_MAX)
    neutral_mass[~in_bound_mask] = 0.

    spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
    norm_intensity = spectrum_intensity / spectrum_intensity_max

    if len(spectrum_mz_list) > args.MAX_NUM_PEAK:
        # get intensity top max_peaks
        top_N_indices = np.argpartition(norm_intensity, -args.MAX_NUM_PEAK)[-args.MAX_NUM_PEAK:]

        spectrum_mz_location = spectrum_mz_location[top_N_indices]
        neutral_mass = neutral_mass[top_N_indices]
        norm_intensity = norm_intensity[top_N_indices]

        # sort mz
        sort_indices = np.argsort(spectrum_mz_location)

        spectrum_mz_location = spectrum_mz_location[sort_indices]
        neutral_mass = neutral_mass[sort_indices]
        norm_intensity = norm_intensity[sort_indices]
    return neutral_mass, norm_intensity
def get_ion_index(peptide_mass, prefix_mass,input_id_list, direction, args):
    """

    :param peptide_mass: neutral mass of a peptide
    :param prefix_mass:
    :param direction: 0 for forward, 1 for backward
    :return: an int32 ndarray of shape [26, 8], each element represent a index of the spectrum embbeding matrix. for out
    of bound position, the index is 0
    """
    if direction == 0:
        candidate_b_mass = prefix_mass + mass_ID_np
        candidate_y_mass = peptide_mass - candidate_b_mass
    elif direction == 1:
        candidate_y_mass = prefix_mass + mass_ID_np
        candidate_b_mass = peptide_mass - candidate_y_mass
    candidate_a_mass = candidate_b_mass - mass_CO

    # b-ions
    candidate_b_H2O = candidate_b_mass - mass_H2O
    candidate_b_NH3 = candidate_b_mass - mass_NH3
    candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * mass_proton) / 2
                                 - mass_H)

    # a-ions
    candidate_a_H2O = candidate_a_mass - mass_H2O
    candidate_a_NH3 = candidate_a_mass - mass_NH3
    candidate_a_plus2_charge1 = ((candidate_a_mass + 2 * mass_proton) / 2
                                 - mass_H)

    # y-ions
    candidate_y_H2O = candidate_y_mass - mass_H2O
    candidate_y_NH3 = candidate_y_mass - mass_NH3
    candidate_y_plus2_charge1 = ((candidate_y_mass + 2 * mass_proton) / 2
                                 - mass_H)

    # ion_8
    b_ions = [candidate_b_mass,
              candidate_b_H2O,
              candidate_b_NH3,
              candidate_b_plus2_charge1]
    y_ions = [candidate_y_mass,
              candidate_y_H2O,
              candidate_y_NH3,
              candidate_y_plus2_charge1]
    a_ions = [candidate_a_mass,
              candidate_a_H2O,
              candidate_a_NH3,
              candidate_a_plus2_charge1]
    # a_ions = [candidate_a_mass]
    internal_ions = mass_ID_np
    IM_ions = mass_ID_np - mass_CO
    internal_aa_sum=sum(config.mass_ID[id] for id in input_id_list[-1:])
    if len(input_id_list) == 1:
        internal_by = [internal_ions] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)]
    elif len(input_id_list) == 2:
        internal_2ions=internal_aa_sum+mass_ID_np
        internal_by = [internal_ions,internal_2ions] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)]
    elif len(input_id_list) == 3:
        internal_2ions=internal_aa_sum+mass_ID_np
        internal_aa_sum2=sum(config.mass_ID[id] for id in input_id_list[-2:])
        internal_3ions=internal_aa_sum2+mass_ID_np
        internal_by = [internal_ions,internal_2ions,internal_3ions] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)]
    elif len(input_id_list) == 4:
        internal_2ions=internal_aa_sum+mass_ID_np
        internal_aa_sum2=sum(config.mass_ID[id] for id in input_id_list[-2:])
        internal_3ions=internal_aa_sum2+mass_ID_np
        internal_aa_sum3=sum(config.mass_ID[id] for id in input_id_list[-3:])
        internal_4ions=internal_aa_sum3+mass_ID_np
        internal_by = [internal_ions,internal_2ions,internal_3ions,internal_4ions] + [[0.0] * len(mass_ID_np)]
    else:
        internal_2ions=internal_aa_sum+mass_ID_np
        internal_aa_sum2=sum(config.mass_ID[id] for id in input_id_list[-2:])
        internal_3ions=internal_aa_sum2+mass_ID_np
        internal_aa_sum3=sum(config.mass_ID[id] for id in input_id_list[-3:])
        internal_4ions=internal_aa_sum3+mass_ID_np
        internal_aa_sum4=sum(config.mass_ID[id] for id in input_id_list[-4:])
        internal_5ions=internal_aa_sum4+mass_ID_np
        internal_by = [internal_ions,internal_2ions,internal_3ions,internal_4ions,internal_5ions]

    ion_mass_list = b_ions + y_ions + a_ions + internal_by + [IM_ions]
    # ion_mass_list = b_ions + y_ions + a_ions
    ion_mass = np.array(ion_mass_list, dtype=np.float32)  # 8 by 26
    # print("ion_mass: ", ion_mass.shape)
    # ion locations
    # ion_location = np.ceil(ion_mass * SPECTRUM_RESOLUTION).astype(np.int64) # 8 by 26

    in_bound_mask = np.logical_and(
        ion_mass > 0,
        ion_mass <= args.MZ_MAX).astype(np.float32)
    ion_location = ion_mass * in_bound_mask  # 8 by 26, out of bound index would have value 0
    return ion_location.transpose()  # 26 by 8

def pad_to_length(data: list, length, pad_token=0.):
    """
    pad data to length if len(data) is smaller than length
    :param data:
    :param length:
    :param pad_token:
    :return:
    """
    for i in range(length - len(data)):
        data.append(pad_token)
