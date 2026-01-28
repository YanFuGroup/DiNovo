import os
import math
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import csv
import re
import logging
from dataclasses import dataclass

import config
from cython_modules import get_ion_index, process_peaks

logger = logging.getLogger(__name__)


def parse_raw_sequence(raw_sequence: str):
    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
        if raw_sequence[index] == "(":
            if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":
                peptide[-1] = "C(Carbamidomethylation)"
                index += 8
            elif peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":
                peptide[-1] = 'M(Oxidation)'
                index += 8
            elif peptide[-1] == 'N' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'N(Deamidation)'
                index += 6
            elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'Q(Deamidation)'
                index += 6
            else:  # unknown modification
                logger.warning(f"unknown modification in seq {raw_sequence}")
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


@dataclass
class DDAFeature:
    feature_id: str
    mz: float
    z: float
    rt_mean: float
    peptide: list
    scan: str
    mass: float
    feature_area: str

@dataclass
class DenovoData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    precursormass: float
    spectrum_representation: np.ndarray
    original_dda_feature: DDAFeature

@dataclass
class MGFfeature:
    PEPMASS: float
    CHARGE: int
    SCANS: str
    SEQ: str
    RTINSECONDS: float
    MOZ_LIST: list
    INTENSITY_LIST: list

@dataclass
class TrainData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    precursormass: float
    spectrum_representation: np.ndarray
    forward_id_target: list
    backward_id_target: list
    forward_ion_location_index_list: list
    backward_ion_location_index_list: list
    forward_id_input: list
    backward_id_input: list


class GCNovoTrainDataset(Dataset):
    def __init__(self, feature_filename, spectrum_filename, args, transform=None):
        """
        read all feature information and store in memory,
        :param feature_filename:
        :param spectrum_filename:
        """
        print('start')
        logger.info(f"input spectrum file: {spectrum_filename}")
        logger.info(f"input feature file: {feature_filename}")
        self.args = args
        self.spectrum_filename = spectrum_filename
        self.input_spectrum_handle = None
        self.feature_list = []
        self.spectrum_location_dict = {}
        self.transform = transform
        self.number_str_set = {str(i) for i in range(10)}
        # read spectrum location file
        spectrum_location_file = spectrum_filename + '.location.pytorch.pkl'
        if os.path.exists(spectrum_location_file):
            logger.info(f"read cached spectrum locations")
            with open(spectrum_location_file, 'rb') as fr:
                self.spectrum_location_dict = pickle.load(fr)
        else:
            logger.info("build spectrum location from scratch")
            spectrum_location_dict = {}
            # line = True
            with open(spectrum_filename, 'r') as f:
                line = f.readline()
                while line:
                    current_location = f.tell()
                    
                    # print(line)
                    if (line[0] == "B") and ("BEGIN IONS" in line):  # "BEGIN IONS" in line:
                        spectrum_location = current_location
                    elif (line[0] == "T") and ("TITLE" in line):
                        # scan = re.split('[=\r\n]', line)[1]
                        scan = line[6:-1]
                        spectrum_location_dict[scan] = spectrum_location
                    line = f.readline()
            self.spectrum_location_dict = spectrum_location_dict
            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(self.spectrum_location_dict, fw)

        # read feature file
        skipped_by_mass = 0
        skipped_by_ptm = 0
        skipped_by_length = 0
        with open(feature_filename, 'r') as fr:
            reader = csv.reader(fr, delimiter=',')
            header = next(reader)
            feature_id_index = header.index(config.col_feature_id)
            mz_index = header.index(config.col_precursor_mz)
            z_index = header.index(config.col_precursor_charge)
            rt_mean_index = header.index(config.col_rt_mean)
            seq_index = header.index(config.col_raw_sequence)
            scan_index = header.index(config.col_scan_list)
            feature_area_index = header.index(config.col_feature_area)
            for line in reader:
                # print(line)
                pep_moz = float(line[mz_index].split(" ")[0])
                mass = (pep_moz - config.mass_H) * float(line[z_index])
                ok, peptide = parse_raw_sequence(line[seq_index])
                if not ok:
                    skipped_by_ptm += 1
                    logger.debug(f"{line[seq_index]} skipped by ptm")
                    continue
                if mass > self.args.MZ_MAX:
                    skipped_by_mass += 1
                    logger.debug(f"{line[seq_index],mass} skipped by mass")
                    continue
                if len(peptide) >= self.args.MAX_LEN:
                    skipped_by_length += 1
                    logger.debug(f"{line[seq_index]} skipped by length")
                    continue
                new_feature = DDAFeature(feature_id=line[feature_id_index],
                                         mz=pep_moz,
                                         z=float(line[z_index]),
                                         rt_mean=float(line[rt_mean_index]),
                                         peptide=peptide,
                                         scan=line[scan_index],
                                         mass=mass,
                                         feature_area=line[feature_area_index])
                self.feature_list.append(new_feature)
        logger.info(f"read {len(self.feature_list)} features, {skipped_by_mass} skipped by mass, "
                    f"{skipped_by_ptm} skipped by unknown modification, {skipped_by_length} skipped by length")

    def __len__(self):
        return len(self.feature_list)

    def close(self):
        self.input_spectrum_handle.close()

    def _parse_spectrum_ion(self, tmp_line):
        mz_list = []
        intensity_list = []
        # line = self.input_spectrum_handle.readline()
        line = tmp_line
        # print("====" * 9)
        # while not "END IONS" in line:
        while line and (line[0] in self.number_str_set):
            # print(line)
            mz, intensity = re.split(' |\r|\n', line)[:2]
            mz_float = float(mz)
            intensity_float = float(intensity)
            # skip an ion if its mass > MZ_MAX
            if mz_float > self.args.MZ_MAX:
                line = self.input_spectrum_handle.readline()
                continue
            mz_list.append(mz_float)
            intensity_list.append(math.sqrt(intensity_float))
            # intensity_list.append(intensity_float)
            line = self.input_spectrum_handle.readline()
        return mz_list, intensity_list

    def _get_feature(self, feature: DDAFeature) -> TrainData:
        spectrum_location = self.spectrum_location_dict[feature.feature_id]
        self.input_spectrum_handle.seek(spectrum_location)
        # parse he

        line = self.input_spectrum_handle.readline()
        while line:
            if line[0] in self.number_str_set:
                break
            line = self.input_spectrum_handle.readline()
        # while True:
        #     if "BEGIN IONS" in line:
        #         line = self.input_spectrum_handle.readline()
        #         continue
        #     elif "TITLE=" in line:
        #         line = self.input_spectrum_handle.readline()
        #         continue
        #     elif "PEPMASS=" in line:
        #         # break
        #         line = self.input_spectrum_handle.readline()
        #         continue
        #     elif "CHARGE=" in line:
        #         line = self.input_spectrum_handle.readline()
        #         continue
        #     elif "SCANS=" in line:
        #         line = self.input_spectrum_handle.readline()
        #         continue
        #     elif "RTINSECONDS=" in line:
        #         line = self.input_spectrum_handle.readline()
        #         continue
        #     elif "SEQ=" in line:
        #         line = self.input_spectrum_handle.readline()
        #         continue
        #     else:
        #         break

        mz_list, intensity_list = self._parse_spectrum_ion(line)

        # mz_list = mgffeature.MOZ_LIST
        # intensity_list = mgffeature.INTENSITY_LIST
        peak_location, peak_intensity, spectrum_representation = process_peaks(mz_list, intensity_list, feature.mass, self.args)

        assert np.max(peak_intensity) < 1.0 + 1e-5

        # print("feature.peptide",feature.peptide,feature.scan)

        peptide_id_list = [config.vocab[x] for x in feature.peptide]
        forward_id_input = [config.GO_ID] + peptide_id_list
        forward_id_target = peptide_id_list + [config.EOS_ID]
        forward_ion_location_index_list = []
        prefix_mass = 0.
        for i, id in enumerate(forward_id_input):
            prefix_mass += config.mass_ID[id]
            ion_location = get_ion_index(feature.mass, prefix_mass, forward_id_input[:i+1], 0, args=self.args)
            forward_ion_location_index_list.append(ion_location)

        backward_id_input = [config.EOS_ID] + peptide_id_list[::-1]
        backward_id_target = peptide_id_list[::-1] + [config.GO_ID]
        backward_ion_location_index_list = []
        suffix_mass = 0
        for i, id in enumerate(backward_id_input):
            suffix_mass += config.mass_ID[id]
            ion_location = get_ion_index(feature.mass, suffix_mass,backward_id_input[:i+1], 1, args=self.args)
            backward_ion_location_index_list.append(ion_location)

        return TrainData(peak_location=peak_location,
                         peak_intensity=peak_intensity,
                         precursormass=feature.mass,
                         spectrum_representation=spectrum_representation,
                         forward_id_target=forward_id_target,
                         backward_id_target=backward_id_target,
                         forward_ion_location_index_list=forward_ion_location_index_list,
                         backward_ion_location_index_list=backward_ion_location_index_list,
                         forward_id_input=forward_id_input,
                         backward_id_input=backward_id_input)

    def __getitem__(self, idx):
        if self.input_spectrum_handle is None:
            self.input_spectrum_handle = open(self.spectrum_filename, 'r')
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
    # print(ion_index_shape, config.vocab_size, config.num_ion)
    # assert ion_index_shape == (config.vocab_size, config.num_ion)

    peak_location = [x.peak_location for x in train_data_list]
    peak_location = np.stack(peak_location) # [batch_size, N]
    peak_location = torch.from_numpy(peak_location)

    peak_intensity = [x.peak_intensity for x in train_data_list]
    peak_intensity = np.stack(peak_intensity) # [batch_size, N]
    peak_intensity = torch.from_numpy(peak_intensity)

    spectrum_representation = [x.spectrum_representation for x in train_data_list]
    spectrum_representation = np.stack(spectrum_representation)  # [batch_size, embed_size]
    spectrum_representation = torch.from_numpy(spectrum_representation)

    batch_forward_ion_index = []
    batch_forward_id_target = []
    batch_forward_id_input = []
    batch_precursormass = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                               np.float32)
        forward_ion_index = np.stack(data.forward_ion_location_index_list)
        ion_index[:forward_ion_index.shape[0], :, :] = forward_ion_index
        batch_forward_ion_index.append(ion_index)

        f_target = np.zeros((batch_max_seq_len,), np.int64)
        forward_target = np.array(data.forward_id_target, np.int64)
        f_target[:forward_target.shape[0]] = forward_target
        batch_forward_id_target.append(f_target)

        f_input = np.zeros((batch_max_seq_len,), np.int64)
        forward_input = np.array(data.forward_id_input, np.int64)
        f_input[:forward_input.shape[0]] = forward_input
        batch_forward_id_input.append(f_input)

        batch_precursormass.append(data.precursormass)

    batch_forward_id_target = torch.from_numpy(np.stack(batch_forward_id_target))  # [batch_size, T]
    batch_forward_ion_index = torch.from_numpy(np.stack(batch_forward_ion_index))  # [batch, T, 26, 8]
    batch_forward_id_input = torch.from_numpy(np.stack(batch_forward_id_input))
    batch_precursormass = torch.from_numpy(np.array(batch_precursormass))

    batch_backward_ion_index = []
    batch_backward_id_target = []
    batch_backward_id_input = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                             np.float32)
        backward_ion_index = np.stack(data.backward_ion_location_index_list)
        ion_index[:backward_ion_index.shape[0], :, :] = backward_ion_index
        batch_backward_ion_index.append(ion_index)

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

    return (peak_location,
            peak_intensity,
            batch_precursormass,
            spectrum_representation,
            batch_forward_id_target,
            batch_backward_id_target,
            batch_forward_ion_index,
            batch_backward_ion_index,
            batch_forward_id_input,
            batch_backward_id_input
            )


# helper functions
def chunks(l, n: int):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class GCNovoDenovoDataset(GCNovoTrainDataset):
    # override _get_feature method
    def _get_feature(self, feature: DDAFeature) -> DenovoData:
        spectrum_location = self.spectrum_location_dict[feature.scan]
        self.input_spectrum_handle.seek(spectrum_location)
        line = self.input_spectrum_handle.readline()
        # print("====" * 9)
        # 此处的函数，旨在获取谱图的离子信息
        # 暂与BEGIN IONS后的谱图基本信息无关，因此可以直接隐去
        # 以数字字符串来替代之
        # ------------------------
        while line:
            if line[0] in self.number_str_set:
                break
            line = self.input_spectrum_handle.readline()

        # print("====" * 9)
        mz_list, intensity_list = self._parse_spectrum_ion(line)

        # mz_list = mgffeature.MOZ_LIST
        # intensity_list = mgffeature.INTENSITY_LIST
        peak_location, peak_intensity, spectrum_representation = process_peaks(mz_list, intensity_list, feature.mass, self.args)

        return DenovoData(peak_location=peak_location,
                          peak_intensity=peak_intensity,
                          precursormass=feature.mass,
                          spectrum_representation=spectrum_representation,
                          original_dda_feature=feature)


if __name__=='__main__':
    train_set = GCNovoTrainDataset(config.input_feature_file_train,
                                     config.input_spectrum_file_train)
    print('finished')