from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

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


#
# def get_sinusoid_encoding_table(n_position, embed_size, padding_idx=0):
#     """ Sinusoid position encoding table
#     n_position: maximum integer that the embedding op could receive
#     embed_size: embed size
#     return
#       a embedding matrix of shape [n_position, embed_size]
#     """
#
#     def cal_angle(position, hid_idx):
#         return position / np.power(config.sinusoid_base, 2 * (hid_idx // 2) / embed_size)
#
#     def get_posi_angle_vec(position):
#         return [cal_angle(position, hid_j) for hid_j in range(embed_size)]
#
#     sinusoid_matrix = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position + 1)], dtype=np.float32)
#
#     sinusoid_matrix[:, 0::2] = np.sin(sinusoid_matrix[:, 0::2])  # dim 2i
#     sinusoid_matrix[:, 1::2] = np.cos(sinusoid_matrix[:, 1::2])  # dim 2i+1
#
#     sinusoid_matrix[padding_idx] = 0.
#     #np.save(config.position_embedding_matrix, sinusoid_matrix)
#
#     return sinusoid_matrix


# if os.path.isfile(config.position_embedding_matrix):

# sinusoid_matrix = np.load(config.position_embedding_matrix)
# else:
# sinusoid_matrix = get_sinusoid_encoding_table(config.n_position, 512, padding_idx=config.PAD_ID)


def get_ion_index(peptide_mass, prefix_mass,input_id_list, direction, args):
    """

    :param peptide_mass: neutral mass of a peptide
    :param prefix_mass:
    :param direction: 0 for forward, 1 for backward
    :return: an int32 ndarray of shape [26, 8], each element represent a index of the spectrum embbeding matrix. for out
    of bound position, the index is 0
    """
    if args.type:
        mass_ID_np[4] = 156.10111
        config.mass_ID[4] = 156.10111

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

    # ion_2
    #~   b_ions = [candidate_b_mass]
    #~   y_ions = [candidate_y_mass]
    #~   ion_mass_list = b_ions + y_ions

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
        #internal_ay = [IM_ions] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)]
    elif len(input_id_list) == 2:
        internal_2ions=internal_aa_sum+mass_ID_np
        #internal_2ions_ay = internal_2ions - mass_CO
        internal_by = [internal_ions,internal_2ions] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)]
        #internal_ay = [IM_ions,internal_2ions_ay] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)]
    elif len(input_id_list) == 3:
        internal_2ions=internal_aa_sum+mass_ID_np
        internal_aa_sum2=sum(config.mass_ID[id] for id in input_id_list[-2:])
        internal_3ions=internal_aa_sum2+mass_ID_np
        #internal_2ions_ay=internal_aa_sum+mass_ID_np-mass_CO
        #internal_3ions_ay=internal_aa_sum2+mass_ID_np-mass_CO
        internal_by = [internal_ions,internal_2ions,internal_3ions] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)]
        #internal_ay = [IM_ions,internal_2ions_ay,internal_3ions_ay] + [[0.0] * len(mass_ID_np)] + [[0.0] * len(mass_ID_np)]
    elif len(input_id_list) == 4:
        internal_2ions=internal_aa_sum+mass_ID_np
        internal_aa_sum2=sum(config.mass_ID[id] for id in input_id_list[-2:])
        internal_3ions=internal_aa_sum2+mass_ID_np
        internal_aa_sum3=sum(config.mass_ID[id] for id in input_id_list[-3:])
        internal_4ions=internal_aa_sum3+mass_ID_np
        #internal_2ions_ay=internal_aa_sum+mass_ID_np-mass_CO
        #internal_3ions_ay=internal_aa_sum2+mass_ID_np-mass_CO
        #internal_4ions_ay=internal_aa_sum3+mass_ID_np-mass_CO
        internal_by = [internal_ions,internal_2ions,internal_3ions,internal_4ions] + [[0.0] * len(mass_ID_np)]
        #internal_ay = [IM_ions,internal_2ions_ay,internal_3ions_ay,internal_4ions_ay] + [[0.0] * len(mass_ID_np)]
    else:
        internal_2ions=internal_aa_sum+mass_ID_np
        internal_aa_sum2=sum(config.mass_ID[id] for id in input_id_list[-2:])
        internal_3ions=internal_aa_sum2+mass_ID_np
        internal_aa_sum3=sum(config.mass_ID[id] for id in input_id_list[-3:])
        internal_4ions=internal_aa_sum3+mass_ID_np
        internal_aa_sum4=sum(config.mass_ID[id] for id in input_id_list[-4:])
        internal_5ions=internal_aa_sum4+mass_ID_np
        #internal_2ions_ay=internal_aa_sum+mass_ID_np-mass_CO
        #internal_3ions_ay=internal_aa_sum2+mass_ID_np-mass_CO
        #internal_4ions_ay=internal_aa_sum3+mass_ID_np-mass_CO
        #internal_5ions_ay=internal_aa_sum4+mass_ID_np-mass_CO
        internal_by = [internal_ions,internal_2ions,internal_3ions,internal_4ions,internal_5ions]
        #internal_ay = [IM_ions,internal_2ions_ay,internal_3ions_ay,internal_4ions_ay,internal_5ions_ay]
    #ion_mass_list = b_ions + y_ions + a_ions + internal_by + internal_ay
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


# class args:
#     MAX_NUM_PEAK: int=10
#     MZ_MAX: float=200.0
#     position_embedding_dim: int=512
def process_peaks(spectrum_mz_list, spectrum_intensity_list, peptide_mass, args):
    charge = 1.0
    spectrum_intensity_max = np.max(spectrum_intensity_list)# if spectrum_intensity_list else 1.0
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
    sort_indices = sorted(enumerate(spectrum_mz_list), key=lambda x: x[1])
    spectrum_mz_list = [index[1] for index in sort_indices]
    spectrum_intensity_list = [spectrum_intensity_list[index[0]] for index in sort_indices]

    pad_to_length(spectrum_mz_list, args.MAX_NUM_PEAK)
    pad_to_length(spectrum_intensity_list, args.MAX_NUM_PEAK)

    spectrum_mz_location = np.array(spectrum_mz_list, dtype=np.float32)
    # spectrum_mz_location = np.ceil(spectrum_mz * config.spectrum_reso).astype(np.int32)

    neutral_mass = spectrum_mz_location - charge * mass_proton
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

    #
    #
    #
    #
    # neutral_mass = spectrum_mz - charge * mass_proton
    # in_bound_mask = np.logical_and(neutral_mass > 0., neutral_mass < args.MZ_MAX)
    # neutral_mass[~in_bound_mask] = 0.
    # # intensity
    # spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
    # norm_intensity = spectrum_intensity / spectrum_intensity_max
    #
    # top_N_indices = np.argpartition(norm_intensity, -args.MAX_NUM_PEAK)[-args.MAX_NUM_PEAK:]
    #
    # spectrum_mz_location = spectrum_mz_location[top_N_indices]
    # mass_location = neutral_mass[top_N_indices]
    # sort_intensity = norm_intensity[top_N_indices]
    #
    # sort_indices = np.argsort(spectrum_mz_location)
    #
    # spectrum_mz_location = spectrum_mz_location[sort_indices]
    # mass_location = mass_location[sort_indices]
    # sort_intensity = sort_intensity[sort_indices]

    # spectrum_representation = np.zeros(args.position_embedding_dim, dtype=np.float32)
    spectrum_representation = np.zeros(args.embedding_size, dtype=np.float32)
    # for i, loc in enumerate(spectrum_mz_location):
    # if loc < 0.5 or loc > config.n_position:
    # continue
    # else:
    # spectrum_representation += sinusoid_matrix[loc] * norm_intensity[i]
    # spectrum_representation[i] = sinusoid_matrix[loc] * norm_intensity[i]

    # print(mass_location, sort_intensity)

    return neutral_mass, norm_intensity, spectrum_representation

# peaks = [100.60915, 101.05979, 104.44301, 105.87823, 111.04419, 111.34513, 125.86681,
#                    129.05455, 132.06154, 134.13451]
#
# intensity = [1, 2, 3, 8, 24, 16, 3.6,
#                    9.7, 7.8, 5.5]
#
# a = args()
#
# peak, inten, embedding = process_peaks(spectrum_mz_list=peaks, spectrum_intensity_list=intensity, peptide_mass=190.00, args=a)
# # spectrum_mz_location = np.ceil(peaks * config.spectrum_reso).astype(np.int32)
# #
# # spectrum_representation = np.zeros([13, 512], dtype=np.float32)
# #
# # for i, loc in enumerate(spectrum_mz_location):
# #     if loc < 0.5 or loc > config.n_position:
# #         continue
# #     else:
# #         print(sinusoid_matrix[loc])
# #         spectrum_representation[i] = sinusoid_matrix[loc]
# # print(spectrum_representation.shape)
