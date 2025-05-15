# DeepNovoV2 is publicly available for non-commercial uses.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import os
# ==============================================================================
# FLAGS (options) for this app
# ==============================================================================



# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
assert PAD_ID == 0
vocab_reverse = ['A',
                 'R',
                 'N',
                 # 'N(Deamidation)',
                 'D',
                 # 'C',
                 'C(Carbamidomethylation)',
                 'E',
                 'Q',
                 # 'Q(Deamidation)',
                 'G',
                 'H',
                 # 'I',
                 'L',
                 'K',
                 'M',
                 'M(Oxidation)',
                 'F',
                 'P',
                 'S',
                 'T',
                 'W',
                 'Y',
                 'V',
                ]

vocab_reverse = _START_VOCAB + vocab_reverse
# print("vocab_reverse ", vocab_reverse)

vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
# print(vocab)
vocab_size = len(vocab_reverse)
# print("vocab_size ", vocab_size, vocab)
use_lstm = True


# ==============================================================================
# GLOBAL VARIABLES for THEORETICAL MASS
# ==============================================================================


mass_proton = 1.007276466621
mass_H = 1.007825
mass_H2O = 18.010564
mass_NH3 = 17.026548
mass_N_terminus = 1.00727
mass_C_terminus = 17.0027
mass_CO = 27.994914
mass_K = 128.0949557
mass_R = 156.1011021

mass_AA = {'_PAD': 0.0,
           '_GO': 0.0,
           '_EOS': mass_H2O,
           'A': 71.03711, # 0
           'R': 156.10111, # 1
           # 'R': 156.10111 + 3.988100,
           'N': 114.04293, # 2
           # 'N(Deamidation)': 115.02695,
           'D': 115.02694, # 3
           # 'C': 160.03065, # 4
           'C(Carbamidomethylation)': 160.03065, # C(+57.02)
           #~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           # 'Q(Deamidation)': 129.0426,
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           # 'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           # 'K': 128.09496 + 8.014200,
           'M': 131.04049, # 12
           'M(Oxidation)': 147.0354,
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           'T': 101.04768, # 16
           'W': 186.07931, # 17
           'Y': 163.06333, # 18
           'V': 99.06841, # 19
          }

mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]

# mass_ID_np = np.array(mass_ID, dtype=np.float32)
# mass_AA_np =mass_ID[2:]
# mass_AA_np.insert(0,mass_NH3)
# mass_AA_np = np.array(mass_AA_np, dtype=np.float32)
# mass_AA_np = np.array(mass_ID_np, dtype=np.float32)
# mass_AA_np_charge2 = mass_AA_np / 2
# print(mass_ID_np)
# print(mass_AA_np.shape)
# print(mass_AA_np)
# print(mass_AA_np_charge2)
mass_AA_min = mass_AA["G"] # 57.02146

MirrorFormat = {
    "A1:K-K":"A_KK",
    "A2:R-R":"A_RR",
    "B: R-K":"B",
    "C: K-R":"C",
    "D: K-X":"D",
    "E: R-X":"E",
    "F: X-K":"F",
    "G: X-R":"G",
    "H: X-X":"H",#MultiNovo
}
# '_GO': 0.0,#1
# 'R': 156.10111,  #4
# 'K': 128.09496,  #13
try_last_aa_mass_dict = {"A_KK":mass_AA["K"], "A_RR":mass_AA["R"], "B": mass_AA["R"], "C": mass_AA["K"], "D": mass_AA["K"], "E":mass_AA["R"], "F": mass_AA["_GO"], "G": mass_AA["_GO"], "H": mass_AA["_GO"]}
lys_first_aa_mass_dict = {"A_KK":mass_AA["K"], "A_RR":mass_AA["R"], "B": mass_AA["K"], "C": mass_AA["R"], "D": mass_AA["_GO"], "E":mass_AA["_GO"], "F": mass_AA["K"], "G": mass_AA["R"], "H": mass_AA["_GO"]}
try_last_aa_id_dict = {"A_KK": 13, "A_RR": 4, "B": 4, "C": 13, "D": 13, "E": 4, "F": 1, "G": 1, "H": 1}
lys_first_aa_id_dict = {"A_KK": 13, "A_RR": 4, "B": 13, "C": 4,"D": 1, "E": 1, "F": 13, "G": 4, "H": 1}



# ==============================================================================
# GLOBAL VARIABLES for PRECISION, RESOLUTION, temp-Limits of MASS & LEN
# ==============================================================================
MAX_LEN = 30

# position_embedding_matrix = "./knapsackfile/embedding_matrix.npy"

WINDOW_SIZE=10
#构建完全背包
KNAPSACK_AA_RESOLUTION = 10000 # 0.0001 Da
mass_AA_min_round = int(round(mass_AA_min * KNAPSACK_AA_RESOLUTION)) # 57.02146
# KNAPSACK_MASS_PRECISION_TOLERANCE = 100 # 0.01 Da
# PRECURSOR_MASS_PRECISION_TOLERANCE = 0.04
PRECURSOR_MASS_PRECISION_TOLERANCE_PPM = 20


# predicted file column format
pcol_feature_id = 0
pcol_feature_area = 1
pcol_sequence = 2
pcol_score = 3
pcol_position_score = 4
pcol_precursor_mz = 5
pcol_precursor_charge = 6
pcol_protein_id = 7
pcol_scan_list_middle = 8
pcol_scan_list_original = 9
pcol_score_max = 10


distance_scale_factor = 100.
sinusoid_base = 300000.0
spectrum_reso = 100
n_position = 3000 * spectrum_reso




