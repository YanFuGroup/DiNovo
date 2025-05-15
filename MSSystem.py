# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSSystem.py
Time: 2022.09.05 Monday
ATTENTION: none
"""
VALUE_MAX_SCAN = 2000000  # int 咱就是说，scan数目最多定这么多
VALUE_ILLEGAL = -7.16     # float 搞成整数有实际意义，不行

IO_NAME_FILE_CONFIG = ('DiNovo.cfg', "[Back-up]DiNovo.cfg")

IO_NAME_FILE_README = ("README.txt",)

# 可以调整名字，但是不要顺便改顺序喔，MSFlow里面有地址计算的方法
IO_NAME_FILE_RESULT = ("[MirrorFinder]MirrorSpecRes.txt", "[MirrorFinder]MirrorSpecDis.txt", # WFLOW_NUMB 2
                       "[DiNovo]MirrorSpecSeq.res", "[DiNovo]MirrorSpecSeq.candidate",       # WFLOW_NUMB 3
                       "[DiNovo]SingleSpecSeq.res", "[DiNovo]SingleSpecSeq.candidate")       # WFLOW_NUMB 4

# [MirrorFinder]MirrorSpecRes.txt
# [MirrorFinder]MirrorSpecDis.txt
# [pNovoM2]MirrorSpecSeq.txt
# [pNovoM2]SingleSpecSeq_A.txt
# [pNovoM2]SingleSpecSeq_B.txt
# [MirrorNovo]MirrorSpecSeq.txt
# [MirrorNovo]SingleSpecSeq_A.txt
# [MirrorNovo]SingleSpecSeq_B.txt
# IO_NAME_FILE_PNOVOM = "[DiNovo]pNovoMRes.txt"
IO_NAME_FILE_PNOVOM = "DenovoResultsFinal.txt"
IO_NAME_FILE_GCNOVO = "MirrorNovoResFinal.txt"
IO_NAME_FILE_PNOVOM_FINAL = "[pNovoM2]MirrorSpecSeq.txt"
IO_NAME_FILE_GCNOVO_FINAL = "[MirrorNovo]MirrorSpecSeq.txt"

IO_NAME_FILE_PNOVOM_SINGLE_A = "DenovoResultsFinal1.txt"
IO_NAME_FILE_PNOVOM_SINGLE_B = "DenovoResultsFinal2.txt"
# IO_NAME_FILE_PNOVOM_SINGLE_NEUCODE = "SingleDenovoFinalResults.txt"
IO_NAME_FILE_GCNOVO_SINGLE_A = "MirrorNovoResFinal1.txt."
IO_NAME_FILE_GCNOVO_SINGLE_B = "MirrorNovoResFinal2.txt."
IO_NAME_FILE_PNOVOM_SINGLE_A_FINAL = "[pNovoM2]SingleSpecSeq_A.txt"
IO_NAME_FILE_PNOVOM_SINGLE_B_FINAL = "[pNovoM2]SingleSpecSeq_B.txt"
IO_NAME_FILE_GCNOVO_SINGLE_A_FINAL = "[MirrorNovo]SingleSpecSeq_A.txt"
IO_NAME_FILE_GCNOVO_SINGLE_B_FINAL = "[MirrorNovo]SingleSpecSeq_B.txt"

IO_NAME_FOLDER_VALIDATION = "validation\\"
IO_NAME_FILE_VALIDATION_STAT = ("[DiNovo]SpectralPairs.validation.stat",
                                "[DiNovo]MirrorSequencing.validation.stat",)

IO_NAME_FOLDER_TEMPORARY = "temp\\"

IO_NAME_FOLDER_PARAMETER = "param\\"

IO_NAME_FOLDER_RESULT = "results\\"

IO_FEATURE_FILE_TP = "[OnlyTP]."

# 20221217 record
# 这部分我想了一下，也很适合作为后缀输出，即考虑分结果输出

# 当 WORK_FLOW_NUMBER 大于 1 时，才考虑输出下面的信息 =====

# WORK_FLOW_NUMBER == 2 && EXPORT_SPECTRAL_PAIR == 1 ######
# WORK_FLOW_NUMBER == 3 && EXPORT_SPECTRAL_PAIR == 1 ######
# 0. [DiNovo]SpectraPairs.res            寻找谱图对结果输出
# 1. [DiNovo]SpectraPairs.dis            寻找谱图对但非结果

# WORK_FLOW_NUMBER == 3
# 2. [DiNovo]MirrorSequencing.res        谱图对测序结果输出
# 3. [DiNovo]MirrorSequencing.candidate  谱图对测序候选结果

# WORK_FLOW_NUMBER == 4
# 4. [DiNovo]SingleSequencing.res        单张谱测序结果输出
# 5. [DiNovo]SingleSequencing.candidate  单张谱测序候选结果

# special log info
DE_NOVO_MODE_STR = {1: "[DiNovo]\t[Direct-read]\t",
                    2: "[DiNovo]\t[GCNovo mode]\t",
                    3: "[DiNovo]\t[pNovoM mode]\t"}

LABEL_TYPE_STR = {
    0: "[LABEL TYPE][None (Unlabeled)]",
    1: "[LABEL TYPE][Neucode  labeled]",
    2: "[LABEL TYPE][14N->15N labeled]",
    3: "[LABEL TYPE][12C->13C labeled]"
}

PREPROCESS_FILE_SUFFIX = ("[DiNovo][Preprocess][1st]",
                          "[DiNovo][Preprocess][2nd]",
                          "[DiNovo][NeuCodeAnnoOnly]")


FEATURE_MGF_FILE_SUFFIX = "[featureNeuCode].mgf"

PMIDX_FILE_SUFFIX = "[PMIDX]"

LOAD_FILE_FLAG = ("@", ".flag")# separate suffix

# 谱图对的类型（一个标记，也对应类型字符串）
SPECTRAL_PAIR_TYPE = {0:"A0:Unknown",
                      1:"A1:K-K", 2:"A2:R-R", 3:"A3:X-X",
                      4:"B: R-K", 5:"C: K-R", 6:"D: K-X",
                      7:"E: R-X", 8:"F: X-K", 9:"G: X-R"}

# reverse spectral pair type
REVERSE_SPECTRAL_PAIR_TYPE = dict()
for key in SPECTRAL_PAIR_TYPE:
    REVERSE_SPECTRAL_PAIR_TYPE[SPECTRAL_PAIR_TYPE[key]] = key
