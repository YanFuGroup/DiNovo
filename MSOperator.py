# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSOperator.py
Time: 2022.09.05 Monday
ATTENTION: none
"""
import os
from MSData import CTagDAGEdge, CMS2Spectrum
from MSData import CINI, Config
from MSTool import toolCountCharInString, toolGetWord
from MSLogging import logGetError
from MSSystem import VALUE_MAX_SCAN, VALUE_ILLEGAL

import locale

# 获取系统默认编码
default_encoding = locale.getpreferredencoding()

# ???
def op_TRANS_PATH_USING_OS_ENCODING(inputPath):

    # 将utf-8编码的文件路径转换为系统默认编码
    return inputPath.encode('utf-8').decode(default_encoding)

# --------------------------------
# input: CFileMS2
# function: initial MS2File
# void
# --------------------------------
def op_INIT_CFILE_MS2(inputMS2):
    inputMS2.INDEX_SCAN = []  # 重新赋值为一个空的 vector<int>
    inputMS2.INDEX_RT = []  # 重新赋值为一个空的 vector<float>

    # 下面三个LIST，都重新赋值为一个 VALUE_MAX_SCAN 长度的 list<float>，相当于是scan的hash table
    inputMS2.LIST_RET_TIME = [VALUE_ILLEGAL] * VALUE_MAX_SCAN
    inputMS2.LIST_ION_INJECTION_TIME = [VALUE_ILLEGAL] * VALUE_MAX_SCAN
    inputMS2.LIST_ACTIVATION_CENTER = [VALUE_ILLEGAL] * VALUE_MAX_SCAN

    # 下面五个MATRIX，都重新赋值为 VALUE_MAX_SCAN 长度的 list<vector<float/string/int>>
    # ATTENTION 这里相当于是个二维的数组了，第一维的个数是固定的 VALUE_MAX_SCAN，相当于是scan的hash table
    # 第二维是vector，存着咱们需要的信息，也就是谱峰质荷比与强度，母离子质量、质荷比与电荷，母离子对应的file name等
    # 这里我不确定用list还是直接用array，反正最外面的第一维，我们用定长的就行
    inputMS2.MATRIX_PEAK_MOZ = [[] * 1] * VALUE_MAX_SCAN  # list<vector<float>>
    inputMS2.MATRIX_PEAK_INT = [[] * 1] * VALUE_MAX_SCAN  # list<vector<float>>
    # 相同的scan，可能有多个母离子状态（质量+电荷）
    inputMS2.MATRIX_FILE_NAME = [[] * 1] * VALUE_MAX_SCAN  # list<vector<string>>
    inputMS2.MATRIX_PRECURSOR_CHARGE = [[] * 1] * VALUE_MAX_SCAN  # list<vector<int>>
    inputMS2.MATRIX_PRECURSOR_MOZ = [[] * 1] * VALUE_MAX_SCAN  # list<vector<float>>

# --------------------------------
# input: dataDAG:CDAG, left_node_index:int, right_node_index:int, mass_tol:float, aa_info_mark:int
# function: add edge to DAG node
# void
# --------------------------------
def op_ADD_EDGE_CTagDAG(dataDAG, left_node_index, right_node_index, mass_tol, aa_info_mark):
    # 在出边的结点处，添加对象
    dataDAG.LIST_NODE[left_node_index].LIST_EDGE.append(CTagDAGEdge())

    # 算一个打分
    dataDAG.LIST_NODE[left_node_index].LIST_EDGE[-1].WEIGHT = dataDAG.LIST_NODE[left_node_index].INTENSITY + \
                                                              dataDAG.LIST_NODE[
                                                                  right_node_index].INTENSITY
    # 标记一下氨基酸组的信息
    dataDAG.LIST_NODE[left_node_index].LIST_EDGE[-1].AA_MARK = aa_info_mark
    # 指向哪一个结点，int类型
    dataDAG.LIST_NODE[left_node_index].LIST_EDGE[-1].LINK_NODE_INDEX = right_node_index
    # 边的差值记录一下
    dataDAG.LIST_NODE[left_node_index].LIST_EDGE[-1].TOL = mass_tol
    # 出入度各加一，这里都是整数哈
    dataDAG.LIST_NODE[left_node_index].OUT += 1
    dataDAG.LIST_NODE[right_node_index].IN += 1


# --------------------------------
# input: CDataPack
# function: initial data package
# void
# --------------------------------
def op_INIT_CFILE_DATAPACK(inputDP):
    inputDP.myINI = CINI()
    inputDP.myCFG = Config()

    # INI
    inputDP.myINI.MASS_ELECTRON = 0.0005485799
    inputDP.myINI.MASS_PROTON_MONO = 1.00727645224  # 1.00782503214-0.0005485799
    inputDP.myINI.MASS_PROTON_ARVG = 1.0025
    inputDP.myINI.MASS_NEUTRON_AVRG = 1.003

    inputDP.myINI.DICT0_ELEMENT_MASS = {}
    inputDP.myINI.DICT0_ELEMENT_ABDC = {}

    inputDP.myINI.DICT1_AA_COM = {}
    inputDP.myINI.DICT1_AA_MASS = {}
    inputDP.myINI.DICT2_MOD_INFO = {}
    inputDP.myINI.DICT2_MOD_MASS = {}

    # Config
    # [INI FILE]
    inputDP.myCFG.I0_INI_PATH_ELEMENT = "element.ini"
    inputDP.myCFG.I1_INI_PATH_AA = "aa.ini"
    inputDP.myCFG.I2_INI_PATH_MOD = "modification.ini"

    # [DATA PARAM]
    inputDP.myCFG.A0_PATH_CFG_FILE = ""
    inputDP.myCFG.A1_PATH_MGF_TRY = ""
    inputDP.myCFG.A2_PATH_MGF_LYS = ""
    inputDP.myCFG.A3_FIX_MOD = ""
    inputDP.myCFG.A4_VAR_MOD = ""
    inputDP.myCFG.A5_MIN_PRECURSOR_MASS = 300.0
    inputDP.myCFG.A6_MAX_PRECURSOR_MASS = 3500.0
    inputDP.myCFG.A7_LABEL_TYPE = 0
    inputDP.myCFG.A8_NEUCODE_OUTPUT_TYPE = 1
    inputDP.myCFG.A9_DOUBLE_PEAK_GAP = 0.036
    inputDP.myCFG.A10_NEUCODE_PEAK_MAX_RATIO = 2.0
    inputDP.myCFG.A11_MIRROR_PROTEASES_APPROACH = 1

    # [Preprocess]
    # piyu preprocess part
    inputDP.myCFG.B1_ADD_ISOTOPE_INTENSITY = 1
    inputDP.myCFG.B2_CHECK_NATURAL_LOSS = 1
    inputDP.myCFG.B3_REMOVE_PRECURSOR_ION = 0
    inputDP.myCFG.B4_CHECK_PRECURSOR_CHARGE = 1
    inputDP.myCFG.B5_ROUND_ONE_HOLD_PEAK_NUM = 200
    # zixuan preprocess part
    inputDP.myCFG.B6_ROUND_TWO_REMOVE_IMMONIUM = 1
    inputDP.myCFG.B7_ROUND_TWO_PEAK_NUM_PER_BIN = 4
    inputDP.myCFG.B8_ROUND_TWO_MASS_BIN_LENTH = 100
    # xinming classification manuscript
    inputDP.myCFG.B9_NEUCODE_DETECT_APPROACH = 1
    inputDP.myCFG.B10_CLASSIFICATION_MODEL_PATH = ""

    # [Pairing]
    inputDP.myCFG.C0_MIRROR_DELTA_RT = 900.0
    inputDP.myCFG.C1_MIRROR_TYPE_A1 = 1
    inputDP.myCFG.C2_MIRROR_TYPE_A2 = 1
    inputDP.myCFG.C3_MIRROR_TYPE_A3 = 0
    inputDP.myCFG.C4_MIRROR_TYPE_B = 1
    inputDP.myCFG.C5_MIRROR_TYPE_C = 1
    inputDP.myCFG.C6_MIRROR_TYPE_D = 1
    inputDP.myCFG.C7_MIRROR_TYPE_E = 1
    inputDP.myCFG.C8_MIRROR_TYPE_F = 1
    inputDP.myCFG.C9_MIRROR_TYPE_G = 1
    inputDP.myCFG.C10_PAIR_FILTER_APPROACH = 3
    inputDP.myCFG.C11_PAIR_P_VALUE_THRESHOLD = 0.05
    inputDP.myCFG.C12_PAIR_DECOY_APPROACH = 2
    inputDP.myCFG.C13_PAIR_DECOY_SHIFTED_IN_DA = 15.0
    inputDP.myCFG.C14_PAIR_FDR_THRESHOLD = 0.02

    # [De Novo]
    inputDP.myCFG.D0_WORK_FLOW_NUMBER = 1
    inputDP.myCFG.D1_MEMORY_LOAD_MODE = 1
    inputDP.myCFG.D2_MULTIPROCESS_NUM = 1
    inputDP.myCFG.D3_MS_TOL = 20
    inputDP.myCFG.D4_MS_TOL_PPM = 1
    inputDP.myCFG.D5_MSMS_TOL = 20
    inputDP.myCFG.D6_MSMS_TOL_PPM = 1
    inputDP.myCFG.D7_HOLD_CANDIDATE_NUM = 400
    inputDP.myCFG.D8_REPORT_PEPTIDE_NUM = 10
    inputDP.myCFG.D9_DE_NOVO_APPROACH = 1
    inputDP.myCFG.D10_MIRROR_NOVO_MODEL_PATH = ""
    inputDP.myCFG.D11_PNOVOM_EXE_PATH = ""
    inputDP.myCFG.D12_DE_NOVO_SINGLE_SPECTRUM = 0
    inputDP.myCFG.D13_BATCH_SIZE = 2

    # [EXPORT]
    inputDP.myCFG.E1_PATH_EXPORT = '.\\test\\'
    inputDP.myCFG.E2_EXPORT_ROUND_ONE_MGF = 0
    inputDP.myCFG.E3_EXPORT_ROUND_TWO_MGF = 0
    inputDP.myCFG.E4_EXPORT_SPECTRAL_PAIR = 1
    inputDP.myCFG.E5_COMBINE_SPLIT_RESULT = 0
    inputDP.myCFG.E6_COMBINE_TOTAL_RESULT = 1
    inputDP.myCFG.E7_EXPORT_FEATURE_MGF = 0

    # [VALIDATION]
    inputDP.myCFG.V0_FLAG_DO_VALIDATION = 0
    inputDP.myCFG.V1_PATH_TRY_PFIND_RES = ""
    inputDP.myCFG.V2_PATH_LYS_PFIND_RES = ""
    inputDP.myCFG.V3_PATH_FASTA_FILE    = ""

    # out param info
    inputDP.LIST_PATH_MGF_TRY = []
    inputDP.LIST_PATH_MGF_LYS = []
    inputDP.LIST_OUTPUT_PATH = []
    inputDP.LIST_MGF_NAME_TRY = []
    inputDP.LIST_MGF_NAME_LYS = []


def op_FILL_LIST_PATH_MS(inputPath, inputList, inputExt):
    separator = '|'
    listStrPath = []

    if len(inputPath) < 1:
        return False
        # logGetError("MSOperator.py, op_FILL_LIST_PATH_MS, Path for MS is empty!")

    if inputPath[-1] == separator:
        pass
    else:
        inputPath = inputPath + separator

    nFile = toolCountCharInString(inputPath, separator)

    for i in range(nFile):
        listStrPath.append(toolGetWord(inputPath, i, separator))

    for strPath in listStrPath:

        if os.path.isdir(strPath):

            for maindir, subdir, file_name_list in os.walk(strPath):

                for filename in file_name_list:

                    tmpPath = os.path.join(maindir, filename)  # 合并成一个完整路径

                    ext = os.path.splitext(tmpPath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容

                    if ext in inputExt:
                        inputList.append(os.path.abspath(tmpPath))

        elif os.path.isfile(strPath):

            inputList.append(os.path.abspath(strPath))

        else:

            logGetError("\n[PATH ERROR][Operator] Path for MGF is illegal!\tpath: " + strPath)

    if inputList:
        return True
    else:
        return False


# ban ----------------------------------------------
# 功能是获取单张Spectrum，且母离子信息只有一个
# inputFileMS2是CFileMS2，这里要引用
# index是整数类型
# [ATTENTION] 这里比较特殊，只把对应的母离子记录下来
def op_getSinglePrecursorMS2Spectrum(self, inputFileMS2, index, precurosr_i):
        outputSpectrum = CMS2Spectrum()
        outputSpectrum.LIST_FILE_NAME = inputFileMS2.MATRIX_FILE_NAME[index]
        outputSpectrum.LIST_PRECURSOR_CHARGE = inputFileMS2.MATRIX_PRECURSOR_CHARGE[index]
        outputSpectrum.LIST_PRECURSOR_MOZ = inputFileMS2.MATRIX_PRECURSOR_MOZ[index]
        outputSpectrum.LIST_PEAK_MOZ = inputFileMS2.MATRIX_PEAK_MOZ[index]
        outputSpectrum.LIST_PEAK_INT = inputFileMS2.MATRIX_PEAK_INT[index]
        outputSpectrum.LIST_PRECURSOR_MASS = []  # 我太迷了
        outputSpectrum.SCAN_RET_TIME = inputFileMS2.LIST_RET_TIME[index]
        outputSpectrum.NEUCODE_LABEL = []       # redirect
        for i in range(len(outputSpectrum.LIST_PRECURSOR_CHARGE)):
            tmpMass = (outputSpectrum.LIST_PRECURSOR_MOZ[i] - self.dp.myINI.MASS_PROTON_MONO) * \
                      outputSpectrum.LIST_PRECURSOR_CHARGE[i] + self.dp.myINI.MASS_PROTON_MONO
            outputSpectrum.LIST_PRECURSOR_MASS.append(tmpMass)

        return outputSpectrum
