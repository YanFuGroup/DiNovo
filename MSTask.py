# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSTask.py
Time: 2022.09.05 Monday
ATTENTION: none
"""
import os, psutil, copy, shutil
import time, pickle
from MSOperator import op_FILL_LIST_PATH_MS
from MSFunctionIO import CFunctionINI, CFunctionParseMGF, CFunctionLoadPKL, CFunctionWritePrepMGF
from MSFunctionIO import CFunctionTempPath
from MSFunctionPreprocess import CFunctionPreprocess, CFunctionPreprocessNeuCode, CFunctionPreprocessForXueLi, CFunctionPreprocessForXueLiNeuCode
from MSFunctionNeuCodeDetection import CFunctionNeuCodeDetection
from MSFunctionMirrorFinder import CFunctionPrecursorMassIndex, CFunctionMirrorFinder
from MSTool import toolGetNameFromPath, toolGenerateMirrorPairTimeDetail
from MSLogging import logGetError, logGetWarning, logToUser, INFO_TO_USER_TaskReadMGF
from MSLogging import INFO_TO_USER_TaskDiNovo, INFO_TO_USER_TaskPrep
from MSLogging import INFO_TO_USER_TaskPMIDX, INFO_TO_USER_TaskPair
from MSSystem import PREPROCESS_FILE_SUFFIX, IO_NAME_FILE_README, IO_NAME_FILE_RESULT, IO_FEATURE_FILE_TP
from MSSystem import SPECTRAL_PAIR_TYPE, REVERSE_SPECTRAL_PAIR_TYPE, DE_NOVO_MODE_STR, IO_NAME_FOLDER_VALIDATION
from MSSystem import IO_NAME_FILE_GCNOVO, IO_NAME_FILE_PNOVOM, IO_NAME_FOLDER_TEMPORARY, IO_NAME_FOLDER_PARAMETER
from MSSystem import IO_NAME_FILE_CONFIG, FEATURE_MGF_FILE_SUFFIX, IO_NAME_FOLDER_RESULT
from MSFunctionDeNovo import CFunctionDeNovo
import multiprocessing as mp
from MSValidationZixuan import CValidationZixuan
from MSValidationXueli import build_location_function, cal_Dinovo_result, cal_Pnovom_result


# 统一做同样的二进制存取！
class CTaskReadMGF:

    def __init__(self, inputDP):
        self.dp = inputDP

    def work(self):
        logToUser(INFO_TO_USER_TaskReadMGF[0])

        label_t = op_FILL_LIST_PATH_MS(self.dp.myCFG.A1_PATH_MGF_TRY, self.dp.LIST_PATH_MGF_TRY, [".mgf", ".MGF"])

        label_l = op_FILL_LIST_PATH_MS(self.dp.myCFG.A2_PATH_MGF_LYS, self.dp.LIST_PATH_MGF_LYS, [".mgf", ".MGF"])

        if label_t and label_l:
            pass

        # 只有一个文件的list存在，另一个是空着的
        elif label_t or label_l:
            if (self.dp.myCFG.D0_WORK_FLOW_NUMBER == 4) or (self.dp.myCFG.D0_WORK_FLOW_NUMBER == 1):

                if label_t:
                    logGetWarning("\n[WARNING]\t[B]-protease MGF file list is NULL(BUT we still go on).\t[WARNING]")
                else:
                    logGetWarning("\n[WARNING]\t[A]-protease MGF file list is NULL(BUT we still go on).\t[WARNING]")

            elif (self.dp.myCFG.D0_WORK_FLOW_NUMBER == 2) or (self.dp.myCFG.D0_WORK_FLOW_NUMBER == 3):

                    if label_t:
                        logGetError("\n[FILE ERROR]\tYou want pairing spectra, but LysargiNase MGF file is NULL! Please check param [PATH_MGF_B]!")
                    else:
                        logGetError("\n[FILE ERROR]\tYou want pairing spectra, but Trypsin MGF file is NULL! Please check param [PATH_MGF_A]!")

        else:
            logGetError("\n[FILE ERROR]\tPaths for MGF file are both NULL! Please check your configuration!")

        functionMS2 = CFunctionParseMGF(self.dp)
        # ATTENTION
        # 这里调用的就是 Cython 封装的函数了
        for path in self.dp.LIST_PATH_MGF_TRY:
            ms2_name = toolGetNameFromPath(path.replace("/", "\\"))
            self.dp.LIST_MGF_NAME_TRY.append(ms2_name)

            logToUser(INFO_TO_USER_TaskReadMGF[1] + "\t[" + ms2_name + "]")
            functionMS2.mgfTOpkl(path)

        for path in self.dp.LIST_PATH_MGF_LYS:
            ms2_name = toolGetNameFromPath(path.replace("/", "\\"))
            self.dp.LIST_MGF_NAME_LYS.append(ms2_name)

            logToUser(INFO_TO_USER_TaskReadMGF[1] + "\t[" + ms2_name + "]")
            functionMS2.mgfTOpkl(path)


# 初始化信息
class CTaskReadINI:

    def __init__(self, inputDP):
        self.dp = inputDP

    def work(self):

        functionINI = CFunctionINI(self.dp)
        # functionINI.ini2file()  # 这个一般不需要运行，是一开始弄符合格式要求的文件才需要调用；
        functionINI.file2ini()

        # 20210907 MOD这里做一下检查 ----------------------------------------------------------
        if self.dp.myCFG.A3_FIX_MOD:
            for mod_name in (self.dp.myCFG.A3_FIX_MOD).split("|"):
                if mod_name == "":
                    pass
                elif mod_name in self.dp.myINI.DICT2_MOD_INFO:
                    pass
                else:
                    logGetError("FIX_MOD: " + self.dp.myCFG.A3_FIX_MOD + " is NOT legal info! please check it again!")

        # exit("I got here")
        # check_list = []
        if self.dp.myCFG.A4_VAR_MOD:
            for mod_name in (self.dp.myCFG.A4_VAR_MOD).split("|"):
                if mod_name == "":
                    pass
                elif mod_name in self.dp.myINI.DICT2_MOD_INFO:
                    # check_list.append(mod_name)
                    pass
                else:
                    logGetError("VAR_MOD: " + self.dp.myCFG.A4_VAR_MOD + " is NOT legal info! please check it again!")

        R_MASS = 156.10111  # Original mass! Do not change it!
        K_MASS = 128.09496  # Original mass! Do not change it!
        self.dp.myINI.DICT1_AA_MASS[ord("K") - ord("A")] = K_MASS
        self.dp.myINI.DICT1_AA_MASS[ord("R") - ord("A")] = R_MASS

        if self.dp.myCFG.A7_LABEL_TYPE == 1:
            self.__neucodeChangeMassKR()

    def __neucodeChangeMassKR(self):

        # basic element [H]  1H:  1.0078246  2H:  2.0141021
        # basic element [C] 12C: 12.0000000 13C: 13.0033554
        # basic element [N] 14N: 14.0030732 15N: 15.0001088
        #  amino acid   [K] C(6)H(12)N(2)O(1)S(0)
        #  amino acid   [R] C(6)H(12)N(4)O(1)S(0)
        # [isotope label]
        add_H = 1.0062775
        # [ 1H ->  2H]: 1.0062775 =  2.0141021 -  1.0078246
        add_C = 1.0033554
        # [12C -> 13C]: 1.0033554 = 13.0033554 - 12.0000000
        add_N = 0.9970356
        # [14N -> 15N]: 0.9970356 = 15.0001088 - 14.0030732
        # [Neuron-encoded]
        # [K602] +8.0142Da, [K080] +8.0502Da | [K602] 13C*6 + 2H*0 + 15N*2. | [K080] 13C*0 + 2H*8 + 15N*0.
        # [R004] +3.9881Da, [R040] +4.0251Da | [R004] 13C*0 + 2H*0 + 15N*4. | [R040] 13C*0 + 2H*4 + 15N*0.
        K602 = (add_C * 6) + (add_H * 0) + (add_N * 2)
        K080 = (add_C * 0) + (add_H * 8) + (add_N * 0)  # heavy label, you could choose it when you want to use
        R004 = (add_C * 0) + (add_H * 0) + (add_N * 4)
        R040 = (add_C * 0) + (add_H * 4) + (add_N * 0)  # heavy label, you could choose it when you want to use

        # print("K:", self.dp.myINI.DICT1_AA_MASS[ord("K") - ord("A")])
        # print("R:", self.dp.myINI.DICT1_AA_MASS[ord("R") - ord("A")])
        self.dp.myINI.DICT1_AA_MASS[ord("K") - ord("A")] += K602
        print("add_K:", K602)

        if self.dp.myCFG.A11_MIRROR_PROTEASES_APPROACH == 1:
            self.dp.myINI.DICT1_AA_MASS[ord("R") - ord("A")] += R004
            print("add_R:", R004)

        #
        # print("K:", self.dp.myINI.DICT1_AA_MASS[ord("K") - ord("A")])
        # print("R:", self.dp.myINI.DICT1_AA_MASS[ord("R") - ord("A")])


# 运行环境检查
class CTaskCheck:

    # 有很多东西可以check：内、外存、文件等等；

    def __init__(self, inputDP):

        self.dp = inputDP

    def work(self):

        self.__checkExportPath()
        self.__backUpParam()

        self.dp.myCFG.E1_PATH_EXPORT += IO_NAME_FOLDER_RESULT

        # exp_path = self.dp.myCFG.E1_PATH_EXPORT
        # self.__cleanAll(exp_path)

        self.__generateTempPath()
        self.__checkInputParam()

        # exit()



    def __backUpParam(self):
        root_path = self.dp.myCFG.E1_PATH_EXPORT
        if (root_path[-1] != "\\") or (root_path[-1] != "/"):
            root_path.replace("/", "\\")
            root_path += "\\"

        # 202305  generate folder to store parameter file
        if os.access(root_path + IO_NAME_FOLDER_PARAMETER, os.F_OK):
            ...
        else:
            os.makedirs(root_path + IO_NAME_FOLDER_PARAMETER)

        #  input param: cfg.mgCFG.A0...
        # output param: export_path\param\...cfg
        with open(self.dp.myCFG.A0_PATH_CFG_FILE, "r", encoding="utf-8") as f:
            buff = f.read()

        outpath = root_path + IO_NAME_FOLDER_PARAMETER + IO_NAME_FILE_CONFIG[0]
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(buff)
        ...


    def __checkExportPath(self):

        if os.access(self.dp.myCFG.E1_PATH_EXPORT, os.F_OK):
            pass

        else:
            os.makedirs(self.dp.myCFG.E1_PATH_EXPORT)

        # ======================================

    def __generateTempPath(self):
        # have the export path
        if self.dp.myCFG.D0_WORK_FLOW_NUMBER > 1:
            if os.access(self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FOLDER_TEMPORARY, os.F_OK):
                pass
            else:
                os.makedirs(self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FOLDER_TEMPORARY)

    def __checkInputParam(self):

        # initial file path, are this files existing?

        if not os.access(self.dp.myCFG.I0_INI_PATH_ELEMENT, os.F_OK):
            logGetError(self.dp.myCFG.I0_INI_PATH_ELEMENT + " is not exist! Please check!\n")

        if not os.access(self.dp.myCFG.I1_INI_PATH_AA, os.F_OK):
            logGetError(self.dp.myCFG.I1_INI_PATH_AA + " is not exist! Please check!\n")

        if not os.access(self.dp.myCFG.I2_INI_PATH_MOD, os.F_OK):
            logGetError(self.dp.myCFG.I2_INI_PATH_MOD + " is not exist! Please check!\n")

        # ----------------------------------------------------------

        if self.dp.myCFG.E1_PATH_EXPORT:
            if (self.dp.myCFG.E1_PATH_EXPORT[-1] != "\\") and (self.dp.myCFG.E1_PATH_EXPORT[-1] != "/"):
                self.dp.myCFG.E1_PATH_EXPORT = (self.dp.myCFG.E1_PATH_EXPORT.replace("/", "\\") + "\\")
        else:
            logGetError("PATH_EXPORT is Empty! Please Fill it!")
        # ----------------------------------------------------------

        # switch option, are they 1 or 0 ?
        # [Spectral Preprocess]
        if (self.dp.myCFG.B1_ADD_ISOTOPE_INTENSITY != 1) and (self.dp.myCFG.B1_ADD_ISOTOPE_INTENSITY != 0):
            logGetError("ADD_ISOTOPE_INTENSITY should fill with 1 or 0! But we got " + str(self.dp.myCFG.B1_ADD_ISOTOPE_INTENSITY) + "\n")

        if (self.dp.myCFG.B2_CHECK_NATURAL_LOSS != 1) and (self.dp.myCFG.B2_CHECK_NATURAL_LOSS != 0):
            logGetError("CHECK_NATURAL_LOSS should fill with 1 or 0! But we got " + str(self.dp.myCFG.B2_CHECK_NATURAL_LOSS) + "\n")

        if (self.dp.myCFG.B3_REMOVE_PRECURSOR_ION != 1) and (self.dp.myCFG.B3_REMOVE_PRECURSOR_ION != 0):
            logGetError("REMOVE_PRECURSOR_ION should fill with 1 or 0! But we got " + str(self.dp.myCFG.B3_REMOVE_PRECURSOR_ION) + "\n")

        if (self.dp.myCFG.B4_CHECK_PRECURSOR_CHARGE != 1) and (self.dp.myCFG.B4_CHECK_PRECURSOR_CHARGE != 0):
            logGetError("CHECK_PRECURSOR_CHARGE should fill with 1 or 0! But we got " + str(self.dp.myCFG.B4_CHECK_PRECURSOR_CHARGE) + "\n")

        if (self.dp.myCFG.B6_ROUND_TWO_REMOVE_IMMONIUM != 1) and (self.dp.myCFG.B6_ROUND_TWO_REMOVE_IMMONIUM != 0):
            logGetError("ROUND_TWO_REMOVE_IMMONIUM should fill with 1 or 0! But we got " + str(self.dp.myCFG.B6_ROUND_TWO_REMOVE_IMMONIUM) + "\n")

        # [Spectral Pair switch]
        if (self.dp.myCFG.C1_MIRROR_TYPE_A1 != 1) and (self.dp.myCFG.C1_MIRROR_TYPE_A1 != 0):
            logGetError("MIRROR_TYPE_A1 should fill with 1 or 0! But we got " + str(self.dp.myCFG.C1_MIRROR_TYPE_A1) + "\n")

        if (self.dp.myCFG.C2_MIRROR_TYPE_A2 != 1) and (self.dp.myCFG.C2_MIRROR_TYPE_A2 != 0):
            logGetError("MIRROR_TYPE_A2 should fill with 1 or 0! But we got " + str(self.dp.myCFG.C2_MIRROR_TYPE_A2) + "\n")

        if (self.dp.myCFG.C3_MIRROR_TYPE_A3 != 1) and (self.dp.myCFG.C3_MIRROR_TYPE_A3 != 0):
            logGetError("MIRROR_TYPE_A3 should fill with 1 or 0! But we got " + str(self.dp.myCFG.C3_MIRROR_TYPE_A3) + "\n")

        if (self.dp.myCFG.C4_MIRROR_TYPE_B != 1) and (self.dp.myCFG.C4_MIRROR_TYPE_B != 0):
            logGetError("MIRROR_TYPE_B should fill with 1 or 0! But we got " + str(self.dp.myCFG.C4_MIRROR_TYPE_B) + "\n")

        if (self.dp.myCFG.C5_MIRROR_TYPE_C != 1) and (self.dp.myCFG.C5_MIRROR_TYPE_C != 0):
            logGetError("MIRROR_TYPE_C should fill with 1 or 0! But we got " + str(self.dp.myCFG.C5_MIRROR_TYPE_C) + "\n")

        if (self.dp.myCFG.C6_MIRROR_TYPE_D != 1) and (self.dp.myCFG.C6_MIRROR_TYPE_D != 0):
            logGetError("MIRROR_TYPE_D should fill with 1 or 0! But we got " + str(self.dp.myCFG.C6_MIRROR_TYPE_D) + "\n")

        if (self.dp.myCFG.C7_MIRROR_TYPE_E != 1) and (self.dp.myCFG.C7_MIRROR_TYPE_E != 0):
            logGetError("MIRROR_TYPE_E should fill with 1 or 0! But we got " + str(self.dp.myCFG.C7_MIRROR_TYPE_E) + "\n")

        if (self.dp.myCFG.C8_MIRROR_TYPE_F != 1) and (self.dp.myCFG.C8_MIRROR_TYPE_F != 0):
            logGetError("MIRROR_TYPE_F should fill with 1 or 0! But we got " + str(self.dp.myCFG.C8_MIRROR_TYPE_F) + "\n")

        if (self.dp.myCFG.C9_MIRROR_TYPE_G != 1) and (self.dp.myCFG.C9_MIRROR_TYPE_G != 0):
            logGetError("MIRROR_TYPE_G should fill with 1 or 0! But we got " + str(self.dp.myCFG.C9_MIRROR_TYPE_G) + "\n")


        # [De Novo Sequencing switch]
        # tol_ppm switch

        if (self.dp.myCFG.D4_MS_TOL_PPM != 0) and (self.dp.myCFG.D4_MS_TOL_PPM != 1):
            logGetError("MS_TOL_PPM should fill with 1 or 0! But we got " + str(self.dp.myCFG.D4_MS_TOL_PPM))

        if (self.dp.myCFG.D6_MSMS_TOL_PPM != 0) and (self.dp.myCFG.D6_MSMS_TOL_PPM != 1):
            logGetError("MSMS_TOL_PPM should fill with 1 or 0! But we got " + str(self.dp.myCFG.D6_MSMS_TOL_PPM))


        # [Export switch] and [Validation switch]
        if (self.dp.myCFG.E2_EXPORT_ROUND_ONE_MGF != 1) and (self.dp.myCFG.E2_EXPORT_ROUND_ONE_MGF != 0):
            logGetError("EXPORT_ROUND_ONE_MGF should fill with 1 or 0! But we got " + str(self.dp.myCFG.E2_EXPORT_ROUND_ONE_MGF) + "\n")

        if (self.dp.myCFG.E3_EXPORT_ROUND_TWO_MGF != 1) and (self.dp.myCFG.E3_EXPORT_ROUND_TWO_MGF != 0):
            logGetError("EXPORT_ROUND_TWO_MGF should fill with 1 or 0! But we got " + str(self.dp.myCFG.E3_EXPORT_ROUND_TWO_MGF) + "\n")

        if (self.dp.myCFG.E4_EXPORT_SPECTRAL_PAIR != 1) and (self.dp.myCFG.E4_EXPORT_SPECTRAL_PAIR != 0):
            logGetError("EXPORT_SPECTRAL_PAIR should fill with 1 or 0! But we got " + str(self.dp.myCFG.E4_EXPORT_SPECTRAL_PAIR) + "\n")

        if (self.dp.myCFG.E5_COMBINE_SPLIT_RESULT != 1) and (self.dp.myCFG.E5_COMBINE_SPLIT_RESULT != 0):
            logGetError("COMBINE_SPLIT_RESULT should fill with 1 or 0! But we got " + str(self.dp.myCFG.E5_COMBINE_SPLIT_RESULT) + "\n")

        if (self.dp.myCFG.E6_COMBINE_TOTAL_RESULT != 1) and (self.dp.myCFG.E6_COMBINE_TOTAL_RESULT != 0):
            logGetError("COMBINE_TOTAL_RESULT should fill with 1 or 0! But we got " + str(self.dp.myCFG.E6_COMBINE_TOTAL_RESULT) + "\n")

        if (self.dp.myCFG.E7_EXPORT_FEATURE_MGF != 1) and (self.dp.myCFG.E7_EXPORT_FEATURE_MGF != 0):
            logGetError("EXPORT_FEATURE_MGF should fill with 1 or 0! But we got " + str(self.dp.myCFG.E7_EXPORT_FEATURE_MGF) + "\n")

        if (self.dp.myCFG.V0_FLAG_DO_VALIDATION < 0) or (self.dp.myCFG.V0_FLAG_DO_VALIDATION > 2):
            logGetError("FLAG_DO_VALIDATION should fill with 0, 1 or 2! But we got " + str(self.dp.myCFG.V0_FLAG_DO_VALIDATION) + "\n")

        # special number check ------------------

        if (self.dp.myCFG.B5_ROUND_ONE_HOLD_PEAK_NUM < 1):
            logGetError("[Round I] HOLD_PEAK_NUM should > 0! But we got " + str(self.dp.myCFG.B5_ROUND_ONE_HOLD_PEAK_NUM) + "\n")

        if (self.dp.myCFG.B7_ROUND_TWO_PEAK_NUM_PER_BIN < 1):
            logGetError("[Round II] PEAK_NUM_PER_BIN should > 0! But we got " + str(self.dp.myCFG.B7_ROUND_TWO_PEAK_NUM_PER_BIN) + "\n")

        if (self.dp.myCFG.B8_ROUND_TWO_MASS_BIN_LENTH < 2):
            logGetError("[Round II] MASS_BIN_LENTH should > 1! But we got " + str(self.dp.myCFG.B8_ROUND_TWO_MASS_BIN_LENTH) + "\n")
        
        if (self.dp.myCFG.D2_MULTIPROCESS_NUM < 1):
            logGetError("MULTIPROCESS_NUM should > 0! But we got " + str(self.dp.myCFG.D2_MULTIPROCESS_NUM))

        # [MS Data]
        if self.dp.myCFG.A5_MIN_PRECURSOR_MASS < 0:
            logGetError("MIN_PRECURSOR_MASS should > 0! But we got " + str(self.dp.myCFG.A5_MIN_PRECURSOR_MASS))

        if self.dp.myCFG.A6_MAX_PRECURSOR_MASS < 0:
            logGetError("MAX_PRECURSOR_MASS should > 0! But we got " + str(self.dp.myCFG.A6_MAX_PRECURSOR_MASS))

        if (self.dp.myCFG.A11_MIRROR_PROTEASES_APPROACH < 1) or (self.dp.myCFG.A11_MIRROR_PROTEASES_APPROACH > 2):
            logGetError("MIRROR_PROTEASES_APPROACH should belong to 1 -> 2! But we got " + str(self.dp.myCFG.A11_MIRROR_PROTEASES_APPROACH))

        if self.dp.myCFG.D8_REPORT_PEPTIDE_NUM < 1:
            logGetError("REPORT_PEPTIDE_NUM should > 0! But we got " + str(self.dp.myCFG.D8_REPORT_PEPTIDE_NUM))

        # number range check ---------------------
        # [MS Data]
        if self.dp.myCFG.A5_MIN_PRECURSOR_MASS >= self.dp.myCFG.A6_MAX_PRECURSOR_MASS:
            print("We should set MIN_PRECURSOR_MASS < MAX_PRECURSOR_MASS! But we got:")
            logGetError("MIN_PRECURSOR_MASS: " + str(self.dp.myCFG.A5_MIN_PRECURSOR_MASS) + "\t\tMAX_PRECURSOR_MASS: " + str(self.dp.myCFG.A6_MAX_PRECURSOR_MASS))

        if self.dp.myCFG.A6_MAX_PRECURSOR_MASS > 10000:
            logGetError("[MASS BOUNDARY]MAX_PRECURSOR_MASS is more than 10000Da!" + "\n")

        if (self.dp.myCFG.A7_LABEL_TYPE < 0) or (self.dp.myCFG.A7_LABEL_TYPE > 1):
            logGetError("LABEL_TYPE should belong to 0 -> 1! But we got " + str(self.dp.myCFG.A7_LABEL_TYPE) + "\n")

        if self.dp.myCFG.A10_NEUCODE_PEAK_MAX_RATIO < 1.0:
            logGetError("NEUCODE_PEAK_MAX_INTENSITY_RATIO should >= 1.0! But we got " + str(self.dp.myCFG.A10_NEUCODE_PEAK_MAX_RATIO) + "\n")

        # [Workflow]
        if (self.dp.myCFG.D0_WORK_FLOW_NUMBER < 1) or (self.dp.myCFG.D0_WORK_FLOW_NUMBER > 5):
            logGetError("WORK_FLOW_NUMBER should belong to 1 -> 4! But we got " + str(self.dp.myCFG.D0_WORK_FLOW_NUMBER) + "\n")

        if (self.dp.myCFG.D1_MEMORY_LOAD_MODE < 1) or (self.dp.myCFG.D1_MEMORY_LOAD_MODE > 3):
            logGetError("MEMORY_LOAD_MODE should belong to 1 -> 3! But we got " + str(self.dp.myCFG.D1_MEMORY_LOAD_MODE) + "\n")

        if (self.dp.myCFG.D2_MULTIPROCESS_NUM < 1):
            logGetError("MULTIPROCESS_NUM should be a positive integer! But we got " + str(self.dp.myCFG.D2_MULTIPROCESS_NUM))

        if (self.dp.myCFG.D2_MULTIPROCESS_NUM > os.cpu_count()):
            logGetWarning("\n[WARNING]\tMULTIPROCESS_NUM is too big, we adjust it from {} to {}.\t[WARNING]".format(str(self.dp.myCFG.D2_MULTIPROCESS_NUM), str(os.cpu_count())))
            self.dp.myCFG.D2_MULTIPROCESS_NUM = os.cpu_count()

        if (self.dp.myCFG.B9_NEUCODE_DETECT_APPROACH < 1) or (self.dp.myCFG.B9_NEUCODE_DETECT_APPROACH > 2):
            logGetError("NEUCODE_PEAK_DETECTION_APPROACH should be 1 or 2! But we got " + str(self.dp.myCFG.B9_NEUCODE_DETECT_APPROACH))

        if (self.dp.myCFG.D9_DE_NOVO_APPROACH < 1) or (self.dp.myCFG.D9_DE_NOVO_APPROACH > 5):
            logGetError("DE_NOVO_APPROACH should be 1 -> 5! But we got " + str(self.dp.myCFG.D9_DE_NOVO_APPROACH))

        # check model path
        if (self.dp.myCFG.B9_NEUCODE_DETECT_APPROACH == 2) and (self.dp.myCFG.A7_LABEL_TYPE == 1):
            if not self.dp.myCFG.B10_CLASSIFICATION_MODEL_PATH:
                logGetError("CLASSIFICATION_MODEL_PATH is NULL while you want to use it!")
            # exist or not?
            else:
                try:
                    with open(self.dp.myCFG.B10_CLASSIFICATION_MODEL_PATH, "r") as f:
                        ...
                except:
                    logGetError("CANNOT OPEN MODEL:\t" + self.dp.myCFG.B10_CLASSIFICATION_MODEL_PATH + "\nPlease check it!")
                ...
        ...

        # check
        if (self.dp.myCFG.C10_PAIR_FILTER_APPROACH < 0) or (self.dp.myCFG.C10_PAIR_FILTER_APPROACH > 3):
            logGetError("PAIR_FILTER_APPROACH should be 0 -> 3! But we got " + str(self.dp.myCFG.C10_PAIR_FILTER_APPROACH) + "\n")


        # check de novo model path
        # work flow number 3  4 is for de novo, 1 and 2 do not use the path of de novo model
        tmpFlag = (self.dp.myCFG.D9_DE_NOVO_APPROACH == 1) or (self.dp.myCFG.D9_DE_NOVO_APPROACH == 3)
        if tmpFlag and (self.dp.myCFG.D0_WORK_FLOW_NUMBER > 2):
            if not self.dp.myCFG.D10_MIRROR_NOVO_MODEL_PATH:
                logGetError("MIRROR_NOVO_MODEL_PATH is NULL while you want to use it!" + "\n")
            else:
                try:
                    with open(self.dp.myCFG.D10_MIRROR_NOVO_MODEL_PATH, "r") as f:
                        ...
                except:
                    logGetError("CANNOT OPEN MODEL:\t" + self.dp.myCFG.D10_MIRROR_NOVO_MODEL_PATH + "\nPlease check it!" + "\n")

        tmpFlag = (self.dp.myCFG.D9_DE_NOVO_APPROACH == 2) or (self.dp.myCFG.D9_DE_NOVO_APPROACH == 3)
        if tmpFlag and (self.dp.myCFG.D0_WORK_FLOW_NUMBER > 2):
            if not self.dp.myCFG.D11_PNOVOM_EXE_PATH:
                logGetError("PNOVOM_EXE_PATH is NULL while you want to use it!" + "\n")
            else:
                # is it an .exe file?
                if self.dp.myCFG.D11_PNOVOM_EXE_PATH[-4:] == ".exe":
                    pass
                else:
                    logGetError("PNOVOM_EXE_PATH is not an .exe file!" + "\n")

                # is its dir path has model model2.exe?
                file_path = os.path.dirname(self.dp.myCFG.D11_PNOVOM_EXE_PATH) + "\\train\\model2.txt"
                try:
                    with open(file_path, "r") as f:
                        ...
                except:
                    print("CANNOT find \"model\\model2.txt\" in directory of specific .exe file.")
                    logGetError("CANNOT OPEN MODEL:\t" + file_path + "\nPlease check it!" + "\n")


        if (self.dp.myCFG.V0_FLAG_DO_VALIDATION == 1):

            if not self.dp.myCFG.V1_PATH_TRY_PFIND_RES:
                logGetError("PATH_A_PFIND_RES is NULL while you want to use it!" + "\n")

            if not self.dp.myCFG.V2_PATH_LYS_PFIND_RES:
                logGetError("PATH_B_PFIND_RES is NULL while you want to use it!" + "\n")

            try:
                with open(self.dp.myCFG.V1_PATH_TRY_PFIND_RES, "r") as f:
                    buff = f.read().split("\n")
                    if len(buff) <= 2:
                        logGetError("[PATH_A_PFIND_RES] Only accept pFind result with: more than 2 lines." + "\n")
            except:
                logGetError("CANNOT OPEN FILE:\t" + self.dp.myCFG.V1_PATH_TRY_PFIND_RES + "\nPlease check it!" + "\n")

            try:
                with open(self.dp.myCFG.V2_PATH_LYS_PFIND_RES, "r") as f:
                    buff = f.read().split("\n")
                    if len(buff) <= 2:
                        logGetError("[PATH_B_PFIND_RES] Only accept pFind result with: more than 2 lines." + "\n")
            except:
                logGetError("CANNOT OPEN FILE:\t" + self.dp.myCFG.V2_PATH_LYS_PFIND_RES + "\nPlease check it!" + "\n")

        if self.dp.myCFG.V0_FLAG_DO_VALIDATION == 2:
            if not self.dp.myCFG.V3_PATH_FASTA_FILE:
                logGetError("PATH_FASTA_FILE is NULL while you want to use it!\n")
            try:
                with open(self.dp.myCFG.V3_PATH_FASTA_FILE, "r") as f:
                    ...
            except:
                logGetError("CANNOT OPEN FILE:\t" + self.dp.myCFG.V3_PATH_FASTA_FILE + "\nPlease check it!" + "\n")

    def __cleanAll(self, inputPath):

        for root, dirs, files in os.walk(inputPath):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir_ in dirs:
                shutil.rmtree(os.path.join(root, dir_))


class CTaskOutputPathFile:

    def __init__(self, inputDP, inputFlowNum):
        self.dp = inputDP
        self.flow_num = inputFlowNum

        if self.flow_num == 2:
            if self.dp.myCFG.E4_EXPORT_SPECTRAL_PAIR == 0:
                self.flow_num = 0

    def work(self, initFile=True):

        root_path = self.dp.myCFG.E1_PATH_EXPORT
        if (root_path[-1] == "\\") or (root_path[-1] == "/"):
            pass
        else:
            root_path += "\\"

        # clean all files and folders in export path
        # self.__cleanAll(root_path)

        for try_name in self.dp.LIST_MGF_NAME_TRY:

            sub_dir = root_path + try_name
            if os.access(sub_dir, os.F_OK):
                pass
            else:
                os.makedirs(sub_dir)

            self.dp.LIST_OUTPUT_PATH.append(sub_dir + "\\")

            if self.flow_num > 1:
                # 这里是很基本的地址运算，和IO_NAME_FILE_RESULT是对应的
                # 可别随便改顺序啊
                head = (self.flow_num - 2) * 2
                tail = head + 2
                for fileName in IO_NAME_FILE_RESULT[head:tail]:
                    filePath = self.dp.LIST_OUTPUT_PATH[-1] + fileName
                    if initFile:
                        with open(filePath, "w", encoding="utf-8") as f:
                            f.write("")
                ...
            ...

        # COMBINE 的设计
        # if self.dp.myC


            # for lys_name in self.dp.LIST_MGF_NAME_LYS:
            #     sub_sub = sub_dir + "\\" + lys_name
            #     if os.access(sub_sub, os.F_OK):
            #         pass
            #     else:
            #         os.makedirs(sub_sub)

    def __geneReadMe(self, rootPath):

        file_path = rootPath + IO_NAME_FILE_README[0]

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n")
            f.write("[DiNovo Result Infomation]\n")
            f.write("ALL the pairing results is based on trypsin data.\n")
            f.write("\n")
            f.write("1.[Preprocess]\n")
            f.write("Preprocessed .mgf files will be generated ")
            f.write("when \"EXPORT_ROUND_ONE_MGF\" or \"EXPORT_ROUND_TWO_MGF\" is set as \"1\" in configure file.\n")
            f.write("The preprocessed files are named as \"OriginalFileName[DiNovo][Preprocessing][1st/2nd].mgf\",")
            f.write("and the \"OriginalFileName\" is the original name of spectra file(mgf).\n")
            f.write("\n")
            f.write("2.[Spectra Pair]\n")
            f.write("\n")
            f.write("3.[De Novo Sequencing]\n")
            f.write("\n")
        ...


class CTaskWritePreprocessMGF:

    def __init__(self, inputDP):
        self.dp = inputDP
        if self.dp.myCFG.A7_LABEL_TYPE == 1:
            self.funcPrep = CFunctionPreprocessNeuCode(self.dp)
        else:
            self.funcPrep = CFunctionPreprocess(self.dp)

        self.funcLoadPKL = CFunctionLoadPKL(self.dp)

        # self.dp.myCFG.A8_NEUCODE_OUTPUT_TYPE = 2

        self.funcWrite = CFunctionWritePrepMGF(self.dp)
        self.mgf_1_suffix = PREPROCESS_FILE_SUFFIX[0]
        self.mgf_2_suffix = PREPROCESS_FILE_SUFFIX[1]
        self.mgf_3_suffix = PREPROCESS_FILE_SUFFIX[2]
        self.funcTempPath = CFunctionTempPath(self.dp)

        # NeuCode && approach
        self.annoFlag = (self.dp.myCFG.A7_LABEL_TYPE == 1) and (self.dp.myCFG.B9_NEUCODE_DETECT_APPROACH == 2)

        self.pklFlag = (self.dp.myCFG.D0_WORK_FLOW_NUMBER > 1)

    def work(self):

        logToUser(INFO_TO_USER_TaskPrep[0])

        # mp.pool()
        # D2_MULTIPROCESS_NUM

        my_pool = mp.Pool(self.dp.myCFG.D2_MULTIPROCESS_NUM)

        for i, tmpPath in enumerate(self.dp.LIST_PATH_MGF_TRY):
            # my_pool.apply_async(func=self.subTask, args=(tmpPath, i, True))
            self.subTask(tmpPath, i, flag=True)

        for i, tmpPath in enumerate(self.dp.LIST_PATH_MGF_LYS):
            my_pool.apply_async(func=self.subTask, args=(tmpPath, i, False))
            # self.subTask(tmpPath, i, flag=False)

        my_pool.close()
        my_pool.join()

    # 20221231 因为并行化的原因，这里不能写成__XXX()形式的函数！否则并行会报错的！
    # 函数名的前面不能带双下划线！
    def subTask(self, inputPath, input_idx, flag=True):
        if flag:
            file_name = self.dp.LIST_MGF_NAME_TRY[input_idx]
        else:
            file_name = self.dp.LIST_MGF_NAME_LYS[input_idx]

        if self.annoFlag:
            logToUser(INFO_TO_USER_TaskPrep[2] + "\t[" + file_name + "]")
        else:
            logToUser(INFO_TO_USER_TaskPrep[1] + "\t[" + file_name + "]")

        mgf_1_path = self.dp.myCFG.E1_PATH_EXPORT + file_name + self.mgf_1_suffix + ".mgf"
        mgf_2_path = self.dp.myCFG.E1_PATH_EXPORT + file_name + self.mgf_2_suffix + ".mgf"
        mgf_3_path = self.dp.myCFG.E1_PATH_EXPORT + file_name + self.mgf_3_suffix + ".mgf"

        if self.annoFlag:
            with open(mgf_3_path, "w", encoding="utf-8") as f:
                f.write("")
            dataMS2 = self.funcLoadPKL.loadMS2PKL(inputPath + ".pkl")
            annoFunc = CFunctionNeuCodeDetection(self.dp)
            for scan in dataMS2.INDEX_SCAN:
                spec = self.funcLoadPKL.getSingleMS2Spectrum(dataMS2, scan)
                anno = annoFunc.detect(spec.LIST_PEAK_MOZ, spec.LIST_PEAK_INT)
                spec.NEUCODE_LABEL = anno
                self.funcWrite.write(mgf_3_path, spec)

        else:
            if self.dp.myCFG.E2_EXPORT_ROUND_ONE_MGF:
                with open(mgf_1_path, "w", encoding="utf-8") as f:
                    f.write("")

            if self.dp.myCFG.E3_EXPORT_ROUND_TWO_MGF:
                with open(mgf_2_path, "w", encoding="utf-8") as f:
                    f.write("")

            dataMS2 = self.funcLoadPKL.loadMS2PKL(inputPath + ".pkl")
            # mass_dict, root = dict(), "D:\\ResearchAssistant@AMSS\\MirrorSpectra\\Pi20221211\\justTestData(informal)\\"
            # with open(root+"YeastNeuCode[TRY]pFind-Filtered[extract][v0315].spectra", "r") as f_try, open(root+"YeastNeuCode[LYS]pFind-Filtered[extract][v0315].spectra", "r") as f_lys:
            #     buff = f_try.read().split("\n")[1:-1] + f_lys.read().split("\n")[1:-1]
            # for line in buff:
            #     mass_dict[line.split("\t")[0]] = float(line.split("\t")[2])
            second_run = self.dp.myCFG.E3_EXPORT_ROUND_TWO_MGF or self.pklFlag
            spec_list = []
            for scan in dataMS2.INDEX_SCAN:
                spec = self.funcLoadPKL.getSingleMS2Spectrum(dataMS2, scan)
                # print(spec.LIST_FILE_NAME)
                mgf_1, mgf_2 = self.funcPrep.preprocess(spec, second=second_run)
                # if (not flag) and mgf_2[0].LIST_FILE_NAME[0] == "Yeast_NeuCode_R29_r1_F1.6594.6594.2.0.dta":
                #     pass
                # else:
                #     continue
                # for i_tmp in range(len(mgf_2)):
                #     mgf_2[i_tmp].LIST_PRECURSOR_MASS[0] = mass_dict[mgf_2[i_tmp].LIST_FILE_NAME[0]]
                if self.pklFlag:
                    spec_list += mgf_2
                if self.dp.myCFG.E2_EXPORT_ROUND_ONE_MGF:
                    self.funcWrite.write(mgf_1_path, mgf_1)
                if self.dp.myCFG.E3_EXPORT_ROUND_TWO_MGF:
                    self.funcWrite.write(mgf_2_path, mgf_2)

            if self.pklFlag:
                path_pkl = self.funcTempPath.geneTempPathSpecList(input_idx, tryFlag=flag)
                fid_pkl = open(path_pkl, 'wb')
                pickle.dump(spec_list, fid_pkl)
                fid_pkl.close()
                logToUser(INFO_TO_USER_TaskPrep[3] + "\t[" + file_name + "]")


class CTaskBuildPMIDX:
    def __init__(self, inputDP):
        self.dp = inputDP
        self.funcLoad  = CFunctionLoadPKL(self.dp)
        self.funcPMIDX = CFunctionPrecursorMassIndex(self.dp)
        self.funcTempPath = CFunctionTempPath(self.dp)

    def work(self):
        logToUser(INFO_TO_USER_TaskPMIDX[0])

        # build lys file idx
        for i, name in enumerate(self.dp.LIST_MGF_NAME_LYS):
            logToUser(INFO_TO_USER_TaskPMIDX[1] + "[" + name + "]")
            self.buildIDX(i)

    def buildIDX(self, i):

        # path of .pkl: list of preprocessed spectra
        pklPath = self.funcTempPath.geneTempPathSpecList(i, tryFlag=False)
        # load it
        specList = self.funcLoad.loadSpecListPKL(pklPath)

        # build idx
        PMIDX = self.funcPMIDX.captainGetPrecursorMassIndex(specList)

        # path of .pkl: PMIDX
        tmpPath = self.funcTempPath.geneTempPathPMIDX(i)

        # save the pmidx as a binary file
        fid_pkl = open(tmpPath, 'wb')
        pickle.dump(PMIDX, fid_pkl)
        fid_pkl.close()

        del PMIDX

# 谱图与谱图的匹配


class CTaskPairing:

    def __init__(self, inputDP):
        self.dp = inputDP

        self.min_mass = self.dp.myCFG.A5_MIN_PRECURSOR_MASS
        self.max_mass = self.dp.myCFG.A6_MAX_PRECURSOR_MASS

        self.memory_mode = self.dp.myCFG.D1_MEMORY_LOAD_MODE
        self.memory_mode_dict = {1:"[less-load mode]",
                                2:"[semi-load mode]",
                                3:"[full-load mode]"}
        self.memory_mode_str = self.memory_mode_dict[self.memory_mode]

        # necessarily function used bellow
        self.funcLoadPKL = CFunctionLoadPKL(self.dp)
        self.dp.myINI.MASS_PROTON_MONO = 1.007276
        self.funcPMIDX = CFunctionPrecursorMassIndex(self.dp)
        if self.dp.myCFG.A7_LABEL_TYPE == 1:
            self.funcPrep = CFunctionPreprocessNeuCode(self.dp)
        else:
            self.funcPrep = CFunctionPreprocess(self.dp)

        self.funcTempPath = CFunctionTempPath(self.dp)
        self.LOAD_TIME_LIMIT = 1200  # 1200 seconds, 20mins

        self.directJudge = True if self.dp.myCFG.C10_PAIR_FILTER_APPROACH == 0 else False
        self.pvFilter = True if (self.dp.myCFG.C10_PAIR_FILTER_APPROACH % 2) == 1 else False
        self.pvThreshold = self.dp.myCFG.C11_PAIR_P_VALUE_THRESHOLD

        self.decoyApproach = self.dp.myCFG.C12_PAIR_DECOY_APPROACH
        self.decoyDeltaMass = self.dp.myCFG.C13_PAIR_DECOY_SHIFTED_IN_DA
        self.fdrThreshold = self.dp.myCFG.C14_PAIR_FDR_THRESHOLD

    def work(self):

        logToUser(INFO_TO_USER_TaskPair[0])

        logToUser(INFO_TO_USER_TaskPair[1] + "\t" + self.memory_mode_str)

        # load one by one
        if self.memory_mode == 1:
            self.less_load_mode()

        # load all LysargiNase data
        # elif self.memory_mode == 2:
        #     self.semi_load_mode()
        #
        # # load all
        # elif self.memory_mode == 3:
        #     self.full_load_mode()

        # elif self.memory_mode == 4:
        #     self.zixuan()

        else:
            logGetError("[ERROR]GET UNKNOWN MEMORY MODE NUMBER!!")

    # 1. 载入单个LysargiNase，生成PMI索引
    # 2. 依次载入单个Trypsin，全部结束后再载入另一个LysargiNase
    # 3. 对当前Trypsin谱图生成specHashTab，若有候选则做匹配打分
    def less_load_mode(self):

        # double for loop
        tryN, lysN = len(self.dp.LIST_PATH_MGF_TRY), len(self.dp.LIST_PATH_MGF_LYS)
        myPool = mp.Pool(self.dp.myCFG.D2_MULTIPROCESS_NUM)

        for i in range(tryN):
            for j in range(lysN):
                # self.oneByOnePairSpec(i, j)
                myPool.apply_async(func=self.oneByOnePairSpec, args=(i, j))

        myPool.close()
        myPool.join()

        # delete all flag file
        self.deleteTempFile(tryN, lysN)
        pass
        # self.__logStatInfo(prep_n, pair_n, mem_record)
        # self.__logTimeCost(use_time)
        ...

    # 1. 载入全部的LysargiNase数据，生成PMI索引
    # 2. 依次载入Trypsin数据，查询LysargiNase
    # 3. 对当前的Trypsin谱图生成specHashTab，若有候选则匹配打分
    def semi_load_mode(self):

        # time record variable                     0         1         2         3         4
        use_time = [0.0, 0.0, 0.0, 0.0, 0.0]  # [ LOAD ], [PM IDX], [PREPOL], [ PAIR ], [ SAVE ]
        mem_record = [0.0, 0.0, 0.0]  # 0:lysData, 1:pm idx, 2:prepData
        prep_n, pair_n = 0, 0
        # load your ms2 files (all LysargiNase)
        mem_used = psutil.virtual_memory().used / 1024.0 ** 2

        logToUser(INFO_TO_USER_TaskPair[2])
        label_time = time.perf_counter()
        ms2_file_list = []
        for i, path in enumerate(self.dp.LIST_PATH_MGF_LYS):
            logToUser(INFO_TO_USER_TaskPair[3] + "[" + self.dp.LIST_MGF_NAME_LYS[i] + "]")
            ms2_file_list.append(self.funcLoadPKL.loadMS2PKL(path + ".pkl"))

        use_time[0] += time.perf_counter() - label_time
        mem_record[0] = psutil.virtual_memory().used / 1024.0 ** 2 - mem_used
        mem_used = psutil.virtual_memory().used / 1024.0 ** 2  # get the new

        # generate: precursor info list, and precursor mass index
        logToUser(INFO_TO_USER_TaskPair[4])
        label_time = time.perf_counter()
        p_tuple_list, PMIDX = self.funcPMIDX.generatePMIDX(ms2_file_list)
        # funcPairing = CFunctionMirrorFinder(self.dp, p_tuple_list, PMIDX)
        use_time[1] += time.perf_counter() - label_time

        mem_record[1] = psutil.virtual_memory().used / 1024.0 ** 2 - mem_used
        mem_used = psutil.virtual_memory().used / 1024.0 ** 2  # get the new

        # [ATTENTION] 如有需要，可对谱图全部做一遍预处理，以避免重复运算的问题
        # 可仅保留 MOZ 和 INT 两个list，记录个数和母离子一致，可用 PMIDX 来寻址
        prep_2nd_spec_moz, prep_2nd_spec_int, prep_2nd_spec_ncl, prep_n = self.__soldierGetOnly2ndPrepData(ms2_file_list, use_time)

        mem_record[2] = psutil.virtual_memory().used / 1024.0 ** 2 - mem_used
        mem_used = psutil.virtual_memory().used / 1024.0 ** 2  # get the new

        # -------------------------------------------------------------------
        # record_dict = dict()
        logToUser(INFO_TO_USER_TaskPair[6])
        # loading, pairing and writing result per trypsin .mgf file
        my_pool = mp.Pool(self.dp.myCFG.D2_MULTIPROCESS_NUM)
        res_value = []
        for i_try in range(len(self.dp.LIST_PATH_MGF_TRY)):

            # online preprocessing, spec pairing, res writing
            tmp_spec_n, tmp_pair_n, time_rec = self.soldierPairSpec(i_try, p_tuple_list, PMIDX, ms2_file_list, prep_2nd_spec_moz, prep_2nd_spec_int, prep_2nd_spec_ncl)
            # (tmp_spec_n, tmp_pair_n, time_rec) = my_pool.apply_async(func=self.soldierPairSpec, args=(i_try, p_tuple_list, PMIDX, ms2_file_list, prep_2nd_spec_moz, prep_2nd_spec_int, prep_2nd_spec_ncl))
            # k = my_pool.apply_async(func=self.soldierPairSpec, args=(i_try, p_tuple_list, PMIDX, ms2_file_list, prep_2nd_spec_moz, prep_2nd_spec_int, prep_2nd_spec_ncl))
            # res_value.append(k)
            prep_n += tmp_spec_n
            pair_n += tmp_pair_n
            for i,t in enumerate(time_rec):
                use_time[i] += t

        my_pool.close()
        my_pool.join()
        # print(res_value)
        for p in res_value:
            (tmp_spec_n, tmp_pair_n, time_rec) = p.get()
            prep_n += tmp_spec_n
            pair_n += tmp_pair_n
            for i,t in enumerate(time_rec):
                use_time[i] += t
            ...

        self.__logStatInfo(prep_n, pair_n, mem_record)
        self.__logTimeCost(use_time)
        # 20221231 这里就不输出啦，因为这里没有记录喔~
        # funcPairing.logTimeCost()
        # print(123)

    # --------------- 暂不实现 --------------
    # 1. 载入全部的Trypsin和全部的LysargiNase
    # 2. 按母离子质量生成两个有序表，顺序遍历
    def full_load_mode(self):
        ...

    # 20230412
    def oneByOnePairSpec(self, i_try, i_lys):

        # start!
        start_time = time.perf_counter()
        processName = "[" + self.dp.LIST_MGF_NAME_TRY[i_try] + " v.s. " + self.dp.LIST_MGF_NAME_LYS[i_lys] + "]"

        # ---- 0.[WAIT] check if last load over(flag file)
        self.__soldierCheckAccessViolation(i_try, i_lys)
        # logToUser(INFO_TO_USER_TaskPair[6] + processName + "READY")
        end_wait = time.perf_counter()

        # ---- 1.[LOAD] load try, lys & idx
        trySpecList, lysSpecList, PMIDX = self.__soldierLoadTLP(i_try, i_lys)
        # logToUser(INFO_TO_USER_TaskPair[2] + processName + " OVER")
        end_load = time.perf_counter()

        # ---- 2.[PAIR] match spectral pair  [ATTENTION: add process bar info]
        match_res = self.__soldierPairAll(trySpecList, lysSpecList, PMIDX, logoStr=processName)
        # logToUser(INFO_TO_USER_TaskPair[6] + processName + " OVER")
        end_pair = time.perf_counter()

        # ---- 3.[SAVE] save result files
        self.__writePairRes(i_try, i_lys, match_res)
        end_all = time.perf_counter()

        all_time  = end_all  - start_time
        wait_time = end_wait - start_time
        load_time = end_load - end_wait
        pair_time = end_pair - end_load
        save_time = end_all  - end_pair
        reportStr = "\tPROCESS\t" + processName
        reportStr += "\tTIME:{:.2f}s | WAIT:{:.2f}s | LOAD:{:.2f}s | PAIR:{:.2f}s | SAVE: {:.2f}s".format(
            all_time, wait_time, load_time, pair_time, save_time)
        # logToUser(INFO_TO_USER_TaskPair[6] + processName + "WRITE OVER")
        logToUser(reportStr)
        ...

    # delete the templete files
    def deleteTempFile(self, tryN, lysN):
        # flag must
        for i in range(tryN):
            for j in range(lysN):
                tmpPath = self.funcTempPath.geneTempPathLoadFlag(i, j)
                self.__soldierCheckAndDeletePath(tmpPath)
                ...

        # delete speclist, if not de novo
        if not ((self.dp.myCFG.D0_WORK_FLOW_NUMBER == 3) and ((self.dp.myCFG.D9_DE_NOVO_APPROACH == 1) or (self.dp.myCFG.D9_DE_NOVO_APPROACH > 3))):
            for i in range(tryN):
                tmpPath = self.funcTempPath.geneTempPathSpecList(i, tryFlag=True)
                self.__soldierCheckAndDeletePath(tmpPath)

            for i in range(lysN):
                tmpPath = self.funcTempPath.geneTempPathSpecList(i, tryFlag=False)
                self.__soldierCheckAndDeletePath(tmpPath)

        # pmidx
        for i in range(lysN):
            tmpPath = self.funcTempPath.geneTempPathPMIDX(i)
            self.__soldierCheckAndDeletePath(tmpPath)

        # delete temp path, if not de novo
        # if self.dp.myCFG.D0_WORK_FLOW_NUMBER < 3:
        #     tmpPath = self.funcTempPath.geneTempFolderPath()
        #     self.__soldierCheckAndDeletePath(tmpPath, folderFlag=True)

    # check if path is exist, delete it
    def __soldierCheckAndDeletePath(self, tmpPath, folderFlag=False):

        if os.access(tmpPath, os.F_OK):
            try:
                if folderFlag:
                    os.removedirs(tmpPath)
                else:
                    os.remove(tmpPath)
            except:
                if folderFlag:
                    logGetWarning("[WARNING] CANNOT delete temp file: " + tmpPath)
                else:
                    logGetWarning("[WARNING] CANNOT delete temp folder: " + tmpPath)

        else:
            if folderFlag:
                logGetWarning("[WARNING] CANNOT Find temp file: " + tmpPath)
            else:
                logGetWarning("[WARNING] CANNOT Find temp folder: " + tmpPath)
        ...

    # check
    def __soldierCheckAccessViolation(self, i_try, i_lys):

        flag_last_try, flag_last_lys = False, False
        if i_try == 0:
            flag_last_try = True
        if i_lys == 0:
            flag_last_lys = True

        path_last_try = self.funcTempPath.geneTempPathLoadFlag(i_try - 1, i_lys)  # have same lys
        path_last_lys = self.funcTempPath.geneTempPathLoadFlag(i_try, i_lys - 1)  # have same try

        if not flag_last_lys:
            flag_last_lys = self.__waitForFileExist(path_last_lys)

        if not flag_last_try:
            flag_last_try = self.__waitForFileExist(path_last_try)

        if not (flag_last_try and flag_last_lys):
            warning_str = "WARNING: Access violation may happpens by: ["
            warning_str += self.dp.LIST_MGF_NAME_TRY[i_try] + "] and ["
            warning_str += self.dp.LIST_MGF_NAME_LYS[i_lys] + "]"
            logGetWarning(warning_str)

    # load
    def __soldierLoadTLP(self, i_try, i_lys):

        privateFuncPKL = CFunctionLoadPKL(self.dp)
        # try spec list
        tmpPath = self.funcTempPath.geneTempPathSpecList(i_try, tryFlag=True)
        # fileT = privateFuncPKL.loadSpecListPKL(tmpPath)
        fileT = copy.deepcopy(privateFuncPKL.loadSpecListPKL(tmpPath))

        # lys spec list
        tmpPath = self.funcTempPath.geneTempPathSpecList(i_lys, tryFlag=False)
        # fileL = privateFuncPKL.loadSpecListPKL(tmpPath)
        fileL = copy.deepcopy(privateFuncPKL.loadSpecListPKL(tmpPath))

        # precursor mass index
        tmpPath = self.funcTempPath.geneTempPathPMIDX(i_lys)
        # fileP = privateFuncPKL.loadPMIDXPKL(tmpPath)
        fileP = copy.deepcopy(privateFuncPKL.loadPMIDXPKL(tmpPath))

        # gene the temp flag file
        self.__generateFlagFile(i_try, i_lys)

        return fileT, fileL, fileP


    def __soldierPairAll(self, trySpecList, lysSpecList, PMIDX, logoStr):
        record_dict = dict()

        p_tuple_list = [[spec.LIST_PRECURSOR_MASS[0]] for spec in lysSpecList]

        funcPairing = CFunctionMirrorFinder(self.dp, p_tuple_list, PMIDX)

        n = (len(trySpecList) // 20) if (len(trySpecList) > 20) else 1

        for i, trySpec in enumerate(trySpecList):
            pmass = trySpec.LIST_PRECURSOR_MASS[0]
            if self.decoyApproach == 1:
                pmass += self.decoyDeltaMass
            # [ATTENTION] 质量越界的不要
            if (pmass < self.min_mass) or (pmass > self.max_mass):
                continue
            # if i % n == 0:
            #     print("temp report: [" + str(i) + "/" + str(len(trySpecList))+ "]\t%.2f" % (i / len(trySpecList) * 100), "%")
            t_rt = trySpec.SCAN_RET_TIME
            tc = trySpec.LIST_PRECURSOR_CHARGE[0]
            candi_idx_dict = funcPairing.recallPrecursorList(pmass)
            record_dict[trySpec.LIST_FILE_NAME[0]] = []

            # initial trypsin spec info, for accelerating
            if self.decoyApproach == 1:
                funcPairing.loadTrypsinSpecInfo(trySpec.LIST_PEAK_MOZ, trySpec.LIST_PEAK_INT, trySpec.NEUCODE_LABEL, pmass-self.decoyDeltaMass)
            else:
                funcPairing.loadTrypsinSpecInfo(trySpec.LIST_PEAK_MOZ, trySpec.LIST_PEAK_INT, trySpec.NEUCODE_LABEL, pmass)

            for int_key in candi_idx_dict:
                sup_str = "A" if (int_key == 0) else chr(ord("A") + int_key - 3)
                for pm_i in candi_idx_dict[int_key]:
                    file_name = lysSpecList[pm_i].LIST_FILE_NAME[0]
                    lc = lysSpecList[pm_i].LIST_PRECURSOR_CHARGE[0]  # for zixuan
                    l_rt = lysSpecList[pm_i].SCAN_RET_TIME

                    # 保留时间过滤，如果差异太大就放过
                    if abs(t_rt - l_rt) > self.dp.myCFG.C0_MIRROR_DELTA_RT:
                        continue
                    ...
                    judge = funcPairing.scoreSpectrumPair(lysSpecList[pm_i].LIST_PEAK_MOZ, lysSpecList[pm_i].LIST_PEAK_INT,
                                                          lysSpecList[pm_i].NEUCODE_LABEL,
                                                          lysSpecList[pm_i].LIST_PRECURSOR_MASS[0], int_key)
                    # [fileName] = [(fileName, )]
                    record_dict[trySpec.LIST_FILE_NAME[0]].append(
                        (file_name, sup_str, pmass, lysSpecList[pm_i].LIST_PRECURSOR_MASS[0], tc, lc, judge))

                # if i % n == 0:
                #     logToUser(INFO_TO_USER_TaskPair[8] + logoStr + ("%.2f" % (i*100.0/len(trySpecList))) + "%")

        return record_dict



    # write
    def __generateFlagFile(self, i_try, i_lys):

        inputPath = self.funcTempPath.geneTempPathLoadFlag(i_try, i_lys)

        try:
            with open(inputPath, "w", encoding="utf-8") as f:
                f.write("")
            # say something?

        except:
            logGetWarning("CANNOT WRITE FLAG FILE: " + inputPath)


    # waiting until time out
    def __waitForFileExist(self, inputPath):
        if not inputPath:
            return True

        res = False
        flag_time = time.perf_counter()

        while not res:
            if os.access(inputPath, os.F_OK):
                res = True
                break
            else:
                # print("\t\twaiting for file:", inputPath)
                time.sleep(1)

            if time.perf_counter() - flag_time > self.LOAD_TIME_LIMIT:
                # print("****wait time limit\n****file:{}\n****time:{}".format(inputPath, time.perf_counter() - flag_time))
                break

        return res




    # param:
    #       i_try               try 文件的idx
    #       p_tuple_list        lys 数据的母离子信息记录
    #       PMIDX               lys 数据的母离子质量倒排
    #       ms2_file_list       lys 全部数据信息的总记录
    #       prep_2nd_spec_moz   lys 全部预处理谱图质荷比
    #       prep_2nd_spec_int   lys 全部预处理谱图的强度
    #       use_time            使用时间信息
    # 20221231 因为并行化的原因，这里不能写成__XXX()形式的函数！否则并行会报错的！
    # 函数名的前面不能带双下划线！
    # def soldierPairSpec(self, i_try, p_tuple_list, PMIDX, ms2_file_list, prep_2nd_spec_moz, prep_2nd_spec_int, prep_2nd_spec_ncl):
    #
    #     # time record variable                     0         1         2         3         4
    #     use_time = [0.0, 0.0, 0.0, 0.0, 0.0]  # [ LOAD ], [PM IDX], [PREPOL], [ PAIR ], [ SAVE ]
    #
    #     path = self.dp.LIST_PATH_MGF_TRY[i_try]
    #     funcPairing = CFunctionMirrorFinder(self.dp, p_tuple_list, PMIDX)
    #     prep_n, pair_n = 0, 0
    #     record_dict = dict()
    #
    #     logToUser(INFO_TO_USER_TaskPair[7] + "[" + self.dp.LIST_MGF_NAME_TRY[i_try] + "]")
    #
    #     # loading trypsin file
    #     label_time = time.perf_counter()
    #     tmpTry = self.funcLoadPKL.loadMS2PKL(path + ".pkl")
    #     use_time[0] += time.perf_counter() - label_time
    #
    #     for tmpScan in tmpTry.INDEX_SCAN[:]:
    #         # for tmpScan in [10006]:
    #
    #         # preprocess online
    #         label_time = time.perf_counter()
    #         tmpSpectrum = self.funcLoadPKL.getSingleMS2Spectrum(tmpTry, tmpScan)
    #         # 预处理后的 moz 和 int list 所组成的 matrix
    #         tryMOZ, tryINT, tryNCL = self.funcPrep.returnTwoListForPair(tmpSpectrum)
    #         prep_n += len(tryMOZ)
    #         use_time[2] += time.perf_counter() - label_time
    #
    #         # pairing
    #         label_time = time.perf_counter()
    #         # candi_idx_dict[0] = [(idx), (idx), ...]
    #         for i, pmass in enumerate(tmpSpectrum.LIST_PRECURSOR_MASS):
    #
    #             # [ATTENTION] 质量越界的不要
    #             if (pmass < self.min_mass) or (pmass > self.max_mass):
    #                 continue
    #             tc = tmpSpectrum.LIST_PRECURSOR_CHARGE[i]  # trypsin spec charge(for zixuan)
    #             candi_idx_dict = funcPairing.recallPrecursorList(pmass)
    #             record_dict[tmpSpectrum.LIST_FILE_NAME[i]] = []
    #
    #             # [20221113]
    #             # initial trypsin spec info, for accelerating
    #             funcPairing.loadTrypsinSpecInfo(tryMOZ[i], tryINT[i], tryNCL[i], pmass)
    #             for int_key in candi_idx_dict:
    #                 sup_str = "A" if (int_key) == 0 else chr(ord("A") + int_key - 3)
    #                 pair_n += len(candi_idx_dict[int_key])
    #                 for pm_i in candi_idx_dict[int_key]:
    #                     ms2_i, scan_i, p_i, lys_mass = p_tuple_list[pm_i]
    #                     file_name = ms2_file_list[ms2_i].MATRIX_FILE_NAME[scan_i][p_i]
    #                     lc = ms2_file_list[ms2_i].MATRIX_PRECURSOR_CHARGE[scan_i][p_i]  # for zixuan
    #                     # print(prep_2nd_spec_moz[pm_i])
    #                     # print(prep_2nd_spec_int[pm_i])
    #                     judge = funcPairing.scoreSpectrumPair(prep_2nd_spec_moz[pm_i], prep_2nd_spec_int[pm_i], prep_2nd_spec_ncl[pm_i],
    #                                                           lys_mass, int_key)
    #                     # [fileName] = [(fileName, )]
    #                     record_dict[tmpSpectrum.LIST_FILE_NAME[i]].append((file_name, sup_str, pmass, lys_mass, tc, lc, judge))
    #                     # break
    #
    #         use_time[3] += time.perf_counter() - label_time
    #
    #     # #####################################################################
    #     label_time = time.perf_counter()
    #     # self.__writePairRes(i_try, record_dict)
    #     use_time[4] += time.perf_counter() - label_time
    #     # #####################################################################
    #
    #     return (prep_n, pair_n, tuple(use_time))

    def __soldierGetOnly2ndPrepData(self, ms2_file_list, use_time):

        prep_2nd_spec_moz = []  # matrix  # 第二轮预处理得到的moz
        prep_2nd_spec_int = []  # matrix  # 第二轮预处理得到的int
        prep_2nd_spec_ncl = []  # matrix  # 第二轮预处理得到的ncl  即 neucode label

        prep_n = 0

        # 直接把两轮预处理结果给存起来！！----------------------------
        label_time = time.perf_counter()
        for i, dataMS2 in enumerate(ms2_file_list):
            logToUser(INFO_TO_USER_TaskPair[5] + "[" + self.dp.LIST_MGF_NAME_LYS[i] + "]")
            for scan in dataMS2.INDEX_SCAN:
                spec = self.funcLoadPKL.getSingleMS2Spectrum(dataMS2, scan)
                tmpMOZ, tmpINT, tmpNCL = self.funcPrep.returnTwoListForPair(spec)
                prep_2nd_spec_moz += tmpMOZ  # [[..], [..], ...]  拼接 list
                prep_2nd_spec_int += tmpINT  # [[..], [..], ...]  拼接 list
                prep_2nd_spec_ncl += tmpNCL  # [[..], [..], ...]  拼接 list
                # prep_2nd_spec_moz[cnt], prep_2nd_spec_int[cnt] = self.funcPrep.returnTwoListForPair(spec)
                # cnt += 1
        use_time[2] += time.perf_counter() - label_time
        prep_n += len(prep_2nd_spec_moz)

        return prep_2nd_spec_moz, prep_2nd_spec_int, prep_2nd_spec_ncl, prep_n

    def __soldierGet1st2ndPrepData(self, ms2_file_list, use_time):
        ...



    def __logStatInfo(self, prep_n, pair_n, mem_record):

        res = "\t\t <Statistics>\n" + "-" * 28
        res += "\n(details)[NUMBER][PREP SPEC] " + str(prep_n)
        res += "\n(details)[NUMBER][EVAL PAIR] " + str(pair_n)
        res += "\n" + "." * 28
        res += "\n(details)[MEM USE][LYS DATA] " + "%.3fMB" % (mem_record[0]+0)
        res += "\n(details)[MEM USE][ PM IDX ] " + "%.3fMB" % (mem_record[1]+0)
        res += "\n(details)[MEM USE][ALL PEAK] " + "%.3fMB" % (mem_record[2]+0)
        res += "\n" + "-" * 28

        logToUser(res)

    def __logTimeCost(self, inputList:list):

        res = toolGenerateMirrorPairTimeDetail(inputList)

        logToUser(res)

    def __writePairRes(self, i_try, i_lys, inputDict):

        # 输出文件路径！
        # trypsin 文件的信息 or index or fileName
        # IO_NAME_FILE_RESULT！

        # res_file_path = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + IO_NAME_FILE_RESULT[0]
        # dis_file_path = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + IO_NAME_FILE_RESULT[1]
        res_file_path = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + IO_NAME_FILE_RESULT[0]
        dis_file_path = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + IO_NAME_FILE_RESULT[1]

        head_list = ["A_DATA_IDX",         # 0
                     "B_DATA_IDX",         # 1
                     "A_TITLE",            # 2
                     "B_TITLE",            # 3
                     "A_CHARGE",           # 2
                     "B_CHARGE",           # 3
                     "A_PM",               # 4
                     "B_PM",               # 5
                     "DELTA_PM",           # 6
                     "TARGET_P-VALUE",    # 7
                     "DECOY_P-VALUE",     # 8
                     "MIRROR_ANNO",       # 9
                     "TARGET_SCORE",      # 10
                     "DECOY_SCORE"]       # 11
        head_str = "\t".join(head_list) + "\n"

        # write the head of file
        with open(res_file_path, "w", encoding="utf-8") as f:
            f.write(head_str)

        with open(dis_file_path, "w", encoding="utf-8") as f:
            f.write(head_str)

        if self.pvFilter:
            key_list = [key for key in inputDict]
            tmpPVIdx = []
            for i, key in enumerate(key_list):
                cnt = 0
                for lys_title, ini_j, try_pm, lys_pm, tc, lc, judge_res in inputDict[key]:
                    tmpPVIdx.append((i, cnt, judge_res[1]))
                    cnt += 1
            tmpPVIdx.sort(key=lambda x: x[2])

            with open(res_file_path, "a", encoding="utf-8") as f_res, open(dis_file_path, "a", encoding="utf-8") as f_dis:
                for key_i, list_i, ev in tmpPVIdx:
                    key = key_list[key_i]
                    lys_title, ini_j, try_pm, lys_pm, tc, lc, judge_res = inputDict[key][list_i]
                    # print(judge_res)
                    writeStr = "{:d}\t{:d}\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:f}\t{:f}\t{:f}\t{}\t{:f}\t{:f}".format(i_try, i_lys, key, lys_title, tc, lc, try_pm, lys_pm, try_pm - lys_pm, judge_res[1], judge_res[2], SPECTRAL_PAIR_TYPE[abs(judge_res[0])], judge_res[3], judge_res[4]) + "\n"

                    if ev <= self.pvThreshold:
                        # a special judge for AKK/ARR/AXX puzzle
                        if judge_res[0] == 0:
                            f_dis.write(writeStr)
                        else:
                            f_res.write(writeStr)
                    else:
                        f_dis.write(writeStr)

            return

        # elif self.directJudge:
        # sort the trypsin spectral title
        key_list = [key for key in inputDict]
        key_list.sort(key=lambda x: (int(x.split(".")[2]), x))# scan, (precursor rank if have)
        # write the context of file
        with open(res_file_path, "a", encoding="utf-8") as f_res, open(dis_file_path, "a", encoding="utf-8") as f_dis:
            for key in key_list:
                if key in inputDict:
                    for lys_title, ini_j, try_pm, lys_pm, tc, lc, judge_res in inputDict[key]:

                        writeStr = "{:d}\t{:d}\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:f}\t{:f}\t{:f}\t{}\t{:f}\t{:f}".format(i_try, i_lys, key, lys_title, tc, lc, try_pm, lys_pm, try_pm - lys_pm, judge_res[1], judge_res[2], SPECTRAL_PAIR_TYPE[abs(judge_res[0])], judge_res[3], judge_res[4]) + "\n"

                        # 0: without p-value-based filtering, without FDR-estimation-based filtering
                        if not self.directJudge:
                            f_res.write(writeStr)

                        elif judge_res[0] <= 0:
                            f_dis.write(writeStr)

                        else:
                            f_res.write(writeStr)

        # stat?


class CTaskDiNovo:
    def __init__(self, inputDP):
        self.dp = inputDP
        self.funcLoadPKL = CFunctionLoadPKL(self.dp)
        self.funcPMIDX = CFunctionPrecursorMassIndex(self.dp)
        if self.dp.myCFG.A7_LABEL_TYPE == 1:
            self.funcPrep = CFunctionPreprocessNeuCode(self.dp)
            # self.funcPrepFeature = CFunctionPreprocessForXueLiNeuCode(self.dp)
        else:
            self.funcPrep = CFunctionPreprocess(self.dp)
        self.funcPrepFeature = CFunctionPreprocessForXueLi(self.dp)
        self.funcDeNovo = CFunctionDeNovo(self.dp)
        self.funcTempPath = CFunctionTempPath(self.dp)
        self.logProcess = True
        self.LOAD_TIME_LIMIT = 1200

    def work(self):
        logToUser(INFO_TO_USER_TaskDiNovo[0])
        tryN, lysN = len(self.dp.LIST_PATH_MGF_TRY), len(self.dp.LIST_PATH_MGF_LYS)
        myPool = mp.Pool(self.dp.myCFG.D2_MULTIPROCESS_NUM)
        logToUser(INFO_TO_USER_TaskDiNovo[1])
        for i in range(tryN):
            for j in range(lysN):
                # self.subTaskForDeNovo(i, j)
                myPool.apply_async(func=self.subTaskForDeNovo, args=(i, j))

        myPool.close()
        myPool.join()

        # delete all flag file
        pass
        self.deleteTempFile(tryN, lysN)

    def subTaskForDeNovo(self, try_i, lys_i):
        logStr = "\tPROCESS\t[" + self.dp.LIST_MGF_NAME_TRY[try_i] + " v.s. " + self.dp.LIST_MGF_NAME_LYS[lys_i]+ "]\t"

        time_start = time.perf_counter()
        # wait
        self.__soldierCheckAccessViolation(try_i, lys_i)
        time_wait = time.perf_counter() - time_start

        # load the spectral pair
        pair_rec = self.__soldierGetMirrorRes(try_i, lys_i)
        # load ori&2nd data, and final: get ori&2nd spec dict
        try_ori_spec_dict = self.__soldierGetOriSpecDict(try_i, pair_rec, tryFlag=True)
        lys_ori_spec_dict = self.__soldierGetOriSpecDict(lys_i, pair_rec, tryFlag=False)
        try_2nd_spec_dict = self.__soldierGet2ndSpecDict(try_i, pair_rec, tryFlag=True)
        lys_2nd_spec_dict = self.__soldierGet2ndSpecDict(lys_i, pair_rec, tryFlag=False)
        self.__generateFlagFile(try_i, lys_i)
        time_load = time.perf_counter() - time_start

        time_prep, time_denovo, time_save = 0.0, 0.0, time.perf_counter()
        test_prt, test_cnt = 0, len(pair_rec)
        # out_path = self.dp.LIST_OUTPUT_PATH[try_i] + self.dp.LIST_MGF_NAME_LYS[lys_i] + IO_NAME_FILE_RESULT[3]  # de novo result (candidate)
        out_path = self.dp.LIST_OUTPUT_PATH[try_i] + str(lys_i) + IO_NAME_FILE_RESULT[3]
        with open(out_path, "w", encoding="utf-8") as f:
            title_list = ["TRY_TITLE", "LYS_TITLE", "TRY_EXPR_MH+", "LYS_EXPR_MH+", "TRY_CHARGE", "LYS_CHARGE",
                          "MIRROR_TYPE"]
            f.write("\t".join(title_list) + "\n")
            title_list = ["CAND_RANK", "TRY_CALC_MH+", "SEQUENCE", "MODIFICATIONS", "PEPTIDE_SCORE", "AA_SCORE"]
            f.write("\t" + "\t".join(title_list) + "\n")
        time_save = time.perf_counter() - time_save

        for try_title in pair_rec:
            if self.logProcess:
                if test_prt % 50 == 0:
                    print(logStr, "........ %.2f" % (0 + test_prt * 100 / test_cnt), "%")
            time_prep_1 = time.perf_counter()
            try_ori_spec = try_ori_spec_dict[try_title]
            try_2nd_spec = try_2nd_spec_dict[try_title]
            try_mass = try_ori_spec.LIST_PRECURSOR_MASS[0]
            try_charge = try_ori_spec.LIST_PRECURSOR_CHARGE[0]
            try_ori_moz = try_ori_spec.LIST_PEAK_MOZ
            try_ori_int = try_ori_spec.LIST_PEAK_INT
            try_2nd_moz = try_2nd_spec.LIST_PEAK_MOZ
            try_2nd_int = try_2nd_spec.LIST_PEAK_INT
            try_label = self.funcPrepFeature.preprocesslabel(try_ori_spec)
            try_iso_label = try_label.CLUSTER
            try_chr_label = try_label.CHARGE
            time_prep += time.perf_counter() - time_prep_1

            for (lys_title, mirror_type) in pair_rec[try_title]:
                # find the lys info
                time_prep_1 = time.perf_counter()
                lys_ori_spec = lys_ori_spec_dict[lys_title]
                lys_2nd_spec = lys_2nd_spec_dict[lys_title]
                lys_mass = lys_ori_spec.LIST_PRECURSOR_MASS[0]
                lys_charge = lys_ori_spec.LIST_PRECURSOR_CHARGE[0]
                lys_ori_moz = lys_ori_spec.LIST_PEAK_MOZ
                lys_ori_int = lys_ori_spec.LIST_PEAK_INT
                lys_2nd_moz = lys_2nd_spec.LIST_PEAK_MOZ
                lys_2nd_int = lys_2nd_spec.LIST_PEAK_INT
                lys_label = self.funcPrepFeature.preprocesslabel(lys_ori_spec)
                lys_iso_label = lys_label.CLUSTER
                lys_chr_label = lys_label.CHARGE
                time_prep += time.perf_counter() - time_prep_1
                # try: mass, charge, ori:moz|int, 1st:moz|int, 2nd:moz|int
                # lys: mass, charge, ori:moz|int, 1st:moz|int, 2nd:moz|int, mirrorType
                time_d1 = time.perf_counter()
                res_obj = self.funcDeNovo.denovo(try_mass, try_charge, try_ori_moz, try_ori_int,
                                                 try_2nd_moz, try_2nd_int, try_iso_label, try_chr_label,
                                                 lys_mass, lys_charge, lys_ori_moz, lys_ori_int,
                                                 lys_2nd_moz, lys_2nd_int, lys_iso_label, lys_chr_label, mirror_type)
                time_denovo += time.perf_counter() - time_d1

                # title_list = ["TRY_TITLE", LYS_TITLE",
                # "CAND_RANK, "TRY_EXPR_MH+", "TRY_CALC_MH+", "MIRROR_TYPE",
                # "SEQUENCE", "MODIFICATIONS", "PEPTIDE_SCORE", "AA_SCORE"]
                time_w1 = time.perf_counter()
                with open(out_path, "a", encoding="utf-8") as f:
                    # spec pair info
                    f.write("\t".join([try_title, lys_title, "%.5f" % (try_mass + 0.0), "%.5f" % (lys_mass + 0.0),
                                       str(try_charge), str(lys_charge), mirror_type]) + "\n")
                    # candidate peptide info per pair
                    for rank in range(len(res_obj.SEQUENCE)):
                        f.write("\t" + "\t".join(
                            [str(rank + 1), "%.5f" % (res_obj.CALCMASS[rank] + 0.0), res_obj.SEQUENCE[rank],
                             res_obj.MODIFICATION[rank], str(res_obj.PEPSCORE[rank]),
                             ",".join(["%.2f" % (t + 0.0) for t in res_obj.AASCORE[rank]])]) + "\n")
                time_save += time.perf_counter() - time_w1
            test_prt += 1

        time_time = time.perf_counter() - time_start

        logStr += "TIME:{:.2f}s | WAIT:{:.2f}s | LOAD:{:.2f}s | DENOVO:{:.2f}s | SAVE: {:.2f}s".format(time_time, time_wait, time_load, time_denovo, time_save)
        logToUser(logStr)

    def __soldierGet1st2ndPrepData(self, ms2_file_list, use_time):

        prep_1st_spec_moz = []  # matrix  # 第一轮预处理得到的moz
        prep_1st_spec_int = []  # matrix  # 第一轮预处理得到的int
        prep_1st_spec_ncl = []  # matrix  # 第一轮预处理得到的ncl  即 neucode label

        prep_2nd_spec_moz = []  # matrix  # 第二轮预处理得到的moz
        prep_2nd_spec_int = []  # matrix  # 第二轮预处理得到的int
        prep_2nd_spec_ncl = []  # matrix  # 第二轮预处理得到的ncl  即 neucode label

        prep_n = 0

        # 直接把两轮预处理结果给存起来！！----------------------------
        label_time = time.perf_counter()
        for i, dataMS2 in enumerate(ms2_file_list):
            logToUser(INFO_TO_USER_TaskPair[5] + "[" + self.dp.LIST_MGF_NAME_LYS[i] + "]")
            for scan in dataMS2.INDEX_SCAN:
                spec = self.funcLoadPKL.getSingleMS2Spectrum(dataMS2, scan)
                tmpMOZ1, tmpINT1, tmpNCL1, tmpMOZ2, tmpINT2, tmpNCL2 = self.funcPrep.returnFourListForPair(spec)

                prep_1st_spec_moz += tmpMOZ1  # matrix  # 第一轮预处理得到的moz
                prep_1st_spec_int += tmpINT1  # matrix  # 第一轮预处理得到的int
                prep_1st_spec_ncl += tmpNCL1  # matrix  # 第一轮预处理得到的ncl  即 neucode label

                prep_2nd_spec_moz += tmpMOZ2  # [[..], [..], ...]  拼接 list
                prep_2nd_spec_int += tmpINT2  # [[..], [..], ...]  拼接 list
                prep_2nd_spec_ncl += tmpNCL2  # [[..], [..], ...]  拼接 list

                # cnt += 1
        use_time[2] += time.perf_counter() - label_time
        prep_n += len(prep_2nd_spec_moz)

        return prep_1st_spec_moz, prep_1st_spec_int, prep_1st_spec_ncl, prep_2nd_spec_moz, prep_2nd_spec_int, prep_2nd_spec_ncl, prep_n

    def __soldierGet2ndSpecDict(self, try_i, pair_rec, tryFlag=True):
        # 读入的是列表，转换为title为key的字典
        res = dict()
        pklPath = self.funcTempPath.geneTempPathSpecList(try_i, tryFlag=tryFlag)
        tmpList = []
        tmpList = copy.deepcopy(self.funcLoadPKL.loadSpecListPKL(pklPath))

        if tryFlag:
            for spec in tmpList:
                if spec.LIST_FILE_NAME[0] in pair_rec:
                    res[spec.LIST_FILE_NAME[0]] = spec

        else:
            titleSet = set()
            for t in pair_rec:
                for tt, _ in pair_rec[t]:
                    titleSet.add(tt)

            for spec in tmpList:
                if spec.LIST_FILE_NAME[0] in titleSet:
                    res[spec.LIST_FILE_NAME[0]] = spec

        return res

    def __soldierGetOriSpecDict(self, try_i, pair_rec, tryFlag=True):

        path = self.dp.LIST_PATH_MGF_TRY[try_i] if tryFlag else self.dp.LIST_PATH_MGF_LYS[try_i]
        inputMS2 = self.funcLoadPKL.loadMS2PKL(path + ".pkl")

        res = dict()
        for scan in inputMS2.INDEX_SCAN:
            for i, t in enumerate(inputMS2.MATRIX_FILE_NAME[scan]):
                res[t] = (scan, i)

        outRes = dict()

        if tryFlag:
            for t in pair_rec:
                scan, i = res[t]
                spec = self.funcLoadPKL.getSingleMS2Spectrum(inputMS2, scan)
                spec.LIST_PRECURSOR_CHARGE = [spec.LIST_PRECURSOR_CHARGE[i]]
                spec.LIST_PRECURSOR_MASS = [spec.LIST_PRECURSOR_MASS[i]]
                spec.LIST_PRECURSOR_MOZ = [spec.LIST_PRECURSOR_MOZ[i]]
                spec.LIST_FILE_NAME = [spec.LIST_FILE_NAME[i]]
                outRes[t] = spec

        # get lys spec dict
        else:
            titleSet = set()
            for t in pair_rec:
                for tt, _ in pair_rec[t]:
                    titleSet.add(tt)

            for t in titleSet:
                scan, i = res[t]
                spec = self.funcLoadPKL.getSingleMS2Spectrum(inputMS2, scan)
                spec.LIST_PRECURSOR_CHARGE = [spec.LIST_PRECURSOR_CHARGE[i], max(spec.LIST_PRECURSOR_CHARGE)]
                spec.LIST_PRECURSOR_MASS = [spec.LIST_PRECURSOR_MASS[i]]
                spec.LIST_PRECURSOR_MOZ = [spec.LIST_PRECURSOR_MOZ[i]]
                spec.LIST_FILE_NAME = [spec.LIST_FILE_NAME[i]]
                outRes[t] = spec

        return outRes

    def __soldierGetMirrorRes(self, try_i, lys_i):

        # res_file_path = self.dp.LIST_OUTPUT_PATH[try_i] + self.dp.LIST_MGF_NAME_LYS[lys_i] + IO_NAME_FILE_RESULT[0]
        res_file_path = self.dp.LIST_OUTPUT_PATH[try_i] + str(lys_i) + IO_NAME_FILE_RESULT[0]
        pair_rec = dict()
        try:
            with open(res_file_path, "r", encoding="utf-8") as f:
                buff = f.read().split("\n")[1:-1]

            for line in buff:
                item_list = line.split("\t")
                # try_t, lys_t, m = item_list[0], item_list[1], REVERSE_SPECTRAL_PAIR_TYPE[item_list[-1]]
                try_t, lys_t, m = item_list[0], item_list[1], item_list[-1]
                if try_t in pair_rec:
                    pair_rec[try_t].append((lys_t, m))
                else:
                    pair_rec[try_t] = [(lys_t, m)]
        except:
            print("CANNOT READ THIS FILE, PLEASE CHECK IT:", res_file_path)

        return pair_rec

    def __easyReportInfo(self, try_cnt, pair_cnt, task_time, denovo_time, write_time):

        logToUser("========== De Novo Task ==========" + "\n")
        logToUser("DeNovoTrySpec:" + str(try_cnt) + "\n")
        logToUser("DeNovoPairNum:" + str(pair_cnt) + "\n")
        logToUser("TotalUsingTime: %.3fs." % (task_time + 0.0) + "\n")
        logToUser("\t>>>De Novo: %.3fs." % (denovo_time + 0.0) + "\n")
        logToUser("\t>>>Output : %.3fs." % (write_time + 0.0) + "\n")
        logToUser("------- de novo test over. -------" + "\n")

    # ----------------------------
    # temp function for parallel de novo
    # delete the templete files
    def deleteTempFile(self, tryN, lysN):
        # de novo parallel flag
        for i in range(tryN):
            for j in range(lysN):
                tmpPath = self.funcTempPath.geneTempPathLoadFlag(i, j)
                self.__soldierCheckAndDeletePath(tmpPath)
                ...

        # speclist
        for i in range(tryN):
            tmpPath = self.funcTempPath.geneTempPathSpecList(i, tryFlag=True)
            self.__soldierCheckAndDeletePath(tmpPath)

        for i in range(lysN):
            tmpPath = self.funcTempPath.geneTempPathSpecList(i, tryFlag=False)
            self.__soldierCheckAndDeletePath(tmpPath)

        # pmidx
        # for i in range(lysN):
        #     tmpPath = self.funcTempPath.geneTempPathPMIDX(i)
        #     self.__soldierCheckAndDeletePath(tmpPath)

        # delete temp path
        tmpPath = self.funcTempPath.geneTempFolderPath()
        self.__soldierCheckAndDeletePath(tmpPath, folderFlag=True)

    # check if path is exist, delete it
    def __soldierCheckAndDeletePath(self, tmpPath, folderFlag=False):

        if os.access(tmpPath, os.F_OK):
            try:
                if folderFlag:
                    for root, dirs, files in os.walk(tmpPath):
                        for file in files:
                            os.remove(os.path.join(root, file))
                        for dir_ in dirs:
                            shutil.rmtree(os.path.join(root, dir_))
                    os.removedirs(tmpPath)
                else:
                    os.remove(tmpPath)
            except:
                if folderFlag:
                    logGetWarning("[WARNING] CANNOT delete temp file: " + tmpPath)
                else:
                    logGetWarning("[WARNING] CANNOT delete temp folder: " + tmpPath)
        else:
            if folderFlag:
                logGetWarning("[WARNING] CANNOT Find temp file: " + tmpPath)
            else:
                logGetWarning("[WARNING] CANNOT Find temp folder: " + tmpPath)
        ...

    # check
    def __soldierCheckAccessViolation(self, i_try, i_lys):

        flag_last_try, flag_last_lys = False, False
        if i_try == 0:
            flag_last_try = True
        if i_lys == 0:
            flag_last_lys = True

        path_last_try = self.funcTempPath.geneTempPathLoadFlag(i_try - 1, i_lys)  # have same lys
        path_last_lys = self.funcTempPath.geneTempPathLoadFlag(i_try, i_lys - 1)  # have same try

        if not flag_last_lys:
            flag_last_lys = self.__waitForFileExist(path_last_lys)

        if not flag_last_try:
            flag_last_try = self.__waitForFileExist(path_last_try)

        if not (flag_last_try and flag_last_lys):
            warning_str = "WARNING: Access violation may happpens by: ["
            warning_str += self.dp.LIST_MGF_NAME_TRY[i_try] + "] and ["
            warning_str += self.dp.LIST_MGF_NAME_LYS[i_lys] + "]"
            logGetWarning(warning_str)


    # write
    def __generateFlagFile(self, i_try, i_lys):

        inputPath = self.funcTempPath.geneTempPathLoadFlag(i_try, i_lys)

        try:
            with open(inputPath, "w", encoding="utf-8") as f:
                f.write("")
            # say something?

        except:
            logGetWarning("CANNOT WRITE FLAG FILE: " + inputPath)


    # waiting until time out
    def __waitForFileExist(self, inputPath):
        if not inputPath:
            return True

        res = False
        flag_time = time.perf_counter()

        while not res:
            if os.access(inputPath, os.F_OK):
                res = True
                break
            else:
                # print("\t\twaiting for file:", inputPath)
                time.sleep(1)

            if time.perf_counter() - flag_time > self.LOAD_TIME_LIMIT:
                # print("****wait time limit\n****file:{}\n****time:{}".format(inputPath, time.perf_counter() - flag_time))
                break

        return res


class CTaskCombineRes:

    def __init__(self, inputDP):
        self.dp = inputDP
        self.PAIR_RES_FILE_NAME = IO_NAME_FILE_RESULT[0]
        self.PAIR_DIS_FILE_NAME = IO_NAME_FILE_RESULT[1]
        self.DE_NOVO_FILE_NAME_TopOne = IO_NAME_FILE_RESULT[2]
        self.DE_NOVO_FILE_NAME_Cand = IO_NAME_FILE_RESULT[3]
        # # 可以调整名字，但是不要顺便改顺序喔，MSFlow里面有地址计算的方法
        # IO_NAME_FILE_RESULT = ("[DiNovo]SpectralPairs.res", "[DiNovo]SpectralPairs.dis",               # WFLOW_NUMB 2
        #                        "[DiNovo]MirrorSequencing.res", "[DiNovo]MirrorSequencing.candidate",   # WFLOW_NUMB 3
        #                        "[DiNovo]SingleSequencing.res", "[DiNovo]SingleSequencing.candidate")   # WFLOW_NUMB 4


    # in flow2
    def combineTotalSpecPair(self):
        # res - spec pair
        # out_path = self.dp.myCFG.E1_PATH_EXPORT + self.PAIR_RES_FILE_NAME
        # ori_path_list = []
        for p in self.dp.LIST_OUTPUT_PATH:
            out_path = p + self.PAIR_RES_FILE_NAME
            ori_path_list = [p + str(ln) + self.PAIR_RES_FILE_NAME for ln in range(len(self.dp.LIST_MGF_NAME_LYS))]
            self.__captainCombineResFile(out_path, ori_path_list, inputBanLineNum=1)
            # self.__captainDeleteResFile(ori_path_list, [])
        out_path = self.dp.myCFG.E1_PATH_EXPORT + self.PAIR_RES_FILE_NAME
        ori_path_list = [p + self.PAIR_RES_FILE_NAME for p in self.dp.LIST_OUTPUT_PATH]
        self.__captainCombineResFile(out_path, ori_path_list, inputBanLineNum=1)

        # dis - spec pair
        # out_path = self.dp.myCFG.E1_PATH_EXPORT + self.PAIR_DIS_FILE_NAME
        # ori_path_list = []
        for p in self.dp.LIST_OUTPUT_PATH:
            out_path = p + self.PAIR_DIS_FILE_NAME
            ori_path_list = [p + str(ln) + self.PAIR_DIS_FILE_NAME for ln in range(len(self.dp.LIST_MGF_NAME_LYS))]
            self.__captainCombineResFile(out_path, ori_path_list, inputBanLineNum=1)
            # self.__captainDeleteResFile(ori_path_list, [])
        out_path = self.dp.myCFG.E1_PATH_EXPORT + self.PAIR_DIS_FILE_NAME
        ori_path_list = [p + self.PAIR_DIS_FILE_NAME for p in self.dp.LIST_OUTPUT_PATH]
        self.__captainCombineResFile(out_path, ori_path_list, inputBanLineNum=1)

        # special for decoy 2
        if self.dp.myCFG.C12_PAIR_DECOY_APPROACH == 2:
            # calculate FDR and refresh the combine res file

            res_out_path = self.dp.myCFG.E1_PATH_EXPORT + self.PAIR_RES_FILE_NAME
            dis_out_path = self.dp.myCFG.E1_PATH_EXPORT + self.PAIR_DIS_FILE_NAME

            FDR, i_pv_t = self.__captainCalcFDR(res_out_path)
            res_tmp_path = res_out_path + ".tmp"

            i2fdr_dict = dict()
            for idx, (i, _, t) in enumerate(i_pv_t):
                i2fdr_dict[i] = [FDR[idx], t]

            with open(res_out_path, "r", encoding="utf-8") as fr, open(dis_out_path, "a", encoding="utf-8") as fw_dis, open(res_tmp_path, "w", encoding="utf-8") as fw_res:
                i = 0
                for line in fr:
                    if i == 0:
                        fw_res.write(line[:-1] + "\tFDR" + "\n")

                    elif i2fdr_dict[i][1]:  # Target result
                        ...
                        if i2fdr_dict[i][0] > self.dp.myCFG.C14_PAIR_FDR_THRESHOLD:
                            fw_dis.write(line)

                        else:
                            fw_res.write(line[:-1] + "\t" + str(i2fdr_dict[i][0]) + "\n")

                    else:  # just Decoy result
                        fw_dis.write(line)

                    i += 1
                ...

            os.remove(res_out_path)
            os.rename(res_tmp_path, res_out_path)  # old name, new name
            ...
            if self.dp.myCFG.D0_WORK_FLOW_NUMBER > 2 or True:
                res_p_matrix = []  # dis 不用管！
                for i, p in enumerate(self.dp.LIST_OUTPUT_PATH):
                    ori_path_list = [p + str(ln) + self.PAIR_RES_FILE_NAME for ln in range(len(self.dp.LIST_MGF_NAME_LYS))]
                    res_p_matrix.append([open(p, "w", encoding="utf-8") for p in ori_path_list])
                with open(res_out_path, "r", encoding="utf-8") as f:
                    i_line = 0
                    for line in f:
                        # title
                        if i_line == 0:
                            for p_list in res_p_matrix:
                                for p in p_list:
                                    p.write(line)

                        # not title and not null
                        elif i_line:
                            item_list = line.split("\t")
                            i, j = int(item_list[0]), int(item_list[1])
                            res_p_matrix[i][j].write(line)

                        # null line
                        else:
                            ...
                        i_line += 1

                for p_list in res_p_matrix:
                    for p in p_list:
                        p.close()
            ...

    # for the future
    def combineTotalDiNovoTopOne(self):
        out_path = self.dp.myCFG.E1_PATH_EXPORT + self.DE_NOVO_FILE_NAME_TopOne
        ori_path_list = [p + self.DE_NOVO_FILE_NAME_TopOne for p in self.dp.LIST_OUTPUT_PATH]
        # [ATTENTION] 这里我还不太确定应该怎么填 Top-1测序结果的 headline
        # 暂时填写为1吧，后面如果调整的话，要记得修改！
        self.__captainCombineResFile(out_path, ori_path_list, inputBanLineNum=1)
        # self.__captainDeleteResFile(ori_path_list, [])

    # in flow3
    def combineTotalDiNovoRes(self, inputN):

        folder_list = []
        if inputN == 0:
            file_name = self.DE_NOVO_FILE_NAME_Cand
            ban_n = 2

        elif inputN == 1:
            file_name = IO_NAME_FILE_GCNOVO
            ban_n = 2

        else:
            file_name = "\\" + IO_NAME_FILE_PNOVOM
            ban_n = 0

        for p in self.dp.LIST_OUTPUT_PATH:
            out_path = p + file_name
            if inputN == 2:
                out_path = p + file_name[1:]
                folder_list = [p + str(i) for i  in range(len(self.dp.LIST_MGF_NAME_LYS))]
            ori_path_list = [p + str(ln) + file_name for ln in range(len(self.dp.LIST_MGF_NAME_LYS))]
            self.__captainCombineResFile(out_path, ori_path_list, inputBanLineNum=ban_n)
            # self.__captainDeleteResFile(ori_path_list, folder_list)

        # de novo!!
        out_path = self.dp.myCFG.E1_PATH_EXPORT + file_name
        ori_path_list = [p + file_name for p in self.dp.LIST_OUTPUT_PATH]

        # [ATTENTION] there are double lines in head !!!!
        self.__captainCombineResFile(out_path, ori_path_list, inputBanLineNum=ban_n)
        # self.__captainDeleteResFile(ori_path_list, [])

    def combineTotalSingleSeq(self, out_path, ori_path_list, mirrorFlag=True):
        # print("***" * 9, "\nMSTask.py  line 1925")
        # print("out_path", out_path)
        # print("ori_path_list\n\t", "\n\t".join(ori_path_list), "***" * 9)
        ban_n = 2 if mirrorFlag else 0
        self.__captainCombineResFile(out_path, ori_path_list, inputBanLineNum=ban_n)

    # FDR, i_pv_t
    def __captainCalcFDR(self, file_path):

        i = 0
        pv_t_idx, pv_d_idx = 9, 10
        i_pv_t = []  # line id, p-value, target_label
        with open(file_path, "r", encoding="utf-8") as f:

            for line in f:

                if i == 0:
                    i += 1
                    continue

                if not line:
                    i += 1
                    continue

                line_list = line.split("\t")
                pv_t, pv_d = float(line_list[pv_t_idx]), float(line_list[pv_d_idx])

                pv, t = (pv_t, True) if pv_t < pv_d else (pv_d, False)
                i_pv_t.append([i, pv, t])

                i += 1

        # calc fdr
        i_pv_t.sort(key=lambda x: x[1])

        FDR = [0] * len(i_pv_t)
        tmpT, tmpD = 0, 1
        for idx, (i, pv, t) in enumerate(i_pv_t):
            if t:
                tmpT += 1
            else:
                tmpD += 1
            FDR[idx] = tmpD / max(tmpT, 1)

        for i in reversed(range(1, len(FDR))):
            if FDR[i-1] > FDR[i]:
                FDR[i-1] = FDR[i]

        return FDR, i_pv_t

    def __captainCombineResFile(self, out_path, ori_path_list, inputBanLineNum):

        self.__soldierCopyFirstFile(ori_path_list[0], out_path)

        for oriFilePath in ori_path_list[1:]:
            self.__soldierCopyOtherFile(oriFilePath, out_path, banLineNum=inputBanLineNum)

    def __captainDeleteResFile(self, inputPathList, newFolderList):

        for p in inputPathList:
            if os.access(p, os.F_OK):
                try:
                    os.remove(p)
                except:
                    logGetWarning("[WARNING] CANNOT delete this file: " + p)
            else:
                logGetWarning("[WARNING] CANNOT find this file: " + p)

        for p in newFolderList:
            # os.removedirs(p)
            if os.path.exists(p):
                try:
                    os.removedirs(p)
                except:
                    logGetWarning("[WARNING] CANNOT delete this folder: " + p)

            else:
                logGetWarning("[WARNING] CANNOT find this folder: " + p)

    def __soldierCopyFirstFile(self, oriFile, newFile):

        try:
            with open(oriFile, "r", encoding="utf-8") as f:
                buff = f.read()

            with open(newFile, "w", encoding="utf-8") as f:
                f.write(buff)
        except:
            with open(newFile, "w", encoding="utf-8") as f:
                f.write("")
            logGetWarning("[WARNING] CANNOT copy file: " + oriFile)


    def __soldierCopyOtherFile(self, oriFile, newFile, banLineNum):
        cnt = 0
        try:
            with open(oriFile, "r", encoding="utf-8") as fi, open(newFile, "a", encoding="utf-8") as fw:
                # buff = f.readlines()[banLineNum:]

                for line in fi:

                    cnt += 1
                    if cnt > banLineNum:
                        fw.write(line)

        except:
            logGetWarning("[WARNING] CANNOT copy file: " + oriFile)

        # with open(newFile, "a", encoding="utf-8") as f:
        #     f.write("".join(buff))

        ...


class CTaskForpNovoMorGCNovo:

    def __init__(self, inputDP):
        self.dp = inputDP
        self.stored_dir = os.getcwd().replace("\\", "/")
        if not (self.stored_dir[-1] == "/"):
            self.stored_dir += "/"

        self.mode_str = DE_NOVO_MODE_STR[self.dp.myCFG.D9_DE_NOVO_APPROACH]

        self.out_dir = os.path.abspath(self.dp.myCFG.E1_PATH_EXPORT) + "\\"
        self.tryMGFPath = "[DiNovo]TRY_COMBINE.mgf"
        self.lysMGFPath = "[DiNovo]LYS_COMBINE.mgf"
        self.specResPath = IO_NAME_FILE_RESULT[0]
        self.deNovoModelPath_MirrorNovo = self.dp.myCFG.D10_MIRROR_NOVO_MODEL_PATH
        self.deNovoModelPath_pNovoM = self.dp.myCFG.D11_PNOVOM_EXE_PATH

        self.dir_path_MirrorNovo = ""
        self.dir_path_pNovoM = ""

        self.deNovoModelPath_MirrorNovo_single = ""
        self.deNovoModelPath_pNovoM_single = ""

        self.dir_path_MirrorNovo_single = ""
        self.dir_path_pNovoM_single = ""

        self.flagRun_MirrorNovo = True if ((self.dp.myCFG.D9_DE_NOVO_APPROACH == 1) or (self.dp.myCFG.D9_DE_NOVO_APPROACH == 3))else False
        self.flagRun_pNovoM = True if ((self.dp.myCFG.D9_DE_NOVO_APPROACH == 2) or (self.dp.myCFG.D9_DE_NOVO_APPROACH == 3))else False

        self.paramPath_MirrorNovo = "config.cfg"
        self.paramPath_pNovoM = "param.txt"

        self.paramPath_MirrorNovo_single = "config_single.cfg"
        self.paramPath_pNovoM_single = "param_single.txt"

        # 0: Direct Mode
        # 1: Mirror Novo
        # 2: pNovoM Novo
        # 3: Combination
        ...

        self.funcTempPath = CFunctionTempPath(self.dp)

        self.LOAD_TIME_LIMIT = 1200


    def work(self):
        logToUser(INFO_TO_USER_TaskDiNovo[0])
        # 0. check if MirrorNovo.py or pNovoM.exe path existes!
        self.__captainCheckFileAndGetPath()

        tryN, lysN = len(self.dp.LIST_PATH_MGF_TRY), len(self.dp.LIST_PATH_MGF_LYS)

        if self.flagRun_MirrorNovo:
            # myPool = mp.Pool(self.dp.myCFG.D2_MULTIPROCESS_NUM)
            logToUser(INFO_TO_USER_TaskDiNovo[1])
            print("[NOTICE] MirrorNovo cannot using multiprocessing function now(to be continued...)")
            for i in range(tryN):
                for j in range(lysN):
                    self.subTask(i, j)
                    # myPool.apply_async(func=self.subTask, args=(i, j, True))

            # myPool.close()
            # myPool.join()


        if self.flagRun_pNovoM:
            myPool = mp.Pool(self.dp.myCFG.D2_MULTIPROCESS_NUM)
            logToUser(INFO_TO_USER_TaskDiNovo[1])
            for i in range(tryN):
                for j in range(lysN):
                    # self.subTask(i, j)
                    myPool.apply_async(func=self.subTask, args=(i, j, False))

            myPool.close()
            myPool.join()
        # exit(123)
        # self.deleteTempFile(tryN, lysN)

        if self.dp.myCFG.D12_DE_NOVO_SINGLE_SPECTRUM == 1:
            # myPool = mp.Pool(self.dp.myCFG.D2_MULTIPROCESS_NUM)
            # [ATTENTION] using GPU, hard to do multiple process
            if self.flagRun_MirrorNovo:
                for i in range(tryN):
                    self.subTaskSingle(i, True, True)

                for i in range(lysN):
                    self.subTaskSingle(i, False, True)


            if self.flagRun_pNovoM:
                myPool = mp.Pool(self.dp.myCFG.D2_MULTIPROCESS_NUM)

                for i in range(tryN):
                    # self.subTaskSingle(i, True, False)
                    myPool.apply_async(func=self.subTaskSingle, args=(i, True, False))

                for i in range(lysN):
                    # self.subTaskSingle(i, False, False)
                    myPool.apply_async(func=self.subTaskSingle, args=(i, False, False))

                myPool.close()
                myPool.join()


    def subTask(self, i_try, i_lys, flag=True):

        timeT = time.perf_counter()
        logStr = "\tPROCESS\t[" + self.dp.LIST_MGF_NAME_TRY[i_try] + " v.s. " + self.dp.LIST_MGF_NAME_LYS[i_lys] + "]\t"

        # 1. combine two .mgf by spectral pair result
        waitT = time.perf_counter()
        self.__soldierCheckAccessViolation(i_try, i_lys)
        waitT = time.perf_counter() - waitT
        # print("wait ok")
        loadT, pickT = self.__captainGeneNewMGF(i_try, i_lys, flag)
        # print("file ok")
        runT = time.perf_counter()
        # 2. generate parameter file in ...
        self.__captainGeneParamFile(i_try, i_lys, flag)
        # print("param ok")
        # 3. run pNovoM / GCNovo ...
        self.__captainRun(i_try, i_lys, flag)
        runT = time.perf_counter() - runT
        timeT = time.perf_counter() - timeT
        # time: total time
        # wait: wait for last two loading tasks fin.
        # load: load ms2 data
        # pick: pick spec and save
        # run: run pNovoM / GCNovo
        logStr += "TIME:{:.2f}s | WAIT:{:.2f}s | LOAD:{:.2f}s | PICK:{:.2f}s | RUN: {:.2f}s".format(timeT, waitT, loadT, pickT, runT)
        logToUser(logStr)


    def subTaskSingle(self, i_data, trypsinFlag=True, mirrorNovoFlag=True):

        timeT = time.perf_counter()
        if trypsinFlag:
            logStr = "\tPROCESS\t" + self.dp.LIST_MGF_NAME_TRY[i_data] + "\t"
        else:
            logStr = "\tPROCESS\t" + self.dp.LIST_MGF_NAME_LYS[i_data] + "\t"
        timeT = time.perf_counter() - timeT

        runT = time.perf_counter()
        # 2. generate parameter file in ...

        self.__captainGeneParamFile_single(i_data, trypsinFlag, mirrorNovoFlag)
        # print("param ok")
        # 3. run pNovoM / GCNovo ...
        self.__captainRun_single(i_data, trypsinFlag, mirrorNovoFlag)
        runT = time.perf_counter() - runT
        timeT = time.perf_counter() - timeT
        # time: total time
        # wait: wait for last two loading tasks fin.
        # load: load ms2 data
        # pick: pick spec and save
        # run: run pNovoM / GCNovo
        logStr += "TIME:{:.2f}s".format(timeT)
        logToUser(logStr)


    def __captainGeneNewMGF(self, i_try, i_lys, mirrorNovoFlag=True):

        if mirrorNovoFlag:
            loadT, pickT = self.__soldierGCNovoMGF(i_try, i_lys)
        else:
            loadT, pickT = self.__soldierpNovoMMGF(i_try, i_lys)

        return loadT, pickT

    # pNovoM
    def __soldierpNovoMMGF(self, i_try, i_lys):

        # logToUser(self.mode_str + "Combine TRY and LYS .mgf files...")
        loadT = time.perf_counter()
        if self.dp.myCFG.A7_LABEL_TYPE == 1:
            try_path = self.dp.myCFG.E1_PATH_EXPORT + self.dp.LIST_MGF_NAME_TRY[i_try] + FEATURE_MGF_FILE_SUFFIX
            lys_path = self.dp.myCFG.E1_PATH_EXPORT + self.dp.LIST_MGF_NAME_LYS[i_lys] + FEATURE_MGF_FILE_SUFFIX
        else:
            try_path = self.dp.LIST_PATH_MGF_TRY[i_try]
            lys_path = self.dp.LIST_PATH_MGF_LYS[i_lys]

        with open(try_path, "r", encoding="utf-8") as f:
            try_line_buff = f.read().split("\n")

        with open(lys_path, "r", encoding="utf-8") as f:
            lys_line_buff = f.read().split("\n")

        self.__generateFlagFile(i_try, i_lys)
        # try_dict, lys_dict = dict(), dict()
        # self.__geneStringDict(try_line_buff, try_dict)
        # self.__geneStringDict(lys_line_buff, lys_dict)
        loadT = time.perf_counter() - loadT

        pickT = time.perf_counter()
        try_path, lys_path = self.__newMgfPath(i_try, i_lys)
        with open(try_path, "w", encoding="utf-8") as f_try, open(lys_path, "w", encoding="utf-8") as f_lys:

            # path = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + IO_NAME_FILE_RESULT[0]
            # path = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + IO_NAME_FILE_RESULT[0]
            # pair_list = self.__getPairList(path)
            f_try.write("\n".join(try_line_buff))
            f_lys.write("\n".join(lys_line_buff))
            # for (try_t, lys_t) in pair_list:
            #     st, ed = try_dict[try_t]
            #     f_try.write("\n".join(try_line_buff[st:ed]) + "\n")
            # 
            #     st, ed = lys_dict[lys_t]
            #     f_lys.write("\n".join(lys_line_buff[st:ed]) + "\n")
        pickT = time.perf_counter() - pickT

        return loadT, pickT

    # GCNovo
    def __soldierGCNovoMGF(self, i_try, i_lys):
        ...
        # logToUser(self.mode_str + "Combine TRY and LYS .mgf files...")
        loadT = time.perf_counter()
        with open(self.dp.LIST_PATH_MGF_TRY[i_try], "r", encoding="utf-8") as f:
            try_line_buff = f.read().split("\n")
        with open(self.dp.LIST_PATH_MGF_LYS[i_lys], "r", encoding="utf-8") as f:
            lys_line_buff = f.read().split("\n")

        self.__generateFlagFile(i_try, i_lys)
        try_dict, lys_dict = dict(), dict()
        self.__geneStringDict(try_line_buff, try_dict)
        self.__geneStringDict(lys_line_buff, lys_dict)
        loadT = time.perf_counter() - loadT

        pickT = time.perf_counter()
        try_path, lys_path = self.__newMgfPath(i_try, i_lys)
        with open(try_path, "w", encoding="utf-8") as f_try, open(lys_path, "w", encoding="utf-8") as f_lys:

            f_try.write("\n".join(try_line_buff))
            f_lys.write("\n".join(lys_line_buff))
            # path = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + IO_NAME_FILE_RESULT[0]
            # path = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + IO_NAME_FILE_RESULT[0]
            # trySet, lysSet = self.__getTwoTitleSet(path)
            #
            # for try_t in trySet:
            #     st, ed = try_dict[try_t]
            #     f_try.write("\n".join(try_line_buff[st:ed]) + "\n")
            #
            # for lys_t in lysSet:
            #     st, ed = lys_dict[lys_t]
            #     f_lys.write("\n".join(lys_line_buff[st:ed]) + "\n")
        pickT = time.perf_counter() - pickT
        return loadT, pickT


    def __getTwoTitleSet(self, pair_path):
        trySet, lysSet = set(), set()

        with open(pair_path, "r", encoding="utf-8") as f:
            buff = f.read().split("\n")[1:-1]

        for i, line in enumerate(buff):
            line_item = line.split("\t")
            trySet.add(line_item[0])
            lysSet.add(line_item[1])

        return trySet, lysSet

    def __geneStringDict(self, rec_list:list, tit_dict:dict):

        st, t = 0, ""
        for i, line in enumerate(rec_list):
            if line:
                if line[0] == "B" and line.startswith("BEGIN IONS"):
                    st = i
                elif line[0] == "E" and line.startswith("END IONS"):
                    tit_dict[t] = (st, i+1)  # 方便后面切片用
                elif line[0] == "T" and line.startswith("TITLE"):
                    t = line[6:]

    def __soldierGetTryStringListDict(self, idx):
        res_list, res_dict = [], dict()
        logToUser(self.mode_str + "\t<Load> [MGF]\t" + self.dp.LIST_PATH_MGF_TRY[idx])
        with open(self.dp.LIST_PATH_MGF_TRY[idx], "r", encoding="utf-8") as f:
            res_list += f.read().split("\n")

        st, t = 0, ""
        for i, line in enumerate(res_list):
            if line:
                if line[0] == "B" and line.startswith("BEGIN IONS"):
                    st = i
                elif line[0] == "E" and line.startswith("END IONS"):
                    res_dict[t] = (st, i+1)  # 方便后面切片用
                elif line[0] == "T" and line.startswith("TITLE"):
                    t = line[6:]

        return res_list, res_dict

    def __captainCheckFileAndGetPath(self):

        # check -------------

        if self.flagRun_MirrorNovo:
            self.deNovoModelPath_MirrorNovo = os.path.abspath(self.deNovoModelPath_MirrorNovo)
            self.dir_path_MirrorNovo = os.path.dirname(self.deNovoModelPath_MirrorNovo) + "\\"
            if self.dp.myCFG.A7_LABEL_TYPE == 1:

                self.dir_path_MirrorNovo_single = self.dir_path_MirrorNovo + "GCNovo_Neucode_Api1.2\\"
                self.deNovoModelPath_MirrorNovo_single = self.dir_path_MirrorNovo_single + "GCNovo.py"

                self.dir_path_MirrorNovo += "NeucodeApi1.1\\"
                self.deNovoModelPath_MirrorNovo = self.dir_path_MirrorNovo + "MirrorNovo.py"

            else:
                self.dir_path_MirrorNovo_single = self.dir_path_MirrorNovo + "GCNovoApi1.1\\"
                self.deNovoModelPath_MirrorNovo_single = self.dir_path_MirrorNovo_single + "GCNovo.py"

                # self.dir_path_MirrorNovo += "NeucodeApi1.0\\"
                # self.deNovoModelPath_MirrorNovo = self.dir_path_MirrorNovo + "MirrorNovo.py"


        if self.flagRun_pNovoM:
            self.deNovoModelPath_pNovoM = os.path.abspath(self.deNovoModelPath_pNovoM)
            self.dir_path_pNovoM = os.path.dirname(self.deNovoModelPath_pNovoM) + "\\"

            self.dir_path_pNovoM_single = self.dir_path_pNovoM + "single\\"
            self.deNovoModelPath_pNovoM_single = self.dir_path_pNovoM_single + "pNovoM2_single.exe"  # "pNovoM2_single.exe"
        # self.tryMGFPath = self.dir_path + self.tryMGFPath
        # self.lysMGFPath = self.dir_path + self.lysMGFPath
        # self.paramPath =  self.dir_path + self.paramPath


    def __captainGeneParamFile(self, i_try, i_lys, mirrorNovoFlag=True):

        if mirrorNovoFlag:
            # logToUser(self.mode_str + "GCNovo param file will be in: " + self.paramPath)
            self.__soldierGeneParamFile_GCNovo(i_try, i_lys)
            # logToUser(self.mode_str + "GCNovo param file has been generated.\n")

            # pNovoM
        else:
            # logToUser(self.mode_str + "pNovoM param file will be in: " + self.paramPath + "\n")
            self.__soldierGeneParamFile_pNovoM(i_try, i_lys)
            # logToUser(self.mode_str + "pNovoM param file has been generated.\n")

        # else:
        #     logGetError("[MSTask] DE_NOVO_APPROACH not 2 or 3!")

    def __captainGeneParamFile_single(self, i_data, trypsinFlag=True, mirrorNovoFlag=True):

        if mirrorNovoFlag:
            self.__soldierGeneParamFile_GCNovo_single(i_data, trypsinFlag)

        else:
            self.__soldierGeneParamFile_pNovoM_single(i_data, trypsinFlag)


    def toolGeneOutPathParamPath_single(self, i_data, trypsinFlag=True, mirrorNovoFlag=True):

        tmpRoot = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FOLDER_TEMPORARY
        if mirrorNovoFlag:
            if trypsinFlag:
                # data_path = self.dp.LIST_PATH_MGF_TRY[i_data]
                tmpParam = tmpRoot + "A_" + str(i_data) + self.paramPath_MirrorNovo_single
                tmpOut = tmpRoot + "A_" + str(i_data)
            else:
                # data_path = self.dp.LIST_PATH_MGF_LYS[i_data]
                tmpParam = tmpRoot + "B_" + str(i_data) + self.paramPath_MirrorNovo_single
                tmpOut = tmpRoot + "B_" + str(i_data)
        else:
            if trypsinFlag:
                tmpParam = tmpRoot + "A_" + str(i_data) + "\\" + self.paramPath_pNovoM_single
                tmpOut = tmpRoot + "A_" + str(i_data) + "\\"
            else:
                tmpParam = tmpRoot + "B_" + str(i_data) + "\\" + self.paramPath_pNovoM_single
                tmpOut = tmpRoot + "B_" + str(i_data) + "\\"

        return tmpOut, tmpParam

    def toolGetSingleSpecSeqResPathList(self, trypsinFlag=True, mirrorNovoFlag=True):

        path_list = []
        # enz_label = "A_" if trypsinFlag else "B_"
        sep_label = "" if mirrorNovoFlag else "\\"

        # tmpOut, _ = self.toolGeneOutPathParamPath_single(0, trypsinFlag, mirrorNovoFlag)

        # MirrorNovo
        if mirrorNovoFlag:
            fileName = "MirrorNovoResFinal.txt.beamsearch.txt"

        # pNovoM2
        else:

            if trypsinFlag:
                fileName = "DenovoResultsSingleSpecFinal1.txt"
            else:
                fileName = "DenovoResultsSingleSpecFinal2.txt"


        data_len = len(self.dp.LIST_PATH_MGF_TRY) if trypsinFlag else len(self.dp.LIST_PATH_MGF_LYS)
        for i in range(data_len):
            tmpOut, _ = self.toolGeneOutPathParamPath_single(i, trypsinFlag, mirrorNovoFlag)
            # tmpPath = tmpOut + enz_label + str(i) + sep_label + fileName
            tmpPath = tmpOut + fileName
            path_list.append(tmpPath)

        return path_list

    def __soldierGeneParamFile_pNovoM(self, i_try, i_lys):

        try_path, lys_path = self.__newMgfPath(i_try, i_lys)
        spec_path = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + IO_NAME_FILE_RESULT[0]
        # tmpParam = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + self.paramPath
        # tmpOut = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + IO_NAME_FILE_PNOVOM
        tmpParam = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + self.paramPath_pNovoM
        # tmpOut = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + IO_NAME_FILE_PNOVOM
        tmpOut = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + "\\"
        if os.access(tmpOut, os.F_OK):
            pass
        else:
            os.mkdir(tmpOut)
        spec_path = os.path.abspath(spec_path)
        tmpOut = os.path.abspath(tmpOut)
        if tmpOut[-1] != "\\":
            tmpOut += "\\"
        # with open(tmpParam, "w", encoding="utf-8") as f:
        with open(tmpParam, "w") as f:  # 20240521  路径中有中文时，pNovoM2使用GB编码读取文件，这里先不指定utf-8
            f.write("\n")
            f.write("TrypsinPath=" + try_path + "\n")
            # f.write("#set path of Ac-LysargiNase mgf file" + "\n")
            f.write("R29Path=" + lys_path + "\n")
            f.write("DiNovoSpecPairsPath=" + spec_path + "\n")
            # f.write("#set output path, the default output file is denovo_result.txt" + "\n")
            f.write("OutputPath=" + tmpOut + "\n")
            # f.write("#the follow three parameters are for cluster, if cluster=no, the three parameters are invalid" + "\n")
            # f.write("#cluster=yes" + "\n")
            # f.write("deltaScore=2" + "\n")
            # f.write("TopN=10" + "\n")
            # f.write("#N-term=27.994915" + "\n")
            ...

            # f.write("#set model, default setting is recommended" + "\n")
            if self.dp.myCFG.A7_LABEL_TYPE == 0:
                f.write("Mode=Normal\n")
            else:
                f.write("Mode=Neucode\n")
                # 1: trypsin - lysargiNase
                # 2: lysC - lysN
                type_num = "K" if self.dp.myCFG.A11_MIRROR_PROTEASES_APPROACH == 2 else "KR"
                f.write("NeucodeSite=" + type_num + "\n")

            f.write("Filter_charge=false\n")
        ...

    def __soldierGeneParamFile_GCNovo(self, i_try, i_lys):

        try_path, lys_path = self.__newMgfPath(i_try, i_lys)
        # tmpParam = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + self.paramPath
        # tmpOut = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + IO_NAME_FILE_GCNOVO
        tmpParam = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + self.paramPath_MirrorNovo
        tmpOut = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + IO_NAME_FILE_GCNOVO
        tmpOut = os.path.abspath(tmpOut)
        # pairPath = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + IO_NAME_FILE_RESULT[0]
        pairPath = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + IO_NAME_FILE_RESULT[0]
        # pairPath = self.dp.LIST_OUTPUT_PATH[i_try] + IO_NAME_FILE_RESULT[1]
        pairPath = os.path.abspath(pairPath)
        with open(tmpParam, "w") as f:
            f.write("[train]" + "\n")
            f.write("engine_model=2" + "\n")
            if self.dp.myCFG.A7_LABEL_TYPE == 1:
                f.write("train_dir=" + self.dir_path_MirrorNovo + "model_file\\test_Ecoli\n")
                f.write("cuda_device=0" + "\n")
            else:
                f.write("train_dir=" + self.dir_path_MirrorNovo + "train_model\n")

            # ------------------------------------
            # 0: trypsin - lysargiNase
            # 1: lysC - lysN
            type_num = "0" if self.dp.myCFG.A11_MIRROR_PROTEASES_APPROACH == 1 else "1"
            f.write("type=" + type_num + "\n")
            # ------------------------------------
            # f.write("cuda_device=0" + "\n")
            f.write("processes=0" + "\n")
            f.write("num_workers=4" + "\n")
            f.write("batch_size=" + str(self.dp.myCFG.D13_BATCH_SIZE) + "\n")
            f.write("num_epoch=20" + "\n")
            f.write("init_lr = 1e-3" + "\n")
            f.write("steps_per_validation = 1000" + "\n")
            f.write("weight_decay = 0.0" + "\n\n")

            f.write("MAX_NUM_PEAK=500" + "\n")
            f.write("MZ_MAX=6000.0" + "\n")
            f.write("MAX_LEN=60" + "\n")
            if self.dp.myCFG.A7_LABEL_TYPE == 1:
                f.write("num_ions=26" + "\n\n")
            else:
                f.write("num_ions=18" + "\n\n")

            f.write("[model]" + "\n")
            # f.write("input_dim=433" + "\n")
            f.write("output_dim=256" + "\n")
            f.write("units=64" + "\n")
            f.write("n_classes=23" + "\n")
            # f.write("edges_classes=25" + "\n")
            # f.write("dropout=0.25" + "\n\n")

            # f.write("[lstm]" + "\n")
            # f.write("use_lstm=True" + "\n")
            # f.write("lstm_hidden_units=512" + "\n")
            # f.write("embedding_size=512" + "\n")
            # f.write("num_lstm_layers=1" + "\n\n")

            f.write("[search]" + "\n")
            f.write("beam_size=" + str(self.dp.myCFG.D8_REPORT_PEPTIDE_NUM) + "\n")
            if self.dp.myCFG.A7_LABEL_TYPE == 1:
                f.write("knapsack=" + self.dir_path_MirrorNovo + "knapsackfile\\neucodeknapsack_C_M_IL.npy" + "\n\n")
            else:
                f.write("knapsack=" + self.dir_path_MirrorNovo + "knapsackfile\\knapsack_C_M_IL.npy" + "\n\n")

            f.write("[data]" + "\n")
            f.write("denovo_input_spectrum_file=" + try_path + "\n")
            f.write("denovo_input_mirror_spectrum_file=" + lys_path + "\n")
            f.write("denovo_input_feature_file=" + pairPath + "\n")
            f.write("denovo_output_file=" + tmpOut + "\n")
        ...

    def __soldierGeneParamFile_pNovoM_single(self, i_data, trypsinFlag=True):
        # print(self.dp.myCFG.E1_PATH_EXPORT)
        self.dp.myCFG.E1_PATH_EXPORT = os.path.abspath(self.dp.myCFG.E1_PATH_EXPORT) + "\\"
        if trypsinFlag:
            try_path, _ = self.__newMgfPath(i_data, 0)
            lys_path = ""
            # if self.dp.myCFG.A7_LABEL_TYPE == 0:
            #     try_path = self.dp.LIST_PATH_MGF_TRY[i_data]
            #     lys_path = ""  # self.dp.LIST_PATH_MGF_LYS[0]  # ""
            # else:
            #     try_path = self.dp.myCFG.E1_PATH_EXPORT + self.dp.LIST_MGF_NAME_TRY[i_data][:] + FEATURE_MGF_FILE_SUFFIX
            #     lys_path = ""

        else:
            _, lys_path = self.__newMgfPath(0, i_data)
            try_path = ""
            # if self.dp.myCFG.A7_LABEL_TYPE == 0:
            #     try_path = ""  # self.dp.LIST_PATH_MGF_TRY[0]  #  ""
            #     lys_path = self.dp.LIST_PATH_MGF_LYS[i_data]
            # else:
            #     try_path = ""
            #     lys_path = self.dp.myCFG.E1_PATH_EXPORT + self.dp.LIST_MGF_NAME_LYS[i_data][:] + FEATURE_MGF_FILE_SUFFIX

        tmpOut, tmpParam = self.toolGeneOutPathParamPath_single(i_data, trypsinFlag, mirrorNovoFlag=False)
        tmpOut = os.path.abspath(tmpOut)
        # print(try_path)
        if os.access(tmpOut, os.F_OK):
            pass
        else:
            os.mkdir(tmpOut)

        if tmpOut[-1] != "\\":
            tmpOut += "\\"

        # if self.dp.myCFG.A7_LABEL_TYPE == 0:
        # with open(tmpParam, "w", encoding="utf-8") as f:
        with open(tmpParam, "w") as f:  # 20240521  路径中有中文时，pNovoM2使用GB编码读取文件，这里先不指定utf-8
            f.write("\n")
            f.write("TrypsinPath=" + try_path + "\n")
            # f.write("#set path of Ac-LysargiNase mgf file" + "\n")
            f.write("R29Path=" + lys_path + "\n")
            # ------------------------------------
            # 0: trypsin - lysargiNase
            # 1: lysC - lysN
            # type_num = "0" if self.dp.myCFG.A11_MIRROR_PROTEASES_APPROACH == 1 else "1"
            # f.write("type=" + type_num + "\n")
            # ------------------------------------
            # f.write("DiNovoSpecPairsPath=" + spec_path + "\n")
            # f.write("#set output path, the default output file is denovo_result.txt" + "\n")
            f.write("DiNovoSpecPairsPath=" + "\n")
            f.write("OutputPath=" + tmpOut + "\n")
            # f.write("#the follow three parameters are for cluster, if cluster=no, the three parameters are invalid" + "\n")
            # f.write("#cluster=yes" + "\n")
            # f.write("deltaScore=2" + "\n")
            # f.write("TopN=10" + "\n")
            # f.write("#N-term=27.994915" + "\n")

            # f.write("#set model, default setting is recommended" + "\n")
            f.write("CleavageTerminal1=C\n")
            f.write("CleavageTerminal2=N\n")
            # special setting
            # 如果C末端有K/R/KR存在
            # 如果N末端有K/R/KR存在
            type_num = "K" if self.dp.myCFG.A11_MIRROR_PROTEASES_APPROACH == 2 else "KR"
            f.write("Site1=" + type_num + "\n")
            f.write("Site2=" + type_num + "\n")

            """
            C1_MIRROR_TYPE_A1 = 1         # xxK - Kxx    considering C:K   N:K
            C2_MIRROR_TYPE_A2 = 1         # xxR - Rxx    considering C:R   N:R
            C3_MIRROR_TYPE_A3 = 0         # xx  -  xx    considering C:_   N:_
            C4_MIRROR_TYPE_B  = 1         # xxR - Kxx    considering C:R   N:K
            C5_MIRROR_TYPE_C  = 1         # xxK - Rxx    considering C:K   N:R
            C6_MIRROR_TYPE_D  = 1         # xxK -  xx    considering C:K   N:_
            C7_MIRROR_TYPE_E  = 1         # xxR -  xx    considering C:R   N:_
            C8_MIRROR_TYPE_F  = 1         # xx  - Kxx    considering C:_   N:K
            C9_MIRROR_TYPE_G  = 1         # xx  - Rxx    considering C:_   N:R
           """
            # C端检测K的配置：A1，C，D
            # C_K = "K" if (self.dp.myCFG.C1_MIRROR_TYPE_A1 or self.dp.myCFG.C5_MIRROR_TYPE_C or self.dp.myCFG.C6_MIRROR_TYPE_D) else ""
            # # C端检测R的配置：A2，B，E
            # C_R = "R" if (self.dp.myCFG.C2_MIRROR_TYPE_A2 or self.dp.myCFG.C4_MIRROR_TYPE_B or self.dp.myCFG.C7_MIRROR_TYPE_E) else ""
            # # N端检测K的配置：A1，B，F
            # N_K = "K" if (self.dp.myCFG.C1_MIRROR_TYPE_A1 or self.dp.myCFG.C4_MIRROR_TYPE_B or self.dp.myCFG.C8_MIRROR_TYPE_F) else ""
            # # N端检测R的配置：A2，C，G
            # N_R = "R" if (self.dp.myCFG.C2_MIRROR_TYPE_A2 or self.dp.myCFG.C5_MIRROR_TYPE_C or self.dp.myCFG.C9_MIRROR_TYPE_G) else ""
            # f.write("Site1=" + C_K + C_R + "\n")
            # f.write("Site2=" + N_K + N_R + "\n")
            # 测试过，site12这里，空着也能跑通
            if self.dp.myCFG.A7_LABEL_TYPE == 0:
                f.write("Mode=Single\n")
            else:
                f.write("Mode=Neucode&Single\n")
            f.write("Filter_charge=false\n")

        # else:
        #     with open(tmpParam, "w") as f:  # 20240521  路径中有中文时，pNovoM2使用GB编码读取文件，这里先不指定utf-8
        #         if trypsinFlag:
        #             f.write("inputMgf=" + try_path + "\n")
        #             f.write("VirtualMgf=" + lys_path + "\n")
        #             f.write("outputPath=" + tmpOut + "\n")
        #             f.write("terminal_neucode=C" + "\n")
        #         else:
        #             f.write("inputMgf=" + lys_path + "\n")
        #             f.write("VirtualMgf=" + try_path + "\n")
        #             f.write("outputPath=" + tmpOut + "\n")
        #             f.write("terminal_neucode=N" + "\n")
        #
        #         # f.write("#the follow three parameters are for cluster, if cluster=no, the three parameters are invalid" + "\n")
        #         f.write("cluster=yes" + "\n")
        #         f.write("deltaScore=2" + "\n")
        #         f.write("TopN=10" + "\n")
        #
        #         # f.write("#set model, default setting is recommended" + "\n")
        #         f.write("ModelPath=" + self.dir_path_pNovoM + "train\\model2.txt" + "\n")
        #         f.write("splitGraph=no\n")
        ...

    def __soldierGeneParamFile_GCNovo_single(self, i_data, trypsinFlag=True):

        if trypsinFlag:
            data_path = self.dp.LIST_PATH_MGF_TRY[i_data]

        else:
            data_path = self.dp.LIST_PATH_MGF_LYS[i_data]

        tmpOut, tmpParam = self.toolGeneOutPathParamPath_single(i_data, trypsinFlag, mirrorNovoFlag=True)
        tmpOut = os.path.abspath(tmpOut) + IO_NAME_FILE_GCNOVO

        single_root = self.dir_path_MirrorNovo_single

        with open(tmpParam, "w") as f:
            f.write("[train]" + "\n")
            f.write("engine_model=2" + "\n")
            if self.dp.myCFG.A7_LABEL_TYPE == 1:
                f.write("train_dir=" + single_root + "train.example\\E.coli_Yeast" + "\n")
            else:
                f.write("train_dir=" + single_root + "train.example\\M30T4V60" + "\n")
            f.write("cuda_device=0" + "\n")
            # -----------------------------
            # 0: trypsin - lysargiNase
            # 1: lysC - lysN
            type_num = "0" if self.dp.myCFG.A11_MIRROR_PROTEASES_APPROACH == 1 else "1"
            f.write("type=" + type_num + "\n")
            # -----------------------------
            f.write("num_workers=2" + "\n")
            f.write("batch_size=" + str(self.dp.myCFG.D13_BATCH_SIZE)  + "\n")
            f.write("num_epoch=20" + "\n")
            f.write("init_lr = 1e-3" + "\n")
            f.write("steps_per_validation = 2000" + "\n")
            f.write("weight_decay = 0.0" + "\n\n")

            f.write("MAX_NUM_PEAK=500" + "\n")
            f.write("MZ_MAX=8000.0" + "\n")
            f.write("MAX_LEN=60" + "\n")
            if self.dp.myCFG.A7_LABEL_TYPE == 1:
                f.write("num_ions=24" + "\n\n")
            else:
                f.write("num_ions=26" + "\n\n")

            f.write("[model]" + "\n")

            if self.dp.myCFG.A7_LABEL_TYPE == 1:
                f.write("input_dim=533" + "\n")
                f.write("output_dim=256" + "\n")
                f.write("units=64" + "\n")
                f.write("n_classes=26" + "\n")
                f.write("edges_classes=27" + "\n")
            else:
                f.write("input_dim=497" + "\n")
                f.write("output_dim=256" + "\n")
                f.write("units=64" + "\n")
                f.write("n_classes=26" + "\n")
                f.write("edges_classes=25" + "\n")

            f.write("use_lstm=False" + "\n")
            f.write("lstm_hidden_units=512" + "\n")
            f.write("embedding_size=512" + "\n")
            f.write("num_lstm_layers=1" + "\n")
            f.write("dropout=0.25" + "\n\n")

            f.write("[search]" + "\n")
            f.write("beam_size=" + str(self.dp.myCFG.D8_REPORT_PEPTIDE_NUM) + "\n")

            if self.dp.myCFG.A7_LABEL_TYPE == 1:
                f.write("knapsack=" + single_root + "knapsackfile\\Neucode.npy" + "\n\n")
            else:
                f.write("knapsack=" + single_root + "knapsackfile\\knapsack_24.npy" + "\n\n")
            f.write("[data]" + "\n")
            f.write("denovo_input_spectrum_file=" + data_path + "\n")
            f.write("denovo_output_file=" + tmpOut + "\n")
        ...

    def __captainRun(self, i_try, i_lys, mirrorNovoFlag=True):

        # logToUser(self.mode_str + "Start runing!" + "\n")
        # hold = time.perf_counter()

        if mirrorNovoFlag:
            self.__soldierRun_GCNovo(i_try, i_lys)

        else:
            self.__soldierRun_pNovoM(i_try, i_lys)
            ...

        # hold = time.perf_counter() - hold
        # logToUser(self.mode_str + "Time cost: %.3fs" % (hold + 0.0))

    def __captainRun_single(self, i_data, trypsinFlag=True, mirrorNovoFlag=True):
        if mirrorNovoFlag:
            self.__soldierRun_GCNovo_single(i_data, trypsinFlag)

        else:
            self.__soldierRun_pNovoM_single(i_data, trypsinFlag)
            ...

    def __soldierRun_pNovoM(self, i_try, i_lys):
        command_line = self.deNovoModelPath_pNovoM + " "
        # param_f_path = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + self.paramPath
        param_f_path = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + self.paramPath_pNovoM
        param_f_path = os.path.abspath(param_f_path)
        command_line += param_f_path
        os.chdir(self.dir_path_pNovoM)
        # print("command_line", command_line)
        receive = os.system(command_line)

        os.chdir(self.stored_dir)

        # logToUser("pNovoM RETURN:\n" + str(receive))
        ...

    def __soldierRun_GCNovo(self, i_try, i_lys):
        command_line = "activate mirrornovo_env && python "
        command_line += self.deNovoModelPath_MirrorNovo + " "
        # param_f_path = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + self.paramPath
        param_f_path = self.dp.LIST_OUTPUT_PATH[i_try] + str(i_lys) + self.paramPath_MirrorNovo
        param_f_path = os.path.abspath(param_f_path)
        command_line += param_f_path
        os.chdir(self.dir_path_MirrorNovo)
        receive = os.system(command_line)
        os.chdir(self.stored_dir)
        # logToUser("GCNovo RETURN:\n" + str(receive))
        ...


    def __soldierRun_pNovoM_single(self, i_data, trypsinFlag=True):
        # if self.dp.myCFG.A7_LABEL_TYPE == 0:
        #     exe_path = self.deNovoModelPath_pNovoM
        # else:
        #     exe_path = os.path.dirname(self.deNovoModelPath_pNovoM) + "\\neuCodeSingle\\neuCodeSingle.exe"
        exe_path = self.deNovoModelPath_pNovoM_single

        command_line = exe_path + " "
        # param_f_path = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + self.paramPath
        # if trypsinFlag:
        _, param_f_path = self.toolGeneOutPathParamPath_single(i_data, trypsinFlag, False)

        param_f_path = os.path.abspath(param_f_path)
        command_line += param_f_path
        os.chdir(self.dir_path_pNovoM_single)
        print("command_line\t", command_line)
        receive = os.system(command_line)

        os.chdir(self.stored_dir)
        # os.remove(param_f_path)
        # logToUser("pNovoM RETURN:\n" + str(receive))
        ...

    def __soldierRun_GCNovo_single(self, i_data, trypsinFlag=True):

        py_path = self.deNovoModelPath_MirrorNovo_single
        py_path = os.path.abspath(py_path)
        command_line = "activate mirrornovo_env && python "
        command_line += py_path + " "
        # param_f_path = self.dp.LIST_OUTPUT_PATH[i_try] + self.dp.LIST_MGF_NAME_LYS[i_lys] + self.paramPath

        _, param_f_path = self.toolGeneOutPathParamPath_single(i_data, trypsinFlag, True)


        param_f_path = os.path.abspath(param_f_path)
        command_line += param_f_path

        os.chdir(self.dir_path_MirrorNovo_single)
        receive = os.system(command_line)
        os.chdir(self.stored_dir)
        os.remove(param_f_path)
        # logToUser("GCNovo RETURN:\n" + str(receive))
        ...

    # ---------------------

    # check

    def __soldierCheckAccessViolation(self, i_try, i_lys):

        flag_last_try, flag_last_lys = False, False
        if i_try == 0:
            flag_last_try = True
        if i_lys == 0:
            flag_last_lys = True

        path_last_try = self.funcTempPath.geneTempPathLoadFlag(i_try - 1, i_lys)  # have same lys
        path_last_lys = self.funcTempPath.geneTempPathLoadFlag(i_try, i_lys - 1)  # have same try

        if not flag_last_lys:
            flag_last_lys = self.__waitForFileExist(path_last_lys)

        if not flag_last_try:
            flag_last_try = self.__waitForFileExist(path_last_try)

        if not (flag_last_try and flag_last_lys):
            warning_str = "WARNING: Access violation may happpens by: ["
            warning_str += self.dp.LIST_MGF_NAME_TRY[i_try] + "] and ["
            warning_str += self.dp.LIST_MGF_NAME_LYS[i_lys] + "]"
            logGetWarning(warning_str)


    # write
    def __generateFlagFile(self, i_try, i_lys):

        inputPath = self.funcTempPath.geneTempPathLoadFlag(i_try, i_lys)

        try:
            with open(inputPath, "w", encoding="utf-8") as f:
                f.write("")
            # say something?

        except:
            logGetWarning("CANNOT WRITE FLAG FILE: " + inputPath)


    # waiting until time out
    def __waitForFileExist(self, inputPath):
        if not inputPath:
            return True

        res = False
        flag_time = time.perf_counter()

        while not res:
            if os.access(inputPath, os.F_OK):
                res = True
                break
            else:
                # print("\t\twaiting for file:", inputPath)
                time.sleep(1)

            if time.perf_counter() - flag_time > self.LOAD_TIME_LIMIT:
                # print("****wait time limit\n****file:{}\n****time:{}".format(inputPath, time.perf_counter() - flag_time))
                break

        return res


    def __newMgfPath(self, i_try, i_lys):

        lab = str(i_try) + "@" + str(i_lys)
        tryP, lysP = lab + self.tryMGFPath, lab + self.lysMGFPath

        tempRoot = self.funcTempPath.geneTempFolderPath()
        tempRoot = os.path.abspath(tempRoot) + "\\"
        tryP, lysP = tempRoot + tryP, tempRoot + lysP

        return tryP, lysP

    # delete the templete files
    def deleteTempFile(self, tryN, lysN):
        # flag must
        for i in range(tryN):
            for j in range(lysN):
                tmpPath = self.funcTempPath.geneTempPathLoadFlag(i, j)
                self.__soldierCheckAndDeletePath(tmpPath)
                ...

        # delete tmp mgf
        tmpRoot = self.funcTempPath.geneTempFolderPath()
        for i in range(tryN):
            for j in range(lysN):
                lab = tmpRoot + str(i) + "@" + str(j)
                if (self.dp.myCFG.D9_DE_NOVO_APPROACH == 2) or (self.dp.myCFG.D9_DE_NOVO_APPROACH == 4):
                    tmpPath = lab + self.tryMGFPath
                    self.__soldierCheckAndDeletePath(tmpPath)
                    self.__soldierCheckAndDeletePath(tmpPath + ".location.pytorch.pkl")
                    tmpPath = lab + self.lysMGFPath
                    self.__soldierCheckAndDeletePath(tmpPath)
                    self.__soldierCheckAndDeletePath(tmpPath + ".location.pytorch.pkl")
                else:
                    tmpPath = lab + self.tryMGFPath
                    self.__soldierCheckAndDeletePath(tmpPath)
                    tmpPath = lab + self.lysMGFPath
                    self.__soldierCheckAndDeletePath(tmpPath)


        # param file
        for i in range(tryN):
            tmpRoot = self.dp.LIST_OUTPUT_PATH[i]
            for j in range(lysN):
                # tmpPath = tmpRoot + self.dp.LIST_MGF_NAME_LYS[j] + self.paramPath
                tmpPath = tmpRoot + str(j) + self.paramPath_MirrorNovo
                self.__soldierCheckAndDeletePath(tmpPath)

        tmpPath = self.funcTempPath.geneTempFolderPath()
        self.__soldierCheckAndDeletePath(tmpPath, folderFlag=True)


    # check if path is exist, delete it
    def __soldierCheckAndDeletePath(self, tmpPath, folderFlag=False):

        if os.access(tmpPath, os.F_OK):
            try:
                if folderFlag:
                    os.removedirs(tmpPath)
                else:
                    os.remove(tmpPath)
            except:
                if folderFlag:
                    logGetWarning("[WARNING] CANNOT delete temp file: " + tmpPath)
                else:
                    logGetWarning("[WARNING] CANNOT delete temp folder: " + tmpPath)
        else:
            if folderFlag:
                logGetWarning("[WARNING] CANNOT Find temp file: " + tmpPath)
            else:
                logGetWarning("[WARNING] CANNOT Find temp folder: " + tmpPath)
        ...


class CTaskIsoFeatureForDeNovo:

    def __init__(self, inputDP):
        self.dp = inputDP
        self.subTask = self.subTaskNeuCode if self.dp.myCFG.A7_LABEL_TYPE == 1 else self.subTaskUnlabel

    def work(self):

        # all_path = self.dp.LIST_PATH_MGF_TRY + self.dp.LIST_PATH_MGF_LYS
        logToUser(INFO_TO_USER_TaskPrep[0])
        myPool = mp.Pool(self.dp.myCFG.D2_MULTIPROCESS_NUM)

        for raw_index, path in enumerate(self.dp.LIST_PATH_MGF_TRY):
            # self.subTask(raw_index, path)
            myPool.apply_async(func=self.subTask, args=(raw_index, path, True))

        for raw_index, path in enumerate(self.dp.LIST_PATH_MGF_LYS):
            # self.subTask(raw_index, path)
            myPool.apply_async(func=self.subTask, args=(raw_index, path, False))

        myPool.close()
        myPool.join()

    def subTaskUnlabel(self, idx, path, flag=True):

        raw_name = self.dp.LIST_MGF_NAME_TRY[idx] if flag else self.dp.LIST_MGF_NAME_LYS[idx]
        logToUser(INFO_TO_USER_TaskPrep[1] + "\t[" + raw_name + "]")

        functionLoadPKL = CFunctionLoadPKL(self.dp)
        funcPreproLabel = CFunctionPreprocessForXueLi(self.dp)
        # 从硬盘中载入，得到CFileMS2，也就是一整个MS2文件
        dataMS2 = functionLoadPKL.loadMS2PKL(path + ".pkl")
        out_path = self.dp.myCFG.E1_PATH_EXPORT + raw_name + "[feature].mgf"

        with open(out_path, "w", encoding="utf-8") as f:
            ...

        for index in dataMS2.INDEX_SCAN[:]:

            # 得到一个新的spectrum，类型是CSpectrum
            spectrum = functionLoadPKL.getSingleMS2Spectrum(dataMS2, index)
            # print(spectrum.LIST_PEAK_MOZ)
            # print(spectrum.LIST_PEAK_INT)
            # 这就是预处理的标注！！哈哈哈哈哈哈！啊哈哈哈哈！
            label = funcPreproLabel.preprocesslabel(spectrum)

            with open(out_path, "a", encoding="utf-8") as f:
                for i, title in enumerate(spectrum.LIST_FILE_NAME):
                    f.write("BEGIN IONS\n")
                    f.write("TITLE=" + title + "\n")
                    f.write("CHARGE=" + str(spectrum.LIST_PRECURSOR_CHARGE[i]) + "+\n")
                    f.write("RTINSECONDS=" + str(spectrum.SCAN_RET_TIME) + "\n")
                    f.write("PEPMASS=" + str(spectrum.LIST_PRECURSOR_MOZ[i]) + "\n")
                    f.write("UNFILTERED=" + ",".join([str(x) for x in label.FILTERED]) + "\n")
                    f.write("ISOTOPE_CLUSTER=" + ",".join([str(x) for x in label.CLUSTER]) + "\n")
                    f.write("ISOTOPE_CHARGE=" + ",".join([str(x) for x in label.CHARGE]) + "\n")
                    f.write("\n".join(
                        ["%.5f" % (spectrum.LIST_PEAK_MOZ[ii] + 0.0) + " " + "%.5f" % (spectrum.LIST_PEAK_INT[ii] + 0.0)
                         for ii in range(len(spectrum.LIST_PEAK_MOZ))]) + "\nEND IONS\n")

            # for i in range(len(label.FILTERED)):
            #     print(i, "\t", spectrum.LIST_PEAK_MOZ[i], "\t", spectrum.LIST_PEAK_INT[i], "\t", label.FILTERED[i],
            #           "\t", label.CLUSTER[i], "\t", label.CHARGE[i])

    def subTaskNeuCode(self, idx, path, flag=True):

        raw_name = self.dp.LIST_MGF_NAME_TRY[idx] if flag else self.dp.LIST_MGF_NAME_LYS[idx]
        logToUser(INFO_TO_USER_TaskPrep[1] + "\t[" + raw_name + "]")

        functionLoadPKL = CFunctionLoadPKL(self.dp)
        funcPreproLabel = CFunctionPreprocessForXueLiNeuCode(self.dp)
        # 从硬盘中载入，得到CFileMS2，也就是一整个MS2文件
        dataMS2 = functionLoadPKL.loadMS2PKL(path + ".pkl")
        out_path = self.dp.myCFG.E1_PATH_EXPORT + raw_name + FEATURE_MGF_FILE_SUFFIX

        with open(out_path, "w", encoding="utf-8") as f:
            ...

        for index in dataMS2.INDEX_SCAN[:]:

            # 得到一个新的spectrum，类型是CSpectrum
            spectrum = functionLoadPKL.getSingleMS2Spectrum(dataMS2, index)
            # print(spectrum.LIST_PEAK_MOZ)
            # print(spectrum.LIST_PEAK_INT)
            # 这里是双峰识别的标注，0是低强度峰，1是判定为单峰，2是判定为双峰（双峰中的低质量峰）
            label = funcPreproLabel.preprocesslabel(spectrum)

            with open(out_path, "a", encoding="utf-8") as f:
                for i, title in enumerate(spectrum.LIST_FILE_NAME):
                    f.write("BEGIN IONS\n")
                    f.write("TITLE=" + title + "\n")
                    f.write("CHARGE=" + str(spectrum.LIST_PRECURSOR_CHARGE[i]) + "+\n")
                    f.write("RTINSECONDS=" + str(spectrum.SCAN_RET_TIME) + "\n")
                    f.write("PEPMASS=" + str(spectrum.LIST_PRECURSOR_MOZ[i]) + "\n")
                    f.write("NEUCODELABEL=" + "".join([str(x) for x in label]) + "\n")
                    f.write("\n".join(
                        ["%.5f" % (spectrum.LIST_PEAK_MOZ[ii] + 0.0) + " " + "%.5f" % (spectrum.LIST_PEAK_INT[ii] + 0.0)
                         for ii in range(len(spectrum.LIST_PEAK_MOZ))]) + "\nEND IONS\n")

            # for i in range(len(label.FILTERED)):
            #     print(i, "\t", spectrum.LIST_PEAK_MOZ[i], "\t", spectrum.LIST_PEAK_INT[i], "\t", label.FILTERED[i],
            #           "\t", label.CLUSTER[i], "\t", label.CHARGE[i])


class CTaskValidation:

    def __init__(self, inputDP):
        self.dp = inputDP
        self.tryResPath = self.dp.myCFG.V1_PATH_TRY_PFIND_RES
        self.lysResPath = self.dp.myCFG.V2_PATH_LYS_PFIND_RES
        self.validationPath = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FOLDER_VALIDATION

        self.oriSpecPairName = IO_NAME_FILE_RESULT[0]
        self.oriDeNovoResName = IO_NAME_FILE_RESULT[2]
        self.oriDeNovoCandName = IO_NAME_FILE_RESULT[3]
        self.oriDeNovopNovoMName = IO_NAME_FILE_PNOVOM
        self.oriDeNovoGCNovoName = IO_NAME_FILE_GCNOVO

        self.outSpecPairName = IO_FEATURE_FILE_TP.join(IO_NAME_FILE_RESULT[0].split("."))
        self.outDeNovoResName = IO_FEATURE_FILE_TP.join(IO_NAME_FILE_RESULT[2].split(".")) # now we dont have it!
        self.outDeNovoCandName = IO_FEATURE_FILE_TP.join(IO_NAME_FILE_RESULT[3].split("."))
        self.outDeNovopNovoMName = IO_FEATURE_FILE_TP.join(IO_NAME_FILE_PNOVOM.split("."))
        self.outDeNovoGCNovoName = IO_FEATURE_FILE_TP.join(IO_NAME_FILE_GCNOVO.split("."))

        self.specPairFlag = True if self.dp.myCFG.D0_WORK_FLOW_NUMBER > 1 else False
        self.deNovoFlag = False
        self.DiNovoFlag = False
        self.GCNovoFlag = False
        self.pNovoMFlag = False
        if self.dp.myCFG.D0_WORK_FLOW_NUMBER == 3:
            # must de novo by mirror spectra
            self.deNovoFlag = True

            # 1 4 5
            if (self.dp.myCFG.D9_DE_NOVO_APPROACH == 1) or (self.dp.myCFG.D9_DE_NOVO_APPROACH > 3):
                self.DiNovoFlag = True
            # 2 4
            if (self.dp.myCFG.D9_DE_NOVO_APPROACH == 2) or (self.dp.myCFG.D9_DE_NOVO_APPROACH == 4):
                self.GCNovoFlag = True
            # 3 5
            if (self.dp.myCFG.D9_DE_NOVO_APPROACH == 3) or (self.dp.myCFG.D9_DE_NOVO_APPROACH == 5):
                self.pNovoMFlag = True

    def work(self):

        logToUser("\n[Validation]\n")

        self.__captainGeneValidationPath()

        if self.specPairFlag:
            self.__captainValidateSpecPair()

        if self.deNovoFlag:
            self.__captainValidateDeNovo()
        ...

    def __captainValidateSpecPair(self):

        specPairResPath = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_RESULT[0]  # 0: .res
        specPairDisPath = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_RESULT[1]  # 1: .dis
        out_path = self.validationPath + self.outSpecPairName
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")

        funcValidation = CValidationZixuan(self.dp, self.outSpecPairName)

        for i in range(len(self.dp.LIST_MGF_NAME_TRY)):
            funcValidation.specPairValidation(i)

        funcValidation.reportAll(out_path)

    def __captainValidateDeNovo(self):

        # build_location_function
        # cal_Dinovo_result
        # cal_Pnovom_result

        pFindResPath = self.tryResPath
        location_dict = build_location_function(pFindResPath)
        specPairTPPath = self.validationPath + self.outSpecPairName
        specPairTP = self.__soldierGetSpecPairTP(specPairTPPath)

        if self.DiNovoFlag:
            ori_path = self.dp.myCFG.E1_PATH_EXPORT + self.oriDeNovoCandName
            out_path = self.validationPath + self.outDeNovoCandName
            self.__soldierMove2ValidationFile(ori_path, out_path, specPairTP)
            cal_Dinovo_result(out_path, pFindResPath, location_dict)

        if self.GCNovoFlag:
            ori_path = self.dp.myCFG.E1_PATH_EXPORT + self.oriDeNovoGCNovoName
            out_path = self.validationPath + self.outDeNovoGCNovoName
            self.__soldierMove2ValidationFile(ori_path, out_path, specPairTP)
            logToUser("[ATTENTION] GCNovo Results will not be validated temporarily.\n")
            # cal_Dinovo_result(out_path, pFindResPath, location_dict)

        if self.pNovoMFlag:
            ori_path = self.dp.myCFG.E1_PATH_EXPORT + self.oriDeNovopNovoMName
            out_path = self.validationPath + self.outDeNovopNovoMName
            self.__soldierMove2ValidationFile(ori_path, out_path, specPairTP, flagDiNovo=False)
            cal_Pnovom_result(out_path, pFindResPath, location_dict)
        ...

    def __captainGeneValidationPath(self):

        # 文件路径存在，忽略即可
        if os.access(self.validationPath, os.F_OK):
            pass

        # 文件路径不存在，生成该路径
        else:
            os.makedirs(self.validationPath)


    def __soldierMove2ValidationFile(self, oriFile, targetFile, accTitlePairSet, flagDiNovo=True):

        if flagDiNovo:
            self.__move2ValidationFileDiNovoFormat(oriFile, targetFile, accTitlePairSet)

        # pNovoM format
        else:
            self.__move2ValidationFilepNovoMFormat(oriFile, targetFile, accTitlePairSet)
            ...

    def __soldierGetSpecPairTP(self, inputPath):

        res = set()
        with open(inputPath, "r", encoding="utf-8") as f:
            buff = f.read().split("\n")[1:-1]

        for line in buff:
            item = line.split("\t")
            res.add((item[0], item[1]))

        return res

    def __move2ValidationFileDiNovoFormat(self, oriFile, targetFile, accTitlePairSet):
        with open(oriFile, "r", encoding="utf-8") as fr, open(targetFile, "w", encoding="utf-8") as fw:
            rbuff = fr.read().split("\n")[:-1]
            fw.write("\n".join(rbuff[:2]) + "\n") # title

            flag = False
            rbuff = rbuff[2:]

            for line in rbuff:

                if not line:
                    continue

                # candidate peptide info
                if line[0] == "\t":
                    if flag:
                        fw.write(line + "\n")

                # spec pair info
                else:
                    item = line.split("\t")
                    if (item[0], item[1]) in accTitlePairSet:
                        flag = True
                        fw.write(line + "\n")
                    else:
                        flag = False


    def __move2ValidationFilepNovoMFormat(self, oriFile, targetFile, accTitlePairSet):

        with open(oriFile, "r", encoding="utf-8") as fr, open(targetFile, "w", encoding="utf-8") as fw:
            rbuff = fr.read().split("\n")[:-1]

            flag = False
            start, end = 0, 0
            tp = ("", "")
            for i,line in enumerate(rbuff):

                if not line:
                    continue

                # candidate peptide info
                if line[0] != "=":

                    if flag:
                        fw.write(line + "\n")

                # spec pair info
                elif "@" in line:
                    # "======TRY_TITLE@LYS_TITLE\t..."
                    item = line[6:].split("\t")[0].split("@")
                    if (item[0], item[1]) in accTitlePairSet:
                        flag = True
                        fw.write(rbuff[i - 1] + "\n")  # keep the original stylw
                        fw.write(line + "\n")

                    else:
                        flag = False

        ...

