# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSFunctionIO.py
Time: 2022.09.05 Monday
ATTENTION: none
"""
import pickle
import os
from MSFunctionComposition import CFunctionComposition
from MSEmass import CEmass
from MSTool import toolGetWord, toolStr2List, toolGetWord1, toolCountCharInString
from MSOperator import op_INIT_CFILE_MS2
from MSData import CDataPack, Config, CModInfo, CFileMS2, CMS2Spectrum
from MSLogging import logGetError, logToUser
from MSSystem import PMIDX_FILE_SUFFIX
from MSSystem import IO_NAME_FOLDER_TEMPORARY, LOAD_FILE_FLAG, PREPROCESS_FILE_SUFFIX
import math
import numpy as np

# 20220912 处理mgf文件尝试加速 ---------------
# 把所有的 toolGetWord(line, 1, '=') 行给换了，
# 这里的函数操作虽然，但是每次都执行，我推测：
# 运行会很慢，但是没有去仔细的做对比实验验证。
# 20220912 -----------------------------------
class CFunctionParseMGF:

    def __init__(self, inputDP):
        self.dp = inputDP
        pass

    # 输入是MS2文件也就是mgf文件的路径
    def mgfTOpkl(self, pathMS2):
        # 生成一个pkl文件的路径，到时候直接转存成二进制文件，以实现快速的read和load
        path_pkl = pathMS2 + ".pkl"
        # 看看能不能读取到文件，能读取到说明不用再重新生成了
        if os.access(path_pkl, os.F_OK):
            pass
        # 如果读取不到的话，那咱们就得重新生成一下
        else:
            # init
            dataMS2 = CFileMS2()
            op_INIT_CFILE_MS2(dataMS2)

            # 20220912 站位用的，数据文件读取完毕之后，这个需要删除掉！！！
            dataMS2.INDEX_SCAN = [-1.23]

            # open
            with open(pathMS2, 'r', encoding='utf8') as f:

                # 这个变量看C++中能不能有用，先保留了
                i_MS2 = -1
                # boolean变量
                # 含义：如果是第一次扫描到这个scan，那就把一些信息记录
                # 否则，我就不再重复记录信息了
                first_time_scan = True
                # 遍历每一行的信息，进行数据的分析处理
                for line in f.readlines():
                    len_line = len(line)
                    line = line.strip()  # 主要是为了去掉末尾的"\n"
                    # 只要行不空，那么我就做下面的操作
                    if len_line > 1:

                        first_char = line[0]

                        # 检测是否为TITLE行
                        if (first_char == "T") and line.startswith("TITLE"):
                            i_MS2 = i_MS2 + 1
                            # 这里用了split函数用 . 来分割，生成一个内容的list
                            tmpScan = int(line.split(".")[-4])
                            # TITLE=xxx...x.SCAN.SCAN.CHARGE.RANK.dta
                            # 逆序主要是怕前面的文件名里有.字符，你可以看情况写

                            # 检查一下，如果这个scan出现过了，那就不记录啦，并且标记first_time_scan为False
                            # 20220912 ------------------------------------------
                            # 20220912 列表查找速度太慢了，这里调整为直接末尾判等
                            # 考虑到表空的问题，直接不判表空了，在前面初始化后加入了一个浮点数字
                            # 这样直接判定末尾数值是否相等，避免无效的列表查找
                            if tmpScan == dataMS2.INDEX_SCAN[-1]:
                                # 已经进行了初始化，且记录了二级谱信息
                                dataMS2.MATRIX_FILE_NAME[tmpScan].append(line[6:])
                                first_time_scan = False
                            else:
                                # 第一次记录信息
                                first_time_scan = True
                                dataMS2.INDEX_SCAN.append(tmpScan)
                                # 这里我就不注释了，按照原来的格式重新赋一下值就好
                                dataMS2.MATRIX_PEAK_MOZ[tmpScan] = []  # 初始化，不能省略
                                dataMS2.MATRIX_PEAK_INT[tmpScan] = []
                                dataMS2.MATRIX_FILE_NAME[tmpScan] = [line[6:]]
                                dataMS2.MATRIX_PRECURSOR_MOZ[tmpScan] = []
                                dataMS2.MATRIX_PRECURSOR_CHARGE[tmpScan] = []

                        elif (first_char == "C") and line.startswith("CHARGE"):
                            # t = line.split("=")[1][:-1]  # 此时字符串尾部没有\n，为了去除+号
                            dataMS2.MATRIX_PRECURSOR_CHARGE[tmpScan].append(int(line.split("=")[1][:-1]))

                        elif (first_char == "R") and line.startswith("RTINSECONDS"):
                            # t = line.split("=")[1]
                            dataMS2.LIST_RET_TIME[tmpScan] = float(line.split("=")[1])
                            # dataMS2.LIST_ION_INJECTION_TIME[tmpScan] = float(t)

                        elif (first_char == "P") and line.startswith("PEPMASS"):
                            t = line.split("=")[1]
                            if " " in t:
                                # t = t.split(" ")[0]
                                dataMS2.MATRIX_PRECURSOR_MOZ[tmpScan].append((float(t.split(" ")[0])))
                            else:
                                dataMS2.MATRIX_PRECURSOR_MOZ[tmpScan].append((float(t)))
                            # dataMS2.MATRIX_PRECURSOR_MOZ[tmpScan].append((float(t)))

                        # 48 是 0 的ASCII码， 57 是 9 的ASCII码,49-57可以出现在第一个字符中嗯
                        elif (ord(first_char) > 48) and (ord(first_char) < 58):

                            # 如果是第一次扫描到这个谱图，那么添加，否则跳过
                            if first_time_scan:

                                # 20220912 new code ----------------------------
                                item_list = line.split(" ")
                                dataMS2.MATRIX_PEAK_MOZ[tmpScan].append(float(item_list[0]))
                                dataMS2.MATRIX_PEAK_INT[tmpScan].append(float(item_list[1]))
                                # 20220912  修改，感觉这里调用费时间 -----------
                                # dataMS2.MATRIX_PEAK_MOZ[tmpScan].append(float(toolGetWord(line, 0, ' ')))
                                # dataMS2.MATRIX_PEAK_INT[tmpScan].append(float(toolGetWord(line, 1, ' ')))
                            else:
                                pass

            # 20220912 -----------------------------------
            # 把之前放在SCAN最开头的那个浮点数给它扔掉！
            dataMS2.INDEX_SCAN.pop(0)

            # 现在我们还在else语句下方
            # 这里要做的操作是write pkl，也就是把二进制文件写入到硬盘里！
            fid_pkl = open(path_pkl, 'wb')
            pickle.dump(dataMS2, fid_pkl)
            fid_pkl.close()
            # 写入完毕后关闭文件指针！！

    # 这个不用写，因为没有被调用过
    def getSingleSpectrum(self, inputFileMS2, index):
        # 把单张谱图的信息，单独存成一个CMS2Spectrum，这里我就不赘述了
        outputSpectrum = CMS2Spectrum()
        outputSpectrum.LIST_PRECURSOR_CHARGE = inputFileMS2.MATRIX_PRECURSOR_CHARGE[index]
        outputSpectrum.LIST_PRECURSOR_MOZ = inputFileMS2.MATRIX_PRECURSOR_MOZ[index]
        outputSpectrum.LIST_PEAK_MOZ = inputFileMS2.MATRIX_PEAK_MOZ[index]
        outputSpectrum.LIST_PEAK_INT = inputFileMS2.MATRIX_PEAK_INT[index]
        outputSpectrum.LIST_PRECURSOR_MASS = []  # 我太迷了
        outputSpectrum.SCAN_RET_TIME = inputFileMS2.LIST_RET_TIME[index]
        for i in range(len(outputSpectrum.LIST_PRECURSOR_CHARGE)):
            tmpMass = (outputSpectrum.LIST_PRECURSOR_MOZ[i] - self.dp.myINI.MASS_PROTON_MONO) * \
                      outputSpectrum.LIST_PRECURSOR_CHARGE[i] + self.dp.myINI.MASS_PROTON_MONO
            outputSpectrum.LIST_PRECURSOR_MASS.append(tmpMass)

        return outputSpectrum


class CFunctionLoadPKL:

    def __init__(self, inputDP):
        self.dp = inputDP
        pass

    # 把CFileMS2类型数据，从硬盘里面给load进来（二进制读写就是快）
    def loadMS2PKL(self, pathPKL):
        dataMS2 = CFileMS2()
        pklFile = open(pathPKL, 'rb')
        dataMS2 = pickle.load(pklFile)
        pklFile.close()
        return dataMS2

    def loadSpecListPKL(self, pathPKL):
        res = []
        pklFile = open(pathPKL, "rb")
        res = pickle.load(pklFile)
        pklFile.close()
        return res

    def loadPMIDXPKL(self, pathPKL):
        res = []
        pklFile = open(pathPKL, "rb")
        res = pickle.load(pklFile)
        pklFile.close()
        return res

    # 功能是获取单张Spectrum
    # inputFileMS2是CFileMS2，这里要引用
    # index是整数类型
    def getSingleMS2Spectrum(self, inputFileMS2, index):
        outputSpectrum = CMS2Spectrum()
        outputSpectrum.LIST_FILE_NAME = inputFileMS2.MATRIX_FILE_NAME[index]
        outputSpectrum.LIST_PRECURSOR_CHARGE = inputFileMS2.MATRIX_PRECURSOR_CHARGE[index]
        outputSpectrum.LIST_PRECURSOR_MOZ = inputFileMS2.MATRIX_PRECURSOR_MOZ[index]
        outputSpectrum.LIST_PEAK_MOZ = inputFileMS2.MATRIX_PEAK_MOZ[index]
        outputSpectrum.LIST_PEAK_INT = inputFileMS2.MATRIX_PEAK_INT[index]
        outputSpectrum.LIST_PRECURSOR_MASS = []  # 我太迷了
        outputSpectrum.SCAN_RET_TIME = inputFileMS2.LIST_RET_TIME[index]
        for i in range(len(outputSpectrum.LIST_PRECURSOR_CHARGE)):
            tmpMass = (outputSpectrum.LIST_PRECURSOR_MOZ[i] - self.dp.myINI.MASS_PROTON_MONO) * \
                      outputSpectrum.LIST_PRECURSOR_CHARGE[i] + self.dp.myINI.MASS_PROTON_MONO
            outputSpectrum.LIST_PRECURSOR_MASS.append(tmpMass)
        outputSpectrum.NEUCODE_LABEL = []
        return outputSpectrum



class CFunctionTempPath:

    def __init__(self, inputDP:CDataPack):
        self.dp = inputDP
        self.spec_list_suffix = PREPROCESS_FILE_SUFFIX[1] + ".pkl"
        self.pmidx_suffix = PMIDX_FILE_SUFFIX + ".pkl"
        self.try_specL_prefix = "[TRY-SPECLIST]"
        self.lys_specL_prefix = "[LYS-SPECLIST]"
        # self.pm_idx_prefix    = "[PM-IDX]"
    def geneTempPathSpecList(self, idx, tryFlag):
        # folder path
        res = self.geneTempFolderPath()

        # add the file name
        if tryFlag:
            # res += self.dp.LIST_MGF_NAME_TRY[idx] + self.spec_list_suffix
            res += self.try_specL_prefix + str(idx) + self.spec_list_suffix
        else:
            # res += self.dp.LIST_MGF_NAME_LYS[idx] + self.spec_list_suffix
            res += self.lys_specL_prefix + str(idx) + self.spec_list_suffix

        return res

    def geneTempPathPMIDX(self, i_lys):

        res = self.geneTempFolderPath()

        # res += self.dp.LIST_MGF_NAME_LYS[i_lys] + self.pmidx_suffix
        res += str(i_lys) + self.pmidx_suffix
        return res

    def geneTempPathLoadFlag(self, i_try, i_lys):

        if (i_try < 0) or (i_lys < 0):
            return ""

        res = self.geneTempFolderPath()

        # res += self.dp.LIST_MGF_NAME_TRY[i_try] + LOAD_FILE_FLAG[0]
        res += str(i_try) + LOAD_FILE_FLAG[0]

        # res += self.dp.LIST_MGF_NAME_LYS[i_lys] + LOAD_FILE_FLAG[1]
        res += str(i_lys) + LOAD_FILE_FLAG[1]
        return res

    def geneTempFolderPath(self):

        res = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FOLDER_TEMPORARY

        return res


class CFunctionINI:

    def __init__(self, inputDP):
        self.dp = inputDP

    def file2ini(self):

        self.__captainFile2Element(self.dp.myCFG.I0_INI_PATH_ELEMENT)
        # 20210923 交换 Mod 和 AA 的顺序
        self.__captainFile2Mod(self.dp.myCFG.I2_INI_PATH_MOD)
        # Mod中 增加了对修饰名称的检查，确保名称都是合法的
        self.__captainFile2AA(self.dp.myCFG.I1_INI_PATH_AA)

        self.__captainCalculateModMonoMass()
        self.__captainCalculateAAMonoMass()


    def __captainFile2Mod(self, path):

        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():

                if len(line) > 1:

                    if line[-1] == '\n':
                        line = line[0:-1]  # 把最后一个\n干掉

                    if line.startswith("@"):
                        continue

                    if line.startswith("name"):

                        str_name = toolGetWord1(line, '=', ' ')
                        if int(toolGetWord(line, 1, " ")) == 1:
                            mod_common = False
                        else:
                            mod_common = True

                    else:

                        str_sites = toolGetWord(toolGetWord(line, 1, '='), 0, " ")

                        str_position = toolGetWord(line, 1, ' ')

                        str_mono_mass = toolGetWord(line, 2, ' ')

                        nBlank = toolCountCharInString(line, ' ')

                        if nBlank > 5:

                            str_comp = toolGetWord(line, 7, ' ')

                        else:

                            str_comp = toolGetWord(line, 5, ' ')

                        self.dp.myINI.DICT2_MOD_INFO[str_name] = CModInfo()
                        self.dp.myINI.DICT2_MOD_INFO[str_name].COMP = str_comp
                        self.dp.myINI.DICT2_MOD_INFO[str_name].SITES = str_sites
                        self.dp.myINI.DICT2_MOD_INFO[str_name].POSITION = str_position
                        self.dp.myINI.DICT2_MOD_INFO[str_name].COMMON = mod_common
                        self.dp.myINI.DICT2_MOD_INFO[str_name].MASS = float(str_mono_mass)


        # check the mod name ------------------------------
        for mod in (self.dp.myCFG.A3_FIX_MOD).split("|"):
            if mod == "":
                continue
            elif mod in self.dp.myINI.DICT2_MOD_INFO:
                pass
            else:
                logGetError("please check the mod name: " + mod)
            # for site in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
        # check the mod name ------------------------------

    def __captainFile2AA(self, path):

        with open(path, 'r', encoding='utf8') as f:

            for line in f.readlines():

                if len(line) > 1:
                    str_name = toolGetWord(line, 0, '|')
                    str_comp = toolGetWord(line, 1, '|')

                    self.dp.myINI.DICT1_AA_COM[str_name] = str_comp

        # self.dp.myINI.DICT2_MOD_INFO[str_name].SITES
        # self.dp.myINI.DICT2_MOD_INFO[str_name].POSITION
        # self.dp.myINI.DICT2_MOD_INFO[str_name].MASS

        # 20210318  I与L的质量相同，没必要枚举两遍
        # 这样对tagExtraction而言可以节省时间
        self.dp.myINI.DICT1_AA_COM["I"] = "C(0)"

    def __captainFile2Element(self, path):

        with open(path, 'r', encoding='utf8') as f:

            for line in f.readlines():

                if len(line) > 1:
                    str_name = toolGetWord(line, 0, '|')
                    str_mass = toolGetWord(line, 1, '|')
                    str_abdc = toolGetWord(line, 2, '|')

                    list_mass = toolStr2List(str_mass, ',')
                    list_abdc = toolStr2List(str_abdc, ',')

                    self.dp.myINI.DICT0_ELEMENT_MASS[str_name] = list_mass
                    self.dp.myINI.DICT0_ELEMENT_ABDC[str_name] = list_abdc

    def __captainCalculateAAMonoMass(self):
        funcComp = CFunctionComposition(self.dp)
        emass = CEmass(self.dp)

        for aa in self.dp.myINI.DICT1_AA_COM:
            elemDict = funcComp.getDictComposition(self.dp.myINI.DICT1_AA_COM[aa])
            aa_mass = emass.getCalculatedMonoMZ(elemDict, 0)
            index = ord(aa) - ord("A")  # int
            self.dp.myINI.DICT1_AA_MASS[index] = round(aa_mass, 9)

        # 将 NORMAL位置的 Mod的质量 也加入到AA中去
        # POSITION信息如下
        # NORMAL
        # PRO_N
        # PEP_N
        # PEP_C
        # ...

        # for order in self.dp.myINI.DICT1_AA_MASS:
        #     print(chr(ord("A") + order), self.dp.myINI.DICT1_AA_MASS[order])

        for mod in (self.dp.myCFG.A3_FIX_MOD).split("|"):
            if mod == "":
                continue

            # 20210923
            # 这里比较特殊，暂时只把NORMAL的给计算上，其他的特殊位置先不考虑
            if self.dp.myINI.DICT2_MOD_INFO[mod].POSITION == "NORMAL":
                pass
                add_mod_mass = self.dp.myINI.DICT2_MOD_MASS[mod]
                for modSite in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                    if modSite == "I":
                        if "L" in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                            continue
                        self.dp.myINI.DICT1_AA_MASS[ord("L") - ord("A")] += add_mod_mass
                    elif math.isclose(0, self.dp.myINI.DICT1_AA_MASS[ord(modSite) - ord("A")]):
                        # 特殊或者不存在的AA，不计算这个
                        pass
                    else:
                        self.dp.myINI.DICT1_AA_MASS[ord(modSite) - ord("A")] += add_mod_mass

        # print("===============================")
        # for order in self.dp.myINI.DICT1_AA_MASS:
        #     print(chr(ord("A") + order), self.dp.myINI.DICT1_AA_MASS[order])

    def __captainCalculateModMonoMass(self):
        funcComp = CFunctionComposition(self.dp)
        emass = CEmass(self.dp)

        for mod in self.dp.myINI.DICT2_MOD_INFO:
            elemDict = funcComp.getDictComposition(self.dp.myINI.DICT2_MOD_INFO[mod].COMP)
            try:
                mod_mass = emass.getCalculatedMonoMZ(elemDict, 0)
                self.dp.myINI.DICT2_MOD_INFO[mod].MASS = round(mod_mass, 9)
                # 这里先留着
                self.dp.myINI.DICT2_MOD_MASS[mod] = round(mod_mass, 9)
            except:
                pass
            # 这里因为有一些标记，并未在元素dict中收录，因此不一定会准确
            # 所以能算的就算，不能算的就先用ini文件中的数据了


# 这里用python就好
class CFunctionConfig:

    def config2file(self, path, config):

        with open(path, 'w', encoding="utf-8") as f:
            f.write("# =================================================================" + "\n")
            f.write("# ------------------ <DiNovo Configuration File> ------------------" + "\n")
            f.write("# =================================================================" + "\n")
            f.write("# [Using Method of DiNovo]" + "\n")
            f.write("# 1. Running DiNovo.exe, get the DiNovo.cfg(it's me!)" + "\n")
            f.write("# 2. Filling this config file(you can change the file name as you wish)" + "\n")
            f.write("# 3. Opening a terminal and running command line: DiNovo.exe xxx.cfg" + "\n")
            f.write("#" + "\n")
            f.write("# Any question, please send e-mail to: yfu@amss.ac.cn" + "\n")
            f.write("# Enjoy your time! Good luck!  :-)" + "\n")
            f.write("# =================================================================" + "\n")

            f.write("\n# [INI Files]\n")
            f.write("INI_PATH_ELEMENT=" + config.I0_INI_PATH_ELEMENT + '\n')
            f.write("INI_PATH_AA=" + config.I1_INI_PATH_AA + '\n')
            f.write("INI_PATH_MOD=" + config.I2_INI_PATH_MOD + '\n')
            f.write("# Please fill above parameters with absolute paths." + "\n")

            f.write("\n# [MS Data]\n")
            f.write("MIRROR_PROTEASES_APPROACH=" + str(config.A11_MIRROR_PROTEASES_APPROACH) + "\n")
            f.write("# Supported mirror proteases approaches:" + "\n")
            f.write("# 1. [A]Trypsin + [B]LysargiNase" + "\n")
            f.write("# 2. [A]LysC    + [B]LysN" + "\n")
            f.write("PATH_MGF_A=" + config.A1_PATH_MGF_TRY + '\n')
            f.write("# Please fill above with path of data cleavaged by protease [A]." + "\n")
            f.write("PATH_MGF_B=" + config.A2_PATH_MGF_LYS + '\n')
            f.write("# Please fill above with path of data cleavaged by protease [B]." + "\n")
            f.write("# Please use \"|\" to separate the .mgf files' path, or input the folder path directly." + "\n")
            f.write("# Example1:\n")
            f.write("# PATH_MGF_A=D:\\test\\Trypsin\\T1.mgf|D:\\test\\Trypsin\\T2.mgf" + "\n")
            f.write("# PATH_MGF_B=D:\\test\\LysargiNase\\L1.mgf|D:\\test\\LysargiNase\\L2.mgf" + "\n")
            f.write("# Example2:\n")
            f.write("# PATH_MGF_A=D:\\test\\Trypsin\\" + "\n")
            f.write("# PATH_MGF_B=D:\\test\\LysargiNase\\" + "\n")

            f.write("FIX_MOD=" + str(config.A3_FIX_MOD) + '\n')
            f.write("VAR_MOD=" + str(config.A4_VAR_MOD) + '\n')
            f.write("# Please use \"|\" to separate the modification names." + "\n")
            f.write("# Please read modification.ini and fill like:" + "\n")
            f.write("# FIX_MOD=Carbamidomethyl[C]" + "\n")
            f.write("# VAR_MOD=Oxidation[M]|Deamidated[N]" + "\n")
            f.write("# If you don't want to fill in, just leave a blank, it's OK  :-)" + "\n")
            f.write("LABEL_TYPE=" + str(config.A7_LABEL_TYPE) + "\n")
            f.write("# 0: Unlabeling." + "\n")
            f.write("# [In vivo labeling]" + "\n")
            f.write("# 1: Neucode(Neuron encoding, double peaks)." + "\n")
            f.write("#    Amino acid \"K\" is labeled as K602:+8.0142Da or K080:+8.0502Da, their mass delta is about 0.036Da." + "\n")
            f.write("#    Amino acid \"R\" is labeled as R004:+3.9881Da or R040:+4.0251Da, their mass delta is about 0.037Da." + "\n")
            # f.write("# 2: 12C/13C labeling." + "\n")  # 混合标记，不用管几比几标记，只需考虑氨基酸质量改变的可变修饰
            # f.write("# 3: 14N/15N labeling." + "\n")  # 混合标记，不用管几比几标记，只需考虑氨基酸质量改变的可变修饰
            # f.write("# 4: SILAC labeling... ..." + "\n")
            # f.write("#    Stable Isotope Labeling by Amino Acids in Cell Culture." + "\n")
            # f.write("# [In vitro labeling]" + "\n")
            # f.write("# 5: TMT labeling... ..." + "\n")
            # f.write("#    Tandem Mass Tag." + "\n")
            f.write("# To be continued..." + "\n")
            # 细胞培养氨基酸稳定同位素标记 Stable Isotope Labeling with Amino Acid in Cell Culture(SILAC)

            f.write("\n# [Work Flow]\n")
            f.write("WORK_FLOW_NUMBER=" + str(config.D0_WORK_FLOW_NUMBER) + "\n")
            f.write("# 1: < Basic Processing > Processing .mgf files only." + "\n")
            f.write("# 2: < Spectral Pairing > Processing .mgf files, finding spectral pairs, but do not de novo sequencing." + "\n")
            f.write("# 3: <De Novo Sequencing> Processing .mgf files, finding spectral pairs, and perform de novo sequencing." + "\n")
            # f.write("# 4: <De Novo Sequencing> Processing .mgf files, and de novo sequencing WITHOUT pairing spectra(allow one data path empty)." + "\n")
            f.write("# [ATTENTION] Remember check mass tol in [De Novo], it works in [Spectral Preprocessing] and [Spectral Pairing]." + "\n")
            # f.write("MEMORY_LOAD_MODE=" + str(config.D1_MEMORY_LOAD_MODE) + "\n")
            # f.write("# For spectral pairing:" + "\n")
            # f.write("# 1: [less-load mode] Load two data files into memory every time(choose only if mode 2 cannot run successfully)." + "\n")
            # f.write("# 2: [semi-load mode] Load half data files into memory, default: load all LysargiNase data files." + "\n")
            # f.write("# 3: [full-load mode] Load total data files into memory once at the beginning." + "\n")
            # f.write("# Time complexity of IO operation are O(a * b), O(a + b) and O(a + b) respectively where a, b are .mgf number." + "\n")
            f.write("MULTIPROCESS_NUM=" + str(config.D2_MULTIPROCESS_NUM) + '\n')
            f.write("# Accelerating! But the memory cost will be higher as the number gets higher." + "\n")

            f.write("\n# [Spectral Preprocessing]\n")
            f.write("# [NeuCode Peaks Detection] Detecting labeling peaks while samples are labeled with NeuCode." + "\n")
            f.write("NEUCODE_PEAK_DETECTION_APPROACH=" + str(config.B9_NEUCODE_DETECT_APPROACH) + "\n")
            f.write("# 1: Default detection approach with deisotope and without classification model, exporting within preprocessed .mgf." + "\n")
            f.write("# 2: Detection with classification model(without deisotope, parameters above expire), exporting with \"[NeuCodeAnnoOnly]\" .mgf." + "\n")
            f.write("NEUCODE_PEAK_INTENSITY_MAX_RATIO=" + str(config.A10_NEUCODE_PEAK_MAX_RATIO) + "\n")
            f.write("# Only work for approach 1. Filling the value of ratio >= 1.0." + "\n")
            f.write("CLASSIFICATION_MODEL_PATH=" + str(config.B10_CLASSIFICATION_MODEL_PATH) + "\n")
            f.write("# The path of classification model for NeuCode peaks detection." + "\n")
            f.write("# [ATTENTION] Detection with classification model only for workflow 1 now." + "\n")
            f.write("# [Deisotope] Hold 350 peaks at most before deisotoping(default).\n")
            f.write("ADD_ISOTOPE_INTENSITY=" + str(config.B1_ADD_ISOTOPE_INTENSITY) + "\n")
            f.write("CHECK_NATURAL_LOSS=" +  str(config.B2_CHECK_NATURAL_LOSS) + "\n")
            f.write("REMOVE_PRECURSOR_ION=" + str(config.B3_REMOVE_PRECURSOR_ION) + "\n")
            f.write("CHECK_PRECURSOR_CHARGE=" + str(config.B4_CHECK_PRECURSOR_CHARGE) + "\n")
            f.write("# Parameters above: 0 means turning off, and 1 means turning on." + "\n")
            f.write("# [Round I] Global filter for low intensity peaks.\n")
            f.write("HOLD_PEAK_NUM=" + str(config.B5_ROUND_ONE_HOLD_PEAK_NUM) + "\n")
            f.write("# [Round II] Local filter for low intensity peaks.\n")
            f.write("REMOVE_IMMONIUM=" + str(config.B6_ROUND_TWO_REMOVE_IMMONIUM) + "\n")
            f.write("PEAK_NUM_PER_BIN=" + str(config.B7_ROUND_TWO_PEAK_NUM_PER_BIN) + "\n")
            f.write("MASS_BIN_LENGTH=" + str(config.B8_ROUND_TWO_MASS_BIN_LENGTH) + "\n")
            f.write("# REMOVE_IMMONIUM: 0 means turning off, and 1 means turning on." + "\n")
            f.write("# Default: Remove immonium ions, and hold 4 peaks per 100Da at most." + "\n")

            f.write("\n# [Spectral Pairing]\n")
            f.write("PAIR_DELTA_RT_THRESHOLD=" + str(config.C0_MIRROR_DELTA_RT) + "\n")
            f.write("# Threshold of delta RT for judging mirror spectral pair, default: 900.0s." + "\n")
            f.write("PAIR_FILTER_APPROACH=" + str(config.C10_PAIR_FILTER_APPROACH) + "\n")
            f.write("# Spectral pair filter approach:" + "\n")
            f.write("# 0: Direct judge." + "\n")
            f.write("# 1: Filter using p-value." + "\n")
            f.write("# 2: Filter using FDR (by simulated null distribution)" + "\n")
            f.write("# 3: Filter using p-value and then using FDR (by simulated null distribution)" + "\n")
            f.write("PAIR_P_VALUE_THRESHOLD=" + str(config.C11_PAIR_P_VALUE_THRESHOLD) + "\n")
            f.write("# Threshold when using p-Value, default: 0.05." + "\n")
            f.write("PAIR_DECOY_APPROACH=" + str(config.C12_PAIR_DECOY_APPROACH) + "\n")
            f.write("# Test mode for decoy generation:" + "\n")
            f.write("# 0: No decoy generated there." + "\n")
            f.write("# 1: Shifted precursor mass with +X Da." + "\n")
            f.write("# 2: Shifted fragment delta with +X Da, within target-decoy competition." + "\n")
            f.write("# 3: Shifted preprocessed peaks with +X Da." + "\n")
            f.write("PAIR_DECOY_SHIFTED_IN_DA=" + str(config.C13_PAIR_DECOY_SHIFTED_IN_DA) + "\n")
            f.write("# X = 15.0 by default." + "\n")
            f.write("PAIR_FDR_THRESHOLD=" + str(config.C14_PAIR_FDR_THRESHOLD) + "\n")
            f.write("# Threshold when using FDR, default: 2%." + "\n")
            f.write("# Estimated FDR := (#D + 1) / max(#T, 1)." + "\n")

            f.write("MIRROR_TYPE_A1=" + str(config.C1_MIRROR_TYPE_A1) + "\n")
            f.write("MIRROR_TYPE_A2=" + str(config.C2_MIRROR_TYPE_A2) + "\n")
            f.write("MIRROR_TYPE_A3=" + str(config.C3_MIRROR_TYPE_A3) + "\n")
            f.write("MIRROR_TYPE_B=" + str(config.C4_MIRROR_TYPE_B) + "\n")
            f.write("MIRROR_TYPE_C=" + str(config.C5_MIRROR_TYPE_C) + "\n")
            f.write("MIRROR_TYPE_D=" + str(config.C6_MIRROR_TYPE_D) + "\n")
            f.write("MIRROR_TYPE_E=" + str(config.C7_MIRROR_TYPE_E) + "\n")
            f.write("MIRROR_TYPE_F=" + str(config.C8_MIRROR_TYPE_F) + "\n")
            f.write("MIRROR_TYPE_G=" + str(config.C9_MIRROR_TYPE_G) + "\n")
            f.write("# 0: Do not consider this mirror type." + "\n")
            f.write("# 1: Consider this mirror type." + "\n")
            f.write("# [MIRROR TYPE INTRODUCTION]([A]-protease - [B]-protease)" + "\n")
            f.write("# TYPE A1: xxxxK - Kxxxx    E(delta(P.M.)) = (K - K) =    0.00Da, E(delta(FragIons_N|C)) = -K|+K = -128.09Da|+128.09Da" + "\n")
            f.write("# TYPE A2: xxxxR - Rxxxx    E(delta(P.M.)) = (R - R) =    0.00Da, E(delta(FragIons_N|C)) = -R|+R = -156.10Da|+156.10Da" + "\n")
            f.write("# TYPE A3: xxxx  -  xxxx    E(delta(P.M.)) = (0 - 0) =    0.00Da, E(delta(FragIons_N|C)) =  0|0  =    0.00Da|   0.00Da" + "\n")
            f.write("# TYPE B:  xxxxR - Kxxxx    E(delta(P.M.)) = (R - K) =  +28.01Da, E(delta(FragIons_N|C)) = -K|+R = -128.09Da|+156.10Da" + "\n")
            f.write("# TYPE C:  xxxxK - Rxxxx    E(delta(P.M.)) = (K - R) =  -28.01Da, E(delta(FragIons_N|C)) = -R|+K = -156.10Da|+128.09Da" + "\n")
            f.write("# TYPE D:  xxxxK -  xxxx    E(delta(P.M.)) = (K - 0) = +128.09Da, E(delta(FragIons_N|C)) =  0|+K =    0.00Da|+128.09Da" + "\n")
            f.write("# TYPE E:  xxxxR -  xxxx    E(delta(P.M.)) = (R - 0) = +156.10Da, E(delta(FragIons_N|C)) =  0|+R =    0.00Da|+156.10Da" + "\n")
            f.write("# TYPE F:  xxxx  - Kxxxx    E(delta(P.M.)) = (0 - K) = -128.09Da, E(delta(FragIons_N|C)) = -K|0  = -128.09Da|   0.00Da" + "\n")
            f.write("# TYPE G:  xxxx  - Rxxxx    E(delta(P.M.)) = (0 - R) = -156.10Da, E(delta(FragIons_N|C)) = -R|0  = -156.10Da|   0.00Da" + "\n")
            f.write("# P.M.: Precursor Mass(Normalized to monovalent state, i.e. peptide mass add single proton mass)." + "\n")
            f.write("# FragIons_N|C: N-terminal fragment ions(like a, b, c) | C-terminal fragment ions(like x, y, z)." + "\n")
            f.write("# Introduction above is based on unlabeled sample." + "\n")
            f.write("# If neuron-encoded sample: Mass(K) should add 8Da, and Mass(R) should add 4Da approximately." + "\n")
            f.write("# The accurate masses of amino acids will be calculated by element composition in DiNovo automatically." + "\n")

            # 离子价态的英文描述（自己查的，咱也不知道对不对嗷）
            # 一价离子  monovalent ion
            # 二价离子    divalent ion
            # 三价离子   trivalent ion
            # 四价离子 tetravalent ion
            # 五价离子 pentavalent ion
            # 六价离子  hexavalent ion
            # 七价离子 heptavalent ion
            # 八价离子  octavalent ion


            f.write("\n# [De Novo]\n")
            f.write("DE_NOVO_APPROACH=" + str(config.D9_DE_NOVO_APPROACH) + "\n")
            f.write("# 0: [Direct mode] Direct-reading de Novo sequencing." + "\n")
            f.write("# 1: [Mirror Novo] De Novo sequencing by MirrorNovo." + "\n")
            f.write("# 2: [  pNovoM2  ] De Novo sequencing by pNovoM2." + "\n")
            f.write("# 3: [Combination] De Novo sequencing by MirrorNovo & pNovoM.exe." + "\n")
            f.write("MIRROR_NOVO_MODEL_PATH=" + config.D10_MIRROR_NOVO_MODEL_PATH + "\n")
            f.write("PNOVOM_EXE_PATH=" + config.D11_PNOVOM_EXE_PATH + "\n")
            f.write("# The path of MirrorNovo / pNovoM.exe for de Novo sequencing." + "\n")
            f.write("# [ATTENTION] if DE_NOVO_APPROACH=1/3, MIRROR_NOVO_MODEL_PATH need to fill path of MirrorNovo.py." + "\n")
            f.write("# [ATTENTION] if DE_NOVO_APPROACH=2/3, PNOVOM_EXE_PATH need to fill path of pNovoM2.exe." + "\n")
            f.write("COMBINE_RESULT=1" + "\n")
            f.write("# [ATTENTION] Default 1: combining two sequencing results when DE_NOVO_APPROACH=3(To be implemented)." + "\n")
            f.write("MIN_PRECURSOR_MASS=" + str(config.A5_MIN_PRECURSOR_MASS) + "\n")
            f.write("MAX_PRECURSOR_MASS=" + str(config.A6_MAX_PRECURSOR_MASS) + "\n")
            f.write("# Default precursor mass range: [300.0Da, 3500.0Da]." + "\n")
            f.write("REPORT_PEPTIDE_NUM=" + str(config.D8_REPORT_PEPTIDE_NUM) + '\n')
            f.write("# The max reporting number of most possible peptides inferred per spectrum/spectral pair." + "\n")
            f.write("MS_TOL=" + str(config.D3_MS_TOL) + '\n')
            f.write("MS_TOL_PPM=" + str(config.D4_MS_TOL_PPM) + '\n')
            f.write("# Precursor mass tolerance is used in spectral pairing(about twice tolerance) and de novo sequencing." + "\n")
            f.write("MSMS_TOL=" + str(config.D5_MSMS_TOL) + '\n')
            f.write("MSMS_TOL_PPM=" + str(config.D6_MSMS_TOL_PPM) + '\n')
            f.write("# MS\\MSMS_TOL_PPM: 0 for Da, and 1 for PPM." + '\n')
            f.write("# Eg. MSMS_TOL=20, MSMS_TOL_PPM=1, it means that fragment mass tolerance is setting as 20 PPM." + '\n')
            f.write("# Fragment mass tolerance is used for scoring in spectral pairing(about twice tolerance) and de novo sequencing." + "\n")
            f.write("DE_NOVO_SINGLE_ENZYME_SPECTRUM=0" + "\n")
            f.write("# 1: De novo sequencing for NeuCode-labelling spectra using pNovoM2." + "\n")
            f.write("PNOVOM2_MODE=1" + "\n")
            f.write("# [ATTENTION] Special function of pNovoM2(To be implemented)." + "\n")
            f.write("BATCH_SIZE=2" + "\n")
            f.write("# [ATTENTION] Accelation for MirrorNovo. Setting it 2 to 10 needs about 4 to 10 GB GPU memory." + "\n")


            f.write("\n# [Export]\n")
            f.write("PATH_EXPORT=" + config.E1_PATH_EXPORT + '\n')
            f.write("# PATH_EXPORT is a path of result folder(Non-existent folder will be generated automatically)." + "\n")
            f.write("# [ATTENTION] If it's an existing folder path, all items in it will be cleared." + "\n")
            f.write("EXPORT_ROUND_ONE_MGF=" + str(config.E2_EXPORT_ROUND_ONE_MGF) + "\n")
            f.write("# Do you want to export 1st-round preprocessed spectra(MGF format)?" + "\n")
            f.write("EXPORT_ROUND_TWO_MGF=" + str(config.E3_EXPORT_ROUND_TWO_MGF) + "\n")
            f.write("# Do you want to export 2nd-round preprocessed spectra(MGF format)?" + "\n")
            # f.write("EXPORT_SPECTRAL_PAIR=" + str(config.E4_EXPORT_SPECTRAL_PAIR) + "\n")
            # f.write("# Do you want to export mirror-spectral pairing results?" + "\n")
            # f.write("COMBINE_SPLIT_RESULT=" + str(config.E5_COMBINE_SPLIT_RESULT) + "\n")
            # f.write("# Combine the de novo results separately." + "\n")
            # f.write("COMBINE_TOTAL_RESULT=" + str(config.E6_COMBINE_TOTAL_RESULT) + "\n")
            # f.write("# Combine ALL de novo results in the end." + "\n")
            # f.write("# Parameters above(except PATH_EXPORT): 0 means turning off, and 1 means turning on." + "\n")
            f.write("EXPORT_FEATURE_MGF=" + str(config.E7_EXPORT_FEATURE_MGF) + "\n")
            f.write("# Do you want to export annotated-and-unpreprocessed spectra(MGF format)?" + "\n")
            f.write("# Annotate Unlabeled: Top-350 or not, isotope cluster, charge state." + "\n")
            f.write("# Annotate NeuCoding: Top-600 or not, NeuCode-labeling double peaks." + "\n")


            f.write("\n# [Validation]\n")
            f.write("FLAG_DO_VALIDATION=" + str(config.V0_FLAG_DO_VALIDATION) + "\n")
            f.write("# 0: [User mode] DiNovo just run properly and nothing more." + "\n")
            f.write("# 1: [Test mode] After finished, results will be validated." + "\n")
            f.write("# 2: [Alignment] After finished, results will be mapped with database." + "\n")
            f.write("PATH_A_PFIND_RES=" + str(config.V1_PATH_TRY_PFIND_RES) + "\n")
            f.write("PATH_B_PFIND_RES=" + str(config.V2_PATH_LYS_PFIND_RES) + "\n")
            f.write("# If FLAG_DO_VALIDATION=1, please fill result paths above(only accept pFind3 result format, pFind-filtered.spectra)." + "\n")
            f.write("# A: result path of [A]-protease data." + "\n")
            f.write("# B: result path of [B]-protease data." + "\n")
            f.write("PATH_FASTA_FILE=" + str(config.V3_PATH_FASTA_FILE) + "\n")
            f.write("# If FLAG_DO_VALIDATION=2, please fill fasta path above." + "\n")

    def file2config(self, path, config):

        config.A0_PATH_CFG_FILE = path

        with open(path, 'r', encoding="utf-8") as f:

            for line in f.readlines():

                if line.startswith("#"):
                    continue

                p_EqualSign = line.find('=')

                if -1 == p_EqualSign:
                    pass
                else:

                    subLine = toolGetWord(line, 0, ';')  # ;后面的是注释
                    self.__soldierParseLine(subLine, config)


    def __soldierParseLine(self, line, cfg: Config):

        str_name = toolGetWord(line, 0, '=')
        str_value = toolGetWord(line, 1, '=').replace("\n", "")


        # [INI FILE INFO]
        # -----------------------------------
        if "INI_PATH_ELEMENT" == str_name:
            cfg.I0_INI_PATH_ELEMENT = str_value

        elif "INI_PATH_AA" == str_name:
            cfg.I1_INI_PATH_AA = str_value

        elif "INI_PATH_MOD" == str_name:
            cfg.I2_INI_PATH_MOD = str_value
        # -----------------------------------

        # [DATA PARAM]
        # -----------------------------------

        elif "MIRROR_PROTEASES_APPROACH" == str_name:
            cfg.A11_MIRROR_PROTEASES_APPROACH = int(str_value)

        elif "PATH_MGF_A" == str_name:
            cfg.A1_PATH_MGF_TRY = str_value.replace("/", "\\")

        elif "PATH_MGF_B" == str_name:
            cfg.A2_PATH_MGF_LYS = str_value.replace("/", "\\")

        elif "PATH_MGF_TRY" == str_name:
            logGetError("\n[Error] Please use the newest .cfg file, eg.just run \"DiNovo.exe\" in command line window to get it.")

        elif "PATH_MGF_LYS" == str_name:
            logGetError("\n[Error] Please use the newest .cfg file, eg.just run \"DiNovo.exe\" in command line window to get it.")

        elif "FIX_MOD" == str_name:
            cfg.A3_FIX_MOD = str_value

        elif "VAR_MOD" == str_name:
            cfg.A4_VAR_MOD = str_value

        elif "LABEL_TYPE" == str_name:
            cfg.A7_LABEL_TYPE = int(str_value)

        elif "NEUCODE_PEAK_INTENSITY_MAX_RATIO" == str_name:
            cfg.A10_NEUCODE_PEAK_MAX_RATIO = float(str_value)
        # -----------------------------------

        # [WORK FLOW]
        # -----------------------------------
        elif "WORK_FLOW_NUMBER" == str_name:
            cfg.D0_WORK_FLOW_NUMBER = int(str_value)

        elif "MEMORY_LOAD_MODE" == str_name:
            cfg.D1_MEMORY_LOAD_MODE = int(str_value)

        elif "MULTIPROCESS_NUM" == str_name:
            cfg.D2_MULTIPROCESS_NUM = int(str_value)
        # -----------------------------------


        # [PREPROCESS SETTING]
        # -----------------------------------
        # [Deisotope]
        elif "NEUCODE_PEAK_DETECTION_APPROACH" == str_name:
            cfg.B9_NEUCODE_DETECT_APPROACH = int(str_value)

        elif "CLASSIFICATION_MODEL_PATH" == str_name:
            cfg.B10_CLASSIFICATION_MODEL_PATH = str_value

        elif "ADD_ISOTOPE_INTENSITY" == str_name:
            cfg.B1_ADD_ISOTOPE_INTENSITY = int(str_value)

        elif "CHECK_NATURAL_LOSS" == str_name:
            cfg.B2_CHECK_NATURAL_LOSS = int(str_value)

        elif "REMOVE_PRECURSOR_ION" == str_name:
            cfg.B3_REMOVE_PRECURSOR_ION = int(str_value)

        elif "CHECK_PRECURSOR_CHARGE" == str_name:
            cfg.B4_CHECK_PRECURSOR_CHARGE = int(str_value)

        # [Round I Filter]
        elif "HOLD_PEAK_NUM" == str_name:
            cfg.B5_ROUND_ONE_HOLD_PEAK_NUM = int(str_value)

        # [Round II Filter]
        elif "REMOVE_IMMONIUM" == str_name:
            cfg.B6_ROUND_TWO_REMOVE_IMMONIUM = int(str_value)

        elif "PEAK_NUM_PER_BIN" == str_name:
            cfg.B7_ROUND_TWO_PEAK_NUM_PER_BIN = int(str_value)

        elif "MASS_BIN_LENGTH" == str_name:
            cfg.B8_ROUND_TWO_MASS_BIN_LENGTH = int(str_value)
        # -----------------------------------

        # [Spectral Pair]
        elif "PAIR_DELTA_RT_THRESHOLD" == str_name:
            cfg.C0_MIRROR_DELTA_RT = abs(float(str_value))

        elif "PAIR_FILTER_APPROACH" == str_name:
            cfg.C10_PAIR_FILTER_APPROACH = int(str_value)

        elif "PAIR_P_VALUE_THRESHOLD" == str_name:
            if str_value:  # not null string
                cfg.C11_PAIR_P_VALUE_THRESHOLD = float(str_value)

        elif "PAIR_DECOY_APPROACH" == str_name:
            cfg.C12_PAIR_DECOY_APPROACH = int(str_value)

        elif "PAIR_DECOY_SHIFTED_IN_DA" == str_name:
            cfg.C13_PAIR_DECOY_SHIFTED_IN_DA = float(str_value)

        elif "PAIR_FDR_THRESHOLD" == str_name:
            if str_value:  # not null string
                cfg.C14_PAIR_FDR_THRESHOLD = float(str_value)

        elif "MIRROR_TYPE_A1" == str_name:
            cfg.C1_MIRROR_TYPE_A1 = int(str_value)

        elif "MIRROR_TYPE_A2" == str_name:
            cfg.C2_MIRROR_TYPE_A2 = int(str_value)

        elif "MIRROR_TYPE_A3" == str_name:
            cfg.C3_MIRROR_TYPE_A3 = int(str_value)

        elif "MIRROR_TYPE_B" == str_name:
            cfg.C4_MIRROR_TYPE_B = int(str_value)

        elif "MIRROR_TYPE_C" == str_name:
            cfg.C5_MIRROR_TYPE_C = int(str_value)

        elif "MIRROR_TYPE_D" == str_name:
            cfg.C6_MIRROR_TYPE_D = int(str_value)

        elif "MIRROR_TYPE_E" == str_name:
            cfg.C7_MIRROR_TYPE_E = int(str_value)

        elif "MIRROR_TYPE_F" == str_name:
            cfg.C8_MIRROR_TYPE_F = int(str_value)

        elif "MIRROR_TYPE_G" == str_name:
            cfg.C9_MIRROR_TYPE_G = int(str_value)
        # -----------------------------------

        # [DE NOVO SETTING]
        # -----------------------------------

        elif "DE_NOVO_APPROACH" == str_name:
            cfg.D9_DE_NOVO_APPROACH = int(str_value)

        elif "MIRROR_NOVO_MODEL_PATH" == str_name:
            cfg.D10_MIRROR_NOVO_MODEL_PATH = str_value

        elif "PNOVOM_EXE_PATH" == str_name:
            cfg.D11_PNOVOM_EXE_PATH = str_value

        elif "MIN_PRECURSOR_MASS" == str_name:
            cfg.A5_MIN_PRECURSOR_MASS = float(str_value)

        elif "MAX_PRECURSOR_MASS" == str_name:
            cfg.A6_MAX_PRECURSOR_MASS = float(str_value)

        elif "REPORT_PEPTIDE_NUM" == str_name:
            cfg.D8_REPORT_PEPTIDE_NUM = int(str_value)

        elif "MS_TOL" == str_name:
            cfg.D3_MS_TOL = float(str_value)

        elif "MS_TOL_PPM" == str_name:
            cfg.D4_MS_TOL_PPM = int(str_value)

        elif "MSMS_TOL" == str_name:
            cfg.D5_MSMS_TOL = float(str_value)

        elif "MSMS_TOL_PPM" == str_name:
            cfg.D6_MSMS_TOL_PPM = int(str_value)

        elif "COMBINE_RESULT" == str_name:
            ...
        elif "DE_NOVO_SINGLE_SPECTRUM" ==str_name:
            cfg.D12_DE_NOVO_SINGLE_SPECTRUM = int(str_value)

        elif "DE_NOVO_SINGLE_SPECTRA" ==str_name:
            cfg.D12_DE_NOVO_SINGLE_SPECTRUM = int(str_value)

        elif "DE_NOVO_SINGLE_ENZYME_SPECTRUM" == str_name:
            cfg.D12_DE_NOVO_SINGLE_SPECTRUM = int(str_value)

        elif "PNOVOM2_MODE" == str_name:
            ...

        elif "BATCH_SIZE" == str_name:
            if str_value:  # not null string
                cfg.D13_BATCH_SIZE = int(str_value)
        # -----------------------------------

        # [EXPORT SETTING]
        # -----------------------------------
        elif "PATH_EXPORT" == str_name:
            cfg.E1_PATH_EXPORT = os.path.abspath(str_value)
            cfg.E1_PATH_EXPORT = cfg.E1_PATH_EXPORT.replace("/", "\\")
            if cfg.E1_PATH_EXPORT[-1] != "\\":
                cfg.E1_PATH_EXPORT += "\\"

        elif "EXPORT_ROUND_ONE_MGF" == str_name:
            cfg.E2_EXPORT_ROUND_ONE_MGF = int(str_value)

        elif "EXPORT_ROUND_TWO_MGF" == str_name:
            cfg.E3_EXPORT_ROUND_TWO_MGF = int(str_value)

        elif "EXPORT_SPECTRAL_PAIR" == str_name:
            cfg.E4_EXPORT_SPECTRAL_PAIR = int(str_value)

        elif "COMBINE_SPLIT_RESULT" == str_name:
            cfg.E5_COMBINE_SPLIT_RESULT = int(str_value)

        elif "COMBINE_TOTAL_RESULT" == str_name:
            cfg.E6_COMBINE_TOTAL_RESULT = int(str_value)

        elif "EXPORT_FEATURE_MGF" == str_name:
            cfg.E7_EXPORT_FEATURE_MGF = int(str_value)

        # [VALIDATION]
        elif "FLAG_DO_VALIDATION" == str_name:
            cfg.V0_FLAG_DO_VALIDATION = int(str_value)

        elif "PATH_A_PFIND_RES" == str_name:
            cfg.V1_PATH_TRY_PFIND_RES = str_value

        elif "PATH_B_PFIND_RES" == str_name:
            cfg.V2_PATH_LYS_PFIND_RES = str_value

        elif "PATH_FASTA_FILE" == str_name:
            cfg.V3_PATH_FASTA_FILE = str_value
        # -----------------------------------

        # some param that we hasn't seen before
        else:

            info = "MSFunction, file2config, " + str_name + " is all right?"
            logGetError(info)
        ...


class CFunctionTransFormat:

    def __init__(self, inputDP):

        self.dp = inputDP

        self.max_rank = self.dp.myCFG.D8_REPORT_PEPTIDE_NUM
        self.MIRROR_TITLE_1 = ["A_TITLE", "B_TITLE", "MIRROR_TYPE"]
        self.MIRROR_TITLE_2 = ["CAND_RANK", "SEQUENCE", "MODIFICATIONS", "PEPTIDE_SCORE", "AA_SCORE"]
        self.neucode_flag = self.dp.myCFG.A7_LABEL_TYPE
        self.try_lys_digest = (1 == self.dp.myCFG.A11_MIRROR_PROTEASES_APPROACH)

    def trans_pNovoM2_mirrorSeq(self, inputPath, outputPath, deleteFlag=True):

        with open(inputPath, "r") as fi, open(outputPath, "w") as fw:

            fw.write("\t".join(self.MIRROR_TITLE_1) + "\n")
            fw.write("\t" + "\t".join(self.MIRROR_TITLE_2[:-1]) + "\n")
            rank = 1

            for line in fi:

                if len(line) < 2:
                    continue

                if line[:2] == "S\t":

                    line_item = line.split("\t")

                    a_t, b_t = line_item[1].split("@")
                    m_t = line_item[2]

                    fw.write("\t".join([a_t, b_t, m_t]) + "\n")

                    rank = 1

                else:
                    if rank > self.max_rank:
                        continue

                    line_item = line.split("\t")
                    mod_str = ""

                    for i, x in enumerate(line_item[0]):

                        if x == "J":
                            mod_str += str(i + 1) + ",Oxidation[M];"

                        elif x == "C":
                            mod_str += str(i + 1) + ",Carbamidomethyl[C];"

                        elif self.neucode_flag == 1 and x == "K":
                            mod_str += str(i + 1) + ",NeuCodeK602[K];"

                        elif self.neucode_flag == 1 and self.try_lys_digest and x == "R":
                            mod_str += str(i + 1) + ",NeuCodeR004[R];"

                    seq = line_item[0].replace("J", "M").replace("I", "L")

                    score = line_item[1]

                    fw.write("\t" + "\t".join([str(rank), seq, mod_str, score]) + "\n")
                    rank += 1

        # delete file
        self.__deleteFile(inputPath, deleteFlag)

    def trans_pNovoM2_singleSeq(self, inputPath, outputPath, deleteFlag=True):

        # title_i = 0 if A_DATA else 1

        with open(inputPath, "r") as fi, open(outputPath, "w") as fw:

            fw.write("TITLE\n")
            fw.write("\t" + "\t".join(self.MIRROR_TITLE_2[:-1]) + "\n")
            rank = 1

            for line in fi:

                if len(line) < 2:
                    continue

                if line[:2] == "S\t":

                    line_item = line.split("\t")

                    single_title = line_item[1].split("@")[0]
                    # m_t = line_item[2]

                    fw.write(single_title + "\n")

                    rank = 1

                else:
                    if rank > self.max_rank:
                        continue

                    line_item = line.split("\t")
                    mod_str = ""

                    for i, x in enumerate(line_item[0]):

                        if x == "J":
                            mod_str += str(i + 1) + ",Oxidation[M];"

                        elif x == "C":
                            mod_str += str(i + 1) + ",Carbamidomethyl[C];"

                        elif self.neucode_flag == 1 and x == "K":
                            mod_str += str(i + 1) + ",NeuCodeK602[K];"

                        elif self.neucode_flag == 1 and self.try_lys_digest and x == "R":
                            mod_str += str(i + 1) + ",NeuCodeR004[R];"

                    seq = line_item[0].replace("J", "M").replace("I", "L")

                    score = line_item[1]

                    fw.write("\t" + "\t".join([str(rank), seq, mod_str, score]) + "\n")
                    rank += 1

        # delete file
        self.__deleteFile(inputPath, deleteFlag)

    def trans_mirrorNovo_mirrorSeq(self, inputPath, outputPath, deleteFlag=True):

        cnt = 0
        with open(inputPath, "r") as fi, open(outputPath, "w") as fw:

            for line in fi:

                cnt += 1
                if line and line[0] == "\t":
                    if cnt < 3:
                        ...
                    elif int(line.split("\t")[1]) > self.max_rank:
                        continue

                fw.write(line)

        # os.rename(inputPath, outputPath)
        self.__deleteFile(inputPath, deleteFlag)
        # delete file
        # self.__deleteFile(inputPath, deleteFlag)

    def trans_mirrorNovo_singleSeq(self, inputPath, outputPath, deleteFlag=True):

        cnt = 0
        with open(inputPath, "r") as fi, open(outputPath, "w") as fw:

            for line in fi:

                cnt += 1
                if line and line[0] == "\t":
                    if cnt < 3:
                        ...
                    elif int(line.split("\t")[1]) > self.max_rank:
                        continue

                fw.write(line)

        # os.rename(inputPath, outputPath)
        # delete file
        self.__deleteFile(inputPath, deleteFlag)

    def __deleteFile(self, filePath, deleteFlag=True):

        if deleteFlag:
            try:
                os.remove(filePath)
            except:
                logToUser("[Warning] Cannot delete file: " + filePath)


class CFunctionWritePrepMGF:

    def __init__(self, inputDP):
        self.dp = inputDP

        if self.dp.myCFG.A7_LABEL_TYPE != 1:
            self.write = self.writeNormal
        # self.dp.myCFG.A8_NEUCODE_OUTPUT_TYPE == 1 and using default approach to detect neucode peaks
        elif self.dp.myCFG.B9_NEUCODE_DETECT_APPROACH == 1:
            self.write = self.writeNeuCodeType1
        # detect neucode peaks using xinming model
        else:
            self.write = self.writeNeuCodeType3

        # 先不考虑其他问题，只看无漏切的谱图吧，不然记录信息太过于复杂了
        self.neucode_gap = self.dp.myCFG.A9_DOUBLE_PEAK_GAP

    def writeNormal(self, inputPath, inputList):

        with open(inputPath, "a", encoding="utf-8") as f:
            for tmpSpec in inputList:
                # tmpSpec = inputList[i]
                f.write("BEGIN IONS\nTITLE=" + tmpSpec.LIST_FILE_NAME[0] + "\n")
                f.write("CHARGE=" + str(tmpSpec.LIST_PRECURSOR_CHARGE[0]) + "+\n")
                f.write("RTINSECONDS=" + str(tmpSpec.SCAN_RET_TIME) + "\n")
                f.write("PEPMASS=" + "%.6f" % (tmpSpec.LIST_PRECURSOR_MOZ[0] +0.0) + "\n")
                if tmpSpec.LIST_PRECURSOR_MOZ:
                    f.write("\n".join(["%.5f" % (tmpSpec.LIST_PEAK_MOZ[ii] + 0.0) + " " + "%.5f" % (tmpSpec.LIST_PEAK_INT[ii] + 0.0) for ii in range(len(tmpSpec.LIST_PEAK_MOZ))]) + "\nEND IONS\n")
                else:
                    f.write("END IONS\n")
    # add a new line to record neucode label of peaks
    def writeNeuCodeType1(self, inputPath, inputList):

        with open(inputPath, "a", encoding="utf-8") as f:
            for i in range(len(inputList)):
                tmpSpec = inputList[i]
                f.write("BEGIN IONS\nTITLE=" + tmpSpec.LIST_FILE_NAME[0] + "\n")
                f.write("CHARGE=" + str(tmpSpec.LIST_PRECURSOR_CHARGE[0]) + "+\n")
                f.write("RTINSECONDS=" + str(tmpSpec.SCAN_RET_TIME) + "\n")
                f.write("PEPMASS=" + "%.6f" % (tmpSpec.LIST_PRECURSOR_MOZ[0] +0.0) + "\n")
                f.write("NEUCODELABEL=" + "".join([str(t) for t in tmpSpec.NEUCODE_LABEL]) + "\n")
                f.write("\n".join(["%.5f" % (tmpSpec.LIST_PEAK_MOZ[ii]) + " " + "%.5f" % (tmpSpec.LIST_PEAK_INT[ii] + 0.0) for ii in range(len(tmpSpec.LIST_PEAK_MOZ))]) + "\nEND IONS\n")

    # generate double peak and then write peaks
    def writeNeuCodeType2(self, inputPath, inputList):

        with open(inputPath, "a", encoding="utf-8") as f:
            for i in range(len(inputList)):
                tmpSpec = inputList[i]

                self.__soldierGenerateDoublePeak(tmpSpec)

                f.write("BEGIN IONS\nTITLE=" + tmpSpec.LIST_FILE_NAME[i] + "\n")
                f.write("CHARGE=" + str(tmpSpec.LIST_PRECURSOR_CHARGE[i]) + "+\n")
                f.write("RTINSECONDS=" + str(tmpSpec.SCAN_RET_TIME) + "\n")
                f.write("PEPMASS=" + "%.6f" % (tmpSpec.LIST_PRECURSOR_MOZ[i] +0.0) + "\n")
                f.write("\n".join(["%.5f" % (tmpSpec.LIST_PEAK_MOZ[ii] + 0.0) + " " + "%.5f" % (tmpSpec.LIST_PEAK_INT[ii] + 0.0) for ii in range(len(tmpSpec.LIST_PEAK_MOZ))]) + "\nEND IONS\n")

    # xinming neucode detection result using model
    def writeNeuCodeType3(self, inputPath, tmpSpec:CMS2Spectrum):
        with open(inputPath, "a", encoding="utf-8") as f:
            for i in range(len(tmpSpec.LIST_FILE_NAME)):
                f.write("BEGIN IONS\nTITLE=" + tmpSpec.LIST_FILE_NAME[i] + "\n")
                f.write("CHARGE=" + str(tmpSpec.LIST_PRECURSOR_CHARGE[i]) + "+\n")
                f.write("RTINSECONDS=" + str(tmpSpec.SCAN_RET_TIME) + "\n")
                f.write("PEPMASS=" + "%.6f" % (tmpSpec.LIST_PRECURSOR_MOZ[i] + 0.0) + "\n")
                f.write("NEUCODELABEL=" + "".join([str(t) for t in tmpSpec.NEUCODE_LABEL]) + "\n")
                f.write("\n".join(["%.5f" % (tmpSpec.LIST_PEAK_MOZ[ii]) + " " + "%.5f" % (tmpSpec.LIST_PEAK_INT[ii] + 0.0) for ii in range(len(tmpSpec.LIST_PEAK_MOZ))]) + "\nEND IONS\n")

    # sub func
    # generate double peaks to
    def __soldierGenerateDoublePeak(self, inputSpec):

        gene_moz = []
        gene_int = []
        for i, label in enumerate(inputSpec.NEUCODE_LABEL):
            if label == 1:
                gene_moz.append(inputSpec.LIST_PEAK_MOZ[i] + self.neucode_gap)
                gene_int.append(inputSpec.LIST_PEAK_INT[i])
        inputSpec.LIST_PEAK_MOZ += gene_moz
        inputSpec.LIST_PEAK_INT += gene_int

        sort_idx = np.argsort(inputSpec.LIST_PEAK_MOZ)

        inputSpec.LIST_PEAK_MOZ = [inputSpec.LIST_PEAK_MOZ[i] for i in sort_idx]
        inputSpec.LIST_PEAK_INT = [inputSpec.LIST_PEAK_INT[i] for i in sort_idx]


class CFunctionCombineMGF:
    def __init__(self, inputDP):
        self.dp = inputDP

    def createEmpty(self, inputPathTry, inputPathLys):
        with open(inputPathTry, "w", encoding="utf-8") as ft, open(inputPathLys, "w", encoding="utf-8") as fl:
            ...



    def writeRepeat(self, inputPath, inputList):
        with open(inputPath, "a", encoding="utf-8") as f:
            f.write("\n".join(inputList) + "\n")