
import time
import heapq
import copy
import numpy as np
import math
from MSData import CPrepLabel
from MSTool import toolMyIntRound
from MSLogging import logGetError, logToUser


class CFunctionPreprocess:

    def __init__(self, inputDP):
        self.dp = inputDP
        # important for zixuan
        self.max_remain_peak_num = self.dp.myCFG.B5_ROUND_ONE_HOLD_PEAK_NUM
        # self.tolerance = 0.02  # Da
        self.msms_tol = self.dp.myCFG.D5_MSMS_TOL
        self.msms_ppm = self.dp.myCFG.D6_MSMS_TOL_PPM
        self.upper_peaks_num = 350
        if self.msms_ppm == 1:
            self.msms_fraction = self.msms_tol / 1e6
        else:
            self.msms_fraction = 1
        self.fold = 1e5
        # mass check
        self.PROTON_MASS_C = self.dp.myINI.DICT0_ELEMENT_MASS["C"][0]
        self.PROTON_MASS_H = self.dp.myINI.DICT0_ELEMENT_MASS["H"][0]
        self.PROTON_MASS_O = self.dp.myINI.DICT0_ELEMENT_MASS["O"][0]
        self.PROTON_MASS_CO = self.PROTON_MASS_C + self.PROTON_MASS_O
        self.PROTON_MASS_OH = self.PROTON_MASS_H + self.PROTON_MASS_O
        self.PROTON_MASS_H2O = self.PROTON_MASS_H + self.PROTON_MASS_OH
        self.PROTON_MASS_NH3 = self.PROTON_MASS_H * 3 + self.dp.myINI.DICT0_ELEMENT_MASS["N"][0]

        # 20230113  special for test [added code in remove precursor mass]
        self.PROTON_MASS_PROTON = self.dp.myINI.MASS_PROTON_MONO
        self.PROTON_MASS_H2O_PROTON = self.PROTON_MASS_H2O + self.dp.myINI.MASS_PROTON_MONO

        self.IMMONIUM_HASH_TAB = self.__getImmoniumIonsHashTab()

    def preprocess(self, dataMS2Spectrum, second=True):
        mgf_1, mgf_2 = [], []
        self.__captainPreprocessMS2_1st(dataMS2Spectrum)
        if second:
            # 20230411  母离子信息去冗余，最大限度减少内存的耗费
            for i, precursorMass in enumerate(dataMS2Spectrum.LIST_PRECURSOR_MASS):
                copyMS2Spectrum = copy.deepcopy(dataMS2Spectrum)
                copyMS2Spectrum.LIST_FILE_NAME = [copyMS2Spectrum.LIST_FILE_NAME[i]]
                copyMS2Spectrum.LIST_PRECURSOR_CHARGE = [copyMS2Spectrum.LIST_PRECURSOR_CHARGE[i]]
                copyMS2Spectrum.LIST_PRECURSOR_MOZ = [copyMS2Spectrum.LIST_PRECURSOR_MOZ[i]]
                copyMS2Spectrum.LIST_PRECURSOR_MASS = [copyMS2Spectrum.LIST_PRECURSOR_MASS[i]]
                # 循环对拷贝的数据，选择是否去除母离子相关峰的信息
                # 5. removing ions with precursor
                if self.dp.myCFG.B3_REMOVE_PRECURSOR_ION == 1:
                    self.__soldierRemovePrecursorPeak(copyMS2Spectrum, precursorMass)

                if self.dp.myCFG.E2_EXPORT_ROUND_ONE_MGF:
                    mgf_1.append(copy.deepcopy(copyMS2Spectrum))
                self.__captainPreprocessMS2_2nd(copyMS2Spectrum)
                mgf_2.append(copyMS2Spectrum)
            return mgf_1, mgf_2

        else:
            for precursorMass in dataMS2Spectrum.LIST_PRECURSOR_MASS:
                copyMS2Spectrum = copy.deepcopy(dataMS2Spectrum)

                # 循环对拷贝的数据，选择是否去除母离子相关峰的信息
                # 4. removing ions with precursor
                # self.__captainPreprocessMS2_1st(copyMS2Spectrum)
                if self.dp.myCFG.B3_REMOVE_PRECURSOR_ION == 1:
                    self.__soldierRemovePrecursorPeak(copyMS2Spectrum, precursorMass)
                mgf_1.append(copy.deepcopy(copyMS2Spectrum))
            return mgf_1, mgf_2

    # only two list of moz and int
    # 20221230  统一NeuCode接口，因此使用三个啦
    def returnTwoListForPair(self, dataMS2Spectrum):
        # 一张谱图有多少个母离子，res_moz和res_int 里就有多少个 list
        res_moz, res_int = [], []

        self.__captainPreprocessMS2_1st(dataMS2Spectrum)
        for precursorMass in dataMS2Spectrum.LIST_PRECURSOR_MASS:
            copyMS2Spectrum = copy.deepcopy(dataMS2Spectrum)

            # 循环对拷贝的数据，选择是否去除母离子相关峰的信息
            # 5. removing ions with precursor
            if self.dp.myCFG.B3_REMOVE_PRECURSOR_ION == 1:
                self.__soldierRemovePrecursorPeak(copyMS2Spectrum, precursorMass)

            self.__captainPreprocessMS2_2nd(copyMS2Spectrum)

            # [ASTTENTION] 20221012record
            # 这里可能不需要copy一遍，以后回头检查代码的时候看一看
            res_moz.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_MOZ))
            res_int.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_INT))

        return res_moz, res_int, [None] * len(res_moz)


    # only two list of moz and int
    # 20221230  统一NeuCode接口，因此使用三个啦
    def returnFourListForPair(self, dataMS2Spectrum):
        # 一张谱图有多少个母离子，res_moz和res_int 里就有多少个 list
        res_moz1, res_int1 = [], []
        res_moz2, res_int2 = [], []
        copyMS2SpectrumTmp = copy.deepcopy(dataMS2Spectrum)
        self.__captainPreprocessMS2_1st(copyMS2SpectrumTmp)
        for precursorMass in dataMS2Spectrum.LIST_PRECURSOR_MASS:
            copyMS2Spectrum = copy.deepcopy(copyMS2SpectrumTmp)

            # 循环对拷贝的数据，选择是否去除母离子相关峰的信息
            # 5. removing ions with precursor
            if self.dp.myCFG.B3_REMOVE_PRECURSOR_ION == 1:
                self.__soldierRemovePrecursorPeak(copyMS2Spectrum, precursorMass)

            res_moz1.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_MOZ))
            res_int1.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_INT))

            self.__captainPreprocessMS2_2nd(copyMS2Spectrum)

            # [ASTTENTION] 20221012record
            # 这里可能不需要copy一遍，以后回头检查代码的时候看一看
            res_moz2.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_MOZ))
            res_int2.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_INT))

        return res_moz1, res_int1, [None]*len(res_moz1), res_moz2, res_int2, [None] * len(res_moz2)


    # piyu preprocessing without de-precursor ions
    def __captainPreprocessMS2_1st(self, dataMS2Spectrum):

        # 20201110  调整执行次序
        # 1. holding top-N peaks
        ori_peaks_num = len(dataMS2Spectrum.LIST_PEAK_INT)

        if dataMS2Spectrum.LIST_PEAK_INT:

            # 若给定参数k则执行保留峰数目为k的操作，且k不小于用户指定数目，否则保留用户指定数目
            if len(dataMS2Spectrum.LIST_PEAK_MOZ) > self.upper_peaks_num:

                rankListByINT = self.__soldierGetRankListByINT(dataMS2Spectrum)

                self.__soldierDeleteWeakPeak(dataMS2Spectrum, rankListByINT, holdNum=self.upper_peaks_num)

            # 如果没有谱峰，也没必要做转换了
            # 2. cluster transforming
            self.__soldierGetSingleChargePeaksMS2TESTING(dataMS2Spectrum)

        # 3. hold top-N peaks
        if dataMS2Spectrum.LIST_PEAK_INT:

            if len(dataMS2Spectrum.LIST_PEAK_MOZ) > self.max_remain_peak_num:

                rankListByINT = self.__soldierGetRankListByINT(dataMS2Spectrum)

                self.__soldierDeleteWeakPeak(dataMS2Spectrum, rankListByINT)

        # 4. check natural loss peaks
        if self.dp.myCFG.B2_CHECK_NATURAL_LOSS == 1:
            self.__soldierClusterNeutronalLossH2OAndNH3(dataMS2Spectrum)

    # zixuan preprocessing part
    def __captainPreprocessMS2_2nd(self, dataMS2Spectrum):

        # 1. 最大最小归一化到10000再开方
        self.__soldierAbsoluteINT2RelativeINT_MAXMIN(dataMS2Spectrum, 0.5)

        # self.__soldierAbsoluteINT2RelativeINT(dataMS2Spectrum)
        # 2. delete the immonium ions info
        if self.dp.myCFG.B6_ROUND_TWO_REMOVE_IMMONIUM and dataMS2Spectrum.LIST_PEAK_INT:
            self.__deleteImmoniumIons(dataMS2Spectrum)

        # control local peaks num
        if self.dp.myCFG.B8_ROUND_TWO_MASS_BIN_LENGTH > 1  and dataMS2Spectrum.LIST_PEAK_INT:
            self.__localPeaksNumControl(dataMS2Spectrum, self.dp.myCFG.B8_ROUND_TWO_MASS_BIN_LENGTH, self.dp.myCFG.B7_ROUND_TWO_PEAK_NUM_PER_BIN)

    # 按照强度排序进行遍历
    # 效果可能会好些(实际上并不是
    # 20210726  该函数固定了20ppm的检测tol，若用户设定可变，则需修改
    def __soldierGetSingleChargePeaksMS2TESTING(self, dataMS2Spectrum):

        # check
        delete_index_list = []
        for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
            if dataMS2Spectrum.LIST_PEAK_MOZ[i] < self.dp.myINI.MASS_PROTON_MONO:
                delete_index_list.append(i)
            else:
                break
        if delete_index_list:
            for index in delete_index_list[::-1]:
                dataMS2Spectrum.LIST_PEAK_MOZ.pop(index)
                dataMS2Spectrum.LIST_PEAK_INT.pop(index)
            delete_index_list = []

        max_charge = max(dataMS2Spectrum.LIST_PRECURSOR_CHARGE)

        if self.dp.myCFG.B4_CHECK_PRECURSOR_CHARGE == 0:
            if max_charge > 1:
                max_charge -= 1

        new_peak_int = []
        new_peak_moz = []
        isotopic_tol = [self.dp.myINI.MASS_NEUTRON_AVRG / c for c in range(1, max_charge + 1)]
        # 1.0030, 0.5015, 0.3334, 0.25075, 0.2016...

        # cluster_start_counter = 0
        charge_state = -1  # 独峰

        while 0 < len(dataMS2Spectrum.LIST_PEAK_MOZ):
            cluster_index_list = []
            # 小于isotopic_tol[-1] - tol 就check下一谱峰
            # 检查开始列表中的任意电荷状态的距离，由index可得电荷状态index+1
            # 超过isotopic_tol[0]就进行归并，并跳转下一峰
            # index = cluster_start_counter
            max_int = -1
            max_int_peak_index = -1
            for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
                if dataMS2Spectrum.LIST_PEAK_INT[i] > max_int:
                    max_int = dataMS2Spectrum.LIST_PEAK_INT[i]
                    max_int_peak_index = i
            # 得到最高峰信号的地址了

            # 而后左右检测是否可有峰簇

            # while tmp_check_index < len(dataMS2Spectrum.LIST_PEAK_MOZ):
            #     if -1 == charge_state:
            #         pass
            # ########################################################
            # not complete

            # -------------------------- right -----------------------------
            # ##############################################################
            tmp_check_index = max_int_peak_index
            # index = 0
            cluster_index_list.append(tmp_check_index)
            cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
            tmp_check_index += 1
            # charge == -1: cluster is not complete

            while tmp_check_index < len(dataMS2Spectrum.LIST_PEAK_MOZ):

                peak_tolerance = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index] - cluster_tail_moz
                if self.msms_ppm == 1:
                    ppm_tol_da = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index] * self.msms_fraction
                else:
                    ppm_tol_da = self.msms_tol
                prep_tolerance = ppm_tol_da
                if peak_tolerance < isotopic_tol[-1] - prep_tolerance:
                    # 小于最小,跳过去
                    # 要继续找，不要break
                    tmp_check_index += 1

                elif peak_tolerance > isotopic_tol[0] + prep_tolerance:
                    # 大于最大
                    # 不要继续找，要break
                    break

                else:

                    # 先确定电荷状态，再继续cluster构建
                    if charge_state == -1:
                        for tol_index in range(len(isotopic_tol)):
                            if math.isclose(peak_tolerance, isotopic_tol[tol_index], abs_tol=prep_tolerance):
                                charge_state = tol_index + 1
                                # 确定质荷比信息
                                cluster_index_list.append(tmp_check_index)
                                cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                                break
                        if charge_state == -1:
                            # 继续向后寻找（MUST）
                            tmp_check_index += 1
                        else:

                            # 连续向后构造
                            tmp_check_index += 1


                    # 已经确定电荷状态
                    else:
                        # 仍然是我的同位素峰，进入峰簇下标列表
                        if math.isclose(peak_tolerance, isotopic_tol[charge_state - 1], abs_tol=prep_tolerance):

                            # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                            nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[-1]]
                            if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] > 1.2 * nearest_int:
                                tmp_check_index += 1
                                continue
                            cluster_index_list.append(tmp_check_index)
                            cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                            tmp_check_index += 1
                        # 超过了枚举电荷的范围，break
                        elif peak_tolerance - isotopic_tol[charge_state - 1] > prep_tolerance:
                            break
                        # 又不匹配，又不越界，啥也不是，继续走吧
                        else:
                            tmp_check_index += 1
            # while index < len(dataMS2Spectrum.LIST_PEAK_MOZ) over.

            # -------------------------- left ------------------------------
            # ##############################################################
            tmp_check_index = max_int_peak_index - 1
            cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[max_int_peak_index]
            while tmp_check_index >= 0 and dataMS2Spectrum.LIST_PEAK_MOZ:

                peak_tolerance = cluster_left_moz - dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                if self.msms_ppm == 1:
                    ppm_tol_da = cluster_left_moz * self.msms_fraction
                else:
                    ppm_tol_da = self.msms_tol
                prep_tolerance = ppm_tol_da
                if peak_tolerance < isotopic_tol[-1] - prep_tolerance:
                    # 小于最小,跳过去
                    # 要继续找，不要break
                    tmp_check_index -= 1

                elif peak_tolerance > isotopic_tol[0] + prep_tolerance:
                    # 大于最大
                    # 不要继续找，要break
                    break

                else:

                    # 先确定电荷状态，再继续cluster构建
                    if charge_state == -1:
                        for tol_index in range(len(isotopic_tol)):
                            if math.isclose(peak_tolerance, isotopic_tol[tol_index], abs_tol=prep_tolerance):

                                # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                                # nearest_int: cluster list中最左端的谱峰信号的强度值
                                nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[0]]
                                # if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.2 * nearest_int:
                                if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.3 * nearest_int:
                                    # break
                                    tmp_check_index -= 1
                                    continue

                                charge_state = tol_index + 1
                                # 确定质荷比信息
                                # cluster_index_list.append(tmp_check_index)
                                cluster_index_list = [tmp_check_index] + cluster_index_list
                                cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                                break
                        if charge_state == -1:
                            # 继续向前寻找（MUST）
                            tmp_check_index -= 1
                        else:

                            # 连续向后构造
                            tmp_check_index -= 1


                    # 已经确定电荷状态
                    else:
                        # 仍然是我的同位素峰，进入峰簇下标列表
                        if math.isclose(peak_tolerance, isotopic_tol[charge_state - 1], abs_tol=prep_tolerance):
                            # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                            nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[0]]
                            # if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.2 * nearest_int:
                            if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.4 * nearest_int:
                                # break
                                tmp_check_index -= 1
                                continue
                            cluster_index_list = [tmp_check_index] + cluster_index_list
                            cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                            tmp_check_index -= 1
                        # 超过了枚举电荷的范围，break
                        elif peak_tolerance - isotopic_tol[charge_state - 1] > prep_tolerance:
                            break
                        # 又不匹配，又不越界，啥也不是，继续走吧
                        else:
                            tmp_check_index -= 1

            # --------------------------------------------------------------

            # cluster
            # cluster 构造结束，开始收工
            if charge_state == -1:
                # 删除，添加
                add_moz = dataMS2Spectrum.LIST_PEAK_MOZ.pop(cluster_index_list[0])
                add_int = dataMS2Spectrum.LIST_PEAK_INT.pop(cluster_index_list[0])
                new_peak_moz.append(add_moz)
                new_peak_int.append(add_int)
            else:

                # ##########################################
                # 20210809 ATTENTION ATTENTION ATTENTION ###
                # pop时，一定一定一定要注意地址的问题！ ####
                # ##########################################
                # --------------- QUESTION --------------- #
                # 这里要把mono峰pop出来，但是一定要放在最后#
                # 否则会改变其他谱峰的地址，导致pop峰错误  #
                # 再就是强度和质荷比错开了，导致tag提取错误#
                # ##########################################
                add_int = 0
                if self.dp.myCFG.B1_ADD_ISOTOPE_INTENSITY:
                    for buf_index in reversed(cluster_index_list[1:]):
                        try:
                            # dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                            # add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)

                            dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                            add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)

                        except:
                            pass
                # ================== DO NOT ADD INTENSITY! ====================
                else:
                    for buf_index in reversed(cluster_index_list[1:]):
                        try:
                            dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                            dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)
                        except:
                            pass

                # 把mono峰的moz提出放在最后，包括mono对应的强度部分也是
                # 如果希望对add的强度做一些操作，可以在for loop try里头去整
                add_moz = dataMS2Spectrum.LIST_PEAK_MOZ.pop(cluster_index_list[0]) * charge_state
                add_moz -= (charge_state - 1) * self.dp.myINI.MASS_PROTON_MONO
                add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(cluster_index_list[0])

                new_peak_moz.append(add_moz)
                new_peak_int.append(add_int)

            charge_state = -1
            # cluster 处理结束
        # while over

        # -------------------------------------------------------------------

        # 排序，检测合并
        index_order = np.argsort(new_peak_moz)
        new_peak_moz = [new_peak_moz[index] for index in index_order]
        new_peak_int = [new_peak_int[index] for index in index_order]

        # -------------------------------------------------------------------

        output_peak_moz = []
        output_peak_int = []

        # 下方：合并转换的谱峰，有可能会影响精度
        add_moz = 0
        add_int = 0
        i = 0
        jump_set = set()
        while i < len(new_peak_moz) - 1:

            if i in jump_set:
                i += 1
                continue

            add_moz = new_peak_moz[i]
            add_int = new_peak_int[i]

            for j in range(i + 1, len(new_peak_moz)):
                if self.msms_ppm == 1:

                    prep_tolerance = new_peak_moz[j] * self.msms_fraction

                else:
                    prep_tolerance = self.msms_tol
                # prep_tolerance = new_peak_moz[j] * 2e-5

                if abs(new_peak_moz[i] - new_peak_moz[j]) < prep_tolerance:
                    add_moz = add_moz * add_int + new_peak_moz[j] * new_peak_int[j]
                    add_int += new_peak_int[j]
                    add_moz /= add_int
                    i = j
                    jump_set.add(j)
                # 仅看左右最远的0.02，两两之间就不看了
                else:
                    output_peak_moz.append(add_moz)
                    output_peak_int.append(add_int)
                    i = j
                    break

            # if abs(new_peak_moz[i] - new_peak_moz[-1]) < prep_tolerance:
            #     output_peak_moz.append(add_moz)
            #     output_peak_int.append(add_int)
            #     i = j
            #     break

        if add_moz in output_peak_moz:
            pass
        else:
            output_peak_moz.append(add_moz)
            output_peak_int.append(add_int)

        # 活久见，图里没谱峰哒，就算辽叭
        if len(new_peak_moz) == 0:
            pass
        # 检测一下最后一根谱峰是否是可以添加的信息
        elif jump_set:
            if max(jump_set) == len(new_peak_moz):
                pass
            else:
                output_peak_moz.append(new_peak_moz[-1])
                output_peak_int.append(new_peak_int[-1])
        else:
            output_peak_moz.append(new_peak_moz[-1])
            output_peak_int.append(new_peak_int[-1])

        # 上方：合并转换的谱峰，有可能会影响精度
        # output_peak_moz = new_peak_moz
        # output_peak_int = new_peak_int

        dataMS2Spectrum.LIST_PEAK_MOZ = output_peak_moz
        dataMS2Spectrum.LIST_PEAK_INT = output_peak_int


    # 同位素峰簇转换结束后，检测失水失氨峰
    # 按照强度为顺序，“向左”对失水失氨的质量依次进行检测
    # 将符合质量的谱峰，构成一个cluster
    # 因为谱峰之间的质量顺序是：ori_peak - H2O, ori_peak - NH3, ori_peak
    # 所以构造的谱峰也将按照此规律检测
    # 20210902  修改质量偏差设定收窄为10ppm
    def __soldierClusterNeutronalLossH2OAndNH3(self, dataMS2Spectrum):

        new_peak_int = []
        new_peak_moz = []

        while 0 < len(dataMS2Spectrum.LIST_PEAK_MOZ):
            cluster_index_list = []

            # 这里进行了修改，即得到最后一根谱峰，然后从后往前找-NH3和-H2O的峰
            tmp_check_index = len(dataMS2Spectrum.LIST_PEAK_MOZ) - 1
            cluster_index_list.append(tmp_check_index)
            cluster_tail_int = dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index]

            # -------------------------- left ------------------------------
            # ##############################################################
            cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
            tmp_check_index -= 1

            while tmp_check_index >= 0 and dataMS2Spectrum.LIST_PEAK_MOZ:

                peak_tolerance = cluster_tail_moz - dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                if self.msms_ppm == 1:
                    # self.msms_fraction
                    ppm_tol_da = cluster_tail_moz * self.msms_fraction
                else:
                    ppm_tol_da = self.msms_tol

                prep_tolerance = ppm_tol_da

                if peak_tolerance < self.PROTON_MASS_NH3 - prep_tolerance:
                    # 小于最小,跳过去
                    # 要继续找，不要break
                    tmp_check_index -= 1

                elif peak_tolerance > self.PROTON_MASS_H2O + prep_tolerance:
                    # 大于最大
                    # 不要继续找，要break
                    break

                else:
                    # 进入自然损失簇下标列表
                    if math.isclose(peak_tolerance, self.PROTON_MASS_NH3, abs_tol=prep_tolerance):

                        cluster_left_int = dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index]
                        if cluster_tail_int > 0.2 * cluster_left_int:
                            cluster_index_list = [tmp_check_index] + cluster_index_list
                        tmp_check_index -= 1
                        # 进入自然损失簇下标列表

                    elif math.isclose(peak_tolerance, self.PROTON_MASS_H2O, abs_tol=prep_tolerance):

                        cluster_left_int = dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index]
                        if cluster_tail_int > 0.2 * cluster_left_int:
                            cluster_index_list = [tmp_check_index] + cluster_index_list
                        tmp_check_index -= 1

                    # 又不匹配，又不越界，啥也不是，继续走吧
                    else:
                        tmp_check_index -= 1

            # --------------------------------------------------------------

            # cluster
            # cluster 构造结束，开始收工
            add_moz = dataMS2Spectrum.LIST_PEAK_MOZ[cluster_index_list[-1]]
            if self.dp.myCFG.B1_ADD_ISOTOPE_INTENSITY:
                add_int = 0
                for buf_index in sorted(cluster_index_list, reverse=True):
                    try:
                        dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                        add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)
                    except:
                        pass
            else:
                add_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[-1]]
                for buf_index in sorted(cluster_index_list, reverse=True):
                    try:
                        dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                        dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)
                    except:
                        pass

            # 如果希望对add的强度做一些操作，可以在for loop try里头去整

            new_peak_moz.append(add_moz)
            new_peak_int.append(add_int)
            # cluster 处理结束
        # while over

        # -------------------------------------------------------------------

        # 排序，检测合并
        if new_peak_moz:
            index_order = np.argsort(new_peak_moz)
            new_peak_moz = [new_peak_moz[index] for index in index_order]
            new_peak_int = [new_peak_int[index] for index in index_order]

        dataMS2Spectrum.LIST_PEAK_MOZ = new_peak_moz
        dataMS2Spectrum.LIST_PEAK_INT = new_peak_int


    def __soldierGetRankListByINT(self, dataMS2Spectrum):

        tmpMS2_np = np.array(dataMS2Spectrum.LIST_PEAK_INT[:])

        rankListByINT = np.argsort(tmpMS2_np)
        # 注意，这里的索引是按照强度，从小到大排列的

        return rankListByINT


    def __soldierRemovePrecursorPeak(self, dataMS2Spectrum, inputPrecursorMass):

        soldier = inputPrecursorMass - 50

        # eg. Precursor -H2O / -NH3, Precursor - 2*H2O
        for i in reversed(range(len(dataMS2Spectrum.LIST_PEAK_MOZ))):

            if dataMS2Spectrum.LIST_PEAK_MOZ[i] >= soldier:
                dataMS2Spectrum.LIST_PEAK_MOZ.pop()
                dataMS2Spectrum.LIST_PEAK_INT.pop()

            else:
                break

        # 20210825 -----------------------------------------------------
        # 去掉质量小于18Da的分子
        soldier = 20

        while dataMS2Spectrum.LIST_PEAK_MOZ:

            if dataMS2Spectrum.LIST_PEAK_MOZ[0] <= soldier:
                dataMS2Spectrum.LIST_PEAK_MOZ.pop(0)
                dataMS2Spectrum.LIST_PEAK_INT.pop(0)

            else:
                break

        maxInt = max(dataMS2Spectrum.LIST_PEAK_INT)
        dataMS2Spectrum.LIST_PEAK_MOZ = [self.PROTON_MASS_PROTON, self.PROTON_MASS_H2O_PROTON] + dataMS2Spectrum.LIST_PEAK_MOZ + [inputPrecursorMass - self.PROTON_MASS_H2O, inputPrecursorMass]
        dataMS2Spectrum.LIST_PEAK_INT = [maxInt, maxInt] + dataMS2Spectrum.LIST_PEAK_INT + [maxInt, maxInt]

    def __soldierDeleteWeakPeak(self, dataMS2Spectrum, rankListByINT, holdNum=0):

        # 获得删除列表，且已经逆序排好顺序，便于删除
        # 20210902 修改判0为判断大小，避免or操作，合并检测的包含关系
        if holdNum < self.max_remain_peak_num:
            deleteList = reversed(np.sort(rankListByINT[:-self.max_remain_peak_num]))
        else:
            deleteList = reversed(np.sort(rankListByINT[:-holdNum]))
        # 逆序删除，不影响索引
        for i in deleteList:
            dataMS2Spectrum.LIST_PEAK_MOZ.pop(i)
            dataMS2Spectrum.LIST_PEAK_INT.pop(i)


    def __soldierINT2RANK(self, dataMS2Spectrum):

        tmpMS2INTList = np.array(dataMS2Spectrum.LIST_PEAK_INT[:])
        # 读取MS2的强度信息

        rankIndexList = np.argsort(-tmpMS2INTList)
        # 将强度从大到小排列，保留地址
        # 加个负号即可

        rank = 1
        for index in rankIndexList:
            dataMS2Spectrum.LIST_PEAK_INT[index] = rank
            rank += 1


    def __soldierAbsoluteINT2RelativeINT(self, dataMS2Spectrum):

        if not dataMS2Spectrum.LIST_PEAK_INT:
            # print("Spectrum empty")
            pass
        # 如果列表不空的话（话说为什么会空呢。。。全为precursor？）
        else:
            maxAbsoluteINTValue = max(dataMS2Spectrum.LIST_PEAK_INT[:])

            # 使用numpy的广播来代替此处的for循环，速度会更快
            for i in range(len(dataMS2Spectrum.LIST_PEAK_INT)):

                tmpAbsINT = dataMS2Spectrum.LIST_PEAK_INT[i]

                dataMS2Spectrum.LIST_PEAK_INT[i] = tmpAbsINT / maxAbsoluteINTValue * 100


    def __soldierAbsoluteINT2RelativeINT_MAXMIN(self, dataMS2Spectrum, exp):

        if not dataMS2Spectrum.LIST_PEAK_INT:
            # print("Spectrum empty")
            pass
        # 如果列表不空的话（话说为什么会空呢。。。全为precursor？）
        else:
            maxAbsoluteINTValue = max(dataMS2Spectrum.LIST_PEAK_INT)
            minAbsoluteINTValue = min(dataMS2Spectrum.LIST_PEAK_INT)

            if maxAbsoluteINTValue == minAbsoluteINTValue:
                # 退化为最大值归一化，避免除零错误
                for i in range(len(dataMS2Spectrum.LIST_PEAK_INT)):
                    tmpAbsINT = dataMS2Spectrum.LIST_PEAK_INT[i]

                    dataMS2Spectrum.LIST_PEAK_INT[i] = (tmpAbsINT / maxAbsoluteINTValue * 10000) ** exp

            else:
                # 最大最小归一化
                maxAbsoluteINTValue -= minAbsoluteINTValue

                # 使用numpy的广播来代替此处的for循环，速度会更快
                for i in range(len(dataMS2Spectrum.LIST_PEAK_INT)):

                    tmpAbsINT = dataMS2Spectrum.LIST_PEAK_INT[i] - minAbsoluteINTValue

                    dataMS2Spectrum.LIST_PEAK_INT[i] = (tmpAbsINT / maxAbsoluteINTValue * 10000) ** exp


    def __localPeaksNumControl(self, dataMS2Spectrum, bin_len, peak_num):
        ...
        res_moz = []
        res_int = []

        # 当列表变为空的时候，我们就不用再循环啦
        while dataMS2Spectrum.LIST_PEAK_INT:
            # 先确定下来因数是多少
            old_soldier = dataMS2Spectrum.LIST_PEAK_MOZ[0] // bin_len
            # 初始化为最小的地址
            tmp_idx = 0

            # find an index range per bin_len
            # 找到第一个和我们预先设定的因数不一致的数字的地址
            for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
                # 一直没找到，那就一直记录即可 -------------
                if old_soldier == (dataMS2Spectrum.LIST_PEAK_MOZ[i] // bin_len):
                    tmp_idx = i
                # 找到了！我们直接break！
                else:
                    break

            # 此时，地址代表了 peak num + 1
            # 因此当tmp_idx 小于 peak_num 时，满足谱峰过滤的要求，直接加进去即可
            if tmp_idx < peak_num:
                res_moz += dataMS2Spectrum.LIST_PEAK_MOZ[:tmp_idx+1]
                dataMS2Spectrum.LIST_PEAK_MOZ = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_idx+1:]
                res_int += dataMS2Spectrum.LIST_PEAK_INT[:tmp_idx+1]
                dataMS2Spectrum.LIST_PEAK_INT = dataMS2Spectrum.LIST_PEAK_INT[tmp_idx+1:]

            # 当前该 bin 中的谱峰数目过多，需要选择最强的peak_num根峰
            else:
                tmp_moz = dataMS2Spectrum.LIST_PEAK_MOZ[:tmp_idx+1]
                tmp_int = dataMS2Spectrum.LIST_PEAK_INT[:tmp_idx+1]
                dataMS2Spectrum.LIST_PEAK_MOZ = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_idx+1:]
                dataMS2Spectrum.LIST_PEAK_INT = dataMS2Spectrum.LIST_PEAK_INT[tmp_idx+1:]

                # 去除强度低的谱峰，这样添加到res_moz / res_int 中时，moz还能保证有序
                while len(tmp_moz) != peak_num:
                    min_idx = tmp_int.index(min(tmp_int))
                    tmp_int.pop(min_idx)
                    tmp_moz.pop(min_idx)
                res_moz += tmp_moz
                res_int += tmp_int

        dataMS2Spectrum.LIST_PEAK_MOZ = res_moz
        dataMS2Spectrum.LIST_PEAK_INT = res_int


    # 获得亚胺离子的质量信息，并生成哈希值的字典
    # 子璇在这里的默认误差范围是0.02Da
    # int((mass+-delta) * 1000)
    def __getImmoniumIonsHashTab(self):
        # self.fold = 1000
        out_set = set()

        # [hash] imonium ions masses with mass tolerance(0.02Da)
        abs_delta_hash = toolMyIntRound(0.02 * self.fold)
        for aa in self.dp.myINI.DICT1_AA_MASS:
            immonium_mass = self.dp.myINI.DICT1_AA_MASS[aa] - self.PROTON_MASS_CO + self.dp.myINI.MASS_PROTON_MONO
            if immonium_mass <= 0:
                continue
            aa_mass_hash = toolMyIntRound(immonium_mass * self.fold)
            out_set.update(set([it for it in range(aa_mass_hash - abs_delta_hash, aa_mass_hash +abs_delta_hash + 1)]))

        # [hash] relative ions masses of immonium ions with mass tolerance(0.05Da)
        abs_delta_hash = toolMyIntRound(0.05 * self.fold)
        tmp_lst = [41.05, 44.05, 55.05, 56.05, 59.05, 61.05, 69.05, 70.05, 72.05, 73.05,
             77.05, 82.05, 84.05, 87.05, 91.05, 100.05, 107.05, 112.05, 117.05, 121.05,
             123.05, 129.05, 130.05, 132.05, 138.05, 166.05, 170.05, 171.05]
        for rel_mass in tmp_lst:
            rel_mass_hash = toolMyIntRound(rel_mass * self.fold)
            out_set.update(set([it for it in range(rel_mass_hash - abs_delta_hash, rel_mass_hash + abs_delta_hash + 1)]))

        return out_set

    def __deleteImmoniumIons(self, dataMS2Spectrum):

        soldier_idx = 0
        for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
            if dataMS2Spectrum.LIST_PEAK_MOZ[i] > 200:
                soldier_idx = i
                break

        # 存在谱峰的质量是小于200的
        if soldier_idx != 0:
            # 逆序删除，不影响地址信息
            for i in reversed(range(soldier_idx)):
                if toolMyIntRound(dataMS2Spectrum.LIST_PEAK_MOZ[i] * self.fold) in self.IMMONIUM_HASH_TAB:
                    dataMS2Spectrum.LIST_PEAK_MOZ.pop(i)
                    dataMS2Spectrum.LIST_PEAK_INT.pop(i)


# =======================================
# [20221123]
# 针对 NeuCode 数据而特别调整的预处理流程
# 原始流程也挺好，就是不太方便修改
# 因为要有特定的标记信息输出
# 感觉上就像冗余代码一样，但是问题不大！
# 如果两边儿中任意一方有什么细节改动
# 可以参考着一起调整，先在此 mark 一下
# =======================================
class CFunctionPreprocessNeuCode:

    def __init__(self, inputDP):
        self.dp = inputDP
        # important for zixuan
        self.max_remain_peak_num = self.dp.myCFG.B5_ROUND_ONE_HOLD_PEAK_NUM
        # self.tolerance = 0.02  # Da
        self.msms_tol = self.dp.myCFG.D5_MSMS_TOL
        self.msms_ppm = self.dp.myCFG.D6_MSMS_TOL_PPM
        self.upper_peaks_num1 = 600
        self.upper_peaks_num2 = 350
        if self.msms_ppm == 1:
            self.msms_fraction = self.msms_tol / 1e6
        else:
            self.msms_fraction = 1
        self.fold = 1e5
        # mass check
        self.PROTON_MASS_C = self.dp.myINI.DICT0_ELEMENT_MASS["C"][0]
        self.PROTON_MASS_H = self.dp.myINI.DICT0_ELEMENT_MASS["H"][0]
        self.PROTON_MASS_O = self.dp.myINI.DICT0_ELEMENT_MASS["O"][0]
        self.PROTON_MASS_CO = self.PROTON_MASS_C + self.PROTON_MASS_O
        self.PROTON_MASS_OH = self.PROTON_MASS_H + self.PROTON_MASS_O
        self.PROTON_MASS_H2O = self.PROTON_MASS_H + self.PROTON_MASS_OH
        self.PROTON_MASS_NH3 = self.PROTON_MASS_H * 3 + self.dp.myINI.DICT0_ELEMENT_MASS["N"][0]

        # self.MAX_INTEN_RATIO = 6.0  # inten_light / inten_heavy
        self.MAX_INTEN_RATIO = self.dp.myCFG.A10_NEUCODE_PEAK_MAX_RATIO

        self.DOUBLE_PEAK_BOUND = 1500
        self.DOUBLE_PEAK_SHIFT = self.dp.myCFG.A9_DOUBLE_PEAK_GAP  # fixed mass shift of (heavy - light) ~0.036 Da
        self.DOUBLE_PEAK_SHIFT2 = self.DOUBLE_PEAK_SHIFT * 2       # 考虑一个漏切，缘分这不就来了吗
        self.DOUBLE_PEAK_TOL_MAX = self.DOUBLE_PEAK_SHIFT2 + 0.5   # 设定了一个检测双峰 tol 的上限，超过即停
        # 我就不信tol能超过 0.5 哼！而且0.5的浮点数表示是1 * 2^(-1)，应该挺好算哈(都是自己的想象-_-未经证实哈...)

        self.DOUBLE_PEAK_MAX_C = 3      # 检测双峰的时候，就按这些电荷的量来检测就行~ 碎片离子再大就检测不到了也

        self.DOUBLE_PEAK_SHIFT_C = [1] + [self.DOUBLE_PEAK_SHIFT/(c+1) for c in range(self.DOUBLE_PEAK_MAX_C)]
        self.DOUBLE_PEAK_SHIFT2_C = [1] + [self.DOUBLE_PEAK_SHIFT2/(c+1) for c in range(self.DOUBLE_PEAK_MAX_C)]

        if self.msms_ppm == 1:
            self.DOUBLE_PEAK_TOL_M = min(10, self.msms_tol)
        else:
            self.DOUBLE_PEAK_TOL_M = min(0.02, self.msms_tol)

        self.DOUBLE_TOL_HASH_TAB_LIST = self.__getDoublePeakTolHashTab()  # [None, set(), set(), set()]
        self.IMMONIUM_HASH_TAB = self.__getImmoniumIonsHashTab()

    # 20221123  nothing need adjust
    def preprocess(self, dataMS2Spectrum, second=True):
        mgf_1, mgf_2 = [], []
        self.__captainPreprocessMS2_1st(dataMS2Spectrum)
        if second:
            # 20230411  母离子信息去冗余，最大限度减少内存的耗费
            for i, precursorMass in enumerate(dataMS2Spectrum.LIST_PRECURSOR_MASS):
                copyMS2Spectrum = copy.deepcopy(dataMS2Spectrum)
                copyMS2Spectrum.LIST_FILE_NAME = [copyMS2Spectrum.LIST_FILE_NAME[i]]
                copyMS2Spectrum.LIST_PRECURSOR_CHARGE = [copyMS2Spectrum.LIST_PRECURSOR_CHARGE[i]]
                copyMS2Spectrum.LIST_PRECURSOR_MOZ = [copyMS2Spectrum.LIST_PRECURSOR_MOZ[i]]
                copyMS2Spectrum.LIST_PRECURSOR_MASS = [copyMS2Spectrum.LIST_PRECURSOR_MASS[i]]
                # 循环对拷贝的数据，选择是否去除母离子相关峰的信息
                # 5. removing ions with precursor
                if self.dp.myCFG.B3_REMOVE_PRECURSOR_ION == 1:
                    self.__soldierRemovePrecursorPeak(copyMS2Spectrum, precursorMass)

                if self.dp.myCFG.E2_EXPORT_ROUND_ONE_MGF:
                    mgf_1.append(copy.deepcopy(copyMS2Spectrum))
                self.__captainPreprocessMS2_2nd(copyMS2Spectrum)
                mgf_2.append(copyMS2Spectrum)
            return mgf_1, mgf_2

        else:
            for i, precursorMass in enumerate(dataMS2Spectrum.LIST_PRECURSOR_MASS):
                copyMS2Spectrum = copy.deepcopy(dataMS2Spectrum)

                # 循环对拷贝的数据，选择是否去除母离子相关峰的信息
                # 4. removing ions with precursor
                # self.__captainPreprocessMS2_1st(copyMS2Spectrum)
                copyMS2Spectrum.LIST_FILE_NAME = [copyMS2Spectrum.LIST_FILE_NAME[i]]
                copyMS2Spectrum.LIST_PRECURSOR_CHARGE = [copyMS2Spectrum.LIST_PRECURSOR_CHARGE[i]]
                copyMS2Spectrum.LIST_PRECURSOR_MOZ = [copyMS2Spectrum.LIST_PRECURSOR_MOZ[i]]
                copyMS2Spectrum.LIST_PRECURSOR_MASS = [copyMS2Spectrum.LIST_PRECURSOR_MASS[i]]

                if self.dp.myCFG.B3_REMOVE_PRECURSOR_ION == 1:
                    self.__soldierRemovePrecursorPeak(copyMS2Spectrum, precursorMass)
                mgf_1.append(copy.deepcopy(copyMS2Spectrum))
            return mgf_1, mgf_2

    # only two list of moz and int
    def returnTwoListForPair(self, dataMS2Spectrum):
        # 一张谱图有多少个母离子，res_moz和res_int 里就有多少个 list
        res_moz, res_int, res_ncl = [], [], []

        self.__captainPreprocessMS2_1st(dataMS2Spectrum)
        for precursorMass in dataMS2Spectrum.LIST_PRECURSOR_MASS:
            copyMS2Spectrum = copy.deepcopy(dataMS2Spectrum)

            # 循环对拷贝的数据，选择是否去除母离子相关峰的信息
            # 5. removing ions with precursor
            if self.dp.myCFG.B3_REMOVE_PRECURSOR_ION == 1:
                self.__soldierRemovePrecursorPeak(copyMS2Spectrum, precursorMass)

            self.__captainPreprocessMS2_2nd(copyMS2Spectrum)

            # [ASTTENTION] 20221012record
            # 这里可能不需要copy一遍，以后回头检查代码的时候看一看
            res_moz.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_MOZ))
            res_int.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_INT))
            res_ncl.append(copy.deepcopy(copyMS2Spectrum.NEUCODE_LABEL))

        return res_moz, res_int, res_ncl


    def returnFourListForPair(self, dataMS2Spectrum):
        # 一张谱图有多少个母离子，res_moz和res_int 里就有多少个 list
        res_moz1, res_int1, res_ncl1 = [], [], []
        res_moz2, res_int2, res_ncl2 = [], [], []

        copyMS2SpectrumTmp = copy.deepcopy(dataMS2Spectrum)
        self.__captainPreprocessMS2_1st(copyMS2SpectrumTmp)
        for precursorMass in dataMS2Spectrum.LIST_PRECURSOR_MASS:
            copyMS2Spectrum = copy.deepcopy(copyMS2SpectrumTmp)

            # 循环对拷贝的数据，选择是否去除母离子相关峰的信息
            # 5. removing ions with precursor
            if self.dp.myCFG.B3_REMOVE_PRECURSOR_ION == 1:
                self.__soldierRemovePrecursorPeak(copyMS2Spectrum, precursorMass)

            # copy 以后不影响值的改变，所以就没有的问题了
            res_moz1.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_MOZ))
            res_int1.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_INT))
            res_ncl1.append(copy.deepcopy(copyMS2Spectrum.NEUCODE_LABEL))

            self.__captainPreprocessMS2_2nd(copyMS2Spectrum)

            # [ASTTENTION] 20221012record
            # 这里可能不需要copy一遍，以后回头检查代码的时候看一看
            res_moz2.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_MOZ))
            res_int2.append(copy.deepcopy(copyMS2Spectrum.LIST_PEAK_INT))
            res_ncl2.append(copy.deepcopy(copyMS2Spectrum.NEUCODE_LABEL))

        return res_moz1, res_int1, res_ncl1, res_moz2, res_int2, res_ncl2

    # 20221123  nothing need adjust
    # piyu preprocessing without de-precursor ions
    def __captainPreprocessMS2_1st(self, dataMS2Spectrum):

        # 20201110  调整执行次序
        # 1. holding top-N peaks
        ori_peaks_num = len(dataMS2Spectrum.LIST_PEAK_INT)

        # len(dataMS2Spectrum.LIST_PEAK_MOZ) ==
        # len(dataMS2Spectrum.LIST_PEAK_INT) ==
        # len(dataMS2Spectrum.NEUCODE_LABEL) == peaks num
        dataMS2Spectrum.NEUCODE_LABEL = [0] * ori_peaks_num

        if dataMS2Spectrum.LIST_PEAK_INT:
            # 若给定参数k则执行保留峰数目为k的操作，且k不小于用户指定数目，否则保留用户指定数目
            if len(dataMS2Spectrum.LIST_PEAK_MOZ) > self.upper_peaks_num1:
                rankListByINT = self.__soldierGetRankListByINT(dataMS2Spectrum)
                self.__soldierDeleteWeakPeak(dataMS2Spectrum, rankListByINT, holdNum=self.upper_peaks_num1)

        # with open(".\\test\\filter600.mgf", "a") as f:
        #     for i in range(len(dataMS2Spectrum.LIST_PRECURSOR_MASS)):
        #         f.write("BEGIN IONS\nTITLE=" + dataMS2Spectrum.LIST_FILE_NAME[i] + "\n")
        #         f.write("CHARGE=" + str(dataMS2Spectrum.LIST_PRECURSOR_CHARGE[i]) + "+\n")
        #         f.write("RTINSECONDS=" + str(dataMS2Spectrum.SCAN_RET_TIME) + "\n")
        #         f.write("PEPMASS=" + "%.6f" % (dataMS2Spectrum.LIST_PRECURSOR_MOZ[i] + 0.0) + "\n")
        #         f.write("\n".join(["%.5f" % (dataMS2Spectrum.LIST_PEAK_MOZ[ii]) + " " + "%.5f" % (dataMS2Spectrum.LIST_PEAK_INT[ii] + 0.0) for ii in range(len(dataMS2Spectrum.LIST_PEAK_MOZ))]) + "\nEND IONS\n")

        # 2. cluster transforming
        self.__soldierGetSingleChargePeaksMS2TESTING(dataMS2Spectrum)

        # for i, t in enumerate(dataMS2Spectrum.NEUCODE_LABEL):
        #     print(i, "\t", t, dataMS2Spectrum.LIST_PEAK_MOZ[i], "\t", dataMS2Spectrum.LIST_PEAK_INT[i])

        # with open(".\\test\\afterMonoTrans.mgf", "a") as f:
        #     for i in range(len(dataMS2Spectrum.LIST_PRECURSOR_MASS)):
        #         f.write("BEGIN IONS\nTITLE=" + dataMS2Spectrum.LIST_FILE_NAME[i] + "\n")
        #         f.write("CHARGE=" + str(dataMS2Spectrum.LIST_PRECURSOR_CHARGE[i]) + "+\n")
        #         f.write("RTINSECONDS=" + str(dataMS2Spectrum.SCAN_RET_TIME) + "\n")
        #         f.write("PEPMASS=" + "%.6f" % (dataMS2Spectrum.LIST_PRECURSOR_MOZ[i] + 0.0) + "\n")
        #         f.write("NEUCODELABEL=" + "".join([str(t) for t in dataMS2Spectrum.NEUCODE_LABEL]) + "\n")
        #         f.write("\n".join(["%.5f" % (dataMS2Spectrum.LIST_PEAK_MOZ[ii]) + " " + "%.5f" % (dataMS2Spectrum.LIST_PEAK_INT[ii] + 0.0) for ii in range(len(dataMS2Spectrum.LIST_PEAK_MOZ))]) + "\nEND IONS\n")


        # 3. hold top-N peaks
        if dataMS2Spectrum.LIST_PEAK_INT:

            if len(dataMS2Spectrum.LIST_PEAK_MOZ) > self.max_remain_peak_num:

                rankListByINT = self.__soldierGetRankListByINT(dataMS2Spectrum)

                self.__soldierDeleteWeakPeak(dataMS2Spectrum, rankListByINT)

        # with open(".\\test\\after350.mgf", "a") as f:
        #     for i in range(len(dataMS2Spectrum.LIST_PRECURSOR_MASS)):
        #         f.write("BEGIN IONS\nTITLE=" + dataMS2Spectrum.LIST_FILE_NAME[i] + "\n")
        #         f.write("CHARGE=" + str(dataMS2Spectrum.LIST_PRECURSOR_CHARGE[i]) + "+\n")
        #         f.write("RTINSECONDS=" + str(dataMS2Spectrum.SCAN_RET_TIME) + "\n")
        #         f.write("PEPMASS=" + "%.6f" % (dataMS2Spectrum.LIST_PRECURSOR_MOZ[i] + 0.0) + "\n")
        #         f.write("NEUCODELABEL=" + "".join([str(t) for t in dataMS2Spectrum.NEUCODE_LABEL]) + "\n")
        #         f.write("\n".join(["%.5f" % (dataMS2Spectrum.LIST_PEAK_MOZ[ii]) + " " + "%.5f" % (dataMS2Spectrum.LIST_PEAK_INT[ii] + 0.0) for ii in range(len(dataMS2Spectrum.LIST_PEAK_MOZ))]) + "\nEND IONS\n")


        # 4. check natural loss peaks
        if self.dp.myCFG.B2_CHECK_NATURAL_LOSS == 1:
            self.__soldierClusterNeutronalLossH2OAndNH3(dataMS2Spectrum)

    # 20221123  nothing need adjust
    # zixuan preprocessing part
    def __captainPreprocessMS2_2nd(self, dataMS2Spectrum):

        # 1. 最大最小归一化到10000再开方
        self.__soldierAbsoluteINT2RelativeINT_MAXMIN(dataMS2Spectrum, 0.5)

        # 2. delete the immonium ions info
        if self.dp.myCFG.B6_ROUND_TWO_REMOVE_IMMONIUM:
            self.__deleteImmoniumIons(dataMS2Spectrum)

        # control local peaks num
        if self.dp.myCFG.B8_ROUND_TWO_MASS_BIN_LENGTH > 1:
            self.__localPeaksNumControl(dataMS2Spectrum, self.dp.myCFG.B8_ROUND_TWO_MASS_BIN_LENGTH, self.dp.myCFG.B7_ROUND_TWO_PEAK_NUM_PER_BIN)

    # 按照强度排序进行遍历
    # 20221123  增加一个新的属性 / 成员，NEUCODE_LABEL
    def __soldierGetSingleChargePeaksMS2TESTING(self, dataMS2Spectrum):

        # check
        delete_index_list = []
        for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
            if dataMS2Spectrum.LIST_PEAK_MOZ[i] < self.dp.myINI.MASS_PROTON_MONO:
                delete_index_list.append(i)
            else:
                break
        if delete_index_list:
            for index in delete_index_list[::-1]:
                dataMS2Spectrum.LIST_PEAK_MOZ.pop(index)
                dataMS2Spectrum.LIST_PEAK_INT.pop(index)
                # dataMS2Spectrum.NEUCODE_LABEL.pop(index)
            delete_index_list = []

        max_charge = max(dataMS2Spectrum.LIST_PRECURSOR_CHARGE)

        if self.dp.myCFG.B4_CHECK_PRECURSOR_CHARGE == 0:
            if max_charge > 1:
                max_charge -= 1

        # [ATTENTION] 20221125
        self.__reshapeNeuCodeLabelPeaks(dataMS2Spectrum, max_charge)
        # [SP_OPERATOR] hold 350 before deconv
        if len(dataMS2Spectrum.LIST_PEAK_MOZ) > self.upper_peaks_num2:
            rankListByINT = self.__soldierGetRankListByINT(dataMS2Spectrum)
            self.__soldierDeleteWeakPeak(dataMS2Spectrum, rankListByINT, holdNum=self.upper_peaks_num2)

        new_peak_int = []
        new_peak_moz = []
        new_nc_label = []
        isotopic_tol = [self.dp.myINI.MASS_NEUTRON_AVRG / c for c in range(1, max_charge + 1)]
        # 1.0030, 0.5015, 0.3334, 0.25075, 0.2016...

        # cluster_start_counter = 0
        charge_state = -1  # 独峰

        while 0 < len(dataMS2Spectrum.LIST_PEAK_MOZ):
            cluster_index_list = []
            # 小于isotopic_tol[-1] - tol 就check下一谱峰
            # 检查开始列表中的任意电荷状态的距离，由index可得电荷状态index+1
            # 超过isotopic_tol[0]就进行归并，并跳转下一峰
            # index = cluster_start_counter
            max_int = -1
            max_int_peak_index = -1
            for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
                if dataMS2Spectrum.LIST_PEAK_INT[i] > max_int:
                    max_int = dataMS2Spectrum.LIST_PEAK_INT[i]
                    max_int_peak_index = i
            # 得到最高峰信号的地址了

            # 而后左右检测是否可有峰簇

            # while tmp_check_index < len(dataMS2Spectrum.LIST_PEAK_MOZ):
            #     if -1 == charge_state:
            #         pass
            # ########################################################
            # not complete

            # -------------------------- right -----------------------------
            # ##############################################################
            tmp_check_index = max_int_peak_index
            # index = 0
            cluster_index_list.append(tmp_check_index)
            cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
            tmp_check_index += 1
            # charge == -1: cluster is not complete

            # if abs(cluster_tail_moz - 830.416) < 0.1 or abs(cluster_tail_moz - 959.415) < 0.1 or abs(cluster_tail_moz - 759.379) < 0.1:
            #     print(cluster_tail_moz)

            while tmp_check_index < len(dataMS2Spectrum.LIST_PEAK_MOZ):

                peak_tolerance = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index] - cluster_tail_moz
                if self.msms_ppm == 1:
                    ppm_tol_da = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index] * self.msms_fraction
                else:
                    ppm_tol_da = self.msms_tol
                prep_tolerance = ppm_tol_da
                if peak_tolerance < isotopic_tol[-1] - prep_tolerance:
                    # 小于最小,跳过去
                    # 要继续找，不要break
                    tmp_check_index += 1

                elif peak_tolerance > isotopic_tol[0] + prep_tolerance:
                    # 大于最大
                    # 不要继续找，要break
                    break

                else:

                    # 先确定电荷状态，再继续cluster构建
                    if charge_state == -1:
                        for tol_index in range(len(isotopic_tol)):
                            if math.isclose(peak_tolerance, isotopic_tol[tol_index], abs_tol=prep_tolerance):
                                charge_state = tol_index + 1
                                # 确定质荷比信息
                                cluster_index_list.append(tmp_check_index)
                                cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                                break
                        if charge_state == -1:
                            # 继续向后寻找（MUST）
                            tmp_check_index += 1
                        else:

                            # 连续向后构造
                            tmp_check_index += 1


                    # 已经确定电荷状态
                    else:
                        # 仍然是我的同位素峰，进入峰簇下标列表
                        if math.isclose(peak_tolerance, isotopic_tol[charge_state - 1], abs_tol=prep_tolerance):

                            # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                            nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[-1]]
                            if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] > 1.2 * nearest_int:
                                tmp_check_index += 1
                                continue
                            cluster_index_list.append(tmp_check_index)
                            cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                            tmp_check_index += 1
                        # 超过了枚举电荷的范围，break
                        elif peak_tolerance - isotopic_tol[charge_state - 1] > prep_tolerance:
                            break
                        # 又不匹配，又不越界，啥也不是，继续走吧
                        else:
                            tmp_check_index += 1
            # while index < len(dataMS2Spectrum.LIST_PEAK_MOZ) over.

            # -------------------------- left ------------------------------
            # ##############################################################
            tmp_check_index = max_int_peak_index - 1
            cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[max_int_peak_index]
            while tmp_check_index >= 0 and dataMS2Spectrum.LIST_PEAK_MOZ:

                peak_tolerance = cluster_left_moz - dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                if self.msms_ppm == 1:
                    ppm_tol_da = cluster_left_moz * self.msms_fraction
                else:
                    ppm_tol_da = self.msms_tol
                prep_tolerance = ppm_tol_da
                if peak_tolerance < isotopic_tol[-1] - prep_tolerance:
                    # 小于最小,跳过去
                    # 要继续找，不要break
                    tmp_check_index -= 1

                elif peak_tolerance > isotopic_tol[0] + prep_tolerance:
                    # 大于最大
                    # 不要继续找，要break
                    break

                else:

                    # 先确定电荷状态，再继续cluster构建
                    if charge_state == -1:
                        for tol_index in range(len(isotopic_tol)):
                            if math.isclose(peak_tolerance, isotopic_tol[tol_index], abs_tol=prep_tolerance):

                                # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                                # nearest_int: cluster list中最左端的谱峰信号的强度值
                                nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[0]]
                                # if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.2 * nearest_int:
                                if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.3 * nearest_int:
                                    # break
                                    tmp_check_index -= 1
                                    continue

                                charge_state = tol_index + 1
                                # 确定质荷比信息
                                # cluster_index_list.append(tmp_check_index)
                                cluster_index_list = [tmp_check_index] + cluster_index_list
                                cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                                break
                        if charge_state == -1:
                            # 继续向前寻找（MUST）
                            tmp_check_index -= 1
                        else:

                            # 连续向后构造
                            tmp_check_index -= 1


                    # 已经确定电荷状态
                    else:
                        # 仍然是我的同位素峰，进入峰簇下标列表
                        if math.isclose(peak_tolerance, isotopic_tol[charge_state - 1], abs_tol=prep_tolerance):
                            # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                            nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[0]]
                            # if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.2 * nearest_int:
                            if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.4 * nearest_int:
                                # break
                                tmp_check_index -= 1
                                continue
                            cluster_index_list = [tmp_check_index] + cluster_index_list
                            cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                            tmp_check_index -= 1
                        # 超过了枚举电荷的范围，break
                        elif peak_tolerance - isotopic_tol[charge_state - 1] > prep_tolerance:
                            break
                        # 又不匹配，又不越界，啥也不是，继续走吧
                        else:
                            tmp_check_index -= 1

            # --------------------------------------------------------------

            # cluster  [如果是独峰，那么用双峰信息来枚举电荷！]
            # cluster 构造结束，开始收工
            if charge_state == -1:
                # 删除，添加
                add_moz = dataMS2Spectrum.LIST_PEAK_MOZ.pop(cluster_index_list[0])
                add_int = dataMS2Spectrum.LIST_PEAK_INT.pop(cluster_index_list[0])
                add_ncl = dataMS2Spectrum.NEUCODE_LABEL.pop(cluster_index_list[0])

                # new_peak_moz.append(add_moz)
                # new_peak_int.append(add_int)
                # new_nc_label.append(add_ncl)
                if add_ncl > 0:
                    for tmp_c in range(1, add_ncl+1):
                        new_peak_moz.append((add_moz - self.dp.myINI.MASS_PROTON_MONO) * tmp_c + self.dp.myINI.MASS_PROTON_MONO)
                        new_peak_int.append(add_int * 2)
                        new_nc_label.append(1)
                else:
                    new_peak_moz.append(add_moz)
                    new_peak_int.append(add_int)
                    new_nc_label.append(0)
            else:

                # ##########################################
                # 20210809 ATTENTION ATTENTION ATTENTION ###
                # pop时，一定一定一定要注意地址的问题！ ####
                # ##########################################
                # --------------- QUESTION --------------- #
                # 这里要把mono峰pop出来，但是一定要放在最后#
                # 否则会改变其他谱峰的地址，导致pop峰错误  #
                # 再就是强度和质荷比错开了，导致tag提取错误#
                # ##########################################
                add_int = 0
                add_ncl = 0
                if self.dp.myCFG.B1_ADD_ISOTOPE_INTENSITY:
                    for buf_index in reversed(cluster_index_list[1:]):
                        try:
                            # dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                            # add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)

                            dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                            tmpT = dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)
                            add_int += tmpT
                            if tmpT < dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[0]]:
                                continue
                            add_ncl = max(dataMS2Spectrum.NEUCODE_LABEL.pop(buf_index), add_ncl)

                        except:
                            pass
                # ================== DO NOT ADD INTENSITY! ====================
                else:
                    for buf_index in reversed(cluster_index_list[1:]):
                        try:
                            dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                            tmpT = dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)
                            if tmpT < dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[0]]:
                                continue
                            add_ncl = max(dataMS2Spectrum.NEUCODE_LABEL.pop(buf_index), add_ncl)
                        except:
                            pass

                # 把mono峰的moz提出放在最后，包括mono对应的强度部分也是
                # 如果希望对add的强度做一些操作，可以在for loop try里头去整
                add_moz = dataMS2Spectrum.LIST_PEAK_MOZ.pop(cluster_index_list[0]) * charge_state
                add_moz -= (charge_state - 1) * self.dp.myINI.MASS_PROTON_MONO
                add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(cluster_index_list[0])
                add_ncl = max(dataMS2Spectrum.NEUCODE_LABEL.pop(cluster_index_list[0]), add_ncl)
                add_ncl = 1 if add_ncl > 0 else 0
                new_peak_moz.append(add_moz)
                if add_ncl == 1:
                    new_peak_int.append(add_int * 2)
                else:
                    new_peak_int.append(add_int)
                new_nc_label.append(add_ncl)

            charge_state = -1
            # cluster 处理结束
        # while over

        # -------------------------------------------------------------------

        # 排序，检测合并
        index_order = np.argsort(new_peak_moz)
        new_peak_moz = [new_peak_moz[index] for index in index_order]
        new_peak_int = [new_peak_int[index] for index in index_order]
        new_nc_label = [new_nc_label[index] for index in index_order]

        # -------------------------------------------------------------------

        output_peak_moz = []
        output_peak_int = []
        output_peak_ncl = []

        # 下方：合并转换的谱峰，有可能会影响精度
        add_moz = 0
        add_int = 0
        add_ncl = 0
        i = 0
        jump_set = set()
        while i < len(new_peak_moz) - 1:

            if i in jump_set:
                i += 1
                continue

            add_moz = new_peak_moz[i]
            add_int = new_peak_int[i]
            add_ncl = new_nc_label[i]
            for j in range(i + 1, len(new_peak_moz)):
                if self.msms_ppm == 1:
                    prep_tolerance = new_peak_moz[j] * self.msms_fraction
                    # prep_tolerance = new_peak_moz[j] * self.msms_fraction * 2

                else:
                    prep_tolerance = self.msms_tol
                    # prep_tolerance = self.msms_tol * 2
                # prep_tolerance = new_peak_moz[j] * 2e-5

                if abs(new_peak_moz[i] - new_peak_moz[j]) < prep_tolerance:
                    add_moz = add_moz * add_int + new_peak_moz[j] * new_peak_int[j]
                    add_int += new_peak_int[j]
                    add_moz /= add_int
                    add_ncl = max(add_ncl, new_nc_label[j])
                    i = j
                    jump_set.add(j)
                # 仅看左右最远的0.02，两两之间就不看了
                else:
                    output_peak_moz.append(add_moz)
                    output_peak_int.append(add_int)
                    output_peak_ncl.append(add_ncl)
                    i = j
                    break

            # if abs(new_peak_moz[i] - new_peak_moz[-1]) < prep_tolerance:
            #     output_peak_moz.append(add_moz)
            #     output_peak_int.append(add_int)
            #     i = j
            #     break

        if add_moz in output_peak_moz:
            pass
        else:
            output_peak_moz.append(add_moz)
            output_peak_int.append(add_int)
            output_peak_ncl.append(add_ncl)

        # 活久见，图里没谱峰哒，就算辽叭
        if len(new_peak_moz) == 0:
            pass
        # 检测一下最后一根谱峰是否是可以添加的信息
        elif jump_set:
            if max(jump_set) == len(new_peak_moz):
                pass
            else:
                output_peak_moz.append(new_peak_moz[-1])
                output_peak_int.append(new_peak_int[-1])
                output_peak_ncl.append(new_nc_label[-1])
        else:
            output_peak_moz.append(new_peak_moz[-1])
            output_peak_int.append(new_peak_int[-1])
            output_peak_ncl.append(new_nc_label[-1])

        # 上方：合并转换的谱峰，有可能会影响精度
        # output_peak_moz = new_peak_moz
        # output_peak_int = new_peak_int

        dataMS2Spectrum.LIST_PEAK_MOZ = output_peak_moz
        dataMS2Spectrum.LIST_PEAK_INT = output_peak_int
        dataMS2Spectrum.NEUCODE_LABEL = output_peak_ncl

    # 20221123  类比上面的预处理流程写的方法，未完成！
    # 但是顺序检测就好啦！不用按强度噻！
    def __reshapeNeuCodeLabelPeaks(self, dataMS2Spectrum, maxCharge:int):

        biggest_c = min(self.DOUBLE_PEAK_MAX_C, maxCharge)
        TOL_HASH_TAB = self.DOUBLE_TOL_HASH_TAB_LIST[biggest_c]

        i = 0
        idx_boundary = len(dataMS2Spectrum.LIST_PEAK_MOZ)

        p_LIST_MOZ = dataMS2Spectrum.LIST_PEAK_MOZ
        p_LIST_INT = dataMS2Spectrum.LIST_PEAK_INT
        heavy_idx = set()  # if light peak in, pop it; add heavy peak in;
        light_int = [0] * idx_boundary
        while i < idx_boundary-1:

            i_moz = p_LIST_MOZ[i]
            i_int = p_LIST_INT[i]
            if i_moz > self.DOUBLE_PEAK_BOUND:
                break

            for j in range(i+1, idx_boundary):

                tol = p_LIST_MOZ[j] - p_LIST_MOZ[i]
                # gap is too big! do not go on again!
                if tol > self.DOUBLE_PEAK_TOL_MAX:
                    break

                # I got a couple of peaks!! In law of mass tol!! XD
                if int(tol * self.fold + 0.5) in TOL_HASH_TAB:
                    if (self.MAX_INTEN_RATIO < p_LIST_INT[j]/ i_int) or (self.MAX_INTEN_RATIO < i_int / p_LIST_INT[j]):
                        continue
                    p_LIST_INT[i] = max(light_int[i], p_LIST_INT[i], p_LIST_INT[j])

                    heavy_idx.add(j)
                    # special scheme: heavy & light
                    if i in heavy_idx:
                        heavy_idx.remove(i)

                    # 只当是提前判断电荷辣！0.036Da 统一按照 1+ 算
                    if tol > self.DOUBLE_PEAK_SHIFT:
                        dataMS2Spectrum.NEUCODE_LABEL[i] = 1
                    else:
                        error = [1] * (biggest_c+1)
                        for c in range(1, biggest_c+1):
                            error[c] = min(error[c],
                                           abs(tol - self.DOUBLE_PEAK_SHIFT_C[c]),
                                           abs(tol - self.DOUBLE_PEAK_SHIFT2_C[c]))
                        dataMS2Spectrum.NEUCODE_LABEL[i] = error.index(min(error))  # max(error.index(min(error)), dataMS2Spectrum.NEUCODE_LABEL[i])

            # while loop is over
            # we need a newer i
            i += 1

        for del_idx in sorted(heavy_idx, reverse=True):
            dataMS2Spectrum.LIST_PEAK_MOZ.pop(del_idx)
            dataMS2Spectrum.LIST_PEAK_INT.pop(del_idx)
            dataMS2Spectrum.NEUCODE_LABEL.pop(del_idx)

        # with open(".\\test\\Y_init_judge.mgf", "a") as f:
        #     for i in range(len(dataMS2Spectrum.LIST_PRECURSOR_MASS)):
        #         f.write("BEGIN IONS\nTITLE=" + dataMS2Spectrum.LIST_FILE_NAME[i] + "\n")
        #         f.write("CHARGE=" + str(dataMS2Spectrum.LIST_PRECURSOR_CHARGE[i]) + "+\n")
        #         f.write("RTINSECONDS=" + str(dataMS2Spectrum.SCAN_RET_TIME) + "\n")
        #         f.write("PEPMASS=" + "%.6f" % (dataMS2Spectrum.LIST_PRECURSOR_MOZ[i] + 0.0) + "\n")
        #         f.write("NEUCODELABEL=" + "".join([str(t) for t in dataMS2Spectrum.NEUCODE_LABEL]) + "\n")
        #         f.write("\n".join(["%.5f" % (dataMS2Spectrum.LIST_PEAK_MOZ[ii]) + " " + "%.5f" % (dataMS2Spectrum.LIST_PEAK_INT[ii] + 0.0) for ii in range(len(dataMS2Spectrum.LIST_PEAK_MOZ))]) + "\nEND IONS\n")
        # print(len(dataMS2Spectrum.NEUCODE_LABEL), "-"*20)
        # for i, t in enumerate(dataMS2Spectrum.NEUCODE_LABEL):
        #     print(i, "\t", t, "\t", dataMS2Spectrum.LIST_PEAK_MOZ[i], "\t", dataMS2Spectrum.LIST_PEAK_INT[i])

        return



    # 同位素峰簇转换结束后，检测失水失氨峰
    # 按照强度为顺序，“向左”对失水失氨的质量依次进行检测
    # 将符合质量的谱峰，构成一个cluster
    # 因为谱峰之间的质量顺序是：ori_peak - H2O, ori_peak - NH3, ori_peak
    # 所以构造的谱峰也将按照此规律检测
    def __soldierClusterNeutronalLossH2OAndNH3(self, dataMS2Spectrum):

        new_peak_int = []
        new_peak_moz = []
        new_nc_label = []

        while 0 < len(dataMS2Spectrum.LIST_PEAK_MOZ):
            cluster_index_list = []

            # 这里进行了修改，即得到最后一根谱峰，然后从后往前找-NH3和-H2O的峰
            tmp_check_index = len(dataMS2Spectrum.LIST_PEAK_MOZ) - 1
            cluster_index_list.append(tmp_check_index)
            cluster_tail_int = dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index]

            # -------------------------- left ------------------------------
            # ##############################################################
            cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
            tmp_check_index -= 1

            while tmp_check_index >= 0 and dataMS2Spectrum.LIST_PEAK_MOZ:

                peak_tolerance = cluster_tail_moz - dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                if self.msms_ppm == 1:
                    # self.msms_fraction
                    ppm_tol_da = cluster_tail_moz * self.msms_fraction
                else:
                    ppm_tol_da = self.msms_tol

                prep_tolerance = ppm_tol_da

                if peak_tolerance < self.PROTON_MASS_NH3 - prep_tolerance:
                    # 小于最小,跳过去
                    # 要继续找，不要break
                    tmp_check_index -= 1

                elif peak_tolerance > self.PROTON_MASS_H2O + prep_tolerance:
                    # 大于最大
                    # 不要继续找，要break
                    break

                else:
                    # 进入自然损失簇下标列表
                    if math.isclose(peak_tolerance, self.PROTON_MASS_NH3, abs_tol=prep_tolerance):

                        cluster_left_int = dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index]
                        if cluster_tail_int > 0.2 * cluster_left_int:
                            cluster_index_list = [tmp_check_index] + cluster_index_list
                        tmp_check_index -= 1
                        # 进入自然损失簇下标列表

                    elif math.isclose(peak_tolerance, self.PROTON_MASS_H2O, abs_tol=prep_tolerance):

                        cluster_left_int = dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index]
                        if cluster_tail_int > 0.2 * cluster_left_int:
                            cluster_index_list = [tmp_check_index] + cluster_index_list
                        tmp_check_index -= 1

                    # 又不匹配，又不越界，啥也不是，继续走吧
                    else:
                        tmp_check_index -= 1

            # --------------------------------------------------------------

            # cluster
            # cluster 构造结束，开始收工
            add_moz = dataMS2Spectrum.LIST_PEAK_MOZ[cluster_index_list[-1]]
            add_ncl = dataMS2Spectrum.NEUCODE_LABEL[cluster_index_list[-1]]
            if self.dp.myCFG.B1_ADD_ISOTOPE_INTENSITY:
                add_int = 0
                for buf_index in sorted(cluster_index_list, reverse=True):
                    try:
                        if add_ncl == dataMS2Spectrum.NEUCODE_LABEL[buf_index]:
                            dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                            add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)
                            dataMS2Spectrum.NEUCODE_LABEL.pop(buf_index)
                            # add_ncl = check_ncl if check_ncl > add_ncl else add_ncl
                    except:
                        pass
            else:
                add_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[-1]]
                for buf_index in sorted(cluster_index_list, reverse=True):
                    try:
                        if add_ncl == dataMS2Spectrum.NEUCODE_LABEL[buf_index]:
                            dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                            dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)
                            dataMS2Spectrum.NEUCODE_LABEL.pop(buf_index)
                    except:
                        pass

            # 如果希望对add的强度做一些操作，可以在for loop try里头去整

            new_peak_moz.append(add_moz)
            new_peak_int.append(add_int)
            new_nc_label.append(add_ncl)
            # cluster 处理结束
        # while over

        # -------------------------------------------------------------------

        # 排序，检测合并
        if new_peak_moz:
            index_order = np.argsort(new_peak_moz)
            new_peak_moz = [new_peak_moz[index] for index in index_order]
            new_peak_int = [new_peak_int[index] for index in index_order]
            new_nc_label = [new_nc_label[index] for index in index_order]

        dataMS2Spectrum.LIST_PEAK_MOZ = new_peak_moz
        dataMS2Spectrum.LIST_PEAK_INT = new_peak_int
        dataMS2Spectrum.NEUCODE_LABEL = new_nc_label

    # 20221123  nothing need adjust
    def __soldierGetRankListByINT(self, dataMS2Spectrum):

        tmpMS2_np = np.array(dataMS2Spectrum.LIST_PEAK_INT[:])

        rankListByINT = np.argsort(tmpMS2_np)
        # 注意，这里的索引是按照强度，从小到大排列的

        return rankListByINT

    # adjust over
    def __soldierRemovePrecursorPeak(self, dataMS2Spectrum, inputPrecursorMass):

        soldier = inputPrecursorMass - 50
        # eg. Precursor -H2O / -NH3, Precursor - 2*H2O
        for i in reversed(range(len(dataMS2Spectrum.LIST_PEAK_MOZ))):

            if dataMS2Spectrum.LIST_PEAK_MOZ[i] >= soldier:
                dataMS2Spectrum.LIST_PEAK_MOZ.pop()
                dataMS2Spectrum.LIST_PEAK_INT.pop()
                dataMS2Spectrum.NEUCODE_LABEL.pop()

            else:
                break

        # 20210825 -----------------------------------------------------
        # 去掉质量小于18Da的分子
        soldier = 20

        while dataMS2Spectrum.LIST_PEAK_MOZ:

            if dataMS2Spectrum.LIST_PEAK_MOZ[0] <= soldier:
                dataMS2Spectrum.LIST_PEAK_MOZ.pop(0)
                dataMS2Spectrum.LIST_PEAK_INT.pop(0)
                dataMS2Spectrum.NEUCODE_LABEL.pop(0)

            else:
                break

    # adjust over
    def __soldierDeleteWeakPeak(self, dataMS2Spectrum, rankListByINT, holdNum=0):

        # 获得删除列表，且已经逆序排好顺序，便于删除
        # 20210902 修改判0为判断大小，避免or操作，合并检测的包含关系
        if holdNum < self.max_remain_peak_num:
            deleteList = reversed(np.sort(rankListByINT[:-self.max_remain_peak_num]))
        else:
            deleteList = reversed(np.sort(rankListByINT[:-holdNum]))
        # 逆序删除，不影响索引
        for i in deleteList:
            dataMS2Spectrum.LIST_PEAK_MOZ.pop(i)
            dataMS2Spectrum.LIST_PEAK_INT.pop(i)
            dataMS2Spectrum.NEUCODE_LABEL.pop(i)

    # 20221123  nothing need adjust
    def __soldierINT2RANK(self, dataMS2Spectrum):

        tmpMS2INTList = np.array(dataMS2Spectrum.LIST_PEAK_INT[:])
        # 读取MS2的强度信息

        rankIndexList = np.argsort(-tmpMS2INTList)
        # 将强度从大到小排列，保留地址
        # 加个负号即可

        rank = 1
        for index in rankIndexList:
            dataMS2Spectrum.LIST_PEAK_INT[index] = rank
            rank += 1

    # 20221123  nothing need adjust
    def __soldierAbsoluteINT2RelativeINT(self, dataMS2Spectrum):

        if not dataMS2Spectrum.LIST_PEAK_INT:
            # print("Spectrum empty")
            pass
        # 如果列表不空的话（话说为什么会空呢。。。全为precursor？）
        else:
            maxAbsoluteINTValue = max(dataMS2Spectrum.LIST_PEAK_INT[:])

            # 使用numpy的广播来代替此处的for循环，速度会更快
            for i in range(len(dataMS2Spectrum.LIST_PEAK_INT)):

                tmpAbsINT = dataMS2Spectrum.LIST_PEAK_INT[i]

                dataMS2Spectrum.LIST_PEAK_INT[i] = tmpAbsINT / maxAbsoluteINTValue * 100

    # 20221123  nothing need adjust
    def __soldierAbsoluteINT2RelativeINT_MAXMIN(self, dataMS2Spectrum, exp):

        if not dataMS2Spectrum.LIST_PEAK_INT:
            # print("Spectrum empty")
            pass
        # 如果列表不空的话（话说为什么会空呢。。。全为precursor？）
        else:
            maxAbsoluteINTValue = max(dataMS2Spectrum.LIST_PEAK_INT)
            minAbsoluteINTValue = min(dataMS2Spectrum.LIST_PEAK_INT)

            if maxAbsoluteINTValue == minAbsoluteINTValue:
                # 退化为最大值归一化，避免除零错误
                for i in range(len(dataMS2Spectrum.LIST_PEAK_INT)):
                    tmpAbsINT = dataMS2Spectrum.LIST_PEAK_INT[i]

                    dataMS2Spectrum.LIST_PEAK_INT[i] = (tmpAbsINT / maxAbsoluteINTValue * 10000) ** exp

            else:
                # 最大最小归一化
                maxAbsoluteINTValue -= minAbsoluteINTValue

                # 使用numpy的广播来代替此处的for循环，速度会更快
                for i in range(len(dataMS2Spectrum.LIST_PEAK_INT)):

                    tmpAbsINT = dataMS2Spectrum.LIST_PEAK_INT[i] - minAbsoluteINTValue

                    dataMS2Spectrum.LIST_PEAK_INT[i] = (tmpAbsINT / maxAbsoluteINTValue * 10000) ** exp

    # 20221123  adjust over
    def __localPeaksNumControl(self, dataMS2Spectrum, bin_len, peak_num):
        ...
        res_moz = []
        res_int = []
        res_ncl=  []  # ncl means neucode label

        # 当列表变为空的时候，我们就不用再循环啦
        while dataMS2Spectrum.LIST_PEAK_INT:
            # 先确定下来因数是多少
            old_soldier = dataMS2Spectrum.LIST_PEAK_MOZ[0] // bin_len
            # 初始化为最小的地址
            tmp_idx = 0

            # find an index range per bin_len
            # 找到第一个和我们预先设定的因数不一致的数字的地址
            for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
                # 一直没找到，那就一直记录即可 -------------
                if old_soldier == (dataMS2Spectrum.LIST_PEAK_MOZ[i] // bin_len):
                    tmp_idx = i
                # 找到了！我们直接break！
                else:
                    break

            # 此时，地址代表了 peak num + 1
            # 因此当tmp_idx 小于 peak_num 时，满足谱峰过滤的要求，直接加进去即可
            if tmp_idx < peak_num:
                res_moz += dataMS2Spectrum.LIST_PEAK_MOZ[:tmp_idx+1]
                dataMS2Spectrum.LIST_PEAK_MOZ = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_idx+1:]
                res_int += dataMS2Spectrum.LIST_PEAK_INT[:tmp_idx+1]
                dataMS2Spectrum.LIST_PEAK_INT = dataMS2Spectrum.LIST_PEAK_INT[tmp_idx+1:]
                res_ncl += dataMS2Spectrum.NEUCODE_LABEL[:tmp_idx+1]
                dataMS2Spectrum.NEUCODE_LABEL = dataMS2Spectrum.NEUCODE_LABEL[tmp_idx+1:]

            # 当前该 bin 中的谱峰数目过多，需要选择最强的peak_num根峰
            else:
                tmp_moz = dataMS2Spectrum.LIST_PEAK_MOZ[:tmp_idx+1]
                tmp_int = dataMS2Spectrum.LIST_PEAK_INT[:tmp_idx+1]
                tmp_ncl = dataMS2Spectrum.NEUCODE_LABEL[:tmp_idx + 1]
                dataMS2Spectrum.LIST_PEAK_MOZ = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_idx+1:]
                dataMS2Spectrum.LIST_PEAK_INT = dataMS2Spectrum.LIST_PEAK_INT[tmp_idx+1:]
                dataMS2Spectrum.NEUCODE_LABEL = dataMS2Spectrum.NEUCODE_LABEL[tmp_idx + 1:]

                # 去除强度低的谱峰，这样添加到res_moz / res_int 中时，moz还能保证有序
                while len(tmp_moz) != peak_num:
                    min_idx = tmp_int.index(min(tmp_int))
                    tmp_int.pop(min_idx)
                    tmp_moz.pop(min_idx)
                    tmp_ncl.pop(min_idx)
                res_moz += tmp_moz
                res_int += tmp_int
                res_ncl += tmp_ncl

        dataMS2Spectrum.LIST_PEAK_MOZ = res_moz
        dataMS2Spectrum.LIST_PEAK_INT = res_int
        dataMS2Spectrum.NEUCODE_LABEL = res_ncl


    # 20221123  nothing need adjust
    # 获得亚胺离子的质量信息，并生成哈希值的字典
    # 子璇在这里的默认误差范围是0.02Da
    # int((mass+-delta) * 1000)
    def __getImmoniumIonsHashTab(self):
        # self.fold = 1000
        out_set = set()

        # [hash] imonium ions masses with mass tolerance(0.02Da)
        abs_delta_hash = toolMyIntRound(0.02 * self.fold)
        for aa in self.dp.myINI.DICT1_AA_MASS:
            immonium_mass = self.dp.myINI.DICT1_AA_MASS[aa] - self.PROTON_MASS_CO + self.dp.myINI.MASS_PROTON_MONO
            if immonium_mass <= 0:
                continue
            aa_mass_hash = toolMyIntRound(immonium_mass * self.fold)
            out_set.update(set([it for it in range(aa_mass_hash - abs_delta_hash, aa_mass_hash +abs_delta_hash + 1)]))

        # [hash] relative ions masses of immonium ions with mass tolerance(0.05Da)
        abs_delta_hash = toolMyIntRound(0.05 * self.fold)
        tmp_lst = [41.05, 44.05, 55.05, 56.05, 59.05, 61.05, 69.05, 70.05, 72.05, 73.05,
             77.05, 82.05, 84.05, 87.05, 91.05, 100.05, 107.05, 112.05, 117.05, 121.05,
             123.05, 129.05, 130.05, 132.05, 138.05, 166.05, 170.05, 171.05]
        for rel_mass in tmp_lst:
            rel_mass_hash = toolMyIntRound(rel_mass * self.fold)
            out_set.update(set([it for it in range(rel_mass_hash - abs_delta_hash, rel_mass_hash + abs_delta_hash + 1)]))

        return out_set

    # 20221123  adjust over
    def __deleteImmoniumIons(self, dataMS2Spectrum):

        soldier_idx = 0
        for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
            if dataMS2Spectrum.LIST_PEAK_MOZ[i] > 200:
                soldier_idx = i
                break

        # 存在谱峰的质量是小于200的
        if soldier_idx != 0:
            # 逆序删除，不影响地址信息
            for i in reversed(range(soldier_idx)):
                if toolMyIntRound(dataMS2Spectrum.LIST_PEAK_MOZ[i] * self.fold) in self.IMMONIUM_HASH_TAB:
                    dataMS2Spectrum.LIST_PEAK_MOZ.pop(i)
                    dataMS2Spectrum.LIST_PEAK_INT.pop(i)
                    dataMS2Spectrum.NEUCODE_LABEL.pop(i)


    def __getDoublePeakTolHashTab(self):

        res = [None]  #  res[0] 是不存在滴，因为电荷不为0

        # 提前计算出来一个固定的质量（相对也让他变成固定的），用于快速的 check 双峰
        # 这里的质荷比，其实是经过缩小的，但是我也不敢缩小的太多
        if self.msms_ppm == 1:
            abs_tol = self.DOUBLE_PEAK_BOUND * self.DOUBLE_PEAK_TOL_M / 1e6
            int_del = int(abs_tol * self.fold + 0.5)  # 四舍五入
        else:
            abs_tol = self.DOUBLE_PEAK_TOL_M
            int_del = int(abs_tol * self.fold + 0.5)  # 四舍五入

        # 最高电荷状态（1+，2+，...）
        for c_stat in range(1, self.DOUBLE_PEAK_MAX_C+1):

            tmp_rec = set()

            # 从1电荷到最高电荷状态（1+，2+，...）
            for i in range(1, c_stat+1):
                # 不含漏切，那么双峰的理论差值就是 0.036Da / c
                tmp_hash_num = int((self.DOUBLE_PEAK_SHIFT / i) * self.fold + 0.5)
                for tmp_delta in range(-int_del, int_del + 1):
                    tmp_rec.add(tmp_hash_num+tmp_delta)

                # 含一个漏切，则双峰的理论差值就是 0.036Da * 2 / c
                tmp_hash_num = int((self.DOUBLE_PEAK_SHIFT2 / i) * self.fold + 0.5)
                for tmp_delta in range(-int_del, int_del + 1):
                    tmp_rec.add(tmp_hash_num+tmp_delta)

            res.append(tmp_rec)

        return res


# 20230216 for xueli

class CFunctionPreprocessForXueLi:

    def __init__(self, inputDP):
        self.dp = inputDP
        # important for zixuan
        self.max_remain_peak_num = self.dp.myCFG.B5_ROUND_ONE_HOLD_PEAK_NUM
        # self.tolerance = 0.02  # Da
        self.msms_tol = self.dp.myCFG.D5_MSMS_TOL
        self.msms_ppm = self.dp.myCFG.D6_MSMS_TOL_PPM
        self.upper_peaks_num = 350
        if self.msms_ppm == 1:
            self.msms_fraction = self.msms_tol / 1e6
        else:
            self.msms_fraction = 1
        self.fold = 1e5
        # mass check
        self.PROTON_MASS_C = self.dp.myINI.DICT0_ELEMENT_MASS["C"][0]
        self.PROTON_MASS_H = self.dp.myINI.DICT0_ELEMENT_MASS["H"][0]
        self.PROTON_MASS_O = self.dp.myINI.DICT0_ELEMENT_MASS["O"][0]
        self.PROTON_MASS_CO = self.PROTON_MASS_C + self.PROTON_MASS_O
        self.PROTON_MASS_OH = self.PROTON_MASS_H + self.PROTON_MASS_O
        self.PROTON_MASS_H2O = self.PROTON_MASS_H + self.PROTON_MASS_OH
        self.PROTON_MASS_NH3 = self.PROTON_MASS_H * 3 + self.dp.myINI.DICT0_ELEMENT_MASS["N"][0]

        # 20230113  special for test [added code in remove precursor mass]
        self.PROTON_MASS_PROTON = self.dp.myINI.MASS_PROTON_MONO
        self.PROTON_MASS_H2O_PROTON = self.PROTON_MASS_H2O + self.dp.myINI.MASS_PROTON_MONO

        self.IMMONIUM_HASH_TAB = self.__getImmoniumIonsHashTab()

        # ====================== can use! ===========================

        self.moz_filtered_1 = []  # 350
        self.int_filtered_1 = []  # 350

        self.moz_iso_cluster = [[]] # [[], [], [], [], ...]  # 同位素峰簇的质荷比列表
        self.int_iso_cluster = [[]] # [[], [], [], [], ...]  # 同位素峰簇的强度列表
        self.c_label_cluster = [0] # [int, int, int, ....]  # 同位素峰簇的电荷

        # ===================== cannot use ==========================

        self.moz_filtered_u = []  # user-defined filtered
        self.int_filtered_u = []  # user-defined filtered

        self.moz_immonium = []    # 亚胺离子
        self.int_immonium = []    # 亚胺离子

        self.moz_filtered_2 = []  # 2nd - local filtered peaks
        self.int_filtered_2 = []  # 2nd - local filtered peaks

    def preprocesslabel(self, dataMS2Spectrum):

        # do not change the ori data
        prep_copy = copy.deepcopy(dataMS2Spectrum)

        label_data = self.__captainPreprocessMS2_1st(prep_copy)

        return label_data

    # piyu preprocessing without de-precursor ions
    def __captainPreprocessMS2_1st(self, dataMS2Spectrum):

        # 20201110  调整执行次序
        # 1. holding top-N peaks
        ori_peaks_num = len(dataMS2Spectrum.LIST_PEAK_INT)

        if dataMS2Spectrum.LIST_PEAK_INT:

            # 若给定参数k则执行保留峰数目为k的操作，且k不小于用户指定数目，否则保留用户指定数目
            if len(dataMS2Spectrum.LIST_PEAK_MOZ) > self.upper_peaks_num:

                rankListByINT = self.__soldierGetRankListByINT(dataMS2Spectrum)

                self.__soldierDeleteWeakPeak(dataMS2Spectrum, rankListByINT, holdNum=self.upper_peaks_num)

        # 2. cluster transforming
        self.__soldierGetSingleChargePeaksMS2TESTING(dataMS2Spectrum)

        # 3. hold top-N peaks
        # if dataMS2Spectrum.LIST_PEAK_INT:
        #
        #     if len(dataMS2Spectrum.LIST_PEAK_MOZ) > self.max_remain_peak_num:
        #
        #         rankListByINT = self.__soldierGetRankListByINT(dataMS2Spectrum)
        #
        #         self.__soldierDeleteWeakPeak(dataMS2Spectrum, rankListByINT)

        # 4. check natural loss peaks
        # if self.dp.myCFG.B2_CHECK_NATURAL_LOSS == 1:
        #     self.__soldierClusterNeutronalLossH2OAndNH3(dataMS2Spectrum)
        res = CPrepLabel()
        self.__soldierFillLabel(res)
        return res


    def __soldierFillLabel(self, label:CPrepLabel):
        label.FILTERED = []  # initial
        label.CLUSTER = []  # initial
        label.CHARGE = []  # initial

        # merge ----
        tmpMOZ = [k for k in self.moz_filtered_1]
        label.FILTERED = [0] * len(self.moz_filtered_1)
        label.CLUSTER = [0] * len(self.moz_filtered_1)
        label.CHARGE = [0] * len(self.moz_filtered_1)
        for i, clu in enumerate(self.moz_iso_cluster):
            label.FILTERED += [1] * len(clu)
            label.CLUSTER += [i] * len(clu)  # 拼接cluster的编号信息
            c = self.c_label_cluster[i]      # 获取电荷状态
            label.CHARGE += [c] * len(clu)   # 拼接这些 moz 对应的电荷
            tmpMOZ += clu                    # 拼接cluster中的 moz 信息
        # sort -----

        sorted_idx = np.argsort(tmpMOZ)

        label.CHARGE = [label.CHARGE[kk] for kk in sorted_idx]
        label.CLUSTER = [label.CLUSTER[kk] for kk in sorted_idx]
        label.FILTERED = [label.FILTERED[kk] for kk in sorted_idx]

        # after use, clean it!
        self.moz_filtered_1 = []  # 350
        self.int_filtered_1 = []  # 350

        self.moz_iso_cluster = [[]]  # [[], [], [], [], ...]  # 同位素峰簇的质荷比列表
        self.int_iso_cluster = [[]]  # [[], [], [], [], ...]  # 同位素峰簇的强度列表
        self.c_label_cluster = [0]  # [int, int, int, ....]  # 同位素峰簇的电荷


    # zixuan preprocessing part
    def __captainPreprocessMS2_2nd(self, dataMS2Spectrum):

        # 1. 最大最小归一化到10000再开方
        self.__soldierAbsoluteINT2RelativeINT_MAXMIN(dataMS2Spectrum, 0.5)

        # 2. delete the immonium ions info
        if self.dp.myCFG.B6_ROUND_TWO_REMOVE_IMMONIUM:
            self.__deleteImmoniumIons(dataMS2Spectrum)

        # control local peaks num
        if self.dp.myCFG.B8_ROUND_TWO_MASS_BIN_LENGTH > 1:
            self.__localPeaksNumControl(dataMS2Spectrum, self.dp.myCFG.B8_ROUND_TWO_MASS_BIN_LENGTH, self.dp.myCFG.B7_ROUND_TWO_PEAK_NUM_PER_BIN)

    # 按照强度排序进行遍历
    # 效果可能会好些(实际上并不是
    # 20210726  该函数固定了20ppm的检测tol，若用户设定可变，则需修改
    def __soldierGetSingleChargePeaksMS2TESTING(self, dataMS2Spectrum):

        # check
        delete_index_list = []
        for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
            if dataMS2Spectrum.LIST_PEAK_MOZ[i] < self.dp.myINI.MASS_PROTON_MONO:
                delete_index_list.append(i)
            else:
                break
        if delete_index_list:
            for index in delete_index_list[::-1]:
                self.moz_filtered_1.append(dataMS2Spectrum.LIST_PEAK_MOZ.pop(index))
                self.int_filtered_1.append(dataMS2Spectrum.LIST_PEAK_INT.pop(index))
            delete_index_list = []

        max_charge = max(dataMS2Spectrum.LIST_PRECURSOR_CHARGE)

        if self.dp.myCFG.B4_CHECK_PRECURSOR_CHARGE == 0:
            if max_charge > 1:
                max_charge -= 1

        new_peak_int = []
        new_peak_moz = []
        isotopic_tol = [self.dp.myINI.MASS_NEUTRON_AVRG / c for c in range(1, max_charge + 1)]
        # 1.0030, 0.5015, 0.3334, 0.25075, 0.2016...

        # cluster_start_counter = 0
        charge_state = -1  # 独峰

        while 0 < len(dataMS2Spectrum.LIST_PEAK_MOZ):
            cluster_index_list = []
            # 小于isotopic_tol[-1] - tol 就check下一谱峰
            # 检查开始列表中的任意电荷状态的距离，由index可得电荷状态index+1
            # 超过isotopic_tol[0]就进行归并，并跳转下一峰
            # index = cluster_start_counter
            max_int = -1
            max_int_peak_index = -1
            for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
                if dataMS2Spectrum.LIST_PEAK_INT[i] > max_int:
                    max_int = dataMS2Spectrum.LIST_PEAK_INT[i]
                    max_int_peak_index = i
            # 得到最高峰信号的地址了

            # 而后左右检测是否可有峰簇

            # while tmp_check_index < len(dataMS2Spectrum.LIST_PEAK_MOZ):
            #     if -1 == charge_state:
            #         pass
            # ########################################################
            # not complete

            # -------------------------- right -----------------------------
            # ##############################################################
            tmp_check_index = max_int_peak_index
            # index = 0
            cluster_index_list.append(tmp_check_index)
            cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
            tmp_check_index += 1
            # charge == -1: cluster is not complete

            while tmp_check_index < len(dataMS2Spectrum.LIST_PEAK_MOZ):

                peak_tolerance = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index] - cluster_tail_moz
                if self.msms_ppm == 1:
                    ppm_tol_da = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index] * self.msms_fraction
                else:
                    ppm_tol_da = self.msms_tol
                prep_tolerance = ppm_tol_da
                if peak_tolerance < isotopic_tol[-1] - prep_tolerance:
                    # 小于最小,跳过去
                    # 要继续找，不要break
                    tmp_check_index += 1

                elif peak_tolerance > isotopic_tol[0] + prep_tolerance:
                    # 大于最大
                    # 不要继续找，要break
                    break

                else:

                    # 先确定电荷状态，再继续cluster构建
                    if charge_state == -1:
                        for tol_index in range(len(isotopic_tol)):
                            if math.isclose(peak_tolerance, isotopic_tol[tol_index], abs_tol=prep_tolerance):
                                charge_state = tol_index + 1
                                # 确定质荷比信息
                                cluster_index_list.append(tmp_check_index)
                                cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                                break
                        if charge_state == -1:
                            # 继续向后寻找（MUST）
                            tmp_check_index += 1
                        else:

                            # 连续向后构造
                            tmp_check_index += 1


                    # 已经确定电荷状态
                    else:
                        # 仍然是我的同位素峰，进入峰簇下标列表
                        if math.isclose(peak_tolerance, isotopic_tol[charge_state - 1], abs_tol=prep_tolerance):

                            # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                            nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[-1]]
                            if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] > 1.2 * nearest_int:
                                tmp_check_index += 1
                                continue
                            cluster_index_list.append(tmp_check_index)
                            cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                            tmp_check_index += 1
                        # 超过了枚举电荷的范围，break
                        elif peak_tolerance - isotopic_tol[charge_state - 1] > prep_tolerance:
                            break
                        # 又不匹配，又不越界，啥也不是，继续走吧
                        else:
                            tmp_check_index += 1
            # while index < len(dataMS2Spectrum.LIST_PEAK_MOZ) over.

            # -------------------------- left ------------------------------
            # ##############################################################
            tmp_check_index = max_int_peak_index - 1
            cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[max_int_peak_index]
            while tmp_check_index >= 0 and dataMS2Spectrum.LIST_PEAK_MOZ:

                peak_tolerance = cluster_left_moz - dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                if self.msms_ppm == 1:
                    ppm_tol_da = cluster_left_moz * self.msms_fraction
                else:
                    ppm_tol_da = self.msms_tol
                prep_tolerance = ppm_tol_da
                if peak_tolerance < isotopic_tol[-1] - prep_tolerance:
                    # 小于最小,跳过去
                    # 要继续找，不要break
                    tmp_check_index -= 1

                elif peak_tolerance > isotopic_tol[0] + prep_tolerance:
                    # 大于最大
                    # 不要继续找，要break
                    break

                else:

                    # 先确定电荷状态，再继续cluster构建
                    if charge_state == -1:
                        for tol_index in range(len(isotopic_tol)):
                            if math.isclose(peak_tolerance, isotopic_tol[tol_index], abs_tol=prep_tolerance):

                                # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                                # nearest_int: cluster list中最左端的谱峰信号的强度值
                                nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[0]]
                                # if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.2 * nearest_int:
                                if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.3 * nearest_int:
                                    # break
                                    tmp_check_index -= 1
                                    continue

                                charge_state = tol_index + 1
                                # 确定质荷比信息
                                # cluster_index_list.append(tmp_check_index)
                                cluster_index_list = [tmp_check_index] + cluster_index_list
                                cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                                break
                        if charge_state == -1:
                            # 继续向前寻找（MUST）
                            tmp_check_index -= 1
                        else:

                            # 连续向后构造
                            tmp_check_index -= 1


                    # 已经确定电荷状态
                    else:
                        # 仍然是我的同位素峰，进入峰簇下标列表
                        if math.isclose(peak_tolerance, isotopic_tol[charge_state - 1], abs_tol=prep_tolerance):
                            # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                            nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[0]]
                            # if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.2 * nearest_int:
                            if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.4 * nearest_int:
                                # break
                                tmp_check_index -= 1
                                continue
                            cluster_index_list = [tmp_check_index] + cluster_index_list
                            cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                            tmp_check_index -= 1
                        # 超过了枚举电荷的范围，break
                        elif peak_tolerance - isotopic_tol[charge_state - 1] > prep_tolerance:
                            break
                        # 又不匹配，又不越界，啥也不是，继续走吧
                        else:
                            tmp_check_index -= 1

            # --------------------------------------------------------------

            # cluster
            # cluster 构造结束，开始收工
            if charge_state == -1:
                # 删除，添加
                add_moz = dataMS2Spectrum.LIST_PEAK_MOZ.pop(cluster_index_list[0])
                add_int = dataMS2Spectrum.LIST_PEAK_INT.pop(cluster_index_list[0])
                new_peak_moz.append(add_moz)
                new_peak_int.append(add_int)
                self.moz_iso_cluster[0].append(add_moz)
                self.int_iso_cluster[0].append(add_int)
            else:

                # ##########################################
                # 20210809 ATTENTION ATTENTION ATTENTION ###
                # pop时，一定一定一定要注意地址的问题！ ####
                # ##########################################
                # --------------- QUESTION --------------- #
                # 这里要把mono峰pop出来，但是一定要放在最后#
                # 否则会改变其他谱峰的地址，导致pop峰错误  #
                # 再就是强度和质荷比错开了，导致tag提取错误#
                # ##########################################
                add_int = 0

                self.moz_iso_cluster.append([dataMS2Spectrum.LIST_PEAK_MOZ[new_i] for new_i in cluster_index_list])
                self.int_iso_cluster.append([dataMS2Spectrum.LIST_PEAK_INT[new_i] for new_i in cluster_index_list])
                self.c_label_cluster.append(charge_state)

                if self.dp.myCFG.B1_ADD_ISOTOPE_INTENSITY:
                    for buf_index in reversed(cluster_index_list[1:]):
                        try:
                            # dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                            # add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)

                            dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                            add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)

                        except:
                            pass
                # ================== DO NOT ADD INTENSITY! ====================
                else:
                    for buf_index in reversed(cluster_index_list[1:]):
                        try:
                            dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                            dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)
                        except:
                            pass

                # 把mono峰的moz提出放在最后，包括mono对应的强度部分也是
                # 如果希望对add的强度做一些操作，可以在for loop try里头去整
                add_moz = dataMS2Spectrum.LIST_PEAK_MOZ.pop(cluster_index_list[0]) * charge_state
                add_moz -= (charge_state - 1) * self.dp.myINI.MASS_PROTON_MONO
                add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(cluster_index_list[0])

                new_peak_moz.append(add_moz)
                new_peak_int.append(add_int)

            charge_state = -1
            # cluster 处理结束
        # while over

        # -------------------------------------------------------------------

        # 排序，检测合并
        index_order = np.argsort(new_peak_moz)
        new_peak_moz = [new_peak_moz[index] for index in index_order]
        new_peak_int = [new_peak_int[index] for index in index_order]

        # -------------------------------------------------------------------

        output_peak_moz = []
        output_peak_int = []

        # 下方：合并转换的谱峰，有可能会影响精度
        add_moz = 0
        add_int = 0
        i = 0
        jump_set = set()
        while i < len(new_peak_moz) - 1:

            if i in jump_set:
                i += 1
                continue

            add_moz = new_peak_moz[i]
            add_int = new_peak_int[i]

            for j in range(i + 1, len(new_peak_moz)):
                if self.msms_ppm == 1:

                    prep_tolerance = new_peak_moz[j] * self.msms_fraction

                else:
                    prep_tolerance = self.msms_tol
                # prep_tolerance = new_peak_moz[j] * 2e-5

                if abs(new_peak_moz[i] - new_peak_moz[j]) < prep_tolerance:
                    add_moz = add_moz * add_int + new_peak_moz[j] * new_peak_int[j]
                    add_int += new_peak_int[j]
                    add_moz /= add_int
                    i = j
                    jump_set.add(j)
                # 仅看左右最远的0.02，两两之间就不看了
                else:
                    output_peak_moz.append(add_moz)
                    output_peak_int.append(add_int)
                    i = j
                    break

            # if abs(new_peak_moz[i] - new_peak_moz[-1]) < prep_tolerance:
            #     output_peak_moz.append(add_moz)
            #     output_peak_int.append(add_int)
            #     i = j
            #     break

        if add_moz in output_peak_moz:
            pass
        else:
            output_peak_moz.append(add_moz)
            output_peak_int.append(add_int)

        # 活久见，图里没谱峰哒，就算辽叭
        if len(new_peak_moz) == 0:
            pass
        # 检测一下最后一根谱峰是否是可以添加的信息
        elif jump_set:
            if max(jump_set) == len(new_peak_moz):
                pass
            else:
                output_peak_moz.append(new_peak_moz[-1])
                output_peak_int.append(new_peak_int[-1])
        else:
            output_peak_moz.append(new_peak_moz[-1])
            output_peak_int.append(new_peak_int[-1])

        # 上方：合并转换的谱峰，有可能会影响精度
        # output_peak_moz = new_peak_moz
        # output_peak_int = new_peak_int

        dataMS2Spectrum.LIST_PEAK_MOZ = output_peak_moz
        dataMS2Spectrum.LIST_PEAK_INT = output_peak_int


    # 同位素峰簇转换结束后，检测失水失氨峰
    # 按照强度为顺序，“向左”对失水失氨的质量依次进行检测
    # 将符合质量的谱峰，构成一个cluster
    # 因为谱峰之间的质量顺序是：ori_peak - H2O, ori_peak - NH3, ori_peak
    # 所以构造的谱峰也将按照此规律检测
    # 20210902  修改质量偏差设定收窄为10ppm
    def __soldierClusterNeutronalLossH2OAndNH3(self, dataMS2Spectrum):

        new_peak_int = []
        new_peak_moz = []

        while 0 < len(dataMS2Spectrum.LIST_PEAK_MOZ):
            cluster_index_list = []

            # 这里进行了修改，即得到最后一根谱峰，然后从后往前找-NH3和-H2O的峰
            tmp_check_index = len(dataMS2Spectrum.LIST_PEAK_MOZ) - 1
            cluster_index_list.append(tmp_check_index)
            cluster_tail_int = dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index]

            # -------------------------- left ------------------------------
            # ##############################################################
            cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
            tmp_check_index -= 1

            while tmp_check_index >= 0 and dataMS2Spectrum.LIST_PEAK_MOZ:

                peak_tolerance = cluster_tail_moz - dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                if self.msms_ppm == 1:
                    # self.msms_fraction
                    ppm_tol_da = cluster_tail_moz * self.msms_fraction
                else:
                    ppm_tol_da = self.msms_tol

                prep_tolerance = ppm_tol_da

                if peak_tolerance < self.PROTON_MASS_NH3 - prep_tolerance:
                    # 小于最小,跳过去
                    # 要继续找，不要break
                    tmp_check_index -= 1

                elif peak_tolerance > self.PROTON_MASS_H2O + prep_tolerance:
                    # 大于最大
                    # 不要继续找，要break
                    break

                else:
                    # 进入自然损失簇下标列表
                    if math.isclose(peak_tolerance, self.PROTON_MASS_NH3, abs_tol=prep_tolerance):

                        cluster_left_int = dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index]
                        if cluster_tail_int > 0.2 * cluster_left_int:
                            cluster_index_list = [tmp_check_index] + cluster_index_list
                        tmp_check_index -= 1
                        # 进入自然损失簇下标列表

                    elif math.isclose(peak_tolerance, self.PROTON_MASS_H2O, abs_tol=prep_tolerance):

                        cluster_left_int = dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index]
                        if cluster_tail_int > 0.2 * cluster_left_int:
                            cluster_index_list = [tmp_check_index] + cluster_index_list
                        tmp_check_index -= 1

                    # 又不匹配，又不越界，啥也不是，继续走吧
                    else:
                        tmp_check_index -= 1

            # --------------------------------------------------------------

            # cluster
            # cluster 构造结束，开始收工
            add_moz = dataMS2Spectrum.LIST_PEAK_MOZ[cluster_index_list[-1]]
            if self.dp.myCFG.B1_ADD_ISOTOPE_INTENSITY:
                add_int = 0
                for buf_index in sorted(cluster_index_list, reverse=True):
                    try:
                        dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                        add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)
                    except:
                        pass
            else:
                add_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[-1]]
                for buf_index in sorted(cluster_index_list, reverse=True):
                    try:
                        dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                        dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)
                    except:
                        pass

            # 如果希望对add的强度做一些操作，可以在for loop try里头去整

            new_peak_moz.append(add_moz)
            new_peak_int.append(add_int)
            # cluster 处理结束
        # while over

        # -------------------------------------------------------------------

        # 排序，检测合并
        if new_peak_moz:
            index_order = np.argsort(new_peak_moz)
            new_peak_moz = [new_peak_moz[index] for index in index_order]
            new_peak_int = [new_peak_int[index] for index in index_order]

        dataMS2Spectrum.LIST_PEAK_MOZ = new_peak_moz
        dataMS2Spectrum.LIST_PEAK_INT = new_peak_int


    def __soldierGetRankListByINT(self, dataMS2Spectrum):

        tmpMS2_np = np.array(dataMS2Spectrum.LIST_PEAK_INT[:])

        rankListByINT = np.argsort(tmpMS2_np)
        # 注意，这里的索引是按照强度，从小到大排列的

        return rankListByINT


    def __soldierRemovePrecursorPeak(self, dataMS2Spectrum, inputPrecursorMass):

        soldier = inputPrecursorMass - 50

        # eg. Precursor -H2O / -NH3, Precursor - 2*H2O
        for i in reversed(range(len(dataMS2Spectrum.LIST_PEAK_MOZ))):

            if dataMS2Spectrum.LIST_PEAK_MOZ[i] >= soldier:
                dataMS2Spectrum.LIST_PEAK_MOZ.pop()
                dataMS2Spectrum.LIST_PEAK_INT.pop()

            else:
                break

        # 20210825 -----------------------------------------------------
        # 去掉质量小于18Da的分子
        soldier = 20

        while dataMS2Spectrum.LIST_PEAK_MOZ:

            if dataMS2Spectrum.LIST_PEAK_MOZ[0] <= soldier:
                dataMS2Spectrum.LIST_PEAK_MOZ.pop(0)
                dataMS2Spectrum.LIST_PEAK_INT.pop(0)

            else:
                break

        maxInt = max(dataMS2Spectrum.LIST_PEAK_INT)
        dataMS2Spectrum.LIST_PEAK_MOZ = [self.PROTON_MASS_PROTON, self.PROTON_MASS_H2O_PROTON] + dataMS2Spectrum.LIST_PEAK_MOZ + [inputPrecursorMass - self.PROTON_MASS_H2O, inputPrecursorMass]
        dataMS2Spectrum.LIST_PEAK_INT = [maxInt, maxInt] + dataMS2Spectrum.LIST_PEAK_INT + [maxInt, maxInt]

    def __soldierDeleteWeakPeak(self, dataMS2Spectrum, rankListByINT, holdNum=0):

        # 获得删除列表，且已经逆序排好顺序，便于删除
        # 20210902 修改判0为判断大小，避免or操作，合并检测的包含关系
        if holdNum < self.max_remain_peak_num:
            deleteList = reversed(np.sort(rankListByINT[:-self.max_remain_peak_num]))
        else:
            deleteList = reversed(np.sort(rankListByINT[:-holdNum]))
        # 逆序删除，不影响索引
        for i in deleteList:
            self.moz_filtered_1.append(dataMS2Spectrum.LIST_PEAK_MOZ.pop(i))
            self.int_filtered_1.append(dataMS2Spectrum.LIST_PEAK_INT.pop(i))


    def __soldierINT2RANK(self, dataMS2Spectrum):

        tmpMS2INTList = np.array(dataMS2Spectrum.LIST_PEAK_INT[:])
        # 读取MS2的强度信息

        rankIndexList = np.argsort(-tmpMS2INTList)
        # 将强度从大到小排列，保留地址
        # 加个负号即可

        rank = 1
        for index in rankIndexList:
            dataMS2Spectrum.LIST_PEAK_INT[index] = rank
            rank += 1


    def __soldierAbsoluteINT2RelativeINT(self, dataMS2Spectrum):

        if not dataMS2Spectrum.LIST_PEAK_INT:
            # print("Spectrum empty")
            pass
        # 如果列表不空的话（话说为什么会空呢。。。全为precursor？）
        else:
            maxAbsoluteINTValue = max(dataMS2Spectrum.LIST_PEAK_INT[:])

            # 使用numpy的广播来代替此处的for循环，速度会更快
            for i in range(len(dataMS2Spectrum.LIST_PEAK_INT)):

                tmpAbsINT = dataMS2Spectrum.LIST_PEAK_INT[i]

                dataMS2Spectrum.LIST_PEAK_INT[i] = tmpAbsINT / maxAbsoluteINTValue * 100


    def __soldierAbsoluteINT2RelativeINT_MAXMIN(self, dataMS2Spectrum, exp):

        if not dataMS2Spectrum.LIST_PEAK_INT:
            # print("Spectrum empty")
            pass
        # 如果列表不空的话（话说为什么会空呢。。。全为precursor？）
        else:
            maxAbsoluteINTValue = max(dataMS2Spectrum.LIST_PEAK_INT)
            minAbsoluteINTValue = min(dataMS2Spectrum.LIST_PEAK_INT)

            if maxAbsoluteINTValue == minAbsoluteINTValue:
                # 退化为最大值归一化，避免除零错误
                for i in range(len(dataMS2Spectrum.LIST_PEAK_INT)):
                    tmpAbsINT = dataMS2Spectrum.LIST_PEAK_INT[i]

                    dataMS2Spectrum.LIST_PEAK_INT[i] = (tmpAbsINT / maxAbsoluteINTValue * 10000) ** exp

            else:
                # 最大最小归一化
                maxAbsoluteINTValue -= minAbsoluteINTValue

                # 使用numpy的广播来代替此处的for循环，速度会更快
                for i in range(len(dataMS2Spectrum.LIST_PEAK_INT)):

                    tmpAbsINT = dataMS2Spectrum.LIST_PEAK_INT[i] - minAbsoluteINTValue

                    dataMS2Spectrum.LIST_PEAK_INT[i] = (tmpAbsINT / maxAbsoluteINTValue * 10000) ** exp


    def __localPeaksNumControl(self, dataMS2Spectrum, bin_len, peak_num):
        ...
        res_moz = []
        res_int = []

        # 当列表变为空的时候，我们就不用再循环啦
        while dataMS2Spectrum.LIST_PEAK_INT:
            # 先确定下来因数是多少
            old_soldier = dataMS2Spectrum.LIST_PEAK_MOZ[0] // bin_len
            # 初始化为最小的地址
            tmp_idx = 0

            # find an index range per bin_len
            # 找到第一个和我们预先设定的因数不一致的数字的地址
            for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
                # 一直没找到，那就一直记录即可 -------------
                if old_soldier == (dataMS2Spectrum.LIST_PEAK_MOZ[i] // bin_len):
                    tmp_idx = i
                # 找到了！我们直接break！
                else:
                    break

            # 此时，地址代表了 peak num + 1
            # 因此当tmp_idx 小于 peak_num 时，满足谱峰过滤的要求，直接加进去即可
            if tmp_idx < peak_num:
                res_moz += dataMS2Spectrum.LIST_PEAK_MOZ[:tmp_idx+1]
                dataMS2Spectrum.LIST_PEAK_MOZ = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_idx+1:]
                res_int += dataMS2Spectrum.LIST_PEAK_INT[:tmp_idx+1]
                dataMS2Spectrum.LIST_PEAK_INT = dataMS2Spectrum.LIST_PEAK_INT[tmp_idx+1:]

            # 当前该 bin 中的谱峰数目过多，需要选择最强的peak_num根峰
            else:
                tmp_moz = dataMS2Spectrum.LIST_PEAK_MOZ[:tmp_idx+1]
                tmp_int = dataMS2Spectrum.LIST_PEAK_INT[:tmp_idx+1]
                dataMS2Spectrum.LIST_PEAK_MOZ = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_idx+1:]
                dataMS2Spectrum.LIST_PEAK_INT = dataMS2Spectrum.LIST_PEAK_INT[tmp_idx+1:]

                # 去除强度低的谱峰，这样添加到res_moz / res_int 中时，moz还能保证有序
                while len(tmp_moz) != peak_num:
                    min_idx = tmp_int.index(min(tmp_int))
                    tmp_int.pop(min_idx)
                    tmp_moz.pop(min_idx)
                res_moz += tmp_moz
                res_int += tmp_int

        dataMS2Spectrum.LIST_PEAK_MOZ = res_moz
        dataMS2Spectrum.LIST_PEAK_INT = res_int


    # 获得亚胺离子的质量信息，并生成哈希值的字典
    # 子璇在这里的默认误差范围是0.02Da
    # int((mass+-delta) * 1000)
    def __getImmoniumIonsHashTab(self):
        # self.fold = 1000
        out_set = set()

        # [hash] imonium ions masses with mass tolerance(0.02Da)
        abs_delta_hash = toolMyIntRound(0.02 * self.fold)
        for aa in self.dp.myINI.DICT1_AA_MASS:
            immonium_mass = self.dp.myINI.DICT1_AA_MASS[aa] - self.PROTON_MASS_CO + self.dp.myINI.MASS_PROTON_MONO
            if immonium_mass <= 0:
                continue
            aa_mass_hash = toolMyIntRound(immonium_mass * self.fold)
            out_set.update(set([it for it in range(aa_mass_hash - abs_delta_hash, aa_mass_hash +abs_delta_hash + 1)]))

        # [hash] relative ions masses of immonium ions with mass tolerance(0.05Da)
        abs_delta_hash = toolMyIntRound(0.05 * self.fold)
        tmp_lst = [41.05, 44.05, 55.05, 56.05, 59.05, 61.05, 69.05, 70.05, 72.05, 73.05,
             77.05, 82.05, 84.05, 87.05, 91.05, 100.05, 107.05, 112.05, 117.05, 121.05,
             123.05, 129.05, 130.05, 132.05, 138.05, 166.05, 170.05, 171.05]
        for rel_mass in tmp_lst:
            rel_mass_hash = toolMyIntRound(rel_mass * self.fold)
            out_set.update(set([it for it in range(rel_mass_hash - abs_delta_hash, rel_mass_hash + abs_delta_hash + 1)]))

        return out_set

    def __deleteImmoniumIons(self, dataMS2Spectrum):

        soldier_idx = 0
        for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
            if dataMS2Spectrum.LIST_PEAK_MOZ[i] > 200:
                soldier_idx = i
                break

        # 存在谱峰的质量是小于200的
        if soldier_idx != 0:
            # 逆序删除，不影响地址信息
            for i in reversed(range(soldier_idx)):
                if toolMyIntRound(dataMS2Spectrum.LIST_PEAK_MOZ[i] * self.fold) in self.IMMONIUM_HASH_TAB:
                    dataMS2Spectrum.LIST_PEAK_MOZ.pop(i)
                    dataMS2Spectrum.LIST_PEAK_INT.pop(i)

# 202402   for xueli neucode version


class CFunctionPreprocessForXueLiNeuCode:

    def __init__(self, inputDP):
        self.dp = inputDP
        # important for zixuan
        self.max_remain_peak_num = self.dp.myCFG.B5_ROUND_ONE_HOLD_PEAK_NUM
        # self.tolerance = 0.02  # Da
        self.msms_tol = self.dp.myCFG.D5_MSMS_TOL
        self.msms_ppm = self.dp.myCFG.D6_MSMS_TOL_PPM
        self.upper_peaks_num1 = 600
        self.upper_peaks_num2 = 350
        if self.msms_ppm == 1:
            self.msms_fraction = self.msms_tol / 1e6
        else:
            self.msms_fraction = 1
        self.fold = 1e5
        # mass check
        self.PROTON_MASS_C = self.dp.myINI.DICT0_ELEMENT_MASS["C"][0]
        self.PROTON_MASS_H = self.dp.myINI.DICT0_ELEMENT_MASS["H"][0]
        self.PROTON_MASS_O = self.dp.myINI.DICT0_ELEMENT_MASS["O"][0]
        self.PROTON_MASS_CO = self.PROTON_MASS_C + self.PROTON_MASS_O
        self.PROTON_MASS_OH = self.PROTON_MASS_H + self.PROTON_MASS_O
        self.PROTON_MASS_H2O = self.PROTON_MASS_H + self.PROTON_MASS_OH
        self.PROTON_MASS_NH3 = self.PROTON_MASS_H * 3 + self.dp.myINI.DICT0_ELEMENT_MASS["N"][0]

        # self.MAX_INTEN_RATIO = 6.0  # inten_light / inten_heavy
        self.MAX_INTEN_RATIO = self.dp.myCFG.A10_NEUCODE_PEAK_MAX_RATIO

        self.DOUBLE_PEAK_BOUND = 1500
        self.DOUBLE_PEAK_SHIFT = self.dp.myCFG.A9_DOUBLE_PEAK_GAP  # fixed mass shift of (heavy - light) ~0.036 Da
        self.DOUBLE_PEAK_SHIFT2 = self.DOUBLE_PEAK_SHIFT * 2  # 考虑一个漏切，缘分这不就来了吗
        self.DOUBLE_PEAK_TOL_MAX = self.DOUBLE_PEAK_SHIFT2 + 0.5  # 设定了一个检测双峰 tol 的上限，超过即停
        # 我就不信tol能超过 0.5 哼！而且0.5的浮点数表示是1 * 2^(-1)，应该挺好算哈(都是自己的想象-_-未经证实哈...)

        self.DOUBLE_PEAK_MAX_C = 3  # 检测双峰的时候，就按这些电荷的量来检测就行~ 碎片离子再大就检测不到了也

        self.DOUBLE_PEAK_SHIFT_C = [1] + [self.DOUBLE_PEAK_SHIFT / (c + 1) for c in range(self.DOUBLE_PEAK_MAX_C)]
        self.DOUBLE_PEAK_SHIFT2_C = [1] + [self.DOUBLE_PEAK_SHIFT2 / (c + 1) for c in range(self.DOUBLE_PEAK_MAX_C)]

        if self.msms_ppm == 1:
            self.DOUBLE_PEAK_TOL_M = min(10, self.msms_tol)
        else:
            self.DOUBLE_PEAK_TOL_M = min(0.02, self.msms_tol)

        self.DOUBLE_TOL_HASH_TAB_LIST = self.__getDoublePeakTolHashTab()  # [None, set(), set(), set()]

        # ====================== can use! ===========================

        self.moz_filtered_1 = []  # inten rank > 600
        self.int_filtered_1 = []  # inten rank > 600

        self.moz_iso_cluster = [[]] # [[], [], [], [], ...]  # 同位素峰簇的质荷比列表
        self.int_iso_cluster = [[]] # [[], [], [], [], ...]  # 同位素峰簇的强度列表
        self.c_label_cluster = [0] # [int, int, int, ....]  # 同位素峰簇的电荷

        # ===================== cannot use ==========================

        # self.moz_filtered_u = []  # user-defined filtered
        # self.int_filtered_u = []  # user-defined filtered
        #
        # self.moz_immonium = []    # 亚胺离子
        # self.int_immonium = []    # 亚胺离子
        #
        # self.moz_filtered_2 = []  # 2nd - local filtered peaks
        # self.int_filtered_2 = []  # 2nd - local filtered peaks

    def preprocesslabel(self, dataMS2Spectrum):

        # do not change the ori data
        prep_copy = copy.deepcopy(dataMS2Spectrum)

        label_data = self.__captainPreprocessMS2_1st(prep_copy)

        return label_data

    # piyu preprocessing without de-precursor ions
    def __captainPreprocessMS2_1st(self, dataMS2Spectrum):

        # 20201110  调整执行次序
        # 1. holding top-N peaks
        ori_peaks_num = len(dataMS2Spectrum.LIST_PEAK_INT)

        if dataMS2Spectrum.LIST_PEAK_INT:

            # 若给定参数k则执行保留峰数目为k的操作，且k不小于用户指定数目，否则保留用户指定数目
            if len(dataMS2Spectrum.LIST_PEAK_MOZ) > self.upper_peaks_num1:

                rankListByINT = self.__soldierGetRankListByINT(dataMS2Spectrum)

                self.__soldierDeleteWeakPeak(dataMS2Spectrum, rankListByINT, holdNum=self.upper_peaks_num1)

        max_charge = max(dataMS2Spectrum.LIST_PRECURSOR_CHARGE)

        dataMS2Spectrum.NEUCODE_LABEL = [1] * len(dataMS2Spectrum.LIST_PEAK_INT)

        self.__reshapeNeuCodeLabelPeaks(dataMS2Spectrum, max_charge)

        if not self.int_filtered_1:
            return dataMS2Spectrum.NEUCODE_LABEL

        res = [0] * len(self.int_filtered_1) + dataMS2Spectrum.NEUCODE_LABEL
        res_moz = self.moz_filtered_1 + dataMS2Spectrum.LIST_PEAK_MOZ
        res_index = np.argsort(res_moz)

        res = [res[i] for i in res_index]

        self.int_filtered_1 = []
        self.moz_filtered_1 = []

        return res


    def __reshapeNeuCodeLabelPeaks(self, dataMS2Spectrum, maxCharge:int):

        biggest_c = min(self.DOUBLE_PEAK_MAX_C, maxCharge)
        TOL_HASH_TAB = self.DOUBLE_TOL_HASH_TAB_LIST[biggest_c]

        i = 0
        idx_boundary = len(dataMS2Spectrum.LIST_PEAK_MOZ)
        # record_list = [1] * idx_boundary

        p_LIST_MOZ = dataMS2Spectrum.LIST_PEAK_MOZ
        p_LIST_INT = dataMS2Spectrum.LIST_PEAK_INT

        while i < idx_boundary-1:

            i_moz = p_LIST_MOZ[i]
            i_int = p_LIST_INT[i]
            if i_moz > self.DOUBLE_PEAK_BOUND:
                break

            for j in range(i+1, idx_boundary):

                tol = p_LIST_MOZ[j] - p_LIST_MOZ[i]
                # gap is too big! do not go on again!
                if tol > self.DOUBLE_PEAK_TOL_MAX:
                    break

                # I got a couple of peaks!! In law of mass tol!! XD
                if int(tol * self.fold + 0.5) in TOL_HASH_TAB:
                    if (self.MAX_INTEN_RATIO < p_LIST_INT[j]/ i_int) or (self.MAX_INTEN_RATIO < i_int / p_LIST_INT[j]):
                        continue

                    # 只当是提前判断电荷辣！0.036Da 统一按照 1+ 算
                    if tol <= self.DOUBLE_PEAK_SHIFT:
                        dataMS2Spectrum.NEUCODE_LABEL[i] = 2

            # while loop is over
            # we need a newer i
            i += 1

        pass



    def __soldierFillLabel(self, label:CPrepLabel):
        label.FILTERED = []  # initial
        label.CLUSTER = []  # initial
        label.CHARGE = []  # initial

        # merge ----
        tmpMOZ = [k for k in self.moz_filtered_1]
        label.FILTERED = [0] * len(self.moz_filtered_1)
        label.CLUSTER = [0] * len(self.moz_filtered_1)
        label.CHARGE = [0] * len(self.moz_filtered_1)
        for i, clu in enumerate(self.moz_iso_cluster):
            label.FILTERED += [1] * len(clu)
            label.CLUSTER += [i] * len(clu)  # 拼接cluster的编号信息
            c = self.c_label_cluster[i]      # 获取电荷状态
            label.CHARGE += [c] * len(clu)   # 拼接这些 moz 对应的电荷
            tmpMOZ += clu                    # 拼接cluster中的 moz 信息
        # sort -----

        sorted_idx = np.argsort(tmpMOZ)

        label.CHARGE = [label.CHARGE[kk] for kk in sorted_idx]
        label.CLUSTER = [label.CLUSTER[kk] for kk in sorted_idx]
        label.FILTERED = [label.FILTERED[kk] for kk in sorted_idx]

        # after use, clean it!
        self.moz_filtered_1 = []  # 350
        self.int_filtered_1 = []  # 350

        self.moz_iso_cluster = [[]]  # [[], [], [], [], ...]  # 同位素峰簇的质荷比列表
        self.int_iso_cluster = [[]]  # [[], [], [], [], ...]  # 同位素峰簇的强度列表
        self.c_label_cluster = [0]  # [int, int, int, ....]  # 同位素峰簇的电荷


    def __soldierGetRankListByINT(self, dataMS2Spectrum):

        tmpMS2_np = np.array(dataMS2Spectrum.LIST_PEAK_INT[:])

        rankListByINT = np.argsort(tmpMS2_np)
        # 注意，这里的索引是按照强度，从小到大排列的

        return rankListByINT


    def __soldierDeleteWeakPeak(self, dataMS2Spectrum, rankListByINT, holdNum=0):

        # 获得删除列表，且已经逆序排好顺序，便于删除
        # 20210902 修改判0为判断大小，避免or操作，合并检测的包含关系
        if holdNum < self.max_remain_peak_num:
            deleteList = reversed(np.sort(rankListByINT[:-self.max_remain_peak_num]))
        else:
            deleteList = reversed(np.sort(rankListByINT[:-holdNum]))
        # 逆序删除，不影响索引
        for i in deleteList:
            self.moz_filtered_1.append(dataMS2Spectrum.LIST_PEAK_MOZ.pop(i))
            self.int_filtered_1.append(dataMS2Spectrum.LIST_PEAK_INT.pop(i))


    def __getDoublePeakTolHashTab(self):

        res = [None]  #  res[0] 是不存在滴，因为电荷不为0

        # 提前计算出来一个固定的质量（相对也让他变成固定的），用于快速的 check 双峰
        # 这里的质荷比，其实是经过缩小的，但是我也不敢缩小的太多
        if self.msms_ppm == 1:
            abs_tol = self.DOUBLE_PEAK_BOUND * self.DOUBLE_PEAK_TOL_M / 1e6
            int_del = int(abs_tol * self.fold + 0.5)  # 四舍五入
        else:
            abs_tol = self.DOUBLE_PEAK_TOL_M
            int_del = int(abs_tol * self.fold + 0.5)  # 四舍五入

        # 最高电荷状态（1+，2+，...）
        for c_stat in range(1, self.DOUBLE_PEAK_MAX_C+1):

            tmp_rec = set()

            # 从1电荷到最高电荷状态（1+，2+，...）
            for i in range(1, c_stat+1):
                # 不含漏切，那么双峰的理论差值就是 0.036Da / c
                tmp_hash_num = int((self.DOUBLE_PEAK_SHIFT / i) * self.fold + 0.5)
                for tmp_delta in range(-int_del, int_del + 1):
                    tmp_rec.add(tmp_hash_num+tmp_delta)

                # 含一个漏切，则双峰的理论差值就是 0.036Da * 2 / c
                tmp_hash_num = int((self.DOUBLE_PEAK_SHIFT2 / i) * self.fold + 0.5)
                for tmp_delta in range(-int_del, int_del + 1):
                    tmp_rec.add(tmp_hash_num+tmp_delta)

            res.append(tmp_rec)

        return res

