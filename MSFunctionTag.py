# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSFunctionTag.py
Time: 2022.09.05 Monday
ATTENTION: none
"""
import heapq
import copy
import numpy as np
import math
from MSTool import toolMyIntRound
from MSOperator import op_ADD_EDGE_CTagDAG
from MSData import CTagDAG, CTagDAGNode, CTagInfo
from MSLogging import logGetError


class CFunctionDiNovo:

    def __init__(self, inputDP, inputAAMarkDict, inputAAMassIndex, inputTermAAMassIndex, maxPathNum=100):

        self.dp = inputDP
        self.max_short_path_num = maxPathNum
        self.msms_tol = self.dp.myCFG.D5_MSMS_TOL
        self.msms_ppm = self.dp.myCFG.D6_MSMS_TOL_PPM

        if self.msms_ppm == 1:
            self.msms_fraction = self.msms_tol / 1e6
        else:
            self.msms_fraction = 1
        self.compare_tol = self.msms_tol
        self.tolerance = self.msms_tol * 1.5  # for score
        # self.tolerance = 0.03  # Da
        self.fold = 1000

        self.tol_score_fm = -2 / (self.tolerance * self.tolerance)
        # self.max_tag_length = 5
        # self.tagLengthList = [3, 4, 5]  # 20220922 后面要调整的
        # generate aa mass index
        self.aaMarkDict = inputAAMarkDict
        self.aaMassIndex = inputAAMassIndex
        self.termMassIndex = inputTermAAMassIndex
        self.lenAAMassIndex = len(self.aaMassIndex)
        self.lenTermMassIndex = len(self.termMassIndex)
        self.PROTON_MASS_C = self.dp.myINI.DICT0_ELEMENT_MASS["C"][0]
        self.PROTON_MASS_H = self.dp.myINI.DICT0_ELEMENT_MASS["H"][0]
        self.PROTON_MASS_O = self.dp.myINI.DICT0_ELEMENT_MASS["O"][0]
        self.PROTON_MASS_CO = self.PROTON_MASS_C + self.PROTON_MASS_O
        self.PROTON_MASS_OH = self.PROTON_MASS_H + self.PROTON_MASS_O
        self.PROTON_MASS_H2O = self.PROTON_MASS_H + self.PROTON_MASS_OH
        self.PROTON_MASS_NH3 = self.PROTON_MASS_H * 3 + self.dp.myINI.DICT0_ELEMENT_MASS["N"][0]


    def sequencing(self, dataMS2Spectrum, complement):
        tag_list = []
        buff_mass = dataMS2Spectrum.LIST_PRECURSOR_MASS
        
        output_ms2_info = []
        for i in range(len(dataMS2Spectrum.LIST_PRECURSOR_MASS)):
            buff_file_name = dataMS2Spectrum.LIST_FILE_NAME[i]
            buff_moz = dataMS2Spectrum.LIST_PRECURSOR_MOZ[i]
            buff_charge = dataMS2Spectrum.LIST_PRECURSOR_CHARGE[i]
            buff_mass = dataMS2Spectrum.LIST_PRECURSOR_MASS[i]
            output_ms2_info.append([buff_file_name, buff_moz, buff_charge, buff_mass])

        for precursorMass in dataMS2Spectrum.LIST_PRECURSOR_MASS:
            # 预处理部分的顺序调整一下，调用的代码应该先解卷积，然后对母离子质量进行去除。
            copyMS2Spectrum = copy.deepcopy(dataMS2Spectrum)
            self.__captainPreprocessMS2(copyMS2Spectrum, precursorMass, complement)

            dataDAG = self.__captainMS2ToDAGbyInvertedIndex(copyMS2Spectrum, precursorMass)

            # 20210930 这里返回的是一个dict！！！用map！！！
            tag_list.append(self.__captainDAG2TagsList(dataDAG, precursorMass))

        return tag_list, output_ms2_info

    # 20211011  output: dict of x-tag
    # key:x     value:[list of tag]
    def __captainDAG2TagsList(self, dataDAG, inputPrecursorMass):

        buff_output_dict = dict()
        for tag_len_key in self.tagLengthList:
            buff_output_dict[tag_len_key] = []
        # inital ---------
        precursorMass = inputPrecursorMass
        # 整个buff_heap_list 是一个以结点数为元素数的顺序结构
        # 第二维是用户指定长度的list（tag_max_len + 1）
        # 第三维则是堆了，这里我们用的是最大堆，也就是score大的靠前
        buff_heap_list = [[[] for _i in range(self.max_tag_length + 1)] for _ in range(dataDAG.NUM_NODE)]
        zero_node_list = []  # buffer
        # 入度0的结点，没有边，所以不用细致记录
        for head_index in dataDAG.IN_ZERO_NODE:
            zero_node_list.append(head_index)
            lf_mass = dataDAG.LIST_NODE[head_index].MASS
            # python里最快的方法是用tuple，这里用类的话实际上会非常慢
            # buff_heap_list[head_index][0].append((lf_mass, -1, "", 0, ""))
            # 适合在C++里写的方法：因为数据类型不一样，所以用了个类，当然可以直接用结构体哈
            buff_heap_list[head_index][0].append((lf_mass, -1, "", 0, ""))  # lf, rf, seq, score, mod

        while zero_node_list:
            # 先提取一波信息
            # time1 = time.perf_counter()
            for zero_node_index in zero_node_list:
                zero_node = dataDAG.LIST_NODE[zero_node_index]
                # node_mass = zero_node.MASS
                # rf_mass = precursorMass - zero_node.MASS - self.dp.myINI.MASS_PROTON_MONO
                # - self.PROTON_MASS_H2O
                # rf_mass = precursorMass - dataDAG.LIST_NODE[link_node_index].MASS + self.dp.myINI.MASS_PROTON_MONO
                for tag_len_key in self.tagLengthList:
                    buff_output_dict[tag_len_key] += copy.deepcopy(buff_heap_list[zero_node_index][tag_len_key])
                # if tmp_buff_max_tag:
                #     buff_output_list += [(lf_mass, tag_name, rf_mass, score)]
            # time2 = time.perf_counter()
            # add_time += time2 - time1
            # 为接下来的操作做准备
            next_zero_node_list = []


            for zero_node_index in zero_node_list:

                zero_node = dataDAG.LIST_NODE[zero_node_index]
                # print("zero node:", zero_node_index)
                # node_mass = zero_node.MASS
                # rf_mass = precursorMass - zero_node.MASS - self.dp.myINI.MASS_PROTON_MONO
                # rf_mass = precursorMass - dataDAG.LIST_NODE[link_node_index].MASS + self.dp.myINI.MASS_PROTON_MONO
                for i in range(self.max_tag_length):
                    if len(buff_heap_list[zero_node_index][i]) > self.max_short_path_num:
                        # time1 = time.perf_counter()
                        # 这里实际上做了堆排序的操作
                        buff_heap_list[zero_node_index][i] = heapq.nlargest(self.max_short_path_num, buff_heap_list[zero_node_index][i], key=lambda x: x[3])
                        # time2 = time.perf_counter()
                        # sort_time += time2 - time1
                ex_path_list = buff_heap_list[zero_node_index]

                for edgeInfo in zero_node.LIST_EDGE:
                    # print("\tedge info:", edgeInfo)
                    # float
                    edge_score = edgeInfo.WEIGHT_INT_AND_DEGREE
                    # int
                    link_node_index = edgeInfo.LINK_NODE_INDEX
                    # float
                    peak_tol = edgeInfo.TOL
                    # next_node_mass = dataDAG.LIST_NODE[link_node_index].MASS
                    if 1 == self.msms_ppm:
                            self.compare_tol = self.msms_fraction * dataDAG.LIST_NODE[link_node_index].MASS
                            self.tolerance = 1.5 * self.compare_tol
                            self.tol_score_fm = -2 / (self.tolerance*self.tolerance)
                    # float
                    rf_mass = precursorMass - dataDAG.LIST_NODE[link_node_index].MASS - self.dp.myINI.MASS_PROTON_MONO
                    # 这里是枚举氨基酸信息的，0代表氨基酸char，1代表质量float，2代表修饰string，具体类型要看索引怎么存的
                    for aa_explain in self.aaMarkDict[edgeInfo.AA_MARK]:
                        # float
                        tmp_tol = abs(peak_tol - aa_explain[1])
                        if tmp_tol > self.compare_tol:
                            continue
                        # float
                        tol_edge_score = edge_score * math.exp(self.tol_score_fm * tmp_tol * tmp_tol)

                        # path_len:int, 代表了tag的长度
                        for path_len in range(self.max_tag_length):
                            # 枚举tag，t是CTagInfo类型
                            for t in ex_path_list[path_len]:
                                # self.lfMass = inputLM       # 左边质量  float
                                #         self.rfMass = inputRM       # 右边质量  float
                                #         self.tagSeq = inputSeq      # 序列标签  string
                                #         self.tagScore = inputScore  # 打分      float
                                #         self.tagMod = inputMod      # 修饰信息  string
                                # 判断一下是不是个空的字符串
                                if aa_explain[2]:
                                    # t_new_mod: string
                                    t_new_mod = t[4] + str(path_len+1) + "," + aa_explain[2] + ";"
                                    buff_heap_list[link_node_index][path_len + 1].append((t[0], rf_mass, t[2] + aa_explain[0], t[3] + tol_edge_score*0.75, t_new_mod))
                                else:
                                    # t_new_mod: string
                                    t_new_mod = t[4]
                                    buff_heap_list[link_node_index][path_len + 1].append((t[0], rf_mass, t[2] + aa_explain[0], t[3] + tol_edge_score, t_new_mod))



                    # 这部分信息整完了，去掉一个入度
                    dataDAG.LIST_NODE[link_node_index].IN -= 1

                    if dataDAG.LIST_NODE[link_node_index].IN == 0:

                        lf_mass = dataDAG.LIST_NODE[link_node_index].MASS

                        buff_heap_list[link_node_index][0].append((lf_mass, -1, "", 0, ""))
                        # lf, rf, seq, score, mod
                        next_zero_node_list.append(link_node_index)



                # [loop over]for edgeInfo in zero_node.LIST_EDGE



            # [loop over]for zero_node_index in zero_node_list
            zero_node_list = next_zero_node_list

        # 功能：如果个数超了，就用堆排整理一下，只取打分top-K的tagInfo
        for tmp_tag_len in self.tagLengthList:
            if len(buff_output_dict[tmp_tag_len]) >= self.max_short_path_num:
                buff_output_dict[tmp_tag_len] = heapq.nlargest(self.max_short_path_num, buff_output_dict[tmp_tag_len], key=lambda x: x[3])
            else:
                buff_output_dict[tmp_tag_len] = sorted(buff_output_dict[tmp_tag_len], key=lambda x: x[3], reverse=True)

        return buff_output_dict

    def __captainPreprocessMS2(self, dataMS2Spectrum, inputPrecursorMass, complement=False):
        # 20201110  调整执行次序
        # 1. holding top-N peaks
        ori_peaks_num = len(dataMS2Spectrum.LIST_PEAK_INT)
        # self.__captainPreprocessMS2_1st(dataMS2Spectrum)

        trans_peaks_num = len(dataMS2Spectrum.LIST_PEAK_INT)

        # 之所以拿出来是因为，上述代码可以复用，就不需要再反复做解卷积操作了
        # 4. removing ions with precursor
        # if self.dp.myCFG.B3_REMOVE_PRECURSOR_ION == 1:
        #     ...
            # self.__soldierRemovePrecursorPeak(dataMS2Spectrum, inputPrecursorMass)

        final_peaks_num = len(dataMS2Spectrum.LIST_PEAK_INT)

        # 5. relative int transforming
        # self.__soldierAbsoluteINT2RelativeINT(dataMS2Spectrum)

        if complement:
            self.__soldierGenerateComplementPeak(dataMS2Spectrum, inputPrecursorMass)

        if final_peaks_num == 0:
            with open(self.dp.myCFG.E1_PATH_EXPORT + "total.emptySpectrum.list", "a", encoding="utf-8") as f:
                precursor_order = dataMS2Spectrum.LIST_PRECURSOR_MASS.index(inputPrecursorMass)
                file_name = dataMS2Spectrum.LIST_FILE_NAME[precursor_order]
                f.write(file_name + "\n")
                f.write("original peaks num:" + str(ori_peaks_num) + "\n")
                f.write("after-trans peaks num:" + str(trans_peaks_num) + "\n\n")


    def __captainMS2ToDAGbyInvertedIndex(self, dataMS2Spectrum, inputPrecursorMass):

        dataDAG = self.__soldierGetInitTagDAG(dataMS2Spectrum, inputPrecursorMass)

        if self.lenTermMassIndex > 0:
            self.__soldierCheckTermAAbyInvertedIndex(dataDAG, inputPrecursorMass)
        # double for loop to add the edge
        for left_node_index in range(dataDAG.NUM_NODE - 1):

            left_mass = dataDAG.LIST_NODE[left_node_index].MASS

            # 右边范围
            for right_node_index in range(left_node_index + 1, dataDAG.NUM_NODE):

                right_mass = dataDAG.LIST_NODE[right_node_index].MASS
                index_by_calc = toolMyIntRound(self.fold * (right_mass - left_mass))
                # calc mass tolerance * 1000,  and then get the int

                # 如果两根峰差距特别大，则直接break；左节点右移，右节点重置
                if index_by_calc >= self.lenAAMassIndex:
                    break

                aa_info_mark = self.aaMassIndex[index_by_calc]

                # 间隔太小，右节点右移；概率应该最大，所以填第一位
                if aa_info_mark is None:
                    continue

                # 符合条件，添加边
                else:
                    op_ADD_EDGE_CTagDAG(dataDAG, left_node_index, right_node_index, right_mass - left_mass, aa_info_mark)

        # 遍历NODE；添加路径起点
        for node_index in range(dataDAG.NUM_NODE):
            # 该点无出边，跳过，找下一个点
            if 0 == dataDAG.LIST_NODE[node_index].OUT:
                continue
            # 该点有出边，干！
            else:
                # 如果有出边且IN为0，则为路径起始点；入0出0，孤立点，直接不考虑
                if 0 == dataDAG.LIST_NODE[node_index].IN:
                    dataDAG.IN_ZERO_NODE.append(node_index)

                # 打分部分，提取tag时再计算TOL：计算边的权值，并更新
                for edge_index in range(dataDAG.LIST_NODE[node_index].OUT):

                    next_node_index = dataDAG.LIST_NODE[node_index].LIST_EDGE[edge_index].LINK_NODE_INDEX
                    dataDAG.LIST_NODE[node_index].LIST_EDGE[edge_index].WEIGHT_INT_AND_DEGREE = dataDAG.LIST_NODE[node_index].LIST_EDGE[edge_index].WEIGHT

                    # tmpN = dataDAG.NUM_NODE
                    tmpInAndOut = dataDAG.LIST_NODE[node_index].OUT + dataDAG.LIST_NODE[next_node_index].IN

                    score_in_and_out = math.sqrt(2 / tmpInAndOut)
                    dataDAG.LIST_NODE[node_index].LIST_EDGE[edge_index].WEIGHT_INT_AND_DEGREE *= score_in_and_out

        return dataDAG


    def __soldierCheckTermAAbyInvertedIndex(self, dataDAG, inputPrecursorMass):

        # left peak, check small ---------------------------------------
        for left_node_index in range(dataDAG.NUM_NODE - 1):
            left_mass = dataDAG.LIST_NODE[left_node_index].MASS
            if math.isclose(left_mass, 0.0):
                pass
            elif math.isclose(left_mass, self.PROTON_MASS_H2O):
                pass
            else:
                continue

            # 右边范围
            for right_node_index in range(left_node_index + 1, dataDAG.NUM_NODE):

                right_mass = dataDAG.LIST_NODE[right_node_index].MASS
                index_by_calc = int(self.fold * (right_mass - left_mass))
                # 如果两根峰差距特别大，则直接break；左节点右移，右节点重置
                if index_by_calc >= self.lenTermMassIndex:
                    break
                # print(index_by_calc, len(self.aaMassIndex))
                aa_info_mark = self.termMassIndex[index_by_calc]
                # 间隔太小，右节点右移；概率应该最大，所以填第一位
                if aa_info_mark is None:
                    continue

                # 符合条件，添加边
                else:
                    op_ADD_EDGE_CTagDAG(dataDAG, left_node_index, right_node_index, right_mass - left_mass, aa_info_mark)

        # right peak, check big ----------------------------------------
        if dataDAG.NUM_NODE > 1:

            for right_node_index in range(1, dataDAG.NUM_NODE):

                right_mass = dataDAG.LIST_NODE[right_node_index].MASS

                if math.isclose(right_mass, inputPrecursorMass - self.dp.myINI.MASS_PROTON_MONO - self.PROTON_MASS_H2O, abs_tol=0.05):
                    pass
                elif math.isclose(right_mass, inputPrecursorMass - self.dp.myINI.MASS_PROTON_MONO, abs_tol=0.05):
                    pass
                else:
                    continue

                # 左边范围, 逆序枚举
                for left_node_index in reversed(range(0, dataDAG.NUM_NODE-1)):

                    left_mass = dataDAG.LIST_NODE[left_node_index].MASS
                    index_by_calc = int(self.fold * (right_mass - left_mass))

                    # 如果两根峰差距特别大，则直接break；左节点右移，右节点重置
                    if index_by_calc >= self.lenTermMassIndex:
                        break
                    # print(index_by_calc, len(self.aaMassIndex))
                    aa_info_mark = self.termMassIndex[index_by_calc]

                    # 间隔太小，右节点右移；概率应该最大，所以填第一位
                    if aa_info_mark is None:
                        continue

                    # 符合条件，添加边
                    else:
                        op_ADD_EDGE_CTagDAG(dataDAG, left_node_index, right_node_index, right_mass - left_mass,
                                            aa_info_mark)

        pass


    def __soldierGenerateComplementPeak(self, dataMS2Spectrum, precursorMass):

        # add the rebuild node(body 2/2)
        lenPeakList = len(dataMS2Spectrum.LIST_PEAK_MOZ)
        if 0 == lenPeakList:
            pass
        else:
            rebuildPeakMOZ = [0 for _ in range(lenPeakList)]
            rebuildPeakINT = [0 for _ in range(lenPeakList)]
            # rebuildPeakNodeList = [CTagDAGNode() for _ in range(lenPeakList)]
            # 质量从小到大排列
            for i in range(lenPeakList):
                # from b/y ions generate y/b ions
                # 因为此时y的质量都是残基了，而b多减去了一个H2O，所以母离子减的时候会多减去一遍H2O
                rebuildPeakMOZ[lenPeakList - i - 1] = precursorMass - dataMS2Spectrum.LIST_PEAK_MOZ[i] + self.dp.myINI.MASS_PROTON_MONO
                rebuildPeakINT[lenPeakList - i - 1] = dataMS2Spectrum.LIST_PEAK_INT[i] / 2
                # 尝试折半构造打分，y离子多，则得到连续的b强度更低

            # merge the body
            # =================================================================
            new_peak_moz = dataMS2Spectrum.LIST_PEAK_MOZ + rebuildPeakMOZ
            new_peak_int = dataMS2Spectrum.LIST_PEAK_INT + rebuildPeakINT
            arg_index_list = np.argsort(new_peak_moz)

            new_peak_moz = [new_peak_moz[i] for i in arg_index_list]
            new_peak_int = [new_peak_int[i] for i in arg_index_list]

            output_peak_moz = []
            output_peak_int = []

            # 注意  下面的代码直接借用了同位素峰簇转换最后的部分，经过许久的测试，我觉得没有什么问题
            # 下方：合并转换的谱峰，有可能会影响精度
            add_moz = 0
            add_int = 0
            i = 0
            jump_set = set()
            while i < len(new_peak_moz) - 1:
                add_moz = new_peak_moz[i]
                add_int = new_peak_int[i]

                if i in jump_set:
                    i += 1
                    continue

                for j in range(i + 1, len(new_peak_moz)):
                    if self.msms_ppm == 1:

                        prep_tolerance = new_peak_moz[j] * self.msms_fraction
                    else:
                        prep_tolerance = self.msms_tol


                    if abs(new_peak_moz[i] - new_peak_moz[j]) < prep_tolerance:
                        add_moz = add_moz * add_int + new_peak_moz[j] * new_peak_int[j]
                        add_int += new_peak_int[j]
                        add_moz /= add_int
                        i = j
                        jump_set.add(j)
                    # 仅看左右最远的0.02，两两之间就不看了
                    elif abs(new_peak_moz[i] - new_peak_moz[j]) >= prep_tolerance:
                        output_peak_moz.append(add_moz)
                        output_peak_int.append(add_int)
                        i = j
                        break

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

            dataMS2Spectrum.LIST_PEAK_MOZ = output_peak_moz
            dataMS2Spectrum.LIST_PEAK_INT = output_peak_int

    # 20210902  尝试：生成互补离子的强度值相比于原始值折半
    def __soldierGetInitTagDAG(self, dataMS2Spectrum, inputPrecursorMass):

        dataDAG = CTagDAG()
        lenPeakList = len(dataMS2Spectrum.LIST_PEAK_MOZ)
        if lenPeakList == 0:
            avrg_int = 0
        else:
            avrg_int = sum(dataMS2Spectrum.LIST_PEAK_INT)/lenPeakList
        # 修改完善20201022
        precursorMass = inputPrecursorMass

        # 必须要有这个，否则会有很奇怪的内存不释放问题
        dataDAG.LIST_NODE = []  # init
        dataDAG.IN_ZERO_NODE = []

        # create the source node and destiny node
        headNode = CTagDAGNode()
        # headNode.MASS = self.dp.myINI.MASS_PROTON_MONO
        # 必须是0！！！否则delta mass就不对了嗷！
        headNode.MASS = 0.0
        headNode.INTENSITY = avrg_int

        tailNode = CTagDAGNode()
        tailNode.MASS = precursorMass - self.dp.myINI.MASS_PROTON_MONO
        #  - self.PROTON_MASS_H2O
        # 不管是b还是y，最终都成为了sigma氨基酸残基质量
        tailNode.INTENSITY = avrg_int

        headNode2 = CTagDAGNode()
        headNode2.MASS = self.PROTON_MASS_H2O
        headNode2.INTENSITY = avrg_int

        tailNode2 = CTagDAGNode()
        tailNode2.MASS = precursorMass - self.dp.myINI.MASS_PROTON_MONO - self.PROTON_MASS_H2O
        tailNode2.INTENSITY = avrg_int


        # 目的：生成互补的离子，若b有缺失，则由precursorMass - y来弥补
        # add the normal node(body 1/2)
        originalPeakNodeList = [CTagDAGNode() for _ in range(lenPeakList)]
        for i in range(lenPeakList):
            # originalPeakNodeList[i].MASS = dataMS2Spectrum.LIST_PEAK_MOZ[i] - self.dp.myINI.MASS_PROTON_MONO
            # b ions extraction
            originalPeakNodeList[i].MASS = dataMS2Spectrum.LIST_PEAK_MOZ[i] - self.dp.myINI.MASS_PROTON_MONO
            #  - self.PROTON_MASS_H2O
            # y ions extraction
            originalPeakNodeList[i].INTENSITY = dataMS2Spectrum.LIST_PEAK_INT[i]


        dataDAG.LIST_NODE = [headNode, headNode2] + originalPeakNodeList + [tailNode2, tailNode]
        # dataDAG.LIST_NODE = [headNode] + originalPeakNodeList + [tailNode]

        # for i in range(1, len(dataDAG.LIST_NODE)):
        #     if abs(dataDAG.LIST_NODE[i].MASS - dataDAG.LIST_NODE[i-1].MASS) <= 0.02:
        #         print(1)

        dataDAG.NUM_NODE = len(dataDAG.LIST_NODE)

        for i in range(dataDAG.NUM_NODE):
            dataDAG.LIST_NODE[i].LIST_EDGE = []

        return dataDAG


class CFunctionAAMassIndex:

    def __init__(self, inputDP):
        self.dp = inputDP
        self.fold = 1000
        
        if 1 == self.dp.myCFG.D6_MSMS_TOL_PPM:
            self.abs_tol = 1250 * self.dp.myCFG.D5_MSMS_TOL / 1e6
        else:
            self.abs_tol = 1.25 * self.dp.myCFG.D5_MSMS_TOL

        self.step_len = int(self.abs_tol * self.fold)


    def generateDictAndIndex(self, inputAAType=1, inputModType=0):

        aaMassSetDict, NCTermSetDict = self.__captainGenerateAAMassSetDict(inputAAType, inputModType)

        outputAAMarkDict, outputAAMassIndex, termAAMassIntertedIndex = self.__captainGenerateAAMassInvertedIndex(aaMassSetDict, NCTermSetDict)

        return outputAAMarkDict, outputAAMassIndex, termAAMassIntertedIndex


    def __captainGenerateAAMassInvertedIndex(self, inputAAMassSetDict, inputNCTermSetDict):
        # find the max num of set dict
        max_mass = -1
        for aaName in inputAAMassSetDict:
            for aaInfoTuple in inputAAMassSetDict[aaName]:
                if aaInfoTuple[0] > max_mass:
                    max_mass = aaInfoTuple[0]

        if max_mass <= 0:
            logGetError("Error: check the aa mass, log get in tag extraction")

        # 从1下标开始，避开0
        outputAAMarkDict = dict()

        # 这里我写的虽然是None，但是因为上面避开了0，所以可以统一为0！
        outputAAMassInvertedIndex = [None for _ in range(int(math.ceil(max_mass * self.fold + self.step_len * 2)))]
        # +1, 对齐index; +3, 对齐向后两个单位；因为设置的是0.03 Da
        # 20210930 设置的多了一些，问题不大

        new_add_index = 1

        for aaName in inputAAMassSetDict:
            for aa_mass, modinfo in inputAAMassSetDict[aaName]:
                trans_index = toolMyIntRound(aa_mass * self.fold)

                # -0.025Da  -->  +0.025Da
                # new: -0.030Da --> +0.030Da
                for tmpIndex in range(trans_index - self.step_len, trans_index + self.step_len + 1):
                    # for tmpIndex in range(trans_index - 1, trans_index + 2):
                    if outputAAMassInvertedIndex[tmpIndex] == None:
                        # first time add this message
                        outputAAMassInvertedIndex[tmpIndex] = new_add_index
                        outputAAMarkDict[new_add_index] = {(aaName, aa_mass, modinfo)}
                        new_add_index += 1
                    else:
                        add_index = outputAAMassInvertedIndex[tmpIndex]
                        outputAAMarkDict[add_index].add((aaName, aa_mass, modinfo))


        if inputNCTermSetDict:
            # find the max num of set dict
            max_mass = -1
            for aaName in inputNCTermSetDict:

                # print(aaName, str(inputNCTermSetDict[aaName]))

                for aaInfoTuple in inputNCTermSetDict[aaName]:
                    if aaInfoTuple[0] > max_mass:
                        max_mass = aaInfoTuple[0]
            # print("max_mass:", max_mass)
            if max_mass <= 0:
                logGetError("Error: check the aa mass, log get in tag extraction")

            specialAAMassIntertedIndex = [None for _ in range(int(math.ceil(max_mass * self.fold + self.step_len * 2)))]
            # print("new index len:", len(specialAAMassIntertedIndex))
            for aaName in inputNCTermSetDict:
                for aa_mass, modinfo in inputNCTermSetDict[aaName]:
                    trans_index = toolMyIntRound(aa_mass * self.fold)
                    # print("aa mass:", aa_mass, "\ttrans index:", trans_index)

                    # -0.025Da  -->  +0.025Da
                    # new: -0.030Da --> +0.030Da
                    for tmpIndex in range(trans_index - self.step_len, trans_index + self.step_len + 1):
                        # for tmpIndex in range(trans_index - 1, trans_index + 2):
                        if specialAAMassIntertedIndex[tmpIndex] == None:
                            # first time add this message
                            specialAAMassIntertedIndex[tmpIndex] = new_add_index
                            outputAAMarkDict[new_add_index] = {(aaName, aa_mass, modinfo)}
                            new_add_index += 1
                        else:
                            add_index = specialAAMassIntertedIndex[tmpIndex]
                            outputAAMarkDict[add_index].add((aaName, aa_mass, modinfo))
        else:
            specialAAMassIntertedIndex = []


        return outputAAMarkDict, outputAAMassInvertedIndex, specialAAMassIntertedIndex


    def __captainGenerateAAMassSetDict(self, inputAAType, inputModType):
        # inputAAType:取1或2，其余的暂时先不考虑
        # 1: 仅仅考虑1个氨基酸
        # 2: 最多2个氨基酸组合
        # inputModType:
        # 0: 不考虑修饰  20210907  考虑用户设定的修饰
        # 1: 考虑common修饰
        # 2: 考虑全部修饰
        # 3: 考虑CNCP排名前10的修饰
        # 4: 考虑用户设定的固定修饰用于标签的搜索  20210907
        # 5: 考虑用户设定的可变修饰用于标签的搜索（再说再说。。）

        # outputNCTermList = []

        tmpSingleAAMassDict = dict()
        # firstly consider single aa residue
        aaMass = self.dp.myINI.DICT1_AA_MASS
        for aa_bias in aaMass:
            if math.isclose(aaMass[aa_bias], 0.0):
                continue
            tmpSingleAAMassDict[chr(aa_bias + ord("A"))] = (aaMass[aa_bias], "")

        # 更新fix mod的信息
        if self.dp.myCFG.A3_FIX_MOD:
            for mod in (self.dp.myCFG.A3_FIX_MOD).split("|"):
                if mod == "":
                    continue
                if self.dp.myINI.DICT2_MOD_INFO[mod].POSITION == "NORMAL":
                    for site in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                        if site in tmpSingleAAMassDict:
                            tmpMass = tmpSingleAAMassDict[site][0]
                            tmpSingleAAMassDict[site] = (tmpMass, mod)
                else:
                    pass
                    # tmpNCTermModDict
                    # outputNCTermList.append(mod)


        # consider case of modification
        tmpModDict = dict()
        tmpNCTermModDict = dict()
        if inputModType == 0:
            # no modification
            pass
        elif inputModType == 1:
            # only common modification
            for mod in self.dp.myINI.DICT2_MOD_INFO:
                if self.dp.myINI.DICT2_MOD_INFO[mod].COMMON and self.dp.myINI.DICT2_MOD_INFO[mod].POSITION == "NORMAL":
                    for site in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                        if site in tmpModDict:
                            tmpModDict[site].add((self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod))
                        else:
                            tmpModDict[site] = {(self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod)}
                else:
                    for site in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                        if site in tmpModDict:
                            tmpNCTermModDict[site].add((self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod))
                        else:
                            tmpNCTermModDict[site] = {(self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod)}

        elif inputModType == 2:
            # all of modification
            for mod in self.dp.myINI.DICT2_MOD_INFO:
                if self.dp.myINI.DICT2_MOD_INFO[mod].POSITION == "NORMAL":
                    for site in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                        if site in tmpModDict:
                            tmpModDict[site].add((self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod))
                        else:
                            tmpModDict[site] = {(self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod)}

                else:
                    for site in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                        if site in tmpModDict:
                            tmpNCTermModDict[site].add((self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod))
                        else:
                            tmpNCTermModDict[site] = {(self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod)}

        elif inputModType == 3:
            # cncp 2020 top10 modifications
            # 20201116补充
            tmp_cncp_top10 = ["Carbamidomethyl[C]", "Gln->pyro-Glu[AnyN-termQ]", "Pyro-carbamidomethyl[AnyN-termC]",
                              "Deamidated[N]", "Oxidation[M]", "Carbamidomethyl[M]", "Met-loss+Acetyl[ProteinN-termM]",
                              "Carbamyl[AnyN-term]", "Carbamidomethyl[AnyN-term]", "Glu->pyro-Glu[AnyN-termE]"]
            tmp_cncp_top10 = ["Carbamidomethyl[C]", "Deamidated[N]", "Oxidation[M]", "Carbamidomethyl[M]"]

            for mod in tmp_cncp_top10:
                    for site in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                        if site in tmpModDict:
                            tmpModDict[site].add((self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod))
                        else:
                            tmpModDict[site] = {(self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod)}

        elif inputModType == 4:
            # the separator char ";" need to be controled in a const
            for mod in (self.dp.myCFG.A3_FIX_MOD).split("|"):
                if mod == "":
                    continue
                if self.dp.myINI.DICT2_MOD_INFO[mod].POSITION == "NORMAL":
                    for site in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                        if site in tmpModDict:
                            tmpModDict[site].add((self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod))
                        else:
                            tmpModDict[site] = {(self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod)}

                else:
                    for site in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                        if site in tmpModDict:
                            tmpNCTermModDict[site].add((self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod))
                        else:
                            tmpNCTermModDict[site] = {(self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod)}


        elif inputModType == 5:
            # the separator char ";" need to be controled in a const

            for mod in (self.dp.myCFG.A4_VAR_MOD).split("|"):
                # print(mod)
                if mod == "":
                    continue
                elif self.dp.myINI.DICT2_MOD_INFO[mod].POSITION == "NORMAL":
                    for site in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                        if site in tmpModDict:
                            tmpModDict[site].add((self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod))
                        else:
                            tmpModDict[site] = {(self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod)}
                else:
                    for site in self.dp.myINI.DICT2_MOD_INFO[mod].SITES:
                        if site in tmpNCTermModDict:
                            tmpNCTermModDict[site].add((self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod))
                        else:
                            tmpNCTermModDict[site] = {(self.dp.myINI.DICT2_MOD_INFO[mod].MASS, mod)}

        else:
            logGetError("check tag extraction param: inputModType")

        outputAAMassSetDict = dict()
        specialMassSetDict = dict()
        # mod dict is not empty
        if tmpModDict:
            for aa in tmpSingleAAMassDict:
                aa_mono_mass = tmpSingleAAMassDict[aa][0]
                if math.isclose(aa_mono_mass, 0.0):
                    continue
                try:
                    add_mass = {(aa_mono_mass + delta_mass[0], delta_mass[1]) for delta_mass in tmpModDict[aa]}
                    add_mass.add((aa_mono_mass, ""))
                except:
                    add_mass = {(aa_mono_mass, "")}
                outputAAMassSetDict[aa] = add_mass
        else:
            for aa in tmpSingleAAMassDict:
                outputAAMassSetDict[aa] = {tmpSingleAAMassDict[aa]}

        if tmpNCTermModDict:
            for aa in tmpSingleAAMassDict:
                aa_mono_mass = tmpSingleAAMassDict[aa][0]
                if math.isclose(aa_mono_mass, 0.0):
                    continue
                if aa in tmpNCTermModDict:
                    if tmpNCTermModDict[aa]:
                        add_mass = {(aa_mono_mass + delta_mass[0], delta_mass[1]) for delta_mass in tmpNCTermModDict[aa]}
                        specialMassSetDict[aa] = add_mass

        if inputAAType == 1:
            # for item in outputAAMassSetDict:
            #     print(item, ":", outputAAMassSetDict[item])
            # 仅考虑1个氨基酸

            return outputAAMassSetDict, specialMassSetDict

        elif inputAAType == 2:
            # 考虑2个氨基酸了
            tmpDoubleAAMassSetDict = dict()
            for first_aa in outputAAMassSetDict:
                for second_aa in outputAAMassSetDict:

                    aa_aa = first_aa + second_aa
                    mass_list = []
                    for first_aa_mass in outputAAMassSetDict[first_aa]:
                        mass_list += [(first_aa_mass[0] + second_aa_mass[0], first_aa_mass[1] + ";" + second_aa_mass[1]) for second_aa_mass in outputAAMassSetDict[second_aa]]

                    tmpDoubleAAMassSetDict[aa_aa] = set(mass_list)

            outputAAMassSetDict.update(tmpDoubleAAMassSetDict)

            return outputAAMassSetDict, specialMassSetDict

        else:
            # 非1非2，先不报错了，暂且这样处理
            return outputAAMassSetDict, specialMassSetDict



