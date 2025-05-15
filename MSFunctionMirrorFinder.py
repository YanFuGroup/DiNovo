import math, time
from MSData import CFileMS2
from MSTool import toolMyIntRound, toolMyIntRoundNegative
from MSLogging import logToUser
import numpy as np

class CFunctionMirrorFinder:

    def __init__(self, inputDP, precursor_list, pm_idx):
        self.dp = inputDP
        self.neucode_label = True if self.dp.myCFG.A7_LABEL_TYPE == 1 else False
        self.fold = 10
        self.evalueFilter = True if self.dp.myCFG.C10_PAIR_FILTER_APPROACH == 1 else False
        self.PRECURSOR_LIST = precursor_list
        self.PMIDX = pm_idx
        self.R_MASS = self.dp.myINI.DICT1_AA_MASS[ord("R") - ord("A")]
        self.K_MASS = self.dp.myINI.DICT1_AA_MASS[ord("K") - ord("A")]
        # self.R_MASS = 156.10111
        # self.K_MASS = 128.09496
        if self.neucode_label:
            self.R_MASS = 160.08925
            self.K_MASS = 136.10916

        # 控制权值的个数
        self.maxWeightNum = max(1000, 3 * self.dp.myCFG.B5_ROUND_ONE_HOLD_PEAK_NUM)
        self.weight = self.__generateWeightOfPeakRank()

        # mirror A 类型里的类型有三种，根据用户设定的类型
        # 后面枚举对应的delta进行匹配，进行谱峰间差值的匹配打分，相互竞争，高分者得
        self.MIRROR_A = self.__getMirrorDeltaInfo_A()

        # mirror B -> G 的类型有六种，根据用户设定的类型
        # 同样枚举对应的delta进行匹配，进行谱峰间差值的匹配打分，但是应该怎么过滤呢？
        # 打分过滤的问题还没有想好，暂时先使用子璇的 N^2 级别的匹配打分
        self.MIRROR_B2G = self.__getMirrorDeltaInfo_B2G()

        self.decoy_shift = self.dp.myCFG.C13_PAIR_DECOY_SHIFTED_IN_DA
        # add 5Da for each expr fragment ion delta mass

        if self.dp.myCFG.C12_PAIR_DECOY_APPROACH == 2:
            self.MIRROR_A_DECOY, self.MIRROR_B2G_DECOY = dict(), dict()
            self.__geneDecoyDelta_A2G()

        self.ms_tol = self.dp.myCFG.D3_MS_TOL
        self.ms_ppm = self.dp.myCFG.D4_MS_TOL_PPM
        self.ms_tol_boundary = 2 * self.ms_tol

        # 与母离子质量相乘，即可转化为绝对质量差
        self.fraction = (self.ms_tol * 1e-6) if self.ms_ppm else 1

        # 直接调用该方法来计算
        self.checkPrecursorMass = self.__checkPrecursorMass_PPM if self.ms_ppm else self.__checkPrecursorMass_Da

        # ********************************************************
        # 子璇的打分部分 -----------------------------------------
        self.bin = 0.0400321 if self.neucode_label else 0.07956211 / 4
        self.interval = 400  # 只考虑[-200, 200]
        self.bin_num = math.ceil(self.interval / self.bin)
        if (self.bin_num % 2 == 0):
            self.bin_num -= 1
        # 保证 bin_num 是奇数

        self.one_over_bin = 1 / self.bin  # to accelerate scoring speed
        self.half_bin_num = (self.bin_num + 1) // 2  # to accelerate scoring speed

        self.half_interval = self.interval // 2
        self.half_bin_add_interval = self.bin/2 + self.half_interval

        self.masslower = - (self.bin_num * self.bin) / 2
        self.massupper = (self.bin_num * self.bin) / 2

        self.alpha1 = 0.3
        self.alpha2 = 0.7

        self.log_label = True
        # self.log_for_evalue = self.__generateWeightOfPeakRank()
        # 子璇的打分部分 -----------------------------------------

        # p-Value calculate
        self.MIRROR_A_DELTA_IDX = self.__getIdxInfoTypeA()
        self.MIRROR_B2G_DELTA_IDX = self.__getIdxInfoTypeB2G()
        # neuCode check score
        self.MIRROR_A2G_BY_DELTA_IDX_SET = self.__getIdxSetTypeAll()

        if self.dp.myCFG.C12_PAIR_DECOY_APPROACH == 2:
            self.MIRROR_A_DECOY_DELTA_IDX = self.__getIdxInfoTypeA(flag=False)
            self.MIRROR_B2G_DECOY_DELTA_IDX = self.__getIdxInfoTypeB2G(flag=False)
            # neuCode check score
            self.MIRROR_A2G_DECOY_BY_DELTA_IDX_SET = self.__getIdxSetTypeAll(flag=False)
            # 20240415 这里先不做调整，算法会变得比较复杂，改动很大

        # ********************************************************

        # 0:gene complement, 1:score, 2:judge
        self.time_record = [0.0, 0.0, 0.0]
        # 0:gene scorePad, 1:double for loop, 2:product, 3:sort in judge
        self.time_score = [0.0, 0.0, 0.0, 0.0]



        # ========================================================
        # [all of them below] for accelerating!!
        # self.one_hot = np.eye((self.bin_num + 1))  # for accelerating
        self.try_moz = None  # [np.array variable]  trypsin spectra peak moz
        self.try_int = None  # [np.array variable]  trypsin spectra peak int
        self.try_ncl = None  # [np.array variable]  trypsin psectra peak neucode label [[ <SPECIAL> ]]
        self.try_wei1 = None # [np.array variable]  trypsin spectra peak moz rank weight * 0.3
        self.try_wei2 = None # [np.array variable]  trypsin spectra peak moz rank weight * 0.7
        self.try_int_x_wei1 = None  # [np.array variable]  trypsin spectra peak int * moz rank weight * 0.3
        self.try_int_x_wei2 = None  # [np.array variable]  trypsin spectra peak int * moz rank weight * 0.7
        self.try_flag = None  # [np.arrat variable]  trypsin spectra peak from ori[True] / com[False]
        self.try_flag_ = None
        # t_wei3 = self.weight[i % len(tryINT)] * 0.3
        # t_wei7 = self.weight[i % len(tryINT)] * 0.7
        # t_int_x_t_wei3 = tryINT[i % len(tryINT)] * t_wei3
        # t_int_x_t_wei7 = tryINT[i % len(tryINT)] * t_wei7
        # ========================================================
        # 20230113 test ===== for remove the score of other aa mass
        # self.zeroIndex = self.__generateZeroIndex()
        # print("function initial OK!")

        # calculate the new judge
        if self.dp.myCFG.C12_PAIR_DECOY_APPROACH == 0:
            self.__soldierGetJudgeAnswerB2G = self.__soldierGetJudgeAnswerB2G0
            self.__soldierGetJudgeAnswerA   = self.__soldierGetJudgeAnswerA0

        elif self.dp.myCFG.C12_PAIR_DECOY_APPROACH == 2:
            self.__soldierGetJudgeAnswerB2G = self.__soldierGetJudgeAnswerB2G2
            self.__soldierGetJudgeAnswerA = self.__soldierGetJudgeAnswerA2

        else:
            self.__soldierGetJudgeAnswerB2G = self.__soldierGetJudgeAnswerB2G13
            self.__soldierGetJudgeAnswerA = self.__soldierGetJudgeAnswerA13


    def logTimeCost(self):
        res = "\n[Total Pair Cost]"
        res += "\n[Total Rec] %.3fs." % sum(self.time_record)
        res += "\n[Comp Peak] %.3fs." % (self.time_record[0] + 0.0)
        res += "\n[Score All] %.3fs." % (self.time_record[1] + 0.0)
        res += "\n[Judge Res] %.3fs." % (self.time_record[2] + 0.0)
        res += "\n[Score And Judge]"
        res += "\n[Score Pad] %.3fs." % (self.time_score[0] + 0.0)
        res += "\n[Doub Loop] %.3fs." % (self.time_score[1] + 0.0)
        res += "\n[Hit X Int] %.3fs." % (self.time_score[2] + 0.0)
        res += "\n[JudgeSort] %.3fs." % (self.time_score[3] + 0.0)
        logToUser(res)

    def loadTrypsinSpecInfo(self, inputTryMOZList:list, inputTryINTList:list, inputTryNCLList, inputTryPM:float):
        # array shape: (N x 1)
        # trans to a matrix! easy to use 2-d idx(i for trypsin, j for lysargiNase)
        self.try_moz = np.array([inputTryMOZList])
        self.try_moz = np.concatenate((self.try_moz, self.__soldierGenerateComponentMOZList(self.try_moz, inputTryPM)), axis=1).T
        ori_num, com_num = len(inputTryINTList), len(self.try_moz) - len(inputTryINTList)
        self.try_int = np.array([inputTryINTList])
        self.try_int = np.concatenate((self.try_int, self.try_int[:, :com_num]), axis=1).T
        # self.try_int = np.array([inputTryINTList + inputTryINTList[:com_num]]).T
        if self.dp.myCFG.C12_PAIR_DECOY_APPROACH == 3:
            self.try_moz -= 5
        if self.neucode_label:
            self.try_ncl = np.array([inputTryNCLList], np.bool)
            self.try_ncl = np.concatenate((self.try_ncl, ~self.try_ncl[:, :com_num]), axis=1).T
        # print("mass1 <- c(", ", ".join([str(t[0]) for t in self.try_moz]), ")")
        # print("inten1 <- c(", ", ".join([str(t[0]) for t in self.try_int]), ")")
        # print("neucode1 <- c(", ", ".join([("1" if t[0] else "0") for t in self.try_ncl]), ")")
        extract_wei = np.concatenate((self.weight[:, :ori_num], self.weight[:, :com_num]), axis=1).T
        self.try_wei1 = extract_wei * self.alpha1  # ori, com        trypsin spectra peak moz rank weight * 0.3
        self.try_wei2 = extract_wei * self.alpha2  # ori, com        trypsin spectra peak moz rank weight * 0.7
        self.try_int_x_wei1 = self.try_int * self.try_wei1  # ori, com   trypsin spectra peak moz rank weight * 0.3
        self.try_int_x_wei2 = self.try_int * self.try_wei2  # ori, com   trypsin spectra peak moz rank weight * 0.7
        self.try_flag = np.array([[True if i < ori_num else False for i in range(len(self.try_moz))]]).T
        # 如果是空矩阵，那么会导致这里报错（取反的操作会报错）；但是又没见过其他的错误类型，所以这里就直接改成如下逻辑了
        try:
            self.try_flag_ = ~self.try_flag
        except:
            self.try_flag_ = np.array([[False if i < ori_num else True for i in range(len(self.try_moz))]]).T

    # [attention] 20221012
    # 这个函数的输入参数太多了，后期看看能不能调整一下
    def scoreSpectrumPair(self, inputLysMOZList:list, inputLysINTList:list, inputLysNCLList, inputLysPM:float, MirrorType:int):

        # [ATTENTION] 20221012 在这一步中，并没有办法使用哈希表加速
        # 子璇这里的算法只能硬生生的使用暴力循环
        # [必要步骤组件]
        # 1. 生成互补峰列表（列表 或 列表对儿？）  PM + Proton - peakMOZ  ** CHECK MASS BIGGER THAN PRECURSOR! **
        # 2. 生成质荷比权重（列表 或 列表对儿？）  1 / (1 + exp(-i))   i 越大权重越大，为 [0.5, 1) 的单调上升函数
        # 3. 初始化分数矩阵（仅用 一个列表 即可）  使用了哈希方法来记录 matching num 和 hitting int，从而计算打分
        # 4. 究极暴力法打分（填充好打分的矩阵！）  double for loop

        if (len(self.try_moz) == 0) or (len(inputLysMOZList) == 0):
            return (MirrorType, np.inf, np.inf, 0, 0)

        if MirrorType == 0:
            return self.__captainScorePairA(inputLysMOZList, inputLysINTList, inputLysNCLList, inputLysPM)

        else:
            return self.__captainScorePairB2G(inputLysMOZList, inputLysINTList, inputLysNCLList, inputLysPM, MirrorType)


    # delta == 0.0 的谱图对，内部也有竞争的关系
    def __captainScorePairA(self, inputLysMOZList:list, inputLysINTList:list, inputLysNCLList, inputLysPM:float):

        time_flag = time.perf_counter()
        # array shape: (1 x N)
        # lysMOZComponent = np.array([inputLysMOZList + self.__soldierGenerateComponentMOZList(inputLysMOZList, inputLysPM)])
        lysMOZComponent = np.array([inputLysMOZList])
        lysMOZComponent = np.concatenate((lysMOZComponent, self.__soldierGenerateComponentMOZList(lysMOZComponent, inputLysPM)), axis=1)
        self.time_record[0] += time.perf_counter() - time_flag
        time_flag = time.perf_counter()
        # print(lysMOZComponent)

        scoreTab = self.__soldierGetScoreList(lysMOZComponent, inputLysINTList, inputLysNCLList, 0)

        self.time_record[1] += time.perf_counter() - time_flag
        time_flag = time.perf_counter()

        judge = self.__soldierGetJudgeAnswerA(scoreTab)
        self.time_record[2] += time.perf_counter() - time_flag
        # time_flag = time.perf_counter()

        return judge


    # delta 的具体取值，可以直接根据类型来获得
    def __captainScorePairB2G(self, inputLysMOZList:list, inputLysINTList:list, inputLysNCLList, inputLysPM:float, mirrorType:int):

        time_flag = time.perf_counter()
        # array shape: (1 x N)
        # lysMOZComponent = np.array([inputLysMOZList + self.__soldierGenerateComponentMOZList(inputLysMOZList, inputLysPM)])
        lysMOZComponent = np.array([inputLysMOZList])
        lysMOZComponent = np.concatenate((lysMOZComponent, self.__soldierGenerateComponentMOZList(lysMOZComponent, inputLysPM)), axis=1)
        self.time_record[0] += time.perf_counter() - time_flag
        time_flag = time.perf_counter()
        # print(lysMOZComponent)
        scoreTab = self.__soldierGetScoreList(lysMOZComponent, inputLysINTList, inputLysNCLList, mirrorType)

        self.time_record[1] += time.perf_counter() - time_flag
        time_flag = time.perf_counter()

        # if self.dp.myCFG.C12_PAIR_DECOY_APPROACH == 2:
        judge = self.__soldierGetJudgeAnswerB2G(scoreTab, mirrorType)
        self.time_record[2] += time.perf_counter() - time_flag

        return judge

    # =======================================================================
    # type, ev, score
    #   |
    # \|/
    # type, t-ev, d-ev, t-score, d-score
    
    # 0: without FDR, only Target PSM
    def __soldierGetJudgeAnswerB2G0(self, scoreTab, mirrorType):

        judge = self.__weaponGetJudgeAnswerB2G(scoreTab, mirrorType)

        return (judge[0], judge[1], np.inf, judge[2], 0)

    def __soldierGetJudgeAnswerA0(self, scoreTab):

        judge = self.__weaponGetJudgeAnswerA(scoreTab)

        return (judge[0], judge[1], np.inf, judge[2], 0)

    # 1: Shifted Precursor Mass && 3: Shifted Preprocessed Peaks
    def __soldierGetJudgeAnswerB2G13(self, scoreTab, mirrorType):
        
        judge = self.__weaponGetJudgeAnswerB2G(scoreTab, mirrorType)
        
        return (judge[0], np.inf, judge[1], 0, judge[2])

    def __soldierGetJudgeAnswerA13(self, scoreTab):

        judge = self.__weaponGetJudgeAnswerA(scoreTab)

        return (judge[0], np.inf, judge[1], 0, judge[2])
    
    # 2: Shifted Score Bin [Special] with competition
    def __soldierGetJudgeAnswerB2G2(self, scoreTab, mirrorType):

        judgeT = self.__weaponGetJudgeAnswerB2G(scoreTab, mirrorType)
        
        judgeD = self.__weaponGetJudgeAnswerB2G(scoreTab, mirrorType, flag=False)
        
        t = judgeT[0] if judgeT[1] < judgeD[1] else judgeD[0]
        
        return (t, judgeT[1], judgeD[1], judgeT[2], judgeD[2])

    def __soldierGetJudgeAnswerA2(self, scoreTab):

        judgeT = self.__weaponGetJudgeAnswerA(scoreTab)

        judgeD = self.__weaponGetJudgeAnswerA(scoreTab, flag=False)

        t = judgeT[0] if judgeT[1] < judgeD[1] else judgeD[0]

        return (t, judgeT[1], judgeD[1], judgeT[2], judgeD[2])
    
    
    
    # =======================================================================

    # 生成互补峰列表，质荷比是倒序的状态
    # （因为原始的是顺序的喔！方便使用子璇的加权方法）
    def __soldierGenerateComponentMOZList(self, inputMOZList:np.array, inputPM:float):

        # [ATTENTION] 20221013
        # 暂时注释掉，先保证和子璇的结果一致
        # for i in reversed(range(len(inputMOZList))):
        #
        #     # 二级谱图里，碎片离子的质量比母离子还高，就更没必要比较辣
        #     if inputMOZList[i] > inputPM:
        #         inputMOZList.pop(i)
        #
        #     else:
        #         break
        # pm_add_p = inputPM + self.dp.myINI.MASS_PROTON_MONO
        # res = [pm_add_p - item for item in inputMOZList]
        #
        # while res and res[-1] < 0:
        #     # inputMOZList.pop()
        #     res.pop()
        res = inputPM + self.dp.myINI.MASS_PROTON_MONO - inputMOZList.T
        return res[(res > 0)[:,0]].T

    # 子璇的打分策略
    def __soldierGetScoreList(self, lysMOZ:np.array, lysINT:list, lysNCL, mirrorType):

        time_flag_0 = time.perf_counter()
        score_wei_inten = np.zeros((self.bin_num + 1))
        score_wei_match = np.zeros((self.bin_num + 1))
        ori_num, com_num = len(lysINT), len(lysMOZ[0]) - len(lysINT)

        extract_wei = np.concatenate((self.weight[:, :ori_num], self.weight[:, :com_num]), axis=1)
        lys_int = np.array([lysINT])
        # lys_int = np.concatenate((lys_int, lys_int[:, :com_num]), axis=1)
        lys_int = np.concatenate((lys_int, lys_int[:, :com_num]), axis=1) * extract_wei
        lys_nclll = np.array([lysNCL])
        lys_nclll = np.concatenate((lys_nclll, lys_nclll[:, :com_num]), axis=1)
        # print("mass2 <- c(", ", ".join([str(t) for t in lysMOZ[0]]), ")")
        # print("inten2 <- c(", ", ".join([str(t) for t in lys_int[0]]), ")")
        # print("neucode2 <- c(", ", ".join([str(t) for t in lys_nclll[0]]), ")")
        # lys_int = lys_int * extract_wei
        lys_flag = np.array([[True if j < ori_num else False for j in range(len(lysMOZ[0]))]])
        lys_flag_ = ~lys_flag
        # 1. every peaks moz delta, numScore(with weight), intScore(with weight) will be calculated by this operator
        delta_matrix = self.try_moz - lysMOZ
        h_scr_matrix1 = self.try_wei1 + extract_wei * self.alpha1  # hit score matrix with 0.3
        h_scr_matrix2 = self.try_wei2 + extract_wei * self.alpha2  # hit score matrix with 0.7
        i_scr_matrix1 = self.try_int_x_wei1 + lys_int * self.alpha1  # int score matrix with 0.3
        i_scr_matrix2 = self.try_int_x_wei2 + lys_int * self.alpha2  # int socre matrix with 0.7

        # 2. every delta who -200 < d < 200 will be choosed as their idx
        legal_idx_ls = (delta_matrix >= self.masslower) * (delta_matrix <= self.massupper)
        # score3_idx_ls = (self.try_flag * lys_flag) | (~self.try_flag * ~lys_flag)
        # score7_idx_ls = (~self.try_flag * lys_flag) | (self.try_flag * ~lys_flag)
        # 3. calc the "score_wei_inten" and "score_wei_match"
        #    index = int(massdiff * self.one_over_bin - 0.5) + self.half_bin_num

        legal_idx_1 = legal_idx_ls * ((self.try_flag * lys_flag) | (self.try_flag_ * lys_flag_))
        legal_idx_2 = legal_idx_ls * ((self.try_flag_ * lys_flag) | (self.try_flag * lys_flag_))
        d1, d2 = delta_matrix[legal_idx_1], delta_matrix[legal_idx_2]
        trans_idx_1 = (d1 * self.one_over_bin + np.sign(d1) * 0.5).astype(np.int) + self.half_bin_num
        trans_idx_2 = (d2 * self.one_over_bin + np.sign(d2) * 0.5).astype(np.int) + self.half_bin_num

        self.time_score[0] += time.perf_counter() - time_flag_0
        time_flag_0 = time.perf_counter()

        # score_wei_match += h_scr_matrix3[legal_idx_ls3].dot(self.one_hot[trans_idx_ls3])
        # score_wei_inten += i_scr_matrix3[legal_idx_ls3].dot(self.one_hot[trans_idx_ls3])
        np.add.at(score_wei_match, trans_idx_1, h_scr_matrix1[legal_idx_1])
        np.add.at(score_wei_inten, trans_idx_1, i_scr_matrix1[legal_idx_1])
        # for idx, h_scr,i_scr in zip(trans_idx_ls3, h_scr_matrix3[legal_idx_ls3], i_scr_matrix3[legal_idx_ls3]):
        #     score_wei_match[idx], score_wei_inten[idx] = score_wei_match[idx] + h_scr, score_wei_inten[idx] + i_scr
        #     score_wei_match[trans_idx_ls3] += h_scr_matrix3[legal_idx_ls3]
        #     score_wei_inten[trans_idx_ls3] += i_scr_matrix3[legal_idx_ls3]

        # score_wei_match += h_scr_matrix7[legal_idx_ls7].dot(self.one_hot[trans_idx_ls7])
        # score_wei_inten += i_scr_matrix7[legal_idx_ls7].dot(self.one_hot[trans_idx_ls7])
        np.add.at(score_wei_match, trans_idx_2, h_scr_matrix2[legal_idx_2])
        np.add.at(score_wei_inten, trans_idx_2, i_scr_matrix2[legal_idx_2])
        # for idx, h_scr,i_scr in zip(trans_idx_ls7, h_scr_matrix7[legal_idx_ls7], i_scr_matrix7[legal_idx_ls7]):
        #     score_wei_match[idx], score_wei_inten[idx] = score_wei_match[idx] + h_scr, score_wei_inten[idx] + i_scr
        #     score_wei_match[trans_idx_ls7] += h_scr_matrix7[legal_idx_ls7]
        #     score_wei_inten[trans_idx_ls7] += i_scr_matrix7[legal_idx_ls7]

        # for i, t_moz in enumerate(tryMOZ):
        #     t_wei3 = self.weight[i % len(tryINT)] * 0.3
        #     t_wei7 = self.weight[i % len(tryINT)] * 0.7
        #     t_int_x_t_wei3 = tryINT[i % len(tryINT)] * t_wei3
        #     t_int_x_t_wei7 = tryINT[i % len(tryINT)] * t_wei7
        #     t_flag = True if i < len(tryINT) else False
        #
        #     for j, l_moz in enumerate(lysMOZ):
        #         massdiff = t_moz - l_moz
        #         if (massdiff < self.masslower) or (massdiff > self.massupper):
        #             continue
        #         if massdiff < 0:
        #             index = int(massdiff * self.one_over_bin - 0.5) + self.half_bin_num
        #         else:
        #             index = int(massdiff * self.one_over_bin + 0.5) + self.half_bin_num
        #
        #         # 均为原始谱峰 或 均为互补峰
        #         if t_flag == lys_flag_lst[j]:
        #             # alpha = 0.3
        #             score_wei_inten[index] += (t_int_x_t_wei3 + lys_int_x_weight_lst3[j])
        #             score_wei_match[index] += (t_wei3 + lys_weight_lst3[j])
        #         # 一方为互补峰，一方为原始峰
        #         else:
        #             # alpha = 0.7
        #             score_wei_inten[index] += (t_int_x_t_wei7 + lys_int_x_weight_lst7[j])
        #             score_wei_match[index] += (t_wei7 + lys_weight_lst7[j])
        if self.neucode_label:
            lys_ncl = np.array([lysNCL], np.bool)
            lys_ncl = np.concatenate((lys_ncl, ~lys_ncl[:, :com_num]), axis=1)
            # xor_matrix = np.logical_xor(self.try_ncl, lys_ncl)  # xor矩阵，一个是双峰，一个不是双峰
            tmp_try_ncl = self.try_ncl | (lys_ncl * False)
            tmp_lys_ncl = (self.try_ncl * False) | lys_ncl
            b_delta_idx, y_delta_idx = self.MIRROR_A2G_BY_DELTA_IDX_SET[mirrorType]  # 里面是b / y离子质量差的地址
            # 两倍强度的条件：
            # |        b ion delta        |        y ion delta        |
            # |     try     |     lys     |     try     |     lys     |
            # |      0      |      1      |      1      |      0      |
            # try: (match b delta & label==0) or (match y delta & label==1)
            # lys: (match b delta & label==1) or (match y delta & label==0)
            # 这样就再累加一遍就好了，筛选：alpha1 和 alpha2 分开
            i_scr_matrix1_try = self.try_int_x_wei1 + lys_int * 0  # int score matrix with 0.3
            i_scr_matrix2_try = self.try_int_x_wei2 + lys_int * 0  # int socre matrix with 0.7
            i_scr_matrix1_lys = self.try_int_x_wei1 * 0 + lys_int * self.alpha1  # int score matrix with 0.3
            i_scr_matrix2_lys = self.try_int_x_wei2 * 0 + lys_int * self.alpha2  # int socre matrix with 0.7
            delta_matrix = (delta_matrix * self.one_over_bin + np.sign(delta_matrix) * 0.5).astype(np.int) + self.half_bin_num

            b_delta_flag, y_delta_flag = (delta_matrix == b_delta_idx[0]), (delta_matrix == y_delta_idx[0])
            if len(b_delta_idx) > 1:
                for i in range(1, len(b_delta_idx)):
                    b_delta_flag, y_delta_flag = b_delta_flag | (delta_matrix == b_delta_idx[i]), y_delta_flag | (delta_matrix == y_delta_idx[i])
            #   try     (doub==0   &   in b delta)  or (doub==1   &   in y delta)
            tmp_idx = (~tmp_try_ncl * b_delta_flag) | (tmp_try_ncl * y_delta_flag)
            k = tmp_idx * legal_idx_1
            np.add.at(score_wei_inten, delta_matrix[tmp_idx * legal_idx_1], i_scr_matrix1_try[tmp_idx * legal_idx_1])  # 0.3
            np.add.at(score_wei_inten, delta_matrix[tmp_idx * legal_idx_2], i_scr_matrix2_try[tmp_idx * legal_idx_2])  # 0.7
            #   lys     (doub==0   &   in y delta)  or (doub==1  &   in b delta)
            tmp_idx = (~tmp_lys_ncl * y_delta_flag) | (tmp_lys_ncl * b_delta_flag)
            np.add.at(score_wei_inten, delta_matrix[tmp_idx * legal_idx_1], i_scr_matrix1_lys[tmp_idx * legal_idx_1])  # 0.3
            np.add.at(score_wei_inten, delta_matrix[tmp_idx * legal_idx_2], i_scr_matrix2_lys[tmp_idx * legal_idx_2])  # 0.7
            # 只增强了intensity，match是没有改变的
        # with open("00hit.txt", "w") as fh, open("00int.txt", "w") as fi:
        #     fh.write("\n".join([str(t) for t in score_wei_match]))
        #     fi.write("\n".join([str(t) for t in score_wei_inten]))
        self.time_score[1] += time.perf_counter() - time_flag_0
        time_flag_0 = time.perf_counter()

        # score = [a * b for a, b in zip(score_wei_inten, score_wei_match)]
        # score = [(score_wei_inten[i] * score_wei_match[i], i) for i in range(len(score_wei_inten))]
        # score_wei_inten[self.zeroIndex] = 0.0

        score = score_wei_inten * score_wei_match
        # score = (h_scr_matrix3[legal_idx_ls3].dot(self.one_hot[trans_idx_ls3]) + h_scr_matrix7[legal_idx_ls7].dot(self.one_hot[trans_idx_ls7])) * (i_scr_matrix3[legal_idx_ls3].dot(self.one_hot[trans_idx_ls3]) + i_scr_matrix7[legal_idx_ls7].dot(self.one_hot[trans_idx_ls7]))
        self.time_score[2] += time.perf_counter() - time_flag_0

        return score

    # 三种类型内部的竞争
    # type_int, t_pv, d_pv, t_score, d_score
    def __weaponGetJudgeAnswerA(self, inputScoreTable: np.array, flag=True):
        time_flag_0 = time.perf_counter()
        # sorted_score = sorted(inputScoreTable, key=lambda x: x[0], reverse=True)
        # sorted_idx = np.argsort(-inputScoreTable)

        # get the top
        rank1_idx = np.argmax(inputScoreTable)
        rank1_score = inputScoreTable[rank1_idx]
        inputScoreTable[rank1_idx] = -rank1_score
        # rank1_idx, rank2_idx = sorted_idx[0], sorted_idx[1]
        rank2_idx = np.argmax(inputScoreTable)
        rank2_score = inputScoreTable[rank2_idx]
        inputScoreTable[rank1_idx] = rank1_score
        # rank1_score, rank2_score = inputScoreTable[rank1_idx], inputScoreTable[rank2_idx]
        # rank1_score, rank1_idx = sorted_score[0]
        # rank2_score, rank2_idx = sorted_score[1]
        # rank3_score, rank3_idx = sorted_score[2]
        self.time_score[3] += time.perf_counter() - time_flag_0

        rank1_trans = (rank1_idx - (self.bin_num + 1) / 2) * self.bin
        rank2_trans = (rank2_idx - (self.bin_num + 1) / 2) * self.bin
        # rank3_trans = (rank3_idx - (self.bin_num + 1) / 2) * self.bin
        # print(rank1_trans, rank1_score, rank2_trans, rank2_score)


        use_dict_m, use_dict_idx = (self.MIRROR_A, self.MIRROR_A_DELTA_IDX) if flag else (self.MIRROR_A_DECOY, self.MIRROR_A_DECOY_DELTA_IDX)


        if 1 in use_dict_m:
            for item in use_dict_m[1]:
                if abs(item - rank1_trans) < self.bin:
                    pv = self.__calcPValue(inputScoreTable, 1, 1, use_dict_idx)
                    # return (1, rank1_idx, rank2_idx, rank1_trans, rank2_trans, rank1_score, rank2_score, ev)
                    return (1, pv, max(inputScoreTable[use_dict_idx[1]]))

        if 2 in use_dict_m:
            for item in use_dict_m[2]:
                if abs(item - rank1_trans) < self.bin:
                    pv = self.__calcPValue(inputScoreTable, 2, 2, use_dict_idx)
                    # return (2, rank1_idx, rank2_idx, rank1_trans, rank2_trans, rank1_score, rank2_score, ev)
                    return (2, pv, max(inputScoreTable[use_dict_idx[2]]))

        if abs(rank1_trans) < self.bin:
            if 1 in use_dict_m:
                for item in use_dict_m[1]:
                    if abs(item - rank2_trans) < self.bin:
                        pv = self.__calcPValue(inputScoreTable, 1, 1, use_dict_idx)
                        # return (1, rank1_idx, rank2_idx, rank1_trans, rank2_trans, rank1_score, rank2_score, ev)
                        return (1, pv, max(inputScoreTable[use_dict_idx[1]]))

            if 2 in use_dict_m:
                for item in use_dict_m[2]:
                    if abs(item - rank2_trans) < self.bin:
                        pv = self.__calcPValue(inputScoreTable, 2, 2, use_dict_idx)
                        # return (2, rank1_idx, rank2_idx, rank1_trans, rank2_trans, rank1_score, rank2_score, ev)
                        return (2, pv, max(inputScoreTable[use_dict_idx[2]]))

            if 3 in use_dict_m:
                pv = self.__calcPValue(inputScoreTable, 3, 3, use_dict_idx)
                # return (3, rank1_idx, rank2_idx, rank1_trans, rank2_trans, rank1_score, rank2_score, ev)
                return (3, pv, max(inputScoreTable[use_dict_idx[3]]))

        pv = self.__calcPValue(inputScoreTable, 0, 0, use_dict_idx)
        # return (0, rank1_idx, rank2_idx, rank1_trans, rank2_trans, rank1_score, rank2_score, ev)

        tmpScore = [0.0, 0.0, 0.0, 0.0]
        for mirrorType in use_dict_idx:
            if mirrorType == 0:
                continue
            for score_idx in use_dict_idx[mirrorType]:
                tmpScore[mirrorType] += inputScoreTable[score_idx]
        if 3 in use_dict_idx:
            tmpScore[3] /= 2
        # 使用负数来标记，不能直接判断
        # 标记的谱图对类型，就看谁的得分更高
        return (-np.argmax(tmpScore), pv, max(tmpScore))
        # 不考虑 3
        # if 3 in self.MIRROR_A:
        #     for item in self.MIRROR_A[3]:
        #         if abs(item - rank1_trans) < self.bin:
        #             return 3

        # res = [0.0, 0.0, 0.0, 0.0]
        #
        # if 1 in self.MIRROR_A:
        #     res[1] += sum([rank1_score if abs(item - rank1_trans) < self.bin else 0.0 for item in self.MIRROR_A[1]])
        #     res[1] += sum([rank2_score if abs(item - rank2_trans) < self.bin else 0.0 for item in self.MIRROR_A[1]])
        #     res[1] += sum([rank3_score if abs(item - rank3_trans) < self.bin else 0.0 for item in self.MIRROR_A[1]])
        #
        # if 2 in self.MIRROR_A:
        #     res[2] += sum([rank1_score if abs(item - rank1_trans) < self.bin else 0.0 for item in self.MIRROR_A[2]])
        #     res[2] += sum([rank2_score if abs(item - rank2_trans) < self.bin else 0.0 for item in self.MIRROR_A[2]])
        #     res[2] += sum([rank3_score if abs(item - rank3_trans) < self.bin else 0.0 for item in self.MIRROR_A[2]])
        #
        # if 3 in self.MIRROR_A:
        #     res[3] += rank1_score if abs(rank1_trans) < self.bin else 0.0
        #     res[3] += rank2_score if abs(rank2_trans) < self.bin else 0.0
        #     res[3] += rank3_score if abs(rank3_trans) < self.bin else 0.0
        # # print(res)
        # # Return first index of value.
        # # 0：没有鉴定出结果！打分不够！说明不对
        # return res.index(max(res))

    # type_int, t_pv, d_pv, t_score, d_score
    def __weaponGetJudgeAnswerB2G(self, inputScoreTable:np.array, inputMirrorType:int, flag=True):
        time_flag_0 = time.perf_counter()
        # sorted_score = sorted(inputScoreTable, key=lambda x: x[0], reverse=True)
        # sorted_idx = np.argsort(-inputScoreTable)
        rank1_idx = np.argmax(inputScoreTable)
        rank1_score = inputScoreTable[rank1_idx]
        inputScoreTable[rank1_idx] = -rank1_score
        # rank1_idx, rank2_idx = sorted_idx[0], sorted_idx[1]
        rank2_idx = np.argmax(inputScoreTable)
        rank2_score = inputScoreTable[rank2_idx]
        inputScoreTable[rank1_idx] = rank1_score
        self.time_score[3] += time.perf_counter() - time_flag_0
        # get the top
        # rank1_idx, rank2_idx = sorted_idx[0], sorted_idx[1]
        # rank1_score, rank2_score = inputScoreTable[rank1_idx], inputScoreTable[rank2_idx]
        # rank1_score, rank1_idx = sorted_score[0]
        # rank2_score, rank2_idx = sorted_score[1]
        # rank3_score, rank3_idx = sorted_score[2]

        rank1_trans = (rank1_idx - (self.bin_num + 1) / 2) * self.bin
        rank2_trans = (rank2_idx - (self.bin_num + 1) / 2) * self.bin
        # rank3_trans = (rank3_idx - (self.bin_num + 1) / 2) * self.bin
        # print(rank1_trans, rank1_score, rank2_trans, rank2_score)
        use_dict_m, use_dict_idx = (self.MIRROR_B2G, self.MIRROR_B2G_DELTA_IDX) if flag else (self.MIRROR_B2G_DECOY, self.MIRROR_B2G_DECOY_DELTA_IDX)
        frag_delta = use_dict_m[inputMirrorType][1:]  # 0: mirror_precursor_delta ; 1 & 2: frag_delta

        for item in frag_delta:
            if abs(rank1_trans - item) < self.bin:
                pv = self.__calcPValue(inputScoreTable, inputMirrorType, inputMirrorType, use_dict_idx)
                # return (inputMirrorType, rank1_idx, rank2_idx, rank1_trans, rank2_trans, rank1_score, rank2_score, ev)
                return (inputMirrorType, pv, max(inputScoreTable[use_dict_idx[inputMirrorType]]))

        # 打分第一名是 0.0 的话  然后再看 第二名 和 第三名
        if abs(rank1_trans) < self.bin:
            for item in frag_delta:
                if abs(rank2_trans - item) < self.bin:
                    pv = self.__calcPValue(inputScoreTable, inputMirrorType, inputMirrorType, use_dict_idx)
                    # return (inputMirrorType, rank1_idx, rank2_idx, rank1_trans, rank2_trans, rank1_score, rank2_score, ev)
                    return (inputMirrorType, pv, max(inputScoreTable[use_dict_idx[inputMirrorType]]))

                # elif abs(rank3_trans - item) < self.bin:
                #     return inputMirrorType
        pv = self.__calcPValue(inputScoreTable, inputMirrorType, inputMirrorType, use_dict_idx)
        # return (0, rank1_idx, rank2_idx, rank1_trans, rank2_trans, rank1_score, rank2_score, ev)

        # 使用负数来标记，直接判断法未通过的类型
        return (-inputMirrorType, pv, max(inputScoreTable[use_dict_idx[inputMirrorType]]))


    def __calcPValue(self, scoreList:np.array, inputMirrorType:int, judgeType:int, use_dict_idx):

        scores = scoreList[scoreList > 0]
        scores.sort()
        n_score = len(scores)
        tail1 = np.around(n_score * 0.9).astype(np.int) - 1  # python里取地址，从0开始，因此index比要少1
        tail2 = n_score - 10  # 因为python切片中:后的index不取值，因此n_score - 10 (- 1 + 1) 结果不变
        tail_num = tail2 - tail1

        if tail_num > 19:
            logs = np.array([(tail_num - i + 1) for i in range(1, tail_num + 1)])
            logs = np.log(0.1 * logs / tail_num)  # using broadcast to accelerate
            # logs = np.log(np.array([0r.1*(tail_num - i)/tail_num for i in range(1, tail_num+1)]))

            t = np.log(scores[tail1:tail2]) if self.log_label else scores[tail1:tail2]
            sumx = np.sum(t)
            sumx2 = np.sum(t*t)
            sumxy = np.sum(logs*t)
            sumy = np.sum(logs)

            if (tail_num * sumx2 != sumx * sumx):
                a1 = (tail_num * sumxy - sumx * sumy) / (tail_num * sumx2 - sumx * sumx)
                a2 = (sumy - (a1 * sumx)) / tail_num
            else:
                a1 = 0
                a2 = 0

            if inputMirrorType < 4:
                ev = self.__calcEValueMirrorTypeA(scoreList, judgeType, a1, a2, use_dict_idx)
            else:
                ev = self.__calcEValueMirrorTypeB2G(scoreList, inputMirrorType, a1, a2, use_dict_idx)

            return ev

        else:
            return np.Inf


    def __calcEValueMirrorTypeA(self, score:np.array, inputMirrorType: int, a1:float, a2:float, use_dict_idx) -> float:

        trans = use_dict_idx[inputMirrorType]
        score_info = score[trans]
        escore_info = np.full((len(score_info),), np.Inf)
        escore_info[score_info>0] = np.exp(a1 * np.log(score_info[score_info>0]) + a2 + 2)

        if inputMirrorType != 0:

            if inputMirrorType == 3:
                return escore_info[0]  # A_none

        return np.min(escore_info)


    def __calcEValueMirrorTypeB2G(self, score:np.array, inputMirrorType: int, a1:float, a2:float, use_dict_idx) -> float:

        trans = use_dict_idx[inputMirrorType]
        score_info = score[trans]
        escore_info = np.full((len(score_info),), np.Inf)
        escore_info[score_info > 0] = np.exp(a1 * np.log(score_info[score_info > 0]) + a2 + 2)

        # 不管是不是0，结果其实都是统一的
        return np.min(escore_info)


    def __getIdxInfoTypeA(self, flag=True):

        use_dict = self.MIRROR_A if flag else self.MIRROR_A_DECOY
        res = {0:[]}

        for key in use_dict:

            delta1, delta2 = use_dict[key]
            trans1 = np.around((delta1 + self.half_bin_add_interval) / self.bin).astype(np.int)
            trans2 = np.around((delta2 + self.half_bin_add_interval) / self.bin).astype(np.int)
            res[key] = np.array([trans1, trans2])
            res[0] += [trans1, trans2]
            # if key == 3:
            #     res[0].append(trans1)
            # else:
            #     res[0] += [trans1, trans2]
        res[0] = np.array(res[0])
        # res[0] = np.array(list(set(res[0])))

        return res

    def __getIdxInfoTypeB2G(self, flag=True):

        use_dict = self.MIRROR_B2G if flag else self.MIRROR_B2G_DECOY
        res = {0:[]}

        for key in use_dict:
            _, delta1, delta2 = use_dict[key]
            trans1 = np.around((delta1 + self.half_bin_add_interval) / self.bin).astype(np.int)
            trans2 = np.around((delta2 + self.half_bin_add_interval) / self.bin).astype(np.int)
            res[key] = np.array([trans1, trans2])
            res[0] += [trans1, trans2]

        res[0] = np.array(list(set(res[0])))

        return res

    def __getIdxSetTypeAll(self, flag=True):

        use_dict_m = self.MIRROR_A if flag else self.MIRROR_A_DECOY  # special for A
        use_dict_idx = self.MIRROR_A_DELTA_IDX if flag else self.MIRROR_A_DECOY_DELTA_IDX

        res = dict()

        if len(use_dict_m) >= 1:
            res[0] = [[], []]
            for i, idx in enumerate(use_dict_idx[0]):
                res[0][i%2].append(idx)

        # B2G
        use_dict_idx = self.MIRROR_B2G_DELTA_IDX if flag else self.MIRROR_B2G_DECOY_DELTA_IDX

        for key in use_dict_idx:
            if key == 0:
                continue
            res[key] = [[], []]
            for i, idx in enumerate(use_dict_idx[key]):
                res[key][i%2].append(idx)

        return res

    # 这里生成以谱峰 moz 的 rank 为基础计算的权重
    # 使用第一轮的全局过滤，实际上谱峰数目也肯定已经到顶了
    # 但为了避免什么奇怪的数字越界问题，我们直接用3倍的保留峰数目
    # 并且和1000作对比，谁大用谁

    def __generateWeightOfPeakRank(self):

        return np.array([[1 / (1 + math.exp(-i)) for i in range(1, self.maxWeightNum+1)]])

    # def __generateZeroIndex(self):
    #     res = set()
    #     # ban = [ord("R") - ord("A"), ord("K") - ord("A")]
    #     for i in self.dp.myINI.DICT1_AA_MASS:
    #         aa_mass = self.dp.myINI.DICT1_AA_MASS[i]
    #         if aa_mass < 1:
    #             continue
    #         if aa_mass > 200:
    #             continue
    #         # index trans
    #         # (d1 * self.one_over_bin + np.sign(d1) * 0.5).astype(np.int) + self.half_bin_num
    #         min_idx = int((aa_mass - self.bin/2) * self.one_over_bin + 0.5) + self.half_bin_num
    #         max_idx = int((aa_mass + self.bin/2) * self.one_over_bin + 0.5) + self.half_bin_num + 1
    #         for index in range(min_idx, min(self.bin_num+1, max_idx)):
    #             res.add(index)
    #         min_idx = int((-aa_mass - self.bin / 2) * self.one_over_bin + 0.5) + self.half_bin_num
    #         max_idx = int((-aa_mass + self.bin / 2) * self.one_over_bin + 0.5) + self.half_bin_num + 1
    #         for index in range(min_idx, min(self.bin_num + 1, max_idx)):
    #             res.add(index)
    #
    #         # for j in self.dp.myINI.DICT1_AA_MASS:
    #         #     aa_mass += self.dp.myINI.DICT1_AA_MASS[j]
    #         #     if aa_mass < 1:
    #         #         continue
    #         #     if aa_mass > 200:
    #         #         continue
    #         #     # index trans
    #         #     # (d1 * self.one_over_bin + np.sign(d1) * 0.5).astype(np.int) + self.half_bin_num
    #         #     min_idx = int((aa_mass - self.bin / 2) * self.one_over_bin + 0.5) + self.half_bin_num
    #         #     max_idx = int((aa_mass + self.bin / 2) * self.one_over_bin + 0.5) + self.half_bin_num + 1
    #         #     for index in range(min_idx, min(self.bin_num + 1, max_idx)):
    #         #         res.add(index)
    #         #     min_idx = int((-aa_mass - self.bin / 2) * self.one_over_bin + 0.5) + self.half_bin_num
    #         #     max_idx = int((-aa_mass + self.bin / 2) * self.one_over_bin + 0.5) + self.half_bin_num + 1
    #         #     for index in range(min_idx, min(self.bin_num + 1, max_idx)):
    #         #         res.add(index)
    #         #
    #         #     for k in self.dp.myINI.DICT1_AA_MASS:
    #         #         aa_mass += self.dp.myINI.DICT1_AA_MASS[k]
    #         #         if aa_mass < 1:
    #         #             continue
    #         #         if aa_mass > 200:
    #         #             continue
    #         #         # index trans
    #         #         # (d1 * self.one_over_bin + np.sign(d1) * 0.5).astype(np.int) + self.half_bin_num
    #         #         min_idx = int((aa_mass - self.bin / 2) * self.one_over_bin + 0.5) + self.half_bin_num
    #         #         max_idx = int((aa_mass + self.bin / 2) * self.one_over_bin + 0.5) + self.half_bin_num + 1
    #         #         for index in range(min_idx, min(self.bin_num + 1, max_idx)):
    #         #             res.add(index)
    #         #         min_idx = int((-aa_mass - self.bin / 2) * self.one_over_bin + 0.5) + self.half_bin_num
    #         #         max_idx = int((-aa_mass + self.bin / 2) * self.one_over_bin + 0.5) + self.half_bin_num + 1
    #         #         for index in range(min_idx, min(self.bin_num + 1, max_idx)):
    #         #             res.add(index)
    #
    #     ban_mass_K = self.dp.myINI.DICT1_AA_MASS[ord("K") - ord("A")]
    #     ban_mass_R = self.dp.myINI.DICT1_AA_MASS[ord("R") - ord("A")]
    #
    #     K_index = int(np.around((ban_mass_K + self.half_bin_add_interval) / self.bin))
    #     R_index = int(np.around((ban_mass_R + self.half_bin_add_interval) / self.bin))
    #     if K_index in res:
    #         res.remove(K_index)
    #     if R_index in res:
    #         res.remove(R_index)
    #
    #     K_index = int(np.around((-ban_mass_K + self.half_bin_add_interval) / self.bin))
    #     R_index = int(np.around((-ban_mass_R + self.half_bin_add_interval) / self.bin))
    #     if K_index in res:
    #         res.remove(K_index)
    #     if R_index in res:
    #         res.remove(R_index)
    #
    #     return np.array(sorted(list(res)))
        # return np.array([])

    def recallPrecursorList(self, inputMass:float):

        res_dict = dict()

        if self.MIRROR_A:
            res_dict.update(self.__captainRecallPrecursorList_A(inputMass))

        if self.MIRROR_B2G:
            res_dict.update(self.__captainRecallPrecursorList_B2G(inputMass))

        return res_dict

    def __captainRecallPrecursorList_A(self, inputMass:float):

        res_dict = {0:[]}

        candi_P = self.PMIDX[int(inputMass * self.fold)]
        if candi_P:
            for pm_i in candi_P:
                if self.checkPrecursorMass(inputMass, self.PRECURSOR_LIST[pm_i][-1]):
                    res_dict[0].append(pm_i)

        return res_dict

    def __captainRecallPrecursorList_B2G(self, inputMass:float):
        # 因为已经检查过了，所以返回的不会空
        res_dict = {item:[] for item in self.MIRROR_B2G}

        for m_type in self.MIRROR_B2G:

            precursor_delta, _, _ = self.MIRROR_B2G[m_type]

            # [ATTENTION] 根据母离子的质量差公式
            # 此处 precursor_delta 的来源是 Trypsin - LysargiNase
            # 即若寻找对应的 LysargiNase 需反向思考计算公式，即有
            # Trypsin - LysargiNase = precursor_delta
            # Trypsin - precursor_delta = LysargiNase
            candi_P = self.PMIDX[int((inputMass - precursor_delta) * self.fold)]
            if candi_P:
                for pm_i in candi_P:
                    # print(candi_P, pm_i)
                    if self.checkPrecursorMass(inputMass, self.PRECURSOR_LIST[pm_i][-1], sup_delta=precursor_delta):
                        res_dict[m_type].append(pm_i)

        return res_dict


    # -----------------------------
    # 对这里的两个算delta的函数进行了统一
    # 实际的计算，都是在和0进行比较
    # -----------------------------
    def __checkPrecursorMass_Da(self, inputMassA: float, inputMassB: float, sup_delta = 0.0):

        return True if (abs(inputMassA - sup_delta - inputMassB) <= self.ms_tol_boundary) else False

    # [ATTENTION] 和子璇公式的推导是一致的（暂时只有ppm的）
    # 因此匹配质量差就变为了 tolerance 和动态的boundary进行比较
    # 比如我们期望，理想状态下 try 和 lys 的质量差为 target_delta
    # 经过计算可得，massdiff_min/max 范围是 (try - lys) ± (try + lys) * 20 / 1000000
    # 若有 massdiff_min <= target_delta <= massdiff_max，则 match
    # 即有 massdiff_min - target_delta <= 0 且 massdiff_max - target_delta >= 0
    # 整理上式 (try - lys) ± (try + lys) * 20 / 1000000 为 (try - lys) ± tol
    # 即有 massdiff_min - target_delta = (try - lys) - tol - target_delta <= 0
    # 以及 massdiff_max - target_delta = (try - lys) + tol - target_delta >= 0
    # 整理得:
    #     (try - lys) - target_delta <= +tol
    #     (try - lys) - target_delta >= -tol
    # 即得  abs((try - lys - target_delta)) <= abs(tol)
    # 其中 tol = (try + lys) * 20 / 1000000
    # (以 20 ppm 为例，公式的证明为上)
    def __checkPrecursorMass_PPM(self, inputMassA: float, inputMassB: float, sup_delta = 0.0):

        return True if (abs(inputMassA - sup_delta - inputMassB) <= (inputMassA + inputMassB) * self.fraction) else False


    # 这个函数返回的是个dict，里面存放着母离子差值在 0.0Da附近的碎片离子差值信息
    # 分别是N端离子和C端离子的差值，也就是b离子和y离子，其他离子类型比较少见，可能会引入噪音
    # 但是计算的方式是一致的，如果后续需要考虑的话，其实也没有问题
    # A类型镜像谱图对儿分为三种，三类的标记分别为1，2，3
    def __getMirrorDeltaInfo_A(self):

        # res_A_dict[type_label] = (b_delta, y_delta)
        res_A_dict = dict()

        # xxxxK - Kxxxx
        if self.dp.myCFG.C1_MIRROR_TYPE_A1 == 1:
            res_A_dict[1] = (-self.K_MASS, self.K_MASS)  # b_delta, y_delta

        # xxxxR - Rxxxx
        if self.dp.myCFG.C2_MIRROR_TYPE_A2 == 1:
            res_A_dict[2] = (-self.R_MASS, self.R_MASS)  # b_delta, y_delta

        # xxxx  -  xxxx
        if self.dp.myCFG.C3_MIRROR_TYPE_A3 == 1:
            res_A_dict[3] = (0.0, 0.0)


        return res_A_dict

    # 与上面的函数类似，字典中的键，对应用户指定的 mirror 类型(int label)
    # 该键对应的值则为(母离子质量差值, N-term离子质量差, C-term离子质量差)
    def __getMirrorDeltaInfo_B2G(self):

        # res_B2G_dict[type_label] = (PM_delta, b_delta, y_delta)
        res_B2G_dict = dict()

        # xxxxR - Kxxxx
        if self.dp.myCFG.C4_MIRROR_TYPE_B == 1:
            res_B2G_dict[4] = (self.R_MASS - self.K_MASS, -self.K_MASS, self.R_MASS)  # pmd, b_delta, y_delta

        # xxxxK - Rxxxx
        if self.dp.myCFG.C5_MIRROR_TYPE_C == 1:
            res_B2G_dict[5] = (self.K_MASS - self.R_MASS, -self.R_MASS, self.K_MASS)  # pmd, b_delta, y_delta

        # xxxxK -  xxxx
        if self.dp.myCFG.C6_MIRROR_TYPE_D == 1:
            res_B2G_dict[6] = (self.K_MASS, 0.0, self.K_MASS)  # pmd, b_delta, y_delta

        # xxxxR -  xxxx
        if self.dp.myCFG.C7_MIRROR_TYPE_E == 1:
            res_B2G_dict[7] = (self.R_MASS, 0.0, self.R_MASS)  # pmd, b_delta, y_delta

        # xxxx  - Kxxxx
        if self.dp.myCFG.C8_MIRROR_TYPE_F == 1:
            res_B2G_dict[8] = (-self.K_MASS, -self.K_MASS, 0.0)  # pmd, b_delta, y_delta

        # xxxx  - Rxxxx
        if self.dp.myCFG.C9_MIRROR_TYPE_G == 1:
            res_B2G_dict[9] = (-self.R_MASS, -self.R_MASS, 0.0)  # pmd, b_delta, y_delta

        return res_B2G_dict


    def __geneDecoyDelta_A2G(self):

        for key in self.MIRROR_A:

            self.MIRROR_A_DECOY[key] = tuple(t + self.decoy_shift for t in self.MIRROR_A[key])

        for key in self.MIRROR_B2G:

            k1, k2, k3 = self.MIRROR_B2G[key]

            self.MIRROR_B2G_DECOY[key] = (k1, k2 + self.decoy_shift, k3 + self.decoy_shift)


# 按照0.1Da的精度间隔来生成 PMI(Precursor Mass Index)
class CFunctionPrecursorMassIndex:

    def __init__(self, inputDP):
        self.dp = inputDP
        self.fold = 10
        self.proton_mass = self.dp.myINI.MASS_PROTON_MONO
        self.min_mass = self.dp.myCFG.A5_MIN_PRECURSOR_MASS
        self.max_mass = self.dp.myCFG.A6_MAX_PRECURSOR_MASS
        self.R_MASS = self.dp.myINI.DICT1_AA_MASS[ord("R") - ord("A")]
        self.K_MASS = self.dp.myINI.DICT1_AA_MASS[ord("K") - ord("A")]

        # self.R_MASS = 156.10111
        # self.K_MASS = 128.09496
        if self.dp.myCFG.A7_LABEL_TYPE == 1:
            print("neucode!")
            self.R_MASS = 160.08925
            self.K_MASS = 136.10916

        self.max_sup = max(self.R_MASS, self.K_MASS)
        # [曾经的注释] 为了计算tol的时候更精确，把156扩充为200Da的质量（万一以后标记了呢）
        # [现在的注释] 精准一点吧，没有必要，万一以后标记质量特别大的话就坏了
        # 目前的想法是直接选择 max(mass_K, mass_R)  稍微准一些免得有什么遗漏  20220926
        if self.max_sup < 20:
            exit("What's wrong with AA mass?? Please check it!\n\nMASS of K or R is not accepted now!")

        self.MAX_BOUNDARY = int((self.dp.myCFG.A6_MAX_PRECURSOR_MASS + self.max_sup) * self.fold)
        # 这里是定下了索引的上界边界，避免的情况发生，
        # 通常MAX_PRECURSOR_MASS是kDa量级，而max_sup是100Da量级，也足够放下tolerance了
        # （Top down 数据在极端情况下可能会有问题，比如越界？可谁会对整条蛋白质做 De Novo 啊喂！）

        # self.ms_tol = self.dp.myCFG.D3_MS_TOL * 2
        # 因为需要考虑Trypsin，所以直接使用两倍质量偏差来计算
        # [ATTENTION] 上面这句话先作废！！！下面听我解释！！
        self.ms_tol = self.dp.myCFG.D3_MS_TOL
        self.ms_ppm = self.dp.myCFG.D4_MS_TOL_PPM
        self.ms_tol *= 2 if (self.ms_ppm == 0) else 1
        # 如果是绝对质量偏差即以道尔顿做单位，则在母离子质量匹配时，直接计算两倍的质量差即可
        # 但如果是相对质量偏差，那么需要考虑什么样的 Trypsin 对应当前 LysargiNase 的误差极限
        # 索引中的 LysargiNase 均需要考虑 Trypsin 肽段的质量的最大情形，即有如下最大形式：
        # 有 Mass_trypsin_max = Mass_lysargiNase + max_sup + maxDelta
        # 且 maxDelta = (Mass_trypsin_max + Mass_lysargiNase) * 20/1000000
        # 由本脚本上方 CFunctionMirrorFinder 函数中的公式推导，tol 由肽质量和来计算，故而
        # 得 maxDelta = (2*Mass_lysargiNase + max_sup) * 20/1000000
        # 其中 max_sup 就是最大的补足质量，目前不出意外的话，基本就是 mass_R
        # 上述公式的质量计算是精确的，而两倍的质量偏差则是一种近似的计算方法

        self.fraction = (self.ms_tol * 1e-6) if (self.ms_ppm == 1) else 1

        if self.ms_ppm == 1:
            # 使用 ppm 作为质量偏差计算的单位
            self.__soldierGetBoundIdx = self.soldierGetBoundIdx_PPM
        else:
            # 使用 Da 作为质量偏差计算的单位
            self.__soldierGetBoundIdx = self.soldierGetBoundIdx_Da


    def generatePMIDX(self, inputList: list):

        idx_tuple_list = self.captainGetIndexTuple(inputList)

        pmidx = self.__captainGetPrecursorMassIndex(idx_tuple_list)

        return idx_tuple_list, pmidx


    # 这样也算是压缩空间，不然直接在索引中记录的话，占空间太多了
    def captainGetIndexTuple(self, inputList: list) -> list:

        res = []

        for i, tmpMS2 in enumerate(inputList):
            # 如果这里的位置是空，则不需要考虑此处的数据
            if tmpMS2 is None:
                continue
            else:
                for tmpScan in tmpMS2.INDEX_SCAN:
                    for pre_i, moz in enumerate(tmpMS2.MATRIX_PRECURSOR_MOZ[tmpScan]):
                        p_mass = (moz - self.proton_mass) * tmpMS2.MATRIX_PRECURSOR_CHARGE[tmpScan][pre_i] + self.proton_mass

                        # 越界的我可不要嗷
                        if (p_mass < self.min_mass) or (p_mass > self.max_mass):
                            continue
                        res.append((i, tmpScan, pre_i, p_mass))

        return res

    # 20230411 外部调用，直接构造
    # 这里注意，不会把超过母离子质量限值的谱图idx给放到索引里
    def captainGetPrecursorMassIndex(self, spec_list: list):
        # initial precursor mass index
        res = [None] * self.MAX_BOUNDARY

        # for loop 的内部不放判断了，直接把判断挪出外面来
        # i: 母离子的idx信息，t: 一个记录着二级谱图地址和母离子质量的tuple
        for i, t in enumerate(spec_list):
            tmpMass = t.LIST_PRECURSOR_MASS[0]
            if tmpMass < self.min_mass or tmpMass > self.max_mass:
                continue
            lower, upper = self.__soldierGetBoundIdx(tmpMass)
            self.__soldierAddIdxInfo(res, lower, upper, i)

        return res

    def __captainGetPrecursorMassIndex(self, tuple_list: list):

        # initial precursor mass index
        res = [None] * self.MAX_BOUNDARY

        # for loop 的内部不放判断了，直接把判断挪出外面来
        # i: 母离子的idx信息，t: 一个记录着二级谱图地址和母离子质量的tuple
        for i, t in enumerate(tuple_list):
            lower, upper = self.__soldierGetBoundIdx(t[-1])
            self.__soldierAddIdxInfo(res, lower, upper, i)

        return res


    # 20221231 因为并行化的原因，这里不能写成__XXX()形式的函数！否则并行会报错的！
    # 函数名的前面不能带双下划线！
    # 使用ppm作为质量偏差计算的单位
    def soldierGetBoundIdx_PPM(self, mass:float):
        delta = (2 * mass + self.max_sup) * self.fraction
        lower = int((mass - delta) * self.fold)
        upper = int((mass + delta) * self.fold + 0.5)

        return lower, upper

    # 20221231 因为并行化的原因，这里不能写成__XXX()形式的函数！否则并行会报错的！
    # 函数名的前面不能带双下划线！
    # 使用Dalton作为质量偏差计算的单位
    def soldierGetBoundIdx_Da(self, mass:float):

        lower = int((mass - self.ms_tol) * self.fold)
        upper = int((mass + self.ms_tol) * self.fold + 0.5)
        return lower, upper

    # 增加母离子信息到PMI索引之中去
    def __soldierAddIdxInfo(self, PMIDX:list, lower:int, upper:int, idx:int):

        for enume_idx in range(lower, upper+1):
            if PMIDX[enume_idx] is None:
                PMIDX[enume_idx] = [idx]
            else:
                PMIDX[enume_idx].append(idx)

