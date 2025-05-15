# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSFlow.py
Time: 2022.09.05 Monday
ATTENTION: none
"""
from MSTask import CTaskReadMGF, CTaskReadINI, CTaskCheck, CTaskOutputPathFile, CTaskCombineRes
from MSTask import CTaskDiNovo, CTaskWritePreprocessMGF, CTaskPairing, CTaskIsoFeatureForDeNovo, CTaskForpNovoMorGCNovo
from MSTask import CTaskBuildPMIDX
from MSSystem import IO_NAME_FILE_CONFIG, LABEL_TYPE_STR, IO_NAME_FOLDER_TEMPORARY
from MSSystem import IO_NAME_FILE_PNOVOM, IO_NAME_FILE_PNOVOM_FINAL
from MSSystem import IO_NAME_FILE_GCNOVO, IO_NAME_FILE_GCNOVO_FINAL
from MSSystem import IO_NAME_FILE_PNOVOM_SINGLE_A, IO_NAME_FILE_PNOVOM_SINGLE_B
from MSSystem import IO_NAME_FILE_GCNOVO_SINGLE_A, IO_NAME_FILE_GCNOVO_SINGLE_B
from MSSystem import IO_NAME_FILE_PNOVOM_SINGLE_A_FINAL, IO_NAME_FILE_PNOVOM_SINGLE_B_FINAL
from MSSystem import IO_NAME_FILE_GCNOVO_SINGLE_A_FINAL, IO_NAME_FILE_GCNOVO_SINGLE_B_FINAL
from MSSystem import FEATURE_MGF_FILE_SUFFIX
from MSFunctionIO import CFunctionConfig, CFunctionTransFormat
from MSData import Config
from MSLogging import INFO_TO_USER_Flow1, INFO_TO_USER_Flow2, logToUser
from MSLogging import INFO_TO_USER_Flow3
from MSTool import toolUsingTimeString
from MSFunctionMapping import CFunctionMapping
import time, os, shutil
import multiprocessing as mp

# a = mp.Manager()

# Output config file
class CFlow0:  # config

    def run(self):
        config = Config()
        functionConfig = CFunctionConfig()
        functionConfig.config2file(IO_NAME_FILE_CONFIG[0], config)

# process .mgf files
class CFlow1:

    def __init__(self, inputDP):
        self.dp = inputDP

    def run(self):

        timeStartFlow1 = time.perf_counter()

        logToUser(INFO_TO_USER_Flow1[0])
        taskCheck = CTaskCheck(self.dp)
        taskCheck.work()

        logToUser(INFO_TO_USER_Flow1[1])
        logToUser(INFO_TO_USER_Flow1[2] + "\t" + LABEL_TYPE_STR[self.dp.myCFG.A7_LABEL_TYPE])
        taskReadINI = CTaskReadINI(self.dp)
        taskReadINI.work()

        # for aa in self.dp.myINI.DICT1_AA_MASS:
        #     if self.dp.myINI.DICT1_AA_MASS[aa] == 0.0:
        #         continue
        #     print(aa, "\t %.6f" % (self.dp.myINI.DICT1_AA_MASS[aa]-0.0), "\tImmonium\t %.6f" % (self.dp.myINI.DICT1_AA_MASS[aa] - 12 -15.99491 + 1.007276))
        #
        # exit()

        timeStartMS2 = time.perf_counter()

        logToUser(INFO_TO_USER_Flow1[3])
        taskFileMS2 = CTaskReadMGF(self.dp)
        taskFileMS2.work()

        timeEndMS2 = time.perf_counter()

        if self.dp.myCFG.E2_EXPORT_ROUND_ONE_MGF or self.dp.myCFG.E3_EXPORT_ROUND_TWO_MGF or ((self.dp.myCFG.D0_WORK_FLOW_NUMBER > 1) and (self.dp.myCFG.D0_WORK_FLOW_NUMBER < 4)):
            logToUser(INFO_TO_USER_Flow1[4])
            taskPreWrite = CTaskWritePreprocessMGF(self.dp)
            taskPreWrite.work()

        if self.dp.myCFG.E7_EXPORT_FEATURE_MGF == 1 or (self.dp.myCFG.D0_WORK_FLOW_NUMBER > 2 and self.dp.myCFG.D9_DE_NOVO_APPROACH > 1 and self.dp.myCFG.A7_LABEL_TYPE == 1):
            logToUser("\n" + ("**" * 5) + " [SPECIAL TASK FOR FEATURE ANNO MGF] " + ("**" * 5) + "\n")
            test = CTaskIsoFeatureForDeNovo(self.dp)
            test.work()

        timeEndFlow1 = time.perf_counter()

        logToUser(toolUsingTimeString("Flow 1", timeStartFlow1, timeEndFlow1))


# process .mgf files
# find spectra pairs
class CFlow2:

    def __init__(self, inputDP):
        self.dp = inputDP
        self.initResFile = True if self.dp.myCFG.D0_WORK_FLOW_NUMBER < 4 else False

    def run(self):
        timeStartFlow2 = time.perf_counter()

        logToUser(INFO_TO_USER_Flow2[0])
        taskOutput = CTaskOutputPathFile(self.dp, inputFlowNum=2)
        taskOutput.work(initFile=self.initResFile)

        logToUser(INFO_TO_USER_Flow2[1])
        taskPMIDX = CTaskBuildPMIDX(self.dp)
        taskPMIDX.work()
        timeEndTaskPMIDX = time.perf_counter()

        logToUser(INFO_TO_USER_Flow2[2])
        taskPairing = CTaskPairing(self.dp)
        taskPairing.work()
        timeEndTaskPairing = time.perf_counter()

        # if self.dp.myCFG.E6_COMBINE_TOTAL_RESULT == 1:
        if self.dp.myCFG.D0_WORK_FLOW_NUMBER == 2:
            logToUser(INFO_TO_USER_Flow2[3])
            taskPairComb = CTaskCombineRes(self.dp)
            taskPairComb.combineTotalSpecPair()


        timeEndFlow2 = time.perf_counter()

        logToUser(toolUsingTimeString("Flow 2", timeStartFlow2, timeEndFlow2))

        # return lysInfo
        """"""

# process .mgf files
# find spectra pairs
# de novo sequencing
class CFlow3:  # De Novo

    def __init__(self, inputDP):
        self.dp = inputDP

    def run(self):

        timeStartFlow3 = time.perf_counter()
        taskPairComb = CTaskCombineRes(self.dp)
        taskPairComb.combineTotalSpecPair()

        logToUser(INFO_TO_USER_Flow3[0])
        # self.dp.myCFG.D9_DE_NOVO_APPROACH = 3
        if self.dp.myCFG.D9_DE_NOVO_APPROACH == 0:
            logToUser(INFO_TO_USER_Flow3[1] + "[Direct-read]")
            taskDiNovo = CTaskDiNovo(self.dp)
            taskDiNovo.work()

            if self.dp.myCFG.E6_COMBINE_TOTAL_RESULT == 1:
                logToUser(INFO_TO_USER_Flow3[2])
                taskPairComb.combineTotalDiNovoRes(0)

        elif self.dp.myCFG.D9_DE_NOVO_APPROACH == 1:
            logToUser(INFO_TO_USER_Flow3[1] + "[MirrorNovo mode]")
            taskGCNovo = CTaskForpNovoMorGCNovo(self.dp)
            taskGCNovo.work()
            if self.dp.myCFG.E6_COMBINE_TOTAL_RESULT == 1:
                logToUser(INFO_TO_USER_Flow3[2])
                taskPairComb.combineTotalDiNovoRes(1)

        elif self.dp.myCFG.D9_DE_NOVO_APPROACH == 2:
            logToUser(INFO_TO_USER_Flow3[1] + "[pNovoM mode]")
            taskpNovoM = CTaskForpNovoMorGCNovo(self.dp)
            taskpNovoM.work()
            if self.dp.myCFG.E6_COMBINE_TOTAL_RESULT == 1:
                logToUser(INFO_TO_USER_Flow3[2])
                taskPairComb.combineTotalDiNovoRes(2)

        # Combine: GCNovo + pNovoM
        elif self.dp.myCFG.D9_DE_NOVO_APPROACH == 3:
            logToUser(INFO_TO_USER_Flow3[1] + "[Combination](MirrorNovo + pNovoM)")
            taskNovo = CTaskForpNovoMorGCNovo(self.dp)
            taskNovo.work()

            if self.dp.myCFG.E6_COMBINE_TOTAL_RESULT == 1:
                logToUser(INFO_TO_USER_Flow3[2])
                taskPairComb.combineTotalDiNovoRes(1)
                taskPairComb.combineTotalDiNovoRes(2)

        else:
            exit("CFlow3, why DE_NOVO_APPROACH is not 1 -> 5? Please check it!")

        if self.dp.myCFG.D12_DE_NOVO_SINGLE_SPECTRUM == 1:
            taskNovo = CTaskForpNovoMorGCNovo(self.dp)

            # MirrorNovo
            if self.dp.myCFG.D9_DE_NOVO_APPROACH % 2 == 1:
                tmpOut = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_GCNOVO_SINGLE_A
                tmpList = taskNovo.toolGetSingleSpecSeqResPathList(True, True)
                taskPairComb.combineTotalSingleSeq(tmpOut, tmpList, True)

                tmpOut = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_GCNOVO_SINGLE_B
                tmpList = taskNovo.toolGetSingleSpecSeqResPathList(False, True)
                taskPairComb.combineTotalSingleSeq(tmpOut, tmpList, True)

            # pNovoM
            if self.dp.myCFG.D9_DE_NOVO_APPROACH > 1:
                tmpOut = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_PNOVOM_SINGLE_A
                tmpList = taskNovo.toolGetSingleSpecSeqResPathList(True, False)
                taskPairComb.combineTotalSingleSeq(tmpOut, tmpList, False)

                tmpOut = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_PNOVOM_SINGLE_B
                tmpList = taskNovo.toolGetSingleSpecSeqResPathList(False, False)
                taskPairComb.combineTotalSingleSeq(tmpOut, tmpList, False)

        # taskPairComb = CTaskCombineRes(self.dp)



        # transformat
        trans = CFunctionTransFormat(self.dp)

        # pNovoM, >1
        if self.dp.myCFG.D9_DE_NOVO_APPROACH > 1:
            ori_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_PNOVOM
            new_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_PNOVOM_FINAL
            trans.trans_pNovoM2_mirrorSeq(ori_path, new_path)

            if self.dp.myCFG.D12_DE_NOVO_SINGLE_SPECTRUM == 1:

                ori_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_PNOVOM_SINGLE_A
                new_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_PNOVOM_SINGLE_A_FINAL
                trans.trans_pNovoM2_singleSeq(ori_path, new_path)

                ori_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_PNOVOM_SINGLE_B
                new_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_PNOVOM_SINGLE_B_FINAL
                trans.trans_pNovoM2_singleSeq(ori_path, new_path)


        if self.dp.myCFG.D9_DE_NOVO_APPROACH % 2 == 1:

            ori_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_GCNOVO
            new_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_GCNOVO_FINAL
            trans.trans_mirrorNovo_mirrorSeq(ori_path, new_path)

            if self.dp.myCFG.D12_DE_NOVO_SINGLE_SPECTRUM == 1:
                ori_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_GCNOVO_SINGLE_A
                new_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_GCNOVO_SINGLE_A_FINAL
                trans.trans_mirrorNovo_singleSeq(ori_path, new_path)

                ori_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_GCNOVO_SINGLE_B
                new_path = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FILE_GCNOVO_SINGLE_B_FINAL
                trans.trans_mirrorNovo_singleSeq(ori_path, new_path)

        # mapping
        if self.dp.myCFG.V0_FLAG_DO_VALIDATION == 2:
            logToUser(INFO_TO_USER_Flow3[3])
            mapFunc = CFunctionMapping(self.dp)
            mapFunc.map()

        timeEndFlow3 = time.perf_counter()
        logToUser(toolUsingTimeString("Flow 3", timeStartFlow3, timeEndFlow3))
        """"""

"[Direct-read]"
"[Intelligent]"
"[GCNovo-mode]"
"[pNovoM-mode]"
# 20221217 record
# 记得要自己生成结果文件嗷！
# 20230216 for xueli single spectra sequencing
class CFlow4:

    def __init__(self, inputDP):

        self.dp = inputDP

    def run(self):

        logToUser(INFO_TO_USER_Flow2[0])
        taskOutput = CTaskOutputPathFile(self.dp, inputFlowNum=2)
        taskOutput.work(initFile=False)

        # logToUser(INFO_TO_USER_Flow2[3])
        # taskPairComb = CTaskCombineRes(self.dp)
        # taskPairComb.combineTotalSpecPair()

