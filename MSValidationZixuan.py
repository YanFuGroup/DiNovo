import json
import os
import time
from MSSystem import IO_NAME_FILE_RESULT, IO_NAME_FILE_VALIDATION_STAT, IO_NAME_FOLDER_VALIDATION, SPECTRAL_PAIR_TYPE
from MSLogging import logToUser
## input: DiNovo.res, DiNovo.dis, pFind_filtered.spectra*2
## output: recall / precision

class CValidationZixuan:

    def __init__(self, inputDP, outSpecPairName):
        self.dp = inputDP
        # total cnt info, only report in the final
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        self.try_resdict = self.__readpFindres(self.dp.myCFG.V1_PATH_TRY_PFIND_RES)
        self.lys_resdict = self.__readpFindres(self.dp.myCFG.V2_PATH_LYS_PFIND_RES)
        # self.try_resdict = dict()
        # self.lys_resdict = dict()
        self.resName = IO_NAME_FILE_RESULT[0]
        self.disName = IO_NAME_FILE_RESULT[1]
        self.TPFName = outSpecPairName
        self.recFile = self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FOLDER_VALIDATION + IO_NAME_FILE_VALIDATION_STAT[0]
        with open(self.recFile, "w") as f:
            f.write("")
        self.typeA = []
        if self.dp.myCFG.C1_MIRROR_TYPE_A1 == 1:
            self.typeA.append(SPECTRAL_PAIR_TYPE[1])
        if self.dp.myCFG.C2_MIRROR_TYPE_A2 == 1:
            self.typeA.append(SPECTRAL_PAIR_TYPE[2])
        if self.dp.myCFG.C3_MIRROR_TYPE_A3 == 1:
            self.typeA.append(SPECTRAL_PAIR_TYPE[3])
    # 对合并后的specPair文件进行validation，以trypsin谱图数据为单元计算
    # [ATTENTION] 随读随算，切莫再缓存，不然规模就太大了
    def specPairValidation(self, i_try):

        res_path = self.dp.LIST_OUTPUT_PATH[i_try] + self.resName
        dis_path = self.dp.LIST_OUTPUT_PATH[i_try] + self.disName
        valiPath = self.dp.LIST_OUTPUT_PATH[i_try] + self.TPFName
        logToUser("\n[Validation][SpectralPair]\t[" + self.dp.LIST_MGF_NAME_TRY[i_try] + "]\tWrite only TP result in: " + valiPath + "\n")

        TruePosi = 0
        FalseNega = 0
        cnt = 0

        with open(res_path, "r") as f:
            buff = f.read().split("\n")
            first_line = buff[0]
            buff = buff[1:-1]

        with open(valiPath, "w") as f:
            f.write(first_line + "\n")
            for line in buff:
                if not line:
                    continue
                cnt += 1
                item_list = line.split("\t")
                try_title = item_list[2]
                lys_title = item_list[3]
                mirr_type = item_list[11]
                if self.__judge_mirror_spectra(try_title, lys_title, mirr_type):
                    TruePosi += 1
                    f.write(line + "\n")
        FalsePosi = cnt - TruePosi

        with open(dis_path, "r") as f:
            buff = f.read().split("\n")[1:-1]
        cnt = 0
        for line in buff:
            if not line:
                continue
            cnt += 1
            item_list = line.split("\t")
            try_title = item_list[2]
            lys_title = item_list[3]
            mirr_type = item_list[11]

            # A 类型，但又区分不出来具体的镜像谱图类型（A1，A2，A3）
            if mirr_type[:2] == "A0":
                tmp = [self.__judge_mirror_spectra(try_title, lys_title, m) for m in self.typeA]
                if (True in tmp):
                    FalseNega += 1
            elif self.__judge_mirror_spectra(try_title, lys_title, mirr_type):
                FalseNega += 1
        TrueNega = cnt - FalseNega

        self.TP += TruePosi
        self.FP += FalsePosi
        self.FN += FalseNega
        self.TN += TrueNega
        # return TruePosi, FalsePosi, FalseNega, TrueNega
        self.__formatReport(TruePosi, FalsePosi, FalseNega, TrueNega, label="["+self.dp.LIST_MGF_NAME_TRY[i_try]+"]")

    # 根据上述分部分计算完毕后，汇总全部的数值进行一次汇报
    def reportAll(self, valiFilePath):

        logToUser("\n[Validation][SpectralPair]\t[ALL DATA COMBINE]\tWrite only TP result in: " + valiFilePath + "\n")

        for i, p in enumerate(self.dp.LIST_OUTPUT_PATH):
            with open(p+self.TPFName) as fr, open(valiFilePath, "a") as fw:
                if i == 0:
                    fw.write(fr.read())
                else:
                    fw.write("\n".join(fr.read().split("\n")[1:]))

        tmpTP, tmpFP, tmpFN, tmpTN = self.TP, self.FP, self.FN, self.TN
        self.__formatReport(tmpTP, tmpFP, tmpFN, tmpTN, label="[ALL DATA COMBINE]", reportFlag=True)

    # 读取pFind的鉴定结果并预存起来（全局维护）
    def __readpFindres(self, path):
        logToUser("Load pFind results: " + path + "\n")
        with open(path, "r") as f:
            buffer_result = f.read().split("\n")[1:]
        result_dict = {}
        for line in buffer_result:
            if line:
                infolist = line.split("\t")
                if len(infolist) > 6:
                    result_dict[infolist[0]] = [infolist[2], infolist[5], infolist[10]]
                    # result_dict[File_Name] = [Exp.MH+, Sequence, Modification]
        return result_dict

    # 判定函数
    def __judge_mirror_mod(self, try_mod, lys_mod, try_len, mirr_type):  # lys序列以KR开头的谱图对类型
        if (not try_mod) and (not lys_mod):  # 无修饰
            return True
        else:
            # 如果是try以KR结尾的谱图对，去掉末尾的KR并检测对应修饰，如果try是X则不管
            # 如果是lys以KR结尾的谱图对，去掉开头的KR并检测对应修饰，如果lys是X则不管（去除则更改后续修饰位置-1）
            try_modlist = [mod_str.split(",") for mod_str in try_mod.split(";")[:-1]]  # [[site,mod], [site,mod], ...]
            lys_modlist = [mod_str.split(",") for mod_str in lys_mod.split(";")[:-1]]  # [[site,mod], [site,mod], ...]
            try_last, lys_firs = mirr_type[-3], mirr_type[-1]

            if try_last != "X":
                while try_modlist:
                    if int(try_modlist[-1][0]) >= try_len:
                        try_modlist = try_modlist[:-1]
                    else:
                        break

            if lys_firs != "X":
                while lys_modlist:
                    if int(lys_modlist[0][0]) < 2:
                        lys_modlist.pop(0)
                    else:
                        break
                if lys_modlist:
                    lys_modlist = [[str(int(t[0])-1), t[1]] for t in lys_modlist]

            # middle sequence mod info
            if try_modlist == lys_modlist:
                return True
            else:
                return False


    # 判定函数
    def __judge_mirror_spectra(self, try_title, lys_title, mirr_type):
        # 有pFind鉴定结果
        if try_title in self.try_resdict and lys_title in self.lys_resdict:  # 有pFind鉴定结果
            try_seq = self.try_resdict[try_title][1]
            lys_seq = self.lys_resdict[lys_title][1]
            try_mod = self.try_resdict[try_title][2]
            lys_mod = self.lys_resdict[lys_title][2]
            try_last = mirr_type[-3]
            lys_firs = mirr_type[-1]
            try_len, lys_len = len(try_seq), len(lys_seq)
            # sequence preprocess - TRY
            if try_last == "X":
                pass
            elif try_seq[-1] == try_last:
                try_seq = try_seq[:-1]
            else:
                return False

            # sequence preprocess - LYS
            if lys_firs == "X":
                pass
            elif lys_seq[0] == lys_firs:
                lys_seq = lys_seq[1:]
            else:
                return False

            # middle sequence judge
            if try_seq != lys_seq:
                return False

            if self.__judge_mirror_mod(try_mod, lys_mod, try_len, mirr_type):
                return True
            else:
                return False

        # 没有pFind鉴定结果
        else:
            return False

    # 格式化输出（替代prettytable）
    def __formatReport(self, TruePosi, FalsePosi, FalseNega, TrueNega, label="", reportFlag=False):
        # resStr = label + "\n"
        # (1) print a table
        # +----------+----------+----------+----------+
        # | SpecPair | Positive | Negative |  Count   |
        # +----------+----------+----------+----------+
        # |   True   |    xxx   |    xxx   |    xxx   |
        # |   False  |    xxx   |    xxx   |    xxx   |
        # |   Count  |    xxx   |    xxx   |    xxx   |
        # +----------+----------+----------+----------+
        head = ["SpecPair", "Positive", "Negative", "Count"]
        column = ["True", "False", "Count"]
        combin = TruePosi + FalsePosi + FalseNega + TrueNega
        maxlen = max([len(t) for t in head + column])
        maxlen = max(maxlen, len(str(combin)))
        bound = ("-" * (maxlen+2)).join(["+"] * 5)

        resStr = bound + "\n"
        resStr += "| " + " | ".join([t.rjust(maxlen) for t in head]) + " |\n"
        resStr += bound + "\n"
        tmpLine = [column[0], str(TruePosi), str(FalseNega), str(TruePosi + FalseNega)]
        resStr += "| " + " | ".join([t.rjust(maxlen) for t in tmpLine]) + " |\n"
        tmpLine = [column[1], str(FalsePosi), str(TrueNega), str(FalsePosi + TrueNega)]
        resStr += "| " + " | ".join([t.rjust(maxlen) for t in tmpLine]) + " |\n"
        tmpLine = [column[2], str(TruePosi + FalsePosi), str(FalseNega + TrueNega), str(combin)]
        resStr += "| " + " | ".join([t.rjust(maxlen) for t in tmpLine]) + " |\n"
        resStr += bound + "\n"

        resStr += "   Recall = " + str(TruePosi).rjust(maxlen) + " / " + str(TruePosi+FalseNega).rjust(maxlen) + " = "
        if TruePosi+FalseNega == 0:
            resStr += "NaN\n"
        else:
            resStr += ("%.4f" % (TruePosi * 100.0 / (TruePosi + FalseNega))).rjust(7) + " %\n"

        resStr += "Precision = " + str(TruePosi).rjust(maxlen) + " / " + str(TruePosi+FalsePosi).rjust(maxlen) + " = "
        if TruePosi+FalsePosi == 0:
            resStr += "NaN\n"
        else:
            resStr += ("%.4f" % (TruePosi * 100.0 / (TruePosi + FalsePosi))).rjust(7) + " %\n"

        resStr += " Accuracy = " + str(TruePosi + TrueNega).rjust(maxlen) + " / " + str(combin).rjust(maxlen) + " = "
        if combin == 0:
            resStr += "NaN\n"
        else:
            resStr += ("%.4f" % ((TruePosi + TrueNega) * 100.0 / combin)).rjust(7) + " %\n"

        # logToUser(resStr + "\n")
        with open(self.recFile, "a") as f:
            f.write(label + "\n" + resStr + "\n\n")

        if reportFlag:
            logToUser(label + "\n" + resStr + "\n\n")
        ...


# try_pFindres_path = r"D:\ResearchAssistant@AMSS\MirrorSpectra\old_res\EColi_try_pFind-Filtered.spectra"
# lys_pFindres_path = r"D:\ResearchAssistant@AMSS\MirrorSpectra\old_res\EColi_lys_pFind-Filtered.spectra"
try_pFindres_path = "D:\\ResearchAssistant@AMSS\\DiNovoData\\YeastRes0315\\Yeast_try_pFind-Filtered[extract][v0315].spectra"
lys_pFindres_path = "D:\\ResearchAssistant@AMSS\\DiNovoData\\YeastRes0315\\Yeast_lys_pFind-Filtered[extract][v0315].spectra"
# print("\n\n", "***"*9, "   ABC   ", "***"*9, "\n")
# res_path = "test20230315/[DiNovo]SpectralPairs.res"
# dis_path = "test20230315/[DiNovo]SpectralPairs.dis"
# specPairValidationZixuan(try_pFindres_path, lys_pFindres_path, res_path, dis_path)
out_TP_path = r"E:\MyPythonWorkSpace\DiNovo\test20230321-99\validation\[DiNovo]SpectralPairs[OnlyTP].res"

# print("\n\n", "***"*9, "   original   ", "***"*9, "\n")
res_path = r"E:\MyPythonWorkSpace\DiNovo\test20230321-99\[DiNovo]SpectralPairs.res"
dis_path = r"E:\MyPythonWorkSpace\DiNovo\test20230321-99\[DiNovo]SpectralPairs.dis"
# specPairValidation(try_pFindres_path, lys_pFindres_path, res_path, dis_path, valiPath="")

# print("\n\n", "***"*9, "   new prep   ", "***"*9, "\n")
# res_path = "test20230315[pre]/UPLC_gel_trypsin_01_HCDFT[extract]/[DiNovo]SpectralPairs.res"
# dis_path = "test20230315[pre]/UPLC_gel_trypsin_01_HCDFT[extract]/[DiNovo]SpectralPairs.dis"
# specPairValidationZixuan(try_pFindres_path, lys_pFindres_path, res_path, dis_path)

# print("\n\n", "***"*9, "   0OtherAA   ", "***"*9, "\n")
# res_path = "test20230315[0aa]/UPLC_gel_trypsin_01_HCDFT[extract]/[DiNovo]SpectralPairs.res"
# dis_path = "test20230315[0aa]/UPLC_gel_trypsin_01_HCDFT[extract]/[DiNovo]SpectralPairs.dis"
# specPairValidationZixuan(try_pFindres_path, lys_pFindres_path, res_path, dis_path)

# print("\n\n", "***"*9, "   prep+0AA   ", "***"*9, "\n")
# res_path = "test20230315[fin]/UPLC_gel_trypsin_01_HCDFT[extract]/[DiNovo]SpectralPairs.res"
# dis_path = "test20230315[fin]/UPLC_gel_trypsin_01_HCDFT[extract]/[DiNovo]SpectralPairs.dis"
# specPairValidationZixuan(try_pFindres_path, lys_pFindres_path, res_path, dis_path)