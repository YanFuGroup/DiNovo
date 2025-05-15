# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSEmass.py
Time: 2022.09.05 Monday
ATTENTION: Re-implement Emass Algorithm, cite:Rockwood AL, Haimi P. Efficient calculation of accurate masses of isotopic peaks. J Am Soc Mass Spectrom. 2006 Mar;17(3):415-9.
"""

from MSLogging import logGetError



class CEmass:

    EMASS_INTENSITY_MAX = 100.0
    EMASS_DUMMY_MASS = -10000000
    EMASS_DUMMY_PRO = 0.0  # 这个是用于填充S33等情况
    # EMASS_CUTOFF_PRO = 0.00001  # 因为很多同位素的丰度就是0.01，两个相乘就是0.00001了。###### 这里是不是说错了？#####
    EMASS_CUTOFF_PRO = 0.0
    # EMASS_CUTOFF_PRO_OUTPUT = 0.01  # 最后输出时，控制一下
    EMASS_CUTOFF_PRO_OUTPUT = 5e-7# 10的-6次方，然后除以2
    # EMASS_CUTOFF_PRO_OUTPUT = 0.0# 不控制输出的数目
    # 最终输出的时候，控制相对强度在1e-6以上才输出

    def __init__(self, inputDP):
        self.dp = inputDP

    def __soldierGenerateVerginList(self, inputElementName):

        # print("MSEmass55: "+str(inputElementID))

        SuperAtomList = []
        numberAtomType = len(self.dp.myINI.DICT0_ELEMENT_MASS[inputElementName])  # 如果元素不在element.ini中，就会报错

        for i in range(numberAtomType):

            tmpMass = self.dp.myINI.DICT0_ELEMENT_MASS[inputElementName][i]
            tmpPro = self.dp.myINI.DICT0_ELEMENT_ABDC[inputElementName][i]
            tmpAtomNode = [tmpPro, tmpMass]  # 先丰度再质量

            SuperAtomList.append(tmpAtomNode)

        return SuperAtomList

    def __soldierGetSuperAtomList(self, dictComposition):  # 这个函数很关键，是java中的admiralGetSuperAtomList

        # init
        salForResult = []
        salForResult.append([1.0, 0.0])

        for name in dictComposition:

            numAtom = dictComposition[name]

            if numAtom > 0:  #  这句话很重要，因为Gln->pyro-Glu[AnyN-termQ]这个修饰，会产生N(-1)。如果是是14N(14)N(-1)，N(-1)是不起作用的
                salForDouble = self.__soldierGenerateVerginList(name)

                while numAtom > 0:

                    if 1 == numAtom % 2:# 判断二进制数尾部是否为1，单数的二进制最末尾就是1 #################################################
                        salForResult = self.__captainMergerSAL(salForResult, salForDouble)# 如果为1的话，那么就是需要计算进入result的
                        salForResult = self.__soldierPrune(salForResult, self.EMASS_CUTOFF_PRO)# 剪枝！！！ATTENTION

                    salForDouble = self.__captainMergerSAL(salForDouble, salForDouble)
                    salForDouble = self.__soldierPrune(salForDouble, self.EMASS_CUTOFF_PRO)  # 这句貌似很重要，不然传进去的都是很大的数组，运算奇慢无比；

                    numAtom = int(numAtom / 2)# 相当于使用右移来控制numAtom #################################################

        # salForResult = self.__soldierPrune(salForResult, self.EMASS_CUTOFF_PRO_OUTPUT)  # EMASS_CUTOFF_PRO太小了，同位素会很多
        # 修改：将该处剪枝操作移除
        return salForResult

    def __soldierPrune(self, inputListNode, inputLimit):

        outputList = []

        for i in range(len(inputListNode)):# 可以直接进行前后判断，按照原论文的判断方法，低强度的同位素峰只在两侧，不过这样也保险，就是慢了些
            tmpNode = inputListNode[i]

            if tmpNode[0] > inputLimit:
                outputList.append(tmpNode)

        return outputList

    def __captainMergerSAL(self, f, g):  # 之所以取名为f、g和h，是为了和论文中公式统一

        h = []

        length_f = len(f)
        length_g = len(g)

        if 0 == length_f or 0 == length_g:

            return h

        for k in range(length_f+length_g):# 这里是否应该需要-1？对应emass151行代码

            sumweight = 0.0
            summass = 0.0

            start = 0
            if k < (length_f - 1):
                start = 0
            else:
                start = k - length_f + 1

            end = k
            if k < length_g - 1:
                end = k
            else:
                end = length_g - 1

            # time20 = time.time()  # 测试用
            for i in range(start, end+1):#这里没问题，对应emass代码155行

                weight = g[i][0] * f[k-i][0]
                mass = g[i][1] + f[k-i][1]

                sumweight = sumweight + weight
                summass = summass + weight * mass

            if sumweight == 0:
                newMass = self.EMASS_DUMMY_MASS
            else:
                newMass = summass / sumweight

            newPro = sumweight
            newAtomNode = [newPro, newMass]

            h.append(newAtomNode)

        return h

    def __soldierMultiListAndNumber(self, inputList, inputNum):
        theFinalResult = [[], []]# 存储通过信号过滤的同位素峰信号
        for i in range(len(inputList[0])):
            inputList[0][i] = inputList[0][i] * inputNum
            if inputList[0][i] >= self.EMASS_CUTOFF_PRO_OUTPUT:
                theFinalResult[0].append(inputList[0][i])
                theFinalResult[1].append(inputList[1][i])
            # 之所以如此考虑，是因为中间部分有可能先被剪（并非完全上凸形态）
            # 修改为返回输出的函数，然后赋值给result
        return theFinalResult

    def getCalculatedIsotopicPeaks(self, inputDictComposition, inputCharge):

        sal = self.__soldierGetSuperAtomList(inputDictComposition)  # sal super atom list

        nPeaks = len(sal)

        result = [[0]*nPeaks, [0]*nPeaks]  # 两位数组，一行有nPeaks个数

        maxPro = 0.0
        i = 0

        for tmpNode in sal:

            result[0][i] = tmpNode[0]

            if tmpNode[0] > maxPro:
                maxPro = tmpNode[0]# 记录出现的最大的概率，便于计算相对强度

            if inputCharge > 0:

                result[1][i] = (tmpNode[1] + inputCharge * self.dp.myINI.MASS_PROTON_MONO) / inputCharge
                # 计算电荷的质量，也就是质子的质量
                # result[1][i] = tmpNode[1] - inputCharge * self.dp.myINI.MASS_ELECTRON / inputCharge
                # 不计算电荷质量，即不补充额外的质量
                # 但是要相应的减去对应的电子的质量，执行此语句，与emass的质荷比计算完全相同

            else:

                logGetError("MSEmass, MK172: " + str(result))

            i = i + 1

        if maxPro == 0:
            logGetError("MSEmass, MK177: " + str(inputDictComposition))
        else:
            factor = self.EMASS_INTENSITY_MAX / maxPro# 除以最大的概率，然后乘以100
            result = self.__soldierMultiListAndNumber(result, factor)# 检查完成，计算没有问题
            # 注意，此处和数据形式不同，计算阶段的数据是以[prob, mass]为元素的列表
            # 现在的列表的长度仅为2，分别是相对强度和质荷比
            # 所以这里不使用剪枝，而是直接在__soldierMultiListAndNumber()函数中解决result

        return result

    def getCalculatedMonoMZ(self, inputDictCom, inputCharge):

        mass = 0.0

        for name in inputDictCom:

            tmpNum = inputDictCom[name]
            list_abdc = self.dp.myINI.DICT0_ELEMENT_ABDC[name]
            list_mass = self.dp.myINI.DICT0_ELEMENT_MASS[name]
            tmpMass = list_mass[list_abdc.index(max(list_abdc))]

            mass += tmpMass * tmpNum

        if inputCharge > 0:
            return (mass + self.dp.myINI.MASS_PROTON_MONO * inputCharge) / inputCharge

        # 仅仅计算元素组成的MONO质量，而不用计算其MOZ
        elif inputCharge == 0:
            return mass
        else:
            info = "MSEMass, getCalculatedMonoMZ, " + str(inputCharge) + " is all right?"
            logGetError(info)

