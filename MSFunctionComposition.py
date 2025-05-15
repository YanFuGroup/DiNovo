# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSFunctionComposition.py
Time: 2022.09.05 Monday
ATTENTION: none
"""
from MSTool import toolCountCharInString, toolGetWord
from MSSystem import VALUE_ILLEGAL
from MSLogging import logGetWarning

"""
getDictComposition
输入的固定格式的字符串，获取元素组成Dict，key=ELEMENT_NAME(string), value=NUMBER(int)
getStrComposition
获取一个氨基酸序列的元素组成，得到固定格式的字符串
"""

class CFunctionComposition:

    def __init__(self, inputDP):
        self.dp = inputDP

    def __captainGetNumberElement8Mod(self, inputMod, inputE, inputDictModCom):

        result = 0
        comMod = inputDictModCom[inputMod]
        iLastRightBracket = 0  # java可以写为-1，python不行
        iELE = VALUE_ILLEGAL

        # find ele
        for i in range(len(comMod)):

            if comMod[i] == '(':

                strELE = comMod[iLastRightBracket:i]

                if strELE == inputE:
                    iELE = i - 1
                    break

            if comMod[i] == ')':
                iLastRightBracket = i + 1

        # get number
        if iELE == VALUE_ILLEGAL:

            return result  # 此时result为0

        else:

            p_bracket_left = 0
            p_bracket_right = 0

            for j in range(iELE + 1, len(comMod)):

                if comMod[j] == '(':
                    p_bracket_left = j

                if comMod[j] == ')':
                    p_bracket_right = j
                    break

            result = int(comMod[p_bracket_left + 1:p_bracket_right])

        return result

    def __captainGetNumberElement8AA(self, inputAA, inputE, inputDictAACom):

        # init
        result = 0
        comAA = inputDictAACom[inputAA]
        iLastRightBracket = 0  # java可以写为-1，python不行
        iELE = VALUE_ILLEGAL

        # find ele
        for i in range(len(comAA)):

            if comAA[i] == '(':

                strELE = comAA[iLastRightBracket:i]

                if strELE == inputE:

                    iELE = i - 1
                    break

            if comAA[i] == ')':

                iLastRightBracket = i + 1

        # get number
        if iELE == VALUE_ILLEGAL:

            return result  # 此时result为0

        else:

            p_bracket_left = 0
            p_bracket_right = 0

            for j in range(iELE+1, len(comAA)):

                if comAA[j] == '(':
                    p_bracket_left = j

                if comAA[j] == ')':
                    p_bracket_right = j
                    break

            result = int(comAA[p_bracket_left+1:p_bracket_right])

        return result

    def getDictComposition(self, inputComposition):

        outputDictComposition = {}  # 这是一个以名字为key，以个数为number的dict

        LastIndexLeftBracket = 0
        LastIndexRightBracket = -1  # 第一开始，+1后从0开始

        lengthInputString = len(inputComposition)
        for i in range(lengthInputString):

            if inputComposition[i] == '(':
                eleName = inputComposition[LastIndexRightBracket + 1:i]
                LastIndexLeftBracket = i

            if inputComposition[i] == ')':

                eleAtomNum = int(inputComposition[LastIndexLeftBracket + 1:i])

                if outputDictComposition.__contains__(eleName):

                    outputDictComposition[eleName] = outputDictComposition[eleName] + eleAtomNum  # eleAtomNum可能是负数

                else:

                    outputDictComposition[eleName] = eleAtomNum

                LastIndexRightBracket = i

        return outputDictComposition

    def getStrComposition(self, inputSeq, inputMod, inputGLC, inputLIK, inputINI):

        result = 'H(2)O(1)'  # 写死的

        for aa in inputSeq:

            # if aa in self.dp.myINI.DICT1_AA_COM:
            if inputINI.DICT1_AA_COM.__contains__(aa):

                # result = result + self.dp.myINI.DICT1_AA_COM[aa]
                result = result + inputINI.DICT1_AA_COM[aa]

            else:

                continue

        # modification
        nMOD = toolCountCharInString(inputMod, ';')  # 标准形式为6,Carbamidomethyl[C];14,Carbamidomethyl[C];
        for iMOD in range(nMOD):
            nameMOD = toolGetWord(toolGetWord(inputMod, iMOD, ';'), 1, ',')
            result = result + inputINI.DICT2_MOD_COM[nameMOD]

        # glyco
        nGLC = toolCountCharInString(inputGLC, ';')
        for iGLC in range(nGLC):
            nameGLC = toolGetWord(toolGetWord(inputGLC, iGLC, ';'), 1, ',')
            if inputINI.DICT3_GLYCO_COM.__contains__(nameGLC):
                result = result + inputINI.DICT3_GLYCO_COM[nameGLC]
            else:
                logGetWarning(inputGLC)

        # linker
        nLINK = toolCountCharInString(inputLIK, ';')  # 标准形式为同修饰
        for iLINK in range(nLINK):
            nameLINK = toolGetWord(toolGetWord(inputLIK, iLINK, ';'), 1, ',')
            result = result + inputINI.DICT4_LINKER_COM[nameLINK]

        return result

