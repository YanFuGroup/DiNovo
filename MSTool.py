# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSTool.py
Time: 2022.09.05 Monday
ATTENTION: none
"""

# ----------------------------------------
# input: float(positive float)
# function: trans to a closest int number
# output: a int number
# ----------------------------------------
def toolMyIntRound(inputNumber:float):

    # 取整
    outputNumber = int(inputNumber)

    # 手动完成四舍五入，因为python在判断位是5的时候，一半舍一半入
    if inputNumber - outputNumber >= 0.5:
        outputNumber += 1

    return outputNumber

def toolMyIntRoundNegative(inputNumber:float):

    outputNumber = int(inputNumber)
    if outputNumber - inputNumber >= 0.5:
        outputNumber -= 1

    return outputNumber

# ----------------------------------------
# input: inputStr:string, index:int, d:char
# function: split the inputStr by d
# output: index-th substring split by d
# 用d分割字符串，输出其中的第index个子序列
# ----------------------------------------
def toolGetWord(inputString, index: int, d):

    if inputString[0] != d:
        inputString = d + inputString

    if inputString[-1] != d:
        inputString = inputString + d

    p_d = []
    
    i = 0
    for c in inputString:
        
        if c == d:
            
            p_d.append(i)
        
        i = i+1
        
    result = inputString[p_d[index]+1:p_d[index+1]]
        
    return result

def toolGetWordForParseEqualLine(inputString: str):

    return inputString.split("=")[1]

# ----------------------------------------
# input: inputStr:string, inputChar:char
# function: split the inputStr by inputChar
# output: number of inputChar in inputStr
# ----------------------------------------
def toolCountCharInString(inputStr, inputChar):

    result = 0

    for c in inputStr:
        if c == inputChar:
            result = result + 1

    return result


# ----------------------------------------
# input: path: string
# function: get file name from file path
# output: a substring of string
# ----------------------------------------
def toolGetNameFromPath(path):

    lenStr = len(path)
    iStart = 0
    iEnd = -1

    for i in range(lenStr):

        j = lenStr - 1 - i
        # 20211014
        # 当iEnd没有设定的时候，检测iEnd
        # 若不修改，如果文件名中包含“.”则文件名会缩短导致输出文件夹路径不对
        # 可能会导致输出结果被覆盖
        if (iEnd == -1) and (path[j] == '.'):
            iEnd = j  # 亲测必须这么写，不用减一

        if path[j] == '\\' or path[j] == '/':
            iStart = j + 1
            break

    return path[iStart:iEnd]


# ----------------------------------------
# input: inputString:string, d1:char, d2:char
# function: get the substring between d1 and d2
# output: substring of inputString
# ----------------------------------------
def toolGetWord1(inputString:str, d1, d2):

    start = 0
    end = len(inputString)

    for i in range(len(inputString)):

        if inputString[i] == d1:
            start = i + 1

        if inputString[i] == d2:
            end = i

    return inputString[start:end]


# ----------------------------------------
# input: inputStr:string, inputSeparator:char
# function: split the inputStr by inputSeparator
# output: vector of float
# ----------------------------------------
def toolStr2List(inputStr, inputSeparator):

    outputList = []

    word = ''

    if inputStr[-1] != inputSeparator:
        inputStr = inputStr + inputSeparator

    for c in inputStr:

        if c == inputSeparator:
            # C++的话，这里大概可以用强制类型转换，或者atof函数
            number = float(word)
            outputList.append(number)
            word = ''

        else:

            word = word + c

    return outputList



def toolUsingTimeString(nameString, timeStart=0, timeEnd=0, padding=25) -> str:

    outString = "\n" + ">" * padding
    outString += " [" + nameString + "] using time: %.3fs." % (timeEnd - timeStart) + "<" * padding
    return outString

# time cost details of mirror spec pairing
def toolGenerateMirrorPairTimeDetail(inputList:list)->str:
    #     0         1         2         3         4         5
    # [ LOAD ], [PM IDX], [PREPOL], [ PAIR ], [ SAVE ]
    # [ LOAD ], [PM IDX], [PREPOL], [ PAIR ], [DiNovo], [ SAVE ]

    if len(inputList) == 5:
        timeStr = ["(details)[TIME COST][ LOAD ] ",
                   "(details)[TIME COST][PM IDX] ",
                   "(details)[TIME COST][PREPOL] ",
                   "(details)[TIME COST][ PAIR ] ",
                   "(details)[TIME COST][ SAVE ] ", ]

    else:
        return ""

    res = "\n\t\t<Spec Pairing>\n" + ("-" * 28) + "\n"
    res += "".join([s + "%.3fs.\n" % (t + 0.0) for s, t in zip(timeStr, inputList)])
    return res + ("-" * 28)

# 没想好怎么写，就先光写俩吧
def toolGenerateDeNovoTimeDetail(inputList: list) -> str:

    if len(inputList) == 2:
        timeStr = ["(details)[TIME COST][DeNovo] ",
                   "(details)[TIME COST][ SAVE ] "]
    else:
        return ""

    res = "\n\t\t<DirectDeNovo>\n" + ("-" * 28) + "\n"
    res += "".join([s + "%.3fs.\n" % (t + 0.0) for s, t in zip(timeStr, inputList)])
    return res + ("-" * 28) + "\n"
