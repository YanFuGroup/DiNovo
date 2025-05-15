# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSLogging.py
Time: 2022.09.05 Monday
ATTENTION: none
"""

import logging
import sys

# 字符©，容易在某些地方出现问题，所以一般直接改用(C)替代
# ----------------------------------------------------------------------------------
# 以下均为静态的变量，可以直接用const list存，这里不必考虑长度的问题
#
INFO_TO_USER_Staff = (
    '\n[DiNovo] Copyright (C) 2024 CAS AMSS. All rights reserved. Version 2024.10',
    '\n[DiNovo] Any question, please send e-mail to: yfu@amss.ac.cn',
    '\n[DiNovo] *************** <<  Welcome to use DiNovo!  >> ******************',
    '\n[DiNovo] Writing config file...',
    '\n[DiNovo] Please fill the .cfg file, and run DiNovo with this command: DiNovo.exe [parameter file]',
    '\n[DiNovo] Finished!\n\n',
)


INFO_TO_USER_Flow1 = (
    "\n[DiNovo] <FLOW BASIC PROCESSING> [INI] Checking runtime environment...",

    "\n[DiNovo] <FLOW BASIC PROCESSING> [INI] Reading initialization files...",

    "\n[DiNovo] <FLOW BASIC PROCESSING> [INI] LABEL INFO: ",

    "\n[DiNovo] <FLOW BASIC PROCESSING> [MGF] Reading MGF files (First time will take a few minutes)...",

    "\n[DiNovo] <FLOW BASIC PROCESSING> [OUT] Preprocessing MGF files and writing ..."
)

INFO_TO_USER_Flow2 = (
    '\n[DiNovo] <FLOW SPECTRAL PAIRING> [INI] Creating output paths and files...',

    '\n[DiNovo] <FLOW SPECTRAL PAIRING> [IDX] Building precursor mass indexes...',

    '\n[DiNovo] <FLOW SPECTRAL PAIRING> [MSP] Matching spectral pairs...',

    '\n[DiNovo] <FLOW SPECTRAL PAIRING> [OUT] Combining results...',  # 3
)

# flow3 中会含有 flow1 和 flow2 中的内容，但是完全没问题
# 直接通过staff调用来完成即可
INFO_TO_USER_Flow3 = (
    '\n[DiNovo] <FLOW DE NOVO SEQUENCING> [SEQ] De Novo sequencing...',

    '\n[DiNovo] <FLOW DE NOVO SEQUENCING> [SEQ] De Novo MODE: ',

    # '\n[DiNovo] <FLOW DE NOVO SEQUENCING> [OUT] Writing reuslts...',

    '\n[DiNovo] <FLOW DE NOVO SEQUENCING> [OUT] Combining...',  # 2

    '\n[DiNovo] <FLOW DE NOVO SEQUENCING> [MAP] Mapping sequencing results...'  # 2
)

INFO_TO_USER_TaskReadMGF = (
    '\n[DiNovo] \t <Task read mgf> START',

    '\n[DiNovo] \t\t <Read mgf> Creating index file for: ',
)


INFO_TO_USER_TaskPrep = (
    "\n[DiNovo] \t <Task preprocess> START",
    
    "\n[DiNovo] \t\t <Preprocess> Preprocess and write: ",

    "\n[DiNovo] \t\t <Preprocess> NeuCode Peaks detection(ONLY) and write for:",

    "\n[DiNovo] \t\t <Preprocess> Save a temp file for: ",
)

INFO_TO_USER_TaskPMIDX = (
    "\n[DiNovo] \t <Task index> START",

    "\n[DiNovo] \t\t <Index> Build PMIDX for: ",

    "\n[DiNovo] \t\t <Index> Check memory cost...",
)

INFO_TO_USER_TaskPair = (

    "\n[DiNovo] \t <Task pairing> Start",

    "\n[DiNovo] \t <Task pairing> Memory mode:",
    
    "\n[DiNovo] \t <Task pairing> Loading ",

    "\n[DiNovo] \t\t <MS data> Loading LysargiNase MGF:\t",

    "\n[DiNovo] \t <Task pairing> Building PMIDX...",

    "\n[DiNovo] \t\t <Index> Preprocessing(online) LysargiNase MGF:\t",

    "\n[DiNovo] \t <Task pairing> Pairing ",

    # "\n[DiNovo] \t\t <PMINDEX> Extracting Trypsin precursor info:\t",  # and building index...

    "\n[DiNovo] \t\t <Pairing> Loading and pairing Trypsin MGF:\t",

    "\n[DiNovo] \t\t <Process> ",

)

INFO_TO_USER_TaskDiNovo = (
    '\n[DiNovo] \t <Task sequencing> Sequencing...',

    '\n[DiNovo] \t\t <Process> ',

    '\n[DiNovo] \tDiNovo sequencing FINISHED!',

    '\n[DiNovo] \tUsing Time:',
)


myLogPath = 'DiNovo.log'

# ---------------------------------------------------------------
# input: string
# 向日志文件里写东西，然后打印到命令行
def logToUser(strInfo, path=myLogPath):

    # if os.access(myLogPath, os.W_OK):  # 当文件被excel打开时，这个东东没啥用

    try:
        print(strInfo)
        f_w = open(path, 'a', encoding='utf8')
        f_w.write(strInfo + '\n')
        f_w.close()
    except IOError:
        print("DiNovo.log is opened! Please close it and run the program again!")
        sys.exit(0)

# ------------------------------------------------------------
# input:string
# log函数
def logGetError(strInfo, path=myLogPath):

    print(strInfo)
    
    logging.basicConfig(filename=path,
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )
    logging.error(strInfo)
    sys.exit(0)


def logGetWarning(strInfo, path=myLogPath):
    
    print(strInfo)
    
    logging.basicConfig(filename=path,
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )
    logging.warning(strInfo)


