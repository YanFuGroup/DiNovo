# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: DiNovo.py
Time: 2022.09.05 Monday
ATTENTION: none
"""
import sys
import multiprocessing
from MSStaff import CStaff
import os

if __name__ == "__main__":
    # get_path = os.path.realpath(sys.argv[0])
    # print(get_path)
    # get_path = os.path.realpath(__file__)
    # print(get_path)
    # get_path = os.path.dirname(get_path) + "\\"
    # print(get_path)
    # exit(3)
    multiprocessing.freeze_support()
    # staff = CStaff(["hah"])
    # staff = CStaff(["hahahaha", "./DiNovo.cfg"])
    # staff = CStaff(["", "D:\\MyPythonWorkSpace\\DiNovo\\DiNovo-1.cfg"])
    staff = CStaff(sys.argv)
    staff.start()


