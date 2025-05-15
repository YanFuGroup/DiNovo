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

    multiprocessing.freeze_support()
    staff = CStaff(sys.argv)
    staff.start()


