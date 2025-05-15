# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSStaff.py
Time: 2022.09.05 Monday
ATTENTION: none
"""

import datetime
from MSLogging import logGetError, INFO_TO_USER_Staff, logToUser
from MSOperator import op_INIT_CFILE_DATAPACK
from MSData import CDataPack
from MSFunctionIO import CFunctionConfig
from MSFlow import CFlow0, CFlow1, CFlow2, CFlow3, CFlow4
from MSTask import CTaskValidation

class CStaff:

    def __init__(self, inputArgv, inputPath=""):

        self.dp = CDataPack()
        self.argv = inputArgv
        self.exe_path = inputPath


    def start(self):



        dateNow = datetime.datetime.now()
        logToUser(str(dateNow))
        # version
        logToUser(INFO_TO_USER_Staff[0])

        # welcome to use it!
        self.__captainReportWelcome()

        # run flow
        self.__captainRunFlow()

        # finish
        dateNow = datetime.datetime.now()
        logToUser(str(dateNow))
        logToUser(INFO_TO_USER_Staff[5])


    def __captainReportWelcome(self):

        logToUser(INFO_TO_USER_Staff[1])

        logToUser(INFO_TO_USER_Staff[2])


    def __captainRunFlow(self):

        n = len(self.argv)

        if n == 1:
            # print("flow1")
            logToUser(INFO_TO_USER_Staff[3])

            # writing a sample file of configuration
            flow0 = CFlow0()
            flow0.run()
            logToUser(INFO_TO_USER_Staff[4])

        elif n == 2:

            # initializing your datapack (include: initial data and configuration)
            op_INIT_CFILE_DATAPACK(self.dp)
            # loading your configuration file (filling by user)
            functionConfig = CFunctionConfig()
            functionConfig.file2config(self.argv[1], self.dp.myCFG)


            # -----------------------------------------------------
            if self.dp.myCFG.B9_NEUCODE_DETECT_APPROACH == 2:
                self.dp.myCFG.D0_WORK_FLOW_NUMBER = 1
                logToUser("\n[ATTENTION] using lightGBM model, change the work flow to number 1.")
            # -----------------------------------------------------

            if self.dp.myCFG.D0_WORK_FLOW_NUMBER == 1:

                # initializing program, reading .mgf files and storing them as .pkl
                flow1 = CFlow1(self.dp)
                flow1.run()

            elif self.dp.myCFG.D0_WORK_FLOW_NUMBER == 2:

                # initializing program, reading .mgf files and storing them as .pkl
                flow1 = CFlow1(self.dp)
                flow1.run()

                # finding mirror spectral pairs
                flow2 = CFlow2(self.dp)
                flow2.run()

            elif self.dp.myCFG.D0_WORK_FLOW_NUMBER == 3:

                # initializing program, reading .mgf files and storing them as .pkl
                flow1 = CFlow1(self.dp)
                flow1.run()

                # finding mirror spectral pairs
                flow2 = CFlow2(self.dp)
                flow2.run()

                # de novo sequencing per spectral pair
                flow3 = CFlow3(self.dp)
                flow3.run()

            # pure de novo sequencing
            elif self.dp.myCFG.D0_WORK_FLOW_NUMBER == 4:

                # initializing program, reading .mgf files and storing them as .pkl
                flow1 = CFlow1(self.dp)
                flow1.run()

                # de novo sequencing directly per spectra without pairing
                flow4 = CFlow4(self.dp)
                flow4.run()

            else:

                logGetError("\n[ATTENTION]\tWrong WORK FLOW NUMBER! Please check your configuration!")

            # clearing
            if self.dp.myCFG.D0_WORK_FLOW_NUMBER > 1:
                # combine spectral pair
                # clearing the temp folders
                import os, shutil
                from MSSystem import IO_NAME_FOLDER_TEMPORARY, FEATURE_MGF_FILE_SUFFIX
                for tmpPath in (self.dp.LIST_OUTPUT_PATH + [self.dp.myCFG.E1_PATH_EXPORT + IO_NAME_FOLDER_TEMPORARY]):
                    # print("Cleaning path:", tmpPath)
                    for root, dirs, files in os.walk(tmpPath):
                        for file in files:
                            try:
                                os.remove(os.path.join(root, file))
                            except:
                                logToUser("[Warning] Cannot delete file: " + file)
                        for dir_ in dirs:
                            try:
                                shutil.rmtree(os.path.join(root, dir_))
                            except:
                                logToUser("[Warning] Cannot delete folder: " + dir_)
                    try:
                        os.removedirs(tmpPath)
                    except:
                        logToUser("[Warning] Cannot delete folder: " + tmpPath)

                # clearing the feature neucode mgf if user don't want to output them
                if self.dp.myCFG.D0_WORK_FLOW_NUMBER > 2 and self.dp.myCFG.D9_DE_NOVO_APPROACH > 1 and self.dp.myCFG.A7_LABEL_TYPE == 1 and self.dp.myCFG.E7_EXPORT_FEATURE_MGF == 0:
                    for raw_name in self.dp.LIST_MGF_NAME_TRY + self.dp.LIST_MGF_NAME_LYS:
                        tmp_path = self.dp.myCFG.E1_PATH_EXPORT + raw_name + FEATURE_MGF_FILE_SUFFIX
                        try:
                            os.remove(tmp_path)
                        except:
                            ...

            if self.dp.myCFG.V0_FLAG_DO_VALIDATION == 1:
                taskValidation = CTaskValidation(self.dp)
                taskValidation.work()


