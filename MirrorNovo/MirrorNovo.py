# -*- coding: utf-8 -*-
import multiprocessing
import os
import time
import warnings

import torch

# 忽略警告
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import datetime
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
from train_func import build_model
from data_loader import GCNovoDenovoDataset
from model import InferenceModelWrapper
from denovo import IonCNNDenovo
from writer import DenovoWriter
from init_args import init_args
import sys
import logging
import logging.config
logger = logging.getLogger(__name__)
def init_log(log_file_name):
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)

def mirrornovo(args, denovo_input_feature_file,denovo_output_file,process_id):

    print("denovo_input_feature_file:",denovo_input_feature_file)
    device = torch.device(f"cuda:{process_id}" if torch.cuda.is_available() and process_id != "-1" else "cpu")
    data_reader = GCNovoDenovoDataset(args,args.denovo_input_spectrum_file,args.denovo_input_mirror_spectrum_file,denovo_input_feature_file)
    denovo_worker = IonCNNDenovo(args=args,device=device)
    forward_deepnovo, backward_deepnovo = build_model(args=args, training=False,device=device)
    model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo)

    writer = DenovoWriter(args=args,denovo_output_file=denovo_output_file)
    candidate = denovo_worker.search_denovo(model_wrapper, data_reader, writer,process_id)

    return None
def rerank(param_name):
    param_path = sys.argv[:]
    logger.info("rerank mode")
    param_path = param_path[1] if len(param_path)>1 else f"./{param_name}.cfg"
    args = init_args(param_path)
    time1 = time.time()
    data_reader = GCNovoDenovoDataset(args,args.denovo_input_spectrum_file,args.denovo_input_mirror_spectrum_file,args.denovo_input_feature_file)
    denovo_worker = IonCNNDenovo(args=args)
    forward_deepnovo, backward_deepnovo = build_model(args=args, training=False)
    model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo)
    writer = DenovoWriter(args,args.denovo_output_file)

    candidate = denovo_worker.search_denovo(model_wrapper, data_reader, writer)
    logger.info(f'using time:{time.time() - time1}')
    return candidate
def split_feature(denovo_input_feature_file, num_processes):
    df = pd.read_csv(denovo_input_feature_file,sep='\t')
    parts = np.array_split(df, num_processes)
    print(df.shape)
    for i, part_df in enumerate(parts):
        part_df.to_csv(f'{denovo_input_feature_file}_tmp{i}', index=False,sep='\t')
def create_file(args):
    knapsack, _ = os.path.split(args.knapsack)
    denovo_output_file, _ = os.path.split(args.denovo_output_file)
    train_dir,_  = os.path.split(args.train_dir)
    log_dir = "./log"
    if not os.path.exists(knapsack):
        os.makedirs(knapsack)
    if not os.path.exists(denovo_output_file):
        os.makedirs(denovo_output_file)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
if __name__ == '__main__':
    logger.info("Welcome to MirrorNovo, Please wait a moment...")

    param_name = "params"
    # search denovo
    param_path = sys.argv[:]
    # print(param_path)
    logger.info("test mode")
    param_path = param_path[1] if len(param_path) > 1 else f"./{param_name}.cfg"
    args = init_args(param_path)
    create_file(args)

    log_file_name = f"./log/{param_name}_" + datetime.datetime.now().strftime("%Y%m%d%H%M") + ".log"
    time1 = time.time()
    init_log(log_file_name=log_file_name)
    mirrornovo(args,args.denovo_input_feature_file, args.denovo_output_file, args.processes)
    # split_feature(args.denovo_input_feature_file,args.processes)
    # process = []
    # for i in range(args.processes):
    #     p = multiprocessing.Process(target=mirrornovo, args=(args,f'{args.denovo_input_feature_file}_tmp{i}',f'{args.denovo_output_file}_file{i}',i))
    #     process.append(p)
    #     p.start()
    # for p in process:
    #     p.join()
    #
    # for i in range(args.processes):
    #     os.remove(f'{args.denovo_input_feature_file}_tmp{i}')
    logger.info(f'using time:{time.time() - time1}')
