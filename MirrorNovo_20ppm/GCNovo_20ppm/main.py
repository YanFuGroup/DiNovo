
# ==============================================================================
import time
import torch
import logging
import logging.config
import config
import os
from train_func1 import train, build_model, validation, perplexity
# from train_func2 import train, build_model, validation, perplexity
# from train_func import train, build_model, validation, perplexity
from data_reader import GCNovoDenovoDataset, collate_func, GCNovoTrainDataset
from model_gcn import InferenceModelWrapper
from denovo import IonCNNDenovo
# from model import InferenceModelWrapper
# from denovo import IonCNNDenovo
# from model_gcn2 import InferenceModelWrapper
# from denovo2 import IonCNNDenovo
from writer import DenovoWriter
from init_args import init_args
import worker_test
from mgf2feature import mgftofeature
import datetime
logger = logging.getLogger(__name__)

def engine_1(args):
    # train + search denovo + test
    start = time.time()
    logger.info(f"training mode")
    torch.cuda.empty_cache()
    train(args=args)
    logger.info(f'using time:{time.time() - start}')
    engine_2(args)

def engine_2(args):
    # search denovo + test
    """
    search denovo
    """
    mgftofeature(args.denovo_input_spectrum_file)
    torch.cuda.empty_cache()
    start = time.time()
    logger.info("denovo mode")
    data_reader = GCNovoDenovoDataset(feature_filename=args.denovo_input_feature_file,
                                        spectrum_filename=args.denovo_input_spectrum_file,
                                        args=args)
    denovo_worker = IonCNNDenovo(args=args)
    forward_GCnovo, backward_GCnovo, init_net = build_model(args=args, training=False)
    model_wrapper = InferenceModelWrapper(forward_GCnovo, backward_GCnovo, init_net)
    writer = DenovoWriter(args=args)
    denovo_worker.search_denovo(model_wrapper, data_reader, writer)
    torch.cuda.empty_cache()
    logger.info(f'using time:{time.time() - start}')

    engine_3(args)

def engine_3(args):
    # test
    logger.info("test mode")
    worker_test = GCnovo_worker_test.WorkerTest(args=args)
    worker_test.test_accuracy()

    # show 95 accuracy score threshold
    accuracy_cutoff = 0.95
    accuracy_file = args.accuracy_file
    # score_cutoff = find_score_cutoff(accuracy_file, accuracy_cutoff)

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

if __name__ == '__main__':
    param_path = [
        "./config/[VTM]Ecoli_lysargiNase_lysN.cfg"
    ]

    log_path = "./log/"
    if isinstance(param_path,list):
        # print(param_path)
        for _param_path in param_path:
            dir, param_file = os.path.split(_param_path)
            # log_file_name = "top5_" + param_file[-4] + ".log"
            now = datetime.datetime.now().strftime("%Y%m%d%H%M")
            args = init_args(_param_path)
            # log_file_name = "./log/" + now + "(" + str(args.engine_model) + ").log"
            log_file_name = log_path + param_file + now + "(" + str(args.engine_model) + ").log"
            init_log(log_file_name=log_file_name)
            if os.path.exists(args.train_dir):
                pass
            else:
                os.makedirs(args.train_dir)
            if args.engine_model == 1:
                # print("engine model 1")
                engine_1(args=args)
            elif args.engine_model == 2:
                engine_2(args=args)
            elif args.engine_model == 3:
                engine_3(args=args)