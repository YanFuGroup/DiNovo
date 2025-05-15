import logging
import os
import torch
from torch import optim, nn
import config
from data_loader import GCNovoDenovoDataset, collate_func
from model import DeepNovoModel
import time
import math
mass_AA_np = config.mass_AA_np
forward_model_save_name = 'forward_deepnovo.pth'
backward_model_save_name = 'backward_deepnovo.pth'

def  build_model(args, training=True,device=0):
    """
    :return:
    """
    forward_deepnovo = DeepNovoModel(args=args)
    backward_deepnovo = DeepNovoModel(args=args)

    # load pretrained params if exist
    if os.path.exists(os.path.join(args.train_dir, forward_model_save_name)):
        assert os.path.exists(os.path.join(args.train_dir, backward_model_save_name))
        try:
            forward_deepnovo.load_state_dict(torch.load(os.path.join(args.train_dir, forward_model_save_name),
                                                        map_location=device))
            backward_deepnovo.load_state_dict(torch.load(os.path.join(args.train_dir, backward_model_save_name),
                                                         map_location=device))
        except Exception as e:
            print(e)
            logging.error("load model failed, try to load model trained on multi-gpu machine")
            exit()

    else:
        assert training, f"building model for testing, but could not found weight under directory " \
                         f"{args.train_dir}"
        # logger.info("initialize a set of new parameters")

    # if args.use_lstm:
    #     # share embedding matrix
    #     backward_deepnovo.embedding.weight = forward_deepnovo.embedding.weight

    backward_deepnovo = backward_deepnovo.to(device)
    forward_deepnovo = forward_deepnovo.to(device)
    # backward_deepnovo = nn.DataParallel(backward_deepnovo,device_ids=[0,1,2,3])
    # forward_deepnovo = nn.DataParallel(forward_deepnovo,device_ids=[0,1,2,3])
    return forward_deepnovo, backward_deepnovo

