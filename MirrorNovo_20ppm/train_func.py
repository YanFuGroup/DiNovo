import logging
import os
import torch
from torch import optim, nn
import config
from data_loader import GCNovoDenovoDataset, collate_func
from model import mirrorNovoModel
import time
import math
mass_AA_np = config.mass_AA_np
forward_model_save_name = 'forward_mirrornovo.pth'
backward_model_save_name = 'backward_mirrornovo.pth'

def  build_model(args, training=True,device=0):
    """
    :return:
    """
    forward_mirrornovo = mirrorNovoModel(args=args)
    backward_mirrornovo = mirrorNovoModel(args=args)

    # load pretrained params if exist
    if os.path.exists(os.path.join(args.train_dir, forward_model_save_name)):
        assert os.path.exists(os.path.join(args.train_dir, backward_model_save_name))
        try:
            forward_mirrornovo.load_state_dict(torch.load(os.path.join(args.train_dir, forward_model_save_name),
                                                        map_location=device))
            backward_mirrornovo.load_state_dict(torch.load(os.path.join(args.train_dir, backward_model_save_name),
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
    #     backward_mirrornovo.embedding.weight = forward_mirrornovo.embedding.weight

    backward_mirrornovo = backward_mirrornovo.to(device)
    forward_mirrornovo = forward_mirrornovo.to(device)
    # backward_mirrornovo = nn.DataParallel(backward_mirrornovo,device_ids=[0,1,2,3])
    # forward_mirrornovo = nn.DataParallel(forward_mirrornovo,device_ids=[0,1,2,3])
    return forward_mirrornovo, backward_mirrornovo

