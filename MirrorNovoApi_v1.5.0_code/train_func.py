import logging
import os
import torch
from torch import optim, nn
import config
from model import MirrorNovo_Model
import time
import math
mass_AA_np = config.mass_AA_np
forward_model_save_name = 'forward_mirrornovo.pth'
backward_model_save_name = 'backward_mirrornovo.pth'

def  build_model(args, training=True,device=0):
    """
    :return:
    """
    forward_mirrornovo = MirrorNovo_Model(args=args)
    backward_mirrornovo = MirrorNovo_Model(args=args)

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
    #     backward_gcnovo.embedding.weight = forward_gcnovo.embedding.weight

    backward_mirrornovo = backward_mirrornovo.to(device)
    forward_mirrornovo = forward_mirrornovo.to(device)
    # backward_gcnovo = nn.DataParallel(backward_gcnovo,device_ids=[0,1,2,3])
    # forward_gcnovo = nn.DataParallel(forward_gcnovo,device_ids=[0,1,2,3])
    return forward_mirrornovo, backward_mirrornovo

