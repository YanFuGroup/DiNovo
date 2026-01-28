import os
import torch
from torch import optim
import torch.nn.functional as F
import config
from data_reader import GCNovoTrainDataset, collate_func
from model_gcn import GCNovoModel, InitNet
import time
import math
import logging

logger = logging.getLogger(__name__)
# mass_AA_np = gcnovo_config.mass_AA_np
forward_model_save_name = 'forward_gcnovo.pth'
backward_model_save_name = 'backward_gcnovo.pth'
init_net_save_name = 'init_net.pth'



def to_one_hot(y, n_dims=None):
    """ Take integer y with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def focal_loss(logits, labels, ignore_index=-100, gamma=2.):
    """

    :param logits: float tensor of shape [batch, T, 26]
    :param labels: long tensor of shape [batch, T]
    :param ignore_index: ignore the loss of those tokens
    :param gamma:
    :return: average loss, num_valid_token
    """
    valid_token_mask = (labels != ignore_index).float()  # [batch, T]
    num_valid_token = torch.sum(valid_token_mask)
    batch_size, T, num_classes = logits.size()
    sigmoid_p = torch.sigmoid(logits)
    target_tensor = to_one_hot(labels, n_dims=num_classes).float().to(device)
    zeros = torch.zeros_like(sigmoid_p)
    pos_p_sub = torch.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)  # [batch, T, 26]
    neg_p_sub = torch.where(target_tensor > zeros, zeros, sigmoid_p)  # [batch, T, 26]

    per_token_loss = - (pos_p_sub ** gamma) * torch.log(torch.clamp(sigmoid_p, 1e-8, 1.0)) - \
                     (neg_p_sub ** gamma) * torch.log(torch.clamp(1.0 - sigmoid_p, 1e-8, 1.0))
    per_entry_loss = torch.sum(per_token_loss, dim=2)  # [batch, T]
    per_entry_loss = per_entry_loss * valid_token_mask  # masking out loss from pad tokens

    per_entry_average_loss = torch.sum(per_entry_loss) / (num_valid_token + 1e-6)
    return per_entry_average_loss, num_valid_token

def  build_model(args, training=True,device=0):
    """
    :return:
    """
    forward_gcnovo = GCNovoModel(args=args,device=device)
    backward_gcnovo = GCNovoModel(args=args,device=device)
    # print(args.use_lstm)
    if False:

        init_net = InitNet(args=args)
    else:
        init_net = None

    # load pretrained params if exist
    if os.path.exists(os.path.join(args.train_dir, forward_model_save_name)):
        assert os.path.exists(os.path.join(args.train_dir, backward_model_save_name))
        logger.info("load pretrained model")
        forward_gcnovo.load_state_dict(torch.load(os.path.join(args.train_dir, forward_model_save_name),
                                                    map_location=device))
        backward_gcnovo.load_state_dict(torch.load(os.path.join(args.train_dir, backward_model_save_name),
                                                     map_location=device))

        if False:
            init_net.load_state_dict(torch.load(os.path.join(args.train_dir, init_net_save_name),
                                                map_location=device))

    else:
        assert training, f"building model for testing, but could not found weight under directory " \
                         f"{args.train_dir}"
        logger.info("initialize a set of new parameters")

    if False:
        # share embedding matrix
        backward_gcnovo.embedding.weight = forward_gcnovo.embedding.weight

    backward_gcnovo = backward_gcnovo.to(device)
    forward_gcnovo = forward_gcnovo.to(device)
    if False:
        init_net = init_net.to(device)
    return forward_gcnovo, backward_gcnovo

def extract_and_move_data(data):
    """

    :param data: result from dataloader
    :return:
    """
    peak_location, \
    peak_intensity, \
    precursormass, \
    spectrum_representation, \
    batch_forward_id_target, \
    batch_backward_id_target, \
    batch_forward_ion_index, \
    batch_backward_ion_index, \
    batch_forward_id_input, \
    batch_backward_id_input = data

    # move to device
    peak_location = peak_location.to(device)
    peak_intensity = peak_intensity.to(device)
    precursormass = precursormass.to(device)
    spectrum_representation = spectrum_representation.to(device)
    batch_forward_id_target = batch_forward_id_target.to(device)
    batch_backward_id_target = batch_backward_id_target.to(device)
    batch_forward_ion_index = batch_forward_ion_index.to(device)
    batch_backward_ion_index = batch_backward_ion_index.to(device)
    batch_forward_id_input = batch_forward_id_input.to(device)
    batch_backward_id_input = batch_backward_id_input.to(device)
    return (peak_location,
            peak_intensity,
            precursormass,
            spectrum_representation,
            batch_forward_id_target,
            batch_backward_id_target,
            batch_forward_ion_index,
            batch_backward_ion_index,
            batch_forward_id_input,
            batch_backward_id_input
            )

def validation(forward_gcnovo, backward_gcnovo,  init_net, valid_loader, args) -> float:
    with torch.no_grad():
        valid_loss = 0
        num_valid_samples = 0
        for data in valid_loader:
            peak_location, \
            peak_intensity, \
            precursormass, \
            spectrum_representation, \
            batch_forward_id_target, \
            batch_backward_id_target, \
            batch_forward_ion_index, \
            batch_backward_ion_index, \
            batch_forward_id_input, \
            batch_backward_id_input = extract_and_move_data(data)
            batch_size = batch_backward_id_target.size(0)

            if False:
                initial_state_tuple = init_net(spectrum_representation)
                forward_logit, _ = forward_gcnovo(batch_forward_ion_index, peak_location, peak_intensity,
                                                    batch_forward_id_input, initial_state_tuple)
                backward_logit, _ = backward_gcnovo(batch_backward_ion_index, peak_location, peak_intensity,
                                                      batch_backward_id_input, initial_state_tuple)
            else:
                # forward_logit = forward_gcnovo(batch_forward_ion_index, peak_location, peak_intensity, spectrum_representation, batch_forward_id_input)
                # backward_logit = backward_gcnovo(batch_backward_ion_index, peak_location, peak_intensity, spectrum_representation, batch_backward_id_input)
                forward_logit = forward_gcnovo(batch_forward_ion_index, peak_location, peak_intensity, precursormass)
                backward_logit = backward_gcnovo(batch_backward_ion_index, peak_location, peak_intensity, precursormass)

            forward_loss, f_num = focal_loss(forward_logit, batch_forward_id_target, ignore_index=0, gamma=2.)
            backward_loss, b_num = focal_loss(backward_logit, batch_backward_id_target, ignore_index=0, gamma=2.)
            valid_loss += forward_loss.item() * f_num.item() + backward_loss.item() * b_num.item()
            num_valid_samples += f_num.item() + b_num.item()
    average_valid_loss = valid_loss / (num_valid_samples + 1e-6)
    return float(average_valid_loss)

def perplexity(log_loss):
    return math.exp(log_loss) if log_loss < 300 else float('inf')

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
    lr = args.init_lr * (0.1 ** ((epoch + 1) // 3))
    logger.info(f"epoch: {epoch}\tlr: {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_model(forward_gcnovo, backward_gcnovo, init_net, args):
    torch.save(forward_gcnovo.state_dict(), os.path.join(args.train_dir,
                                                           forward_model_save_name))
    torch.save(backward_gcnovo.state_dict(), os.path.join(args.train_dir,
                                                            backward_model_save_name))
    if False:
        torch.save(init_net.state_dict(), os.path.join(args.train_dir,
                                                   init_net_save_name))


def train(args):
    train_set = GCNovoTrainDataset(args.input_feature_file_train,
                                     args.input_spectrum_file_train,
                                     args=args)
    num_train_features = len(train_set)
    steps_per_epoch = int(num_train_features / args.batch_size)
    print('+++++++++++++++++++++++++++')
    logger.info(f"{steps_per_epoch} steps per epoch")
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers,
                                                    collate_fn=collate_func)
    valid_set = GCNovoTrainDataset(args.input_feature_file_valid,
                                     args.input_spectrum_file_valid,
                                     args=args)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers,
                                                    collate_fn=collate_func)
    print('finished read data--------------')
    forward_gcnovo, backward_gcnovo, init_net = build_model(args=args)

    dense_params = list(forward_gcnovo.parameters()) + list(backward_gcnovo.parameters())
    print(dense_params)
    dense_optimizer = optim.Adam(dense_params,
                                 lr=args.init_lr,
                                 weight_decay=args.weight_decay)

    dense_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dense_optimizer, 'min', factor=0.5, verbose=True,
                                                                 threshold=1e-4, cooldown=10, min_lr=1e-5)


    best_valid_loss = float("inf")
    # train loop
    best_epoch = None
    best_step = None
    start_time = time.time()
    for epoch in range(args.num_epoch):
        # learning rate schedule
        # adjust_learning_rate(optimizer, epoch)
        for i, data in enumerate(train_data_loader):
            dense_optimizer.zero_grad()
            # sparse_optimizer.zero_grad()
            # start_time = time.time()
            peak_location, \
            peak_intensity, \
            precursormass, \
            spectrum_representation, \
            batch_forward_id_target, \
            batch_backward_id_target, \
            batch_forward_ion_index, \
            batch_backward_ion_index, \
            batch_forward_id_input, \
            batch_backward_id_input = extract_and_move_data(data)
            batch_size = batch_backward_id_target.size(0)
            # print('-------------')

            if False:
                initial_state_tuple = init_net(spectrum_representation)
                forward_logit, _ = forward_gcnovo(batch_forward_ion_index, peak_location, peak_intensity, precursormass,
                                                    batch_forward_id_input, initial_state_tuple)
                backward_logit, _ = backward_gcnovo(batch_backward_ion_index, peak_location, peak_intensity,precursormass,
                                                      batch_backward_id_input, initial_state_tuple)
            else:
                forward_logit = forward_gcnovo(batch_forward_ion_index, peak_location, peak_intensity, precursormass)
                backward_logit = backward_gcnovo(batch_backward_ion_index, peak_location, peak_intensity, precursormass)

            # print(forward_logit.size())
            forward_loss, _ = focal_loss(forward_logit, batch_forward_id_target, ignore_index=0, gamma=2.)
            backward_loss, _ = focal_loss(backward_logit, batch_backward_id_target, ignore_index=0, gamma=2.)
            total_loss = (forward_loss + backward_loss) / 2.
            # compute gradient
            total_loss.backward()
            # clip gradient
            # torch.nn.utils.clip_grad_norm_(dense_params, gcnovo_config.max_gradient_norm)

            dense_optimizer.step()
            # sparse_optimizer.step()

            if (i + 1) % args.steps_per_validation == 0:

                duration = time.time() - start_time

                step_time = duration / args.steps_per_validation
                loss_cpu = total_loss.item()
                # evaluation mode
                forward_gcnovo.eval()
                backward_gcnovo.eval()
                validation_loss = validation(forward_gcnovo, backward_gcnovo, init_net, valid_data_loader, args=args)
                dense_scheduler.step(validation_loss)
                # sparse_scheduler.step(validation_loss)

                logger.info(f"epoch {epoch} step {i}/{steps_per_epoch}, "
                            f"train perplexity: {perplexity(loss_cpu)}\t"
                            f"validation perplexity: {perplexity(validation_loss)}\tstep time: {step_time}")

                if validation_loss < best_valid_loss:
                    best_valid_loss = validation_loss
                    logger.info(f"best valid loss achieved at epoch {epoch} step {i}")
                    best_epoch = epoch
                    best_step = i
                    # save model if achieve a new best valid loss
                    print('save model+++++++++++++++++++++++++++++++')
                    save_model(forward_gcnovo, backward_gcnovo, init_net, args=args)

                # back to train model
                forward_gcnovo.train()
                backward_gcnovo.train()

                start_time = time.time()
            # observed that most of gpu memory is unoccupied cache, so clear cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(f"best model at epoch {best_epoch} step {best_step}")
