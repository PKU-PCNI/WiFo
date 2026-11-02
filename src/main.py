# coding=utf-8
import argparse
import random
import os
from model import WiFo_model
from train import TrainLoop

import setproctitle
import torch

from DataLoader import data_load_main
from utils import *

import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

def dev(device_id='0'):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():
        return th.device('cuda:{}'.format(device_id))
    return th.device("cpu")

def create_argparser():
    defaults = dict(
        # experimental settings
        note = '',
        task = 'short',
        file_load_path = '',
        dataset = 'DS1',
        used_data = '',
        process_name = 'process_name',
        his_len = 6,
        pred_len = 6,
        few_ratio = 0.5,
        stage = 0,

        # model settings
        mask_ratio = 0.5,
        patch_size = 4,
        t_patch_size = 2,
        size = 'middle',
        no_qkv_bias = 0,
        pos_emb = 'SinCos',
        conv_num = 3,

        # pretrain settings
        random=True,
        mask_strategy = 'random',
        mask_strategy_random = 'batch', # ['none','batch']
        
        # training parameters
        lr=1e-3,
        min_lr = 1e-5,
        early_stop = 5,
        weight_decay=0.05,
        batch_size=256,
        log_interval=5,
        total_epoches = 10000,
        device_id='1',
        machine = 'machine_name',
        clip_grad = 0.05,  # 0.05
        lr_anneal_steps = 200,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')

def main():

    th.autograd.set_detect_anomaly(True)
    
    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-{}".format(args.process_name, args.device_id))
    setup_init(100)  # 随机种子设定100

    test_data = data_load_main(args) # 加载数据

    args.folder = 'Dataset_{}_Task_{}_FewRatio_{}_{}_{}/'.format(args.dataset, args.task, args.few_ratio, args.size, args.note)


    args.folder = 'Test_'+args.folder

    if args.mask_strategy_random != 'batch':
        args.folder = '{}_{}'.format(args.mask_strategy,args.mask_ratio) + args.folder
    args.model_path = './experiments/{}'.format(args.folder)
    logdir = "./logs/{}".format(args.folder)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        os.makedirs(args.model_path+'model_save/')

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)
    device = dev(args.device_id)

    model = WiFo_model(args=args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    if args.file_load_path != '':
        model.load_state_dict(torch.load('{}.pkl'.format(args.file_load_path),map_location=device, weights_only=True), strict=False)
        print('pretrained model loaded'+args.file_load_path)
    TrainLoop(
        args=args,
        writer=writer,
        model=model,
        test_data=test_data,
        device=device,
        early_stop=args.early_stop,
    ).run_loop()


if __name__ == "__main__":
    main()