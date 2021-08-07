from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import PIL
import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz
import numpy as np
import functools
import pororo_data as data
import pororo_data_test as data_test

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.datasets import TextDataset
from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import GANTrainer




def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./cfg/pororo_s1.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    dir_path = '/media/bigguy/yl353/pororo_png/'
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    print('Using config:')
    pprint.pprint(cfg)
    random.seed(0)
    torch.manual_seed(0)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(0)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = './output/%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME)

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        image_transforms = transforms.Compose([
            PIL.Image.fromarray,
            transforms.Resize((cfg.IMSIZE, cfg.IMSIZE) ),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # dataset = TextDataset(cfg.DATA_DIR, 'train',
        #                       imsize=cfg.IMSIZE,
        #                       transform=image_transform)
        #assert dataset
        def video_transform(video, image_transform):
            vid = []
            for im in video:
                vid.append(image_transform(im))

            vid = torch.stack(vid).permute(1, 0, 2, 3)

            return vid
        video_len = 5
        n_channels = 3
        video_transforms = functools.partial(video_transform, image_transform=image_transforms)

        counter = np.load(dir_path + 'frames_counter.npy').item()
        base = data.VideoFolderDataset(dir_path, counter = counter, cache = dir_path, min_len = 4)
        storydataset = data.StoryDataset(base,dir_path, video_transforms)
        imagedataset = data.ImageDataset(base, dir_path, image_transforms)

        imageloader = torch.utils.data.DataLoader(
            imagedataset, batch_size=cfg.TRAIN.IM_BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        storyloader = torch.utils.data.DataLoader(
            storydataset, batch_size=cfg.TRAIN.ST_BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        base_test = data_test.VideoFolderDataset(dir_path, counter, dir_path, 4, False)
        testdataset = data_test.StoryDataset(base_test, dir_path, video_transforms)
        testloader = torch.utils.data.DataLoader(
            testdataset, batch_size=24,
            drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))
        algo = GANTrainer(output_dir, ratio = 1.0)
        algo.train(imageloader, storyloader, testloader, cfg.STAGE)
    else:
        datapath= '%s/test/val_captions.t7' % (cfg.DATA_DIR)
        algo = GANTrainer(output_dir)
        algo.sample(datapath, cfg.STAGE)
