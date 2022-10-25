""" Deformable Aggregation of Covariance Matrices for Few-shot Segmentation 
    Adapted from Hypercorrelation Squeeze code """
    
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from model.base.normalize import Normalize
import cv2
import numpy as np

import math
from model.hsnet_pgp import PGPNetwork
import os
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
import pdb
from model.base.normalize import Normalize
from gpytorch.mlls import SumMarginalLogLikelihood
import time
import random

def seed_torch(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(epoch, model, dataloader, mll, optimizer_gp, optimizer, training):

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)
    cls_loss = torch.nn.CrossEntropyLoss()

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)
        logit_mask, normed_s_feats, scaled_s_masks = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)
        #For debug || Load pre-trained GP hyper-parameters
        #print(model.module.gp_model.models[0].covar_module.kernels[1].base_kernel.lengthscale)
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])

        if training:
            optimizer_gp.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_gp.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':
    
    seed_torch(42)
    # Arguments parsing
    parser = argparse.ArgumentParser(description='DACM-Pytorch')
    parser.add_argument('--datapath', type=str, default='Datasets_HSN')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='pascal2l')
    parser.add_argument('--bsz', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101'])
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = PGPNetwork(args.backbone, False)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.module.hpn_learner.parameters(), "lr": args.lr}])
    optimizer_gp = optim.Adam([{"params": model.module.gp_model.parameters(), "lr": args.lr * 10}])
    scheduler_gp = optim.lr_scheduler.StepLR(optimizer_gp, step_size=2, gamma=1e-1)

    mll = SumMarginalLogLikelihood(model.module.likelihoods, model.module.gp_model)

    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.niter):

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, mll, optimizer_gp, optimizer, training=True)
        scheduler_gp.step()
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, mll, optimizer_gp, optimizer, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')


