import logging
import os
from datetime import datetime

import numpy as np
import torch


def setup_logger(args, current_time, log_dir="../logs"):
    os.makedirs(log_dir, exist_ok=True)

    log_filename = f"{args.dataset}_{args.task}_{current_time}.log"
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if args.lradj == 'type7':
        lr_adjust = {epoch: args.learning_rate * (0.7 ** ((epoch - 1) // 1))}
    if args.lradj == 'type6':
        lr_adjust = {epoch: args.learning_rate * (0.6 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, logger, patience=5, verbose=False, delta=0):
        self.logger = logger
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, model_name):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, model_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, model_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, model_name):
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + model_name)
        self.val_loss_min = val_loss