import argparse
import csv
import time
from datetime import datetime
from logging import critical
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, auc, roc_auc_score, precision_recall_curve, \
    average_precision_score, precision_score, recall_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from fastdtw import fastdtw
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

from STAno import STAno
from tools import EarlyStopping, setup_logger, adjust_learning_rate
from data_provider.dataloader import load_data, collate_fn, GraphDataset
from huggingface_hub import login

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#torch.autograd.set_detect_anomaly(True)

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)







class Trainer:
    def __init__(self, train_loader, test_loader, test_labels, args):
        self.args = args
        self.device = args.device

        self.train_loader = train_loader

        self.test_loader = test_loader
        self.test_labels = test_labels

        self.model = STAno(self.args).to(self.device)
        self.optimizer = self._select_optimizer()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.8)
        self.drop_patch_prob = args.drop_patch_prob

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        return model_optim

    def _select_criterion(self, task):
        if task == 'classification':
            criterion = nn.CrossEntropyLoss(reduction='mean')
        elif task == 'reconstruction':
            criterion = nn.MSELoss(reduction='mean')
        else:
            raise ValueError("Invalid criterion: {task}")
        return criterion

    def drop_patches(self, nodes_tensor, edge_indices_tensor, edge_attrs_tensor, adj_tensor):
        if self.drop_patch_prob > 0:
            nodes_mask = torch.bernoulli((1 - self.drop_patch_prob) * torch.ones_like(nodes_tensor, dtype=torch.float))
            nodes_tensor = nodes_tensor * nodes_mask.to(torch.int)

            edge_mask = torch.bernoulli((1 - self.drop_patch_prob) * torch.ones_like(edge_attrs_tensor, dtype=torch.float))

            edge_indices_tensor = edge_indices_tensor * edge_mask.unsqueeze(-1).to(torch.int)
            edge_attrs_tensor = edge_attrs_tensor * edge_mask.to(torch.int)

            adj_tensor = adj_tensor * edge_mask.unsqueeze(1).to(torch.float)

        return nodes_tensor, edge_indices_tensor, edge_attrs_tensor, adj_tensor

    def train(self, current_time):
        best_model_folder = os.path.join("checkpoints", f"{self.args.task}_{self.args.dataset}_{current_time}")
        os.makedirs(best_model_folder, exist_ok=True)

        ce_criterion = self._select_criterion('classification')
        recon_criterion = self._select_criterion('reconstruction')
        scaler = torch.amp.GradScaler('cuda')
        early_stopping = EarlyStopping(logger=logger, patience=self.args.patience, verbose=True)

        train_steps = len(self.train_loader)
        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_list = []
            self.model.train()
            epoch_start_time = time.time()

            consecutive_errors = 0
            stop_training = False
            # --- We added tqdm here to track batch progress! ---
            for i, batch_data in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}")):
                if stop_training:
                    break
                try:
                    iter_count += 1
                    (nodes_tensor, edge_indices_tensor, edge_attrs_tensor, nodes_mask_tensor, edge_indices_mask_tensor, edge_attrs_mask_tensor,
                     adj_tensor, adj_mask) = [d.to(self.device) for d in batch_data]
                    nodes_tensor, edge_indices_tensor, edge_attrs_tensor, adj_tensor = self.drop_patches(
                        nodes_tensor, edge_indices_tensor, edge_attrs_tensor, adj_tensor
                    )

                    last_true_indices = (nodes_mask_tensor.sum(dim=1) - 1).long()
                    with torch.amp.autocast('cuda'):
                        mask_nodes = nodes_tensor * nodes_mask_tensor
                        mask_edge_indices = edge_indices_tensor * edge_indices_mask_tensor.unsqueeze(-1)
                        mask_edge_attrs = edge_attrs_tensor * edge_attrs_mask_tensor
                        node_logits, edge_indices_logits, edge_attr_logits, decoded_nodes, decoded_edge_indices, decoded_edge_attrs, embedded_input, embedded_output= self.model(
                            mask_nodes, mask_edge_indices, mask_edge_attrs, adj_tensor, mode='train')

                        input_nodes_src = mask_nodes[:, 0]
                        output_nodes_src = node_logits[:, 0, :]
                        input_nodes_dst = nodes_tensor[torch.arange(nodes_tensor.shape[0]), last_true_indices]
                        output_nodes_dst = node_logits[torch.arange(node_logits.shape[0]), last_true_indices]
                        src_loss = ce_criterion(output_nodes_src, input_nodes_src.long())
                        dst_loss = ce_criterion(output_nodes_dst, input_nodes_dst.long())
                        loss1 = 0.1 * src_loss.mean() + 0.9 * dst_loss.mean()
                        node_loss = ce_criterion(
                            node_logits.permute(0, 2, 1).float(), nodes_tensor.long()
                        )
                        edge_indices_loss = ce_criterion(
                            edge_indices_logits.permute(0, 1, 3, 2).reshape(-1, edge_indices_logits.shape[2]),
                            edge_indices_tensor.long().reshape(-1)
                        )
                        edge_attrs_loss = ce_criterion(
                            edge_attr_logits.permute(0, 2, 1).float(), edge_attrs_tensor.long()
                        )
                        loss2 = 0.2 * node_loss.mean() + 0.6 * edge_indices_loss.mean() + 0.2 * edge_attrs_loss.mean()
                        loss3 = recon_criterion(embedded_output, embedded_input)
                        loss = 0.1 * loss1 + 0.5 * loss2 + 0.4 * loss3

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                    train_loss_list.append(loss.item())

                    if (i + 1) % 400 == 0:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        logger.info(f"iters={i + 1}, epoch={epoch + 1}, loss={loss.item():.6f}, "
                                    f"speed={speed:.4f}s/iter, left={left_time:.4f}s")
                        iter_count = 0
                        time_now = time.time()
                    consecutive_errors = 0

                except Exception as e:
                    logger.warning(f"Exception occurred at epoch {epoch + 1}, batch {i + 1}: {e}")
                    raise e  # --- WE ADDED THIS TO REVEAL THE HIDDEN ERROR ---
                    
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        logger.error("Three consecutive errors occurred. Stopping training.")
                        stop_training = True
                        break

            if stop_training:
                logger.info("Training stopped due to consecutive errors.")
                break
            epoch_time_cost = time.time() - epoch_start_time
            logger.info(f"[Train] Epoch={epoch + 1} cost time={epoch_time_cost:.2f}s")

            if not stop_training:
                train_loss_avg = float(np.mean(train_loss_list))
                val_loss = self.val()
                logger.info(f"Epoch={epoch + 1}, Steps={train_steps}, "
                            f"TrainLoss={train_loss_avg:.6f}, ValLoss={val_loss:.6f}")

                early_stopping(val_loss, self.model, best_model_folder, model_name="checkpoint.pth")
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered.")
                    break
                self.scheduler.step()

        checkpoints_path = os.path.join(best_model_folder, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(checkpoints_path, weights_only=True, map_location=self.device))
        return self.model

    def val(self):
        total_loss = []
        self.model.eval()
        val_time = time.time()
        ce_criterion = self._select_criterion('classification')
        recon_criterion = self._select_criterion('reconstruction')
        with torch.no_grad():
            for i, batch_data in enumerate(self.test_loader):
                (nodes_tensor, edge_indices_tensor, edge_attrs_tensor,
                 nodes_mask_tensor, edge_indices_mask_tensor, edge_attrs_mask_tensor,
                 adj_tensor, adj_mask) = [d.to(self.device) for d in batch_data]
                last_true_indices = (nodes_mask_tensor.sum(dim=1) - 1).long()
                with torch.amp.autocast('cuda'):
                    mask_nodes = nodes_tensor * nodes_mask_tensor
                    mask_edge_indices = edge_indices_tensor * edge_indices_mask_tensor.unsqueeze(-1)
                    mask_edge_attrs = edge_attrs_tensor * edge_attrs_mask_tensor
                    node_logits, edge_indices_logits, edge_attr_logits, decoded_nodes, decoded_edge_indices, decoded_edge_attrs, embedded_input, embedded_output = self.model(
                        mask_nodes, mask_edge_indices, mask_edge_attrs, adj_tensor, mode='train')

                    input_nodes_src = mask_nodes[:, 0]
                    output_nodes_src = node_logits[:, 0, :]
                    input_nodes_dst = nodes_tensor[torch.arange(nodes_tensor.shape[0]), last_true_indices]
                    output_nodes_dst = node_logits[torch.arange(node_logits.shape[0]), last_true_indices]
                    src_loss = ce_criterion(output_nodes_src, input_nodes_src.long())
                    dst_loss = ce_criterion(output_nodes_dst, input_nodes_dst.long())
                    loss1 = 0.1 * src_loss.mean() + 0.9 * dst_loss.mean()
                    node_loss = ce_criterion(
                        node_logits.permute(0, 2, 1).float(), nodes_tensor.long()
                    )
                    edge_indices_loss = ce_criterion(
                        edge_indices_logits.permute(0, 1, 3, 2).reshape(-1, edge_indices_logits.shape[2]),
                        edge_indices_tensor.long().reshape(-1)
                    )
                    edge_attrs_loss = ce_criterion(
                        edge_attr_logits.permute(0, 2, 1).float(), edge_attrs_tensor.long()
                    )
                    loss2 = 0.2 * node_loss.mean() + 0.6 * edge_indices_loss.mean() + 0.2 * edge_attrs_loss.mean()
                    loss3 = recon_criterion(embedded_output, embedded_input)
                    loss = 0.1 * loss1 + 0.5 * loss2 + 0.4 * loss3

                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        logger.info(f"[Val] cost time={time.time() - val_time:.2f}s")
        return total_loss

    def test(self, evaluation=False):
        if evaluation:
            self.model.load_state_dict(
                torch.load(self.args.ckpt_path, weights_only=False))
        self.model = self.model.eval()
        anomaly_scores  = []
        test_labels = []
        test_labels_tensor = torch.tensor(self.test_labels).to(self.device)
        start_time = time.time()
        with torch.no_grad():
            for i, batch_data in enumerate(self.test_loader):
                (nodes_tensor, edge_indices_tensor, edge_attrs_tensor,
                 nodes_mask_tensor, edge_indices_mask_tensor, edge_attrs_mask_tensor,
                 adj_tensor, adj_mask) = [d.to(self.device) for d in batch_data]
                batch_labels = test_labels_tensor[i * nodes_tensor.size(0):(i + 1) * nodes_tensor.size(0)].unsqueeze(1)

                with torch.amp.autocast('cuda'):
                    mask_nodes = nodes_tensor * nodes_mask_tensor
                    mask_edge_indices = edge_indices_tensor * edge_indices_mask_tensor.unsqueeze(-1)
                    mask_edge_attrs = edge_attrs_tensor * edge_attrs_mask_tensor
                    decoded_nodes, decoded_edge_indices, decoded_edge_attrs, embedded_input = self.model(mask_nodes, mask_edge_indices, mask_edge_attrs, adj_tensor, mode='test')

                    if self.args.task == 'detour' or self.args.task == 'loop' or self.args.task == 'switch':
                        embedded_output = self.model.embedding_layer(
                            decoded_nodes, decoded_edge_indices, mask_edge_attrs, adj_tensor
                        )
                    elif self.args.task == 'time':
                        embedded_output = self.model.embedding_layer(
                            mask_nodes, mask_edge_indices, decoded_edge_attrs, adj_tensor
                        )
                    else:
                        raise ValueError(f"Invalid task={self.args.task}")

                    embedded_input_np = embedded_input.detach().cpu().numpy()
                    embedded_output_np = embedded_output.detach().cpu().numpy()
                    batch_anomaly_scores = []
                    for j in range(embedded_input_np.shape[0]):
                        dtw_distance, _ = fastdtw(embedded_input_np[j], embedded_output_np[j])
                        batch_anomaly_scores.append(dtw_distance)
                    anomaly_scores.append(batch_anomaly_scores)

                    test_labels.append(batch_labels.detach().cpu().numpy())

                    if i % 100 == 0:
                        logger.info(f"Batch {i} processed")

        anomaly_scores = np.concatenate(anomaly_scores, axis=0).reshape(-1)
        threshold = np.percentile(anomaly_scores, 100 - self.args.anomaly_ratio)
        logger.info(f"Threshold: {threshold}")

        pred = (anomaly_scores > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        logger.info("pred:   {}".format(pred.shape))
        logger.info("gt:     {}".format(gt.shape))

        p, r, f1, _ = precision_recall_fscore_support(gt, pred, average='binary')
        pre, rec, _t = precision_recall_curve(gt, pred)
        pr_auc = auc(rec, pre)
        roc_auc = roc_auc_score(gt, pred)
        end_time = time.time()
        logger.info(f"Test Precision: {p:.4f}, Test Recall: {r:.4f}, Test F1-Score: {f1:.4f}, Test PR-AUC: {pr_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Test Time: {end_time - start_time:.2f}s")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--LLM_path', type=str, required=True, default='./LLAMA', help='LLM path')
    parser.add_argument('--dataset', type=str, required=True, default='chengdu', help="Dataset name (e.g., 'chengdu', 'porto')")
    parser.add_argument('--noise', type=bool, required=True, default=False, help="Noise")
    parser.add_argument('--is_training', type=str, required=True, default=True, help="Status")
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/chengdu/checkpoint.pth', help="Checkpoint path")
    parser.add_argument('--task', type=str, required=True, default='detour', help="Task to perform (e.g., 'detour', 'switch', 'time', 'loop')")
    parser.add_argument('--anomaly_ratio', type=float, default=5, help="Anomaly ratio (%)")
    parser.add_argument('--detour_level', type=float, default=3, help="Detour level")
    parser.add_argument('--point_prob', type=float, default=0.3, help="Point prob")
    parser.add_argument('--switch_level', type=float, default=0.3, help="Switch level")
    parser.add_argument('--time_level', type=float, default=15, help="Time level")
    parser.add_argument('--loop_level', type=float, default=1, help="Loop level")

    # GPU
    parser.add_argument('--use_gpu', type=bool, default='True', help="Use GPU or not")
    parser.add_argument('--device', type=str, default='cuda')

    # Optimization
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='optimizer learning rate in train')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--drop_out', type=float, default=0.05, help="Dropout rate")
    parser.add_argument('--drop_patch_prob', type=float, default=0.2, help="Drop patch prob")

    return parser.parse_args()


def load_task_data(dataset, training, noise, task, level, prob=None):
    # Hardcoded absolute path based on your exact folder structure
    base_dir = r"C:\Users\test\Documents\GitHub\FOTraj\FOTraj\data_provider\datasets"
    
    if training:
        if noise:
            train_data = load_data(f'{base_dir}\\{dataset}\\processed_{dataset}_noise_train.csv')
        else:
            train_data = load_data(f'{base_dir}\\{dataset}\\processed_{dataset}_train.csv')
    else:
        train_data = None
        
    task_path = f'{base_dir}\\{dataset}\\{task}\\{task}_{level}'
    if prob:
        task_path += f'_prob_{prob}'
        
    test_data = load_data(f'{task_path}\\{dataset}_test_{task}.csv')
    test_label_id = load_data(f'{task_path}\\{dataset}_test_{task}_idx.csv')
    return train_data, test_data, test_label_id


def main(args, current_time):
    # --- HARDCODE OVERRIDES TO FIX ARGPARSE BUG ---
    args.noise = False
    args.is_training = True 
    
    data_load_start_time = time.time()

    if args.task == 'detour':
        train_data, test_data, test_label_id = load_task_data(args.dataset, args.is_training, args.noise, args.task, args.detour_level, args.point_prob)
    elif args.task == 'switch':
        train_data, test_data, test_label_id = load_task_data(args.dataset, args.is_training, args.noise, args.task, args.switch_level)
    elif args.task == 'time':
        train_data, test_data, test_label_id = load_task_data(args.dataset, args.is_training, args.noise, args.task, args.time_level)
    elif args.task == 'loop':
        train_data, test_data, test_label_id = load_task_data(args.dataset, args.is_training, args.noise, args.task, args.loop_level)
    else:
        raise ValueError("Invalid task")
        
    test_labels = np.zeros(len(test_data))
    test_labels[test_label_id] = 1
    
    # --- THE FIXED DATALOADERS ---
    if args.is_training:
        train_loader = DataLoader(dataset=GraphDataset(train_data), batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True) 
    else:
        train_loader = None
    test_loader = DataLoader(dataset=GraphDataset(test_data), batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    data_load_end_time = time.time()
    logger.info("Data loading time: {}".format(data_load_end_time - data_load_start_time))

    # --- THE MISSING INITIALIZATION ---
    if args.is_training:
        trainer = Trainer(train_loader,  test_loader, test_labels, args)
    else:
        trainer = Trainer(None, test_loader, test_labels, args)

    # --- RUN THE TRAINING & TESTING (Lowercase 'trainer') ---
    if args.is_training:
        trainer.train(current_time)
        
    trainer.test()

if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(args, current_time)
    logger.info(args)
    main(args, current_time)
    logger.info(f"All Time: {time.time() - start_time}")
