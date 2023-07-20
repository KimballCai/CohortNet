import os
import time
from datetime import datetime
import numpy as np
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from util import *

import warnings

warnings.filterwarnings("ignore")

# define env setting
sets = ['train', 'valid', 'test']

from argument import get_parser
from main import ModelTrainer

from model import cluster


# this trainer is for the cohort modeling
class CohortModelTrainer(ModelTrainer):
    def __init__(self, args):
        super(CohortModelTrainer, self).__init__(args)

    def set_config(self):
        # loss metrics
        self.metrics = ['bceloss', 'auroc', 'auprc', 'accu', 'f1', 'minpse']
        self.train_mode = "CO"

    def learning(self):
        for i_fold in self.args.folds:
            logging.info('============= {}-th fold ============='.format(i_fold))
            self.trainer_mode = "train"
            self.dataset = {}
            for name in sets:
                self.dataset[name] = self.set_dataset(name, i_fold)
            self.input_dim = self.dataset['train'].input_dim
            self.output_dim = self.dataset['train'].output_dim
            self.set_model(self.args)
            self.recorders = Recorders(sets[1:], self.metrics, self.args.patience)

            if self.args.fix:
                self.args.cohort_iter = -1
                self.fix_param()

            if self.args.mode == "train":
                self.criterion = self.set_criterion(type=self.args.criterion)
                self.writer = SummaryWriter(self.log_path)

                cohort_epoch = self.args.cohort_epoch
                for epoch in range(1, self.args.epochs + 1):
                    adjust_learning_rate(self.args, self.optimizer, epoch)

                    if epoch > cohort_epoch:
                        with torch.no_grad():
                            self.cal_clusters(self.args, self.dataset['train'])

                            if self.args.cohort_iter == -1:
                                cohort_epoch = self.args.epochs + 1
                            else:
                                cohort_epoch += self.args.cohort_iter
                        self.save_results(0, i_fold, None)

                    time1 = time.time()
                    loss = self.train(epoch)
                    time2 = time.time()
                    save_flag, results = self.validate(epoch)
                    time3 = time.time()
                    logging.info('Epoch {}, lr {:.6f}, train loss {:.4f}, '
                                 'train time {:.2f}s, valid time {:.2f}s, total time {:.2f}s.'.format(
                        epoch,
                        self.optimizer.param_groups[0]['lr'],
                        loss,
                        time2 - time1,
                        time3 - time2,
                        time3 - time1
                    ))
                    logging.info(self.recorders.to_string())
                    # save results
                    for subset in range(len(self.recorders.sets)):
                        for m in range(len(self.recorders.metrics)):
                            self.writer.add_scalar(tag="%s/%s/%s" % (self.train_mode,
                                                                     self.recorders.sets[subset],
                                                                     self.recorders.metrics[m]),
                                                   scalar_value=results[subset, m], global_step=epoch)
                    if save_flag == 1:
                        self.save_results(epoch, i_fold, results)
                    elif save_flag == 0:
                        logging.info("[*] Overfitting... Stop!")
                        break
                self.trainer_mode = "eval"
                self.eval_model(os.path.join(self.log_path, 'ckpt_{i}.pth'.format(i=i_fold)))

            elif self.args.mode == "eval":
                self.trainer_mode = "eval"
                assert args.model_path != "#", "[x] Please provide a valid model path!"
                if not os.path.exists(args.model_path):
                    logging.info("[x] Model path is invalid: %s" % self.args.model_path)
                self.eval_model(args.model_path)

    def save_results(self, epoch, i_fold, results):
        logging.info("[*] Saving files...")
        state = {
            'args': self.args,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        if type(self.model) == torch.nn.DataParallel:
            state['cohorts'] = self.model.module.cohorts
        else:
            state['cohorts'] = self.model.cohorts

        save_file = os.path.join(self.log_path, 'ckpt_{i}.pth'.format(i=i_fold))
        torch.save(state, save_file)
        if epoch != 0:
            np.save(os.path.join(self.log_path, "best_valid_{i}".format(i=i_fold)), results)
            self.recorders.save(os.path.join(self.log_path, "recorders_{i}.npz".format(i=i_fold)))
            self.recorders.record_to_csv(self.log_path)

    def fix_param(self):
        patterns = ['cohort']
        logging.info("[*] non-fix pattern: " + str(patterns))
        fix_w_cnt = 0
        fix_param_cnt = 0
        for name, value in self.model.named_parameters():
            value.requires_grad = False
            fix_w_cnt += 1
            fix_param_cnt += value.numel()
            for p in patterns:
                low_name = name.lower()
                if p in low_name:
                    value.requires_grad = True
                    fix_w_cnt -= 1
                    fix_param_cnt -= value.numel()
                    continue
        if self.args.debug:
            for name, value in self.model.named_parameters():
                logging.info("%s %s"%(name, value.requires_grad))
        logging.info("[*] No.weight is fixed: %d" % fix_w_cnt)
        logging.info("[*] No.params is fixed: %d" % fix_param_cnt)
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info("[*] No.params is trainable: %d" % trainable_num)

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        state_dict = ckpt['model']
        model_dict = self.model.state_dict()
        new_state_dict = {}
        skipcount, loadedcount = 0, 0
        for k, v in state_dict.items():
            k2 = k.replace("module.", "")
            if ("cohort" in k.lower()) and self.trainer_mode == "train":
                if self.args.debug:
                    logging.info("%s is skipped"%(k))
                skipcount += 1
                continue
            if k in model_dict.keys():
                new_state_dict[k] = v
                loadedcount += 1
            elif k2 in model_dict.keys():
                new_state_dict[k2] = v
                loadedcount += 1
            else:
                if self.args.debug:
                    logging.info("%s is skipped" % (k))
                skipcount += 1
        model_dict.update(new_state_dict)
        self.model.load_state_dict(model_dict)

        if "cohorts" in ckpt.keys():
            logging.info("[*] Cohorts loaded.")
            if type(self.model) == torch.nn.DataParallel:
                self.model.module.cohorts = ckpt['cohorts']
            else:
                self.model.cohorts = ckpt['cohorts']

        logging.info("[*] Model loaded!")
        logging.info("[*] skipped: %d  loaded: %d" % (skipcount, loadedcount))

    def cal_clusters(self, args, dataset):
        logging.info("[*] Start to calculate clusters. (totally GPU)")
        start = time.time()
        self.model.eval()
        attention = []

        batch_size = 200
        for batch_id, batch_x, batch_y in dataset.get_generator(batch_size, shuffle=False):
            info, tdata, tmask, input_t = batch_x
            tdata = torch.tensor(tdata).float()
            tmask = torch.tensor(tmask).float()
            if torch.cuda.is_available():
                tdata = tdata.cuda()
                tmask = tmask.cuda()
            results = self.model([tdata, tmask])

            if attention == []:
                attention = results[2].detach().cpu()  # attention for interaction
                t_rep = results[1].detach().cpu()  # B * T * F * H
                pt_mask = tmask.detach().cpu()
                label = batch_y
            else:
                attention = torch.cat((attention, results[2].detach().cpu()), 0)
                t_rep = torch.cat((t_rep, results[1].detach().cpu()), 0)
                pt_mask = torch.cat((pt_mask, tmask.detach().cpu()), 0)
                label = np.concatenate((label, batch_y), 0)

            # if batch_id == steps:
            #     break
        # cluster_in = torch.flatten(cluster_in, start_dim=2)
        logging.info("[*] Get all representations.")
        logging.info("[*] attention shape: %s" % str(attention.shape))
        logging.info("[*] t rep shape: %s" % str(t_rep.shape))

        f_num = dataset.input_dim
        time_dim = dataset.time_dim

        pt_a = torch.flatten(attention, start_dim=0, end_dim=1)
        pt_clusterin = torch.flatten(t_rep, start_dim=0, end_dim=1)  # B*T * F * Cp
        pt_mask = pt_mask.any(1).byte().unsqueeze(1).repeat(1, time_dim, 1)  # B * T * F
        pt_mask = torch.flatten(pt_mask, start_dim=0, end_dim=1)  # B*T * F
        pt_rep = torch.flatten(t_rep, start_dim=0, end_dim=1)  # B*T * F * H

        dup_label = np.tile(label[:, np.newaxis], (1, time_dim))
        dup_label = torch.Tensor(dup_label.reshape(-1)).cuda()  # B*T

        f_dim = t_rep.shape[-1]
        num_clusters = np.ones(dataset.input_dim-1, dtype=np.int) * args.k
        logging.info("[*] No. clusters: %s"%str(num_clusters))
        top_n_a = args.topn
        min_freq = args.min_freq
        min_sample_freq = args.min_sample_freq

        f_centers = []
        f_codes = None
        for fid in range(f_num):
            centers = None if self.model.module.cohorts == None else self.model.module.cohorts[fid]['centers'].cuda()
            mask = pt_mask[:, fid].cuda()
            sub_f_centers, sub_pos_f_codes = cluster(pt_clusterin[mask, fid, :].cuda(), num_clusters[fid], centers)
            f_centers.append(sub_f_centers.detach())
            sub_pos_f_codes = (sub_pos_f_codes + 2).detach()
            mask_f_codes = torch.ones(pt_clusterin.shape[0], dtype=torch.long).cuda()
            sub_f_codes = torch.zeros(pt_clusterin.shape[0], dtype=torch.long).cuda()
            sub_f_codes = sub_f_codes.scatter_add_(0, torch.where(mask)[0], sub_pos_f_codes)
            sub_f_codes = sub_f_codes.scatter_add_(0, torch.where(1 - mask)[0], mask_f_codes)
            if fid == 0:
                f_codes = sub_f_codes.unsqueeze(1)  # B*T * 1
            else:
                f_codes = torch.cat((f_codes, sub_f_codes.unsqueeze(1)), 1).detach()
                # logging.info(torch.unique(f_codes))
                # logging.info("[*] Get one feature state. %d", fid)

        logging.info("[*] Get all feature states.")

        # calculate the cohorts of features
        cohorts = {}
        cohort_size1, cohort_size2, cohort_size3, cohort_size4, cohort_size5 = [], [], [], [], []
        cohort_freq, cohort_sam_freq = [], []
        for fid in range(f_num):
            process_start = time.time()
            sub_pt = pt_a[:, fid, :].cuda()
            f_a_index = torch.argsort(sub_pt)[:, -1 * top_n_a:]
            f_a_top_mask = torch.eye(f_num)[f_a_index].sum(-2).cuda()
            f_a_top_mask[:, fid] = 1
            f_a_top_mask = f_a_top_mask.byte()
            f_top = f_a_top_mask * f_codes  # B*T * F

            logging.info("%d %s"%(fid,f_top.shape))
            if fid == 0: logging.info("remove the missing code for origin feature")
            # f_top_idx = torch.where(~(f_top == 1).any(1))[0]
            f_top_idx = torch.where(f_top[:,fid] != 1)[0]
            f_top = f_top[f_top_idx]
            logging.info("%d %s" % (fid, f_top.shape))

            pattern, pat_index, pat_cnt = torch.unique(f_top, dim=0, return_counts=True, return_inverse=True)

            # current version:  rep is based on pattern within codes
            count_index = torch.argsort(pat_cnt)
            pattern = pattern[count_index]
            pat_cnt = pat_cnt[count_index]
            # pat_index = pat_index[count_index]
            cohort_size1.append(len(pattern))

            match_id = torch.where(pat_cnt > min_freq)[0][0]
            # filter1: minimal frequency
            selected = match_id
            pattern = pattern[selected:]  # C * F
            cohort_size2.append(len(pattern))

            # time-consuming when calculate cohorts
            if fid == 0: logging.info("[*] use all patients with this pattern to learn cohorts")
            chunk_sample = 200
            chunk_size = chunk_sample*time_dim  # 9600
            num_records = int(f_codes.shape[0])
            pat_chunk_size = 600
            num_patterns = int(pattern.shape[0])
            pat_cnt = torch.zeros(num_patterns, dtype=torch.float).cuda()
            sample_pat_cnt = torch.zeros(num_patterns, dtype=torch.float).cuda()
            pat_rep = torch.zeros(num_patterns, f_dim, dtype=torch.float).cuda()
            pos_cnt = torch.zeros(num_patterns, dtype=torch.float).cuda()
            neg_cnt = torch.zeros(num_patterns, dtype=torch.float).cuda()
            pos_cnt_sample = torch.zeros(num_patterns, dtype=torch.float).cuda()
            neg_cnt_sample = torch.zeros(num_patterns, dtype=torch.float).cuda()
            pos_pat_rep = torch.zeros(num_patterns, f_dim, dtype=torch.float).cuda()
            neg_pat_rep = torch.zeros(num_patterns, f_dim, dtype=torch.float).cuda()
            for i in range(0, num_records, chunk_size):
                begin = i
                end = min(begin + chunk_size, num_records)
                sub_tlabel = dup_label[begin:end]  # M*T
                sub_f_codes = f_codes[begin:end, :]  # M * F
                f_rep = pt_rep[begin:end, fid, :].cuda()  # M * H
                for j in range(0, num_patterns, pat_chunk_size):
                    pat_begin = j
                    pat_end = min(pat_begin+pat_chunk_size, num_patterns)
                    sub_pattern = pattern[pat_begin:pat_end]
                    sub_pattern_mask = (sub_pattern != 0).byte()
                    pattern_code = sub_f_codes.unsqueeze(1).byte() * sub_pattern_mask  # M * C * F
                    pattern_match = (pattern_code == sub_pattern).all(2).byte()  # M * C
                    # logging.info(pattern_match.shape)
                    # logging.info("%d %d"%(pat_begin,pat_end))
                    subs_pat_match = pattern_match.reshape(int((end-begin)/48), time_dim, pat_end-pat_begin)  # S * T * C

                    # general pattern rep
                    sub_pat_rep = pattern_match.unsqueeze(-1) * f_rep.unsqueeze(1)  # M * C * H
                    pat_rep[pat_begin:pat_end] += sub_pat_rep.sum(0)  # C * H
                    pat_cnt[pat_begin:pat_end] += pattern_match.sum(0)  # C
                    sample_pat_cnt[pat_begin:pat_end] += subs_pat_match.any(1).sum(0)  # C

                    # pos and neg sample pattern rep
                    pos_pat_match = sub_tlabel.unsqueeze(1) * pattern_match  # M * C
                    neg_pat_match = (1-sub_tlabel.unsqueeze(1)) * pattern_match  # M * C
                    pos_pat_rep[pat_begin:pat_end] += (pos_pat_match.unsqueeze(2) * f_rep.unsqueeze(1)).sum(0)
                    neg_pat_rep[pat_begin:pat_end] += (neg_pat_match.unsqueeze(2) * f_rep.unsqueeze(1)).sum(0)
                    pos_cnt[pat_begin:pat_end] += pos_pat_match.sum(0)
                    neg_cnt[pat_begin:pat_end] += neg_pat_match.sum(0)
                    pos_pat_match_sample = pos_pat_match.reshape(-1, time_dim, pat_end - pat_begin).any(1)  # S * C
                    pos_cnt_sample[pat_begin:pat_end] += pos_pat_match_sample.sum(0)  # C
                    neg_pat_match_sample = neg_pat_match.reshape(-1, time_dim, pat_end - pat_begin).any(1)  # S * C
                    neg_cnt_sample[pat_begin:pat_end] += neg_pat_match_sample.sum(0)  # C

            pat_rep = pat_rep/pat_cnt.unsqueeze(1)  # C * H
            pos_pat_rep = pos_pat_rep/(pos_cnt+10e-6).unsqueeze(1)  # C * H
            neg_pat_rep = neg_pat_rep/(neg_cnt+10e-6).unsqueeze(1)  # C * H

            # filter2: pattern exists on more than x samples
            filter2_idx = torch.where(sample_pat_cnt >= min_sample_freq)[0]
            pattern = pattern[filter2_idx]
            pat_cnt = pat_cnt[filter2_idx]
            sample_pat_cnt = sample_pat_cnt[filter2_idx]
            pat_rep = pat_rep[filter2_idx]
            pos_cnt = pos_cnt[filter2_idx]
            pos_pat_rep = pos_pat_rep[filter2_idx]
            pos_cnt_sample = pos_cnt_sample[filter2_idx]
            neg_cnt = neg_cnt[filter2_idx]
            neg_pat_rep = neg_pat_rep[filter2_idx]
            neg_cnt_sample = neg_cnt_sample[filter2_idx]
            cohort_size3.append(len(filter2_idx))

            # sorted by No.samples
            sample_cnt_idx = torch.argsort(sample_pat_cnt)
            pattern = pattern[sample_cnt_idx]
            pat_cnt = pat_cnt[sample_cnt_idx]
            sample_pat_cnt = sample_pat_cnt[sample_cnt_idx]
            pat_rep = pat_rep[sample_cnt_idx]
            pos_cnt = pos_cnt[sample_cnt_idx]
            pos_pat_rep = pos_pat_rep[sample_cnt_idx]
            neg_cnt = neg_cnt[sample_cnt_idx]
            neg_pat_rep = neg_pat_rep[sample_cnt_idx]
            pos_cnt_sample = pos_cnt_sample[sample_cnt_idx]
            neg_cnt_sample = neg_cnt_sample[sample_cnt_idx]

            if len(pat_cnt) == 0:
                cohort_freq.append(0)
                cohort_sam_freq.append(0)
            else:
                cohort_freq.append(min(pat_cnt).cpu().numpy())
                cohort_sam_freq.append(min(sample_pat_cnt).cpu().numpy())

            process_end = time.time()
            logging.info("[*] Process Feature %d with %.3f seconds" % (fid, process_end-process_start))
            cohorts[fid] = {
                "pat": pattern.cpu().detach(),
                "pat_count": pat_cnt.cpu().detach(),
                "sample_pat_cnt": sample_pat_cnt.cpu().detach(),
                "pat_rep": pat_rep.cpu().detach(),
                "labeled": {
                    "pos_cnt": pos_cnt.cpu().detach(),
                    "pos_rep": pos_pat_rep.cpu().detach(),
                    "neg_cnt": neg_cnt.cpu().detach(),
                    "neg_rep": neg_pat_rep.cpu().detach(),
                    "pos_cnt_sample": pos_cnt_sample.cpu().detach(),
                    "neg_cnt_sample": neg_cnt_sample.cpu().detach(),
                },
                "centers": f_centers[fid].cpu().detach(),
            }

        end = time.time()
        logging.info("[*] No.cohort in total")
        logging.info(np.array(cohort_size1))
        logging.info("[*] Filter1: No.cohort filtered by org freq")
        logging.info(np.array(cohort_size2))
        logging.info("[*] Filter3: No.cohort filtered by sample freq")
        logging.info(np.array(cohort_size3))
        logging.info("[*] Filter4: No.cohort filtered by max size")
        logging.info(np.array(cohort_size5))
        logging.info("[*] minimal cohort freq")
        logging.info(np.array(cohort_freq))
        logging.info("[*] minimal cohort freq (sample-level)")
        logging.info(np.array(cohort_sam_freq))
        logging.info("[*] Build all feature clusters: total time {:.2f}s ".format(end - start))
        self.model.module.cohorts = cohorts


if __name__ == '__main__':
    args = get_parser()
    args.epochs = 80
    if args.debug:
        args.batch_size = 128
    trainer = CohortModelTrainer(args)