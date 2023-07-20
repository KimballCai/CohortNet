import numpy as np
from sklearn import metrics
import logging

EPS = 1e-10

# binary_eval can evaluate binary prediction results
def binary_eval(y_true, y_pred, verbose=False):
    y_pred = np.reshape(y_pred, (-1))
    y_pred = np.stack([1 - y_pred, y_pred]).transpose((1, 0))

    out = {}
    cf = metrics.confusion_matrix(y_true, y_pred.argmax(axis=1))
    if verbose:
        logging.info("confusion matrix:")
        logging.info(cf)
    cf = cf.astype(np.float32)

    out['accu'] = (cf[0][0] + cf[1][1]) / np.sum(cf)
    out['prec0'] = cf[0][0] / (cf[0][0] + cf[1][0] + EPS)
    out['prec1'] = cf[1][1] / (cf[1][1] + cf[0][1] + EPS)
    out['rec0'] = cf[0][0] / (cf[0][0] + cf[0][1] + EPS)
    out['rec1'] = cf[1][1] / (cf[1][1] + cf[1][0] + EPS)
    out['auroc'] = metrics.roc_auc_score(y_true, y_pred[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, y_pred[:, 1])
    out['auprc'] = metrics.auc(recalls, precisions)
    out['minpse'] = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    y_pred_label = (y_pred[:, 1] > 0.5)
    out['prec'] = metrics.precision_score(y_true, y_pred_label)
    out['recall'] = metrics.recall_score(y_true, y_pred_label)
    out['f1'] = metrics.f1_score(y_true, y_pred_label)

    bce = []
    for i in range(len(y_true)):
        bce.append(-y_true[i] * np.log(y_pred[i, 1] + EPS) - (1 - y_true[i]) * np.log(1 - y_pred[i, 1] + EPS))
    out['bceloss'] = np.mean(bce)

    if verbose:
        logging.info("accuracy = {}".format(out['accu']))
        logging.info("precision class 0 = {}".format(out['prec0']))
        logging.info("precision class 1 = {}".format(out['prec1']))
        logging.info("recall class 0 = {}".format(out['rec0']))
        logging.info("recall class 1 = {}".format(out['rec1']))
        logging.info("AUC of ROC = {}".format(out['auroc']))
        logging.info("AUC of PRC = {}".format(out['auprc']))
        logging.info("min(+P, Se) = {}".format(out['minpse']))
        logging.info("BCEloss = {}".format(out['bceloss']))

    return out

import math
import os

import numpy as np

metrics_better = {
    "lower": ['loss', 'bceloss', 'ham', 'klloss', 'supconloss'],
    "higher": ['auroc', 'auprc', 'abs_accu', 'accu', 'prec', 'recall', 'f1', 'minpse']
}

def calculate_warm(args):
    if args.batch_size >= 256:
        args.warm = True
    if args.warm:
        args.warmup_from = args.lr*0.05
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.warmup_to = (args.lr - eta_min) * (1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.lr
    return args

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        if args.warm:
            epoch = max(0, epoch - args.warm_epochs)
            total_epoch = max(0, args.epochs - args.warm_epochs)
        else:
            total_epoch = args.epochs
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / total_epoch)) / 2
    else:
        steps = np.sum(epoch > np.array([int(x) for x in args.lr_decay_epochs.split(',')]))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class Recorder(object):
    def __init__(self, name, patience=20):
        self.name = name
        self.epoch = 0
        self.values = []

        self.patience = patience
        self.patience_account = 0

        self.better = None
        self.best_epoch = -1
        self.best_value = 0

        self.reset(name)

    def reset(self, name):
        self.epoch = 0
        self.values = []
        self.patience_account = 0
        if name in metrics_better['lower']:
            self.better = min
            self.best_value = 100
        elif name in metrics_better['higher']:
            self.better = max
            self.best_value = -100
        else:
            raise NotImplementedError("[x] No such matrix recorder: %s" % name)
        self.best_epoch = -1

    def insert(self, value):
        self.epoch += 1
        self.values.append(value)
        if self.better(value, self.best_value) == value:
            self.best_value = value
            self.best_epoch = self.epoch
            self.patience_account = 0
        else:
            self.patience_account += 1
            if self.patience_account == self.patience:
                return 0
            return 2
        return 1


class Recorders(object):
    def __init__(self, sets, metrics, patience=20):
        self.sets = sets
        self.metrics = metrics
        self.epoch = 0
        self.recorders = {
            sub_set: {
                name: Recorder(name, patience) for name in metrics
            } for sub_set in sets
        }

    # def __init__(self, *args):
        # if len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], list):
        #     sets = args[0]
        #     metrics = args[1]
        #     self.sets = sets
        #     self.metrics = metrics
        #     self.epoch = 0
        #     self.recorders = {
        #         sub_set: {
        #             name: Recorder(name) for name in metrics
        #         } for sub_set in sets
        #     }
        # elif len(args) == 1 and isinstance(args[0], str):
        #     file_name = args[0]
        #     file = np.load(file_name)
        #     self.sets = file['sets']
        #     self.metrics = file['metrics']
        #     self.epoch = file['epoch']
        #     self.recorders = file['recorders']

    def reset(self):
        for k, sub_set in self.recorders.items():
            for _, sub_recorder in sub_set.items():
                sub_recorder.reset()

    def insert(self, results):
        self.epoch += 1
        save_flag = False
        for sub_set, sub_result in results.items():
            for m, value in sub_result.items():
                if m in self.metrics:
                    out = self.recorders[sub_set][m].insert(value)
                    if sub_set == "valid" and m == self.metrics[0]:
                        save_flag = out
        return save_flag

    def get_epoch_result(self, epoch=-1):
        if epoch == -1: epoch = self.epoch
        assert self.epoch > 0, "[x] No epoch recorded!"
        assert self.epoch <= epoch, "[x] target epoch {} is bigger than recorded epoch {}".format(epoch, self.epoch)

        out = []
        for k, sub_set in self.recorders.items():
            sub_out = []
            for m in self.metrics:
                assert len(sub_set[m].values) == epoch, "Metrics %s is not recorded!" % m
                sub_out.append(sub_set[m].values[epoch - 1])
            out.append(sub_out)
        return np.array(out)

    def to_string(self, epoch=-1):
        if epoch == -1: epoch = self.epoch
        assert self.epoch > 0, "[x] No epoch recorded!"

        data = self.get_epoch_result(epoch)
        caption = "\n%6s" % ""
        for m in self.metrics:
            caption += "%-9s" % m
        out = caption + "\n"
        content = ""
        for i, sub_set in enumerate(self.sets):
            content += "%-6s" % sub_set
            for j, m in enumerate(self.metrics):
                content += "%-8.4f " % data[i][j]
            content += "\n"
        out += content
        return out

    def get_best(self, name, sub_set='valid'):
        assert self.epoch > 0, "[x] No epoch recorded!"
        best_epoch = self.recorders[sub_set][name].best_epoch
        return self.get_epoch_result(best_epoch), self.to_string(best_epoch)

    def save(self, file_name):
        np.savez(file_name, sets=self.sets, metrics=self.metrics, epoch=self.epoch, recorders=self.recorders)

    def record_to_csv(self, out_path):
        import pandas as pd
        result = pd.DataFrame()
        for m in self.metrics:
            tem = pd.DataFrame(columns=['Epoch']+self.metrics,index=self.sets)
            best_epoch = self.recorders["valid"][m].best_epoch-1
            for set in self.sets:
                if set == "train":
                    tem.loc['train', 'Epoch'] = self.recorders["train"][m].epoch
                elif set == "valid":
                    tem.loc['valid', 'Epoch'] = best_epoch
                elif set == "test":
                    tem.loc['test', 'Epoch'] = self.recorders["test"][m].best_epoch

            for dataset in self.sets:
                for m in self.metrics:
                    tem.loc[dataset,m] = "%.4f" % self.recorders[dataset][m].values[best_epoch]
            result = pd.concat([result,tem],0)
        result.to_csv(os.path.join(out_path, 'result.csv'))



