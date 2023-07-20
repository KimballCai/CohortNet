import os
import time
import logging
import time
import warnings
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from util import *

warnings.filterwarnings("ignore")

# define env setting
sets = ['train', 'valid', 'test']

from argument import get_parser


def set_random_seed(seed=2000):
    logging.info("[*]random seed: %d" % seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn

# this is for the supervised training
class ModelTrainer():
    def __init__(self, args):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        args = calculate_warm(args)
        self.args = args
        # print(self.args)

        # the gpu setting
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu)

        self.set_config()

        # get the file path for the logging
        log_root = "Please update the url for log file"
        if os.path.exists(log_root):
            self.log_root = log_root
        else:
            raise FileNotFoundError("[x] Log path does not exist.")
        self.log_path = self.set_log_path(timestamp)

        # the logger setting
        handlers = [logging.FileHandler(self.log_path + 'log_{}.txt'.format(timestamp), mode='w'),
                    logging.StreamHandler()]
        logging.basicConfig(level=logging.INFO, datefmt='%m-%d-%y %H:%M', format='%(asctime)s:%(message)s',
                            handlers=handlers)
        logging.info("================== Start %s =================="%timestamp)

        # set the random seed
        if not self.args.random:
            random_seed = 2000
        else:
            random_seed = np.random.randint(10e6)
        set_random_seed(random_seed)

        logging.info('Timestamp: {}'.format(timestamp))
        logging.info('Arguments')
        for k, v in sorted(vars(self.args).items()):
            logging.info("%s = %s" % (k, str(v)))

        self.learning()

    def set_config(self):
        # for binary-label classification
        self.metrics = ['bceloss', 'auroc', 'auprc', 'accu', 'f1', 'minpse']
        self.train_mode = "SP"

    def learning(self):
        for i_fold in self.args.folds:
            logging.info('============= {}-th fold ============='.format(i_fold))
            self.dataset = {}
            for name in sets:
                self.dataset[name] = self.set_dataset(name, i_fold)
            self.input_dim = self.dataset['train'].input_dim
            self.output_dim = self.dataset['train'].output_dim
            self.set_model(self.args)
            self.recorders = Recorders(sets, self.metrics, self.args.patience)

            if self.args.mode == "train":
                self.criterion = self.set_criterion(type=self.args.criterion)
                self.writer = SummaryWriter(self.log_path)
                for epoch in range(1, self.args.epochs+1):
                    adjust_learning_rate(self.args, self.optimizer, epoch)

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

                self.eval_model(os.path.join(self.log_path, 'ckpt_{i}.pth'.format(i=i_fold)))

            elif self.args.mode == "eval":
                assert self.args.model_path != "#", "[x] Please provide a valid model path!"
                if not os.path.exists(args.model_path):
                    logging.info("[x] Model path is invalid: %s" % self.args.model_path)
                self.eval_model(args.model_path)

    def train(self, epoch):
        self.model.train()
        losses = []
        batch_time = []
        total_steps = (self.dataset['train'].sample_size-1)//self.args.batch_size+1
        for batch_id, batch_x, batch_y in self.dataset['train'].get_generator(self.args.batch_size, shuffle=True):
            start = time.time()
            warmup_learning_rate(self.args, epoch, batch_id+1, total_steps, self.optimizer)

            label = torch.tensor(batch_y).float()
            tdata = torch.tensor(batch_x[1]).float()
            tmask = torch.tensor(batch_x[2]).float()
            if self.args.dataset_mode == "regular":
                stime = torch.tensor(batch_x[3]).float()
                if torch.cuda.is_available():
                    stime = stime.cuda()
            if torch.cuda.is_available():
                tdata = tdata.cuda()
                tmask = tmask.cuda()
                label = label.cuda()

            if self.args.model in ["CohortNet"]:
                out = self.model([tdata, tmask])
                # prediction
                prediction = out[0]
                # representation
                rep = out[1]
            loss = self.criterion(prediction, label)
            losses.append(loss.item())

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            end = time.time()
            batch_time.append(end - start)

            if batch_id % self.args.print_freq == 0:
                logging.info('Train: [{0}][{1}/{2}]\t'
                             'BT avg {batch_time:.3f}\t'
                             'loss avg {loss:.4f}'.format(
                    epoch, batch_id, total_steps, batch_time=np.average(batch_time), loss=np.average(losses))
                )
        return np.average(losses)

    def validate(self, epoch=-1):
        self.model.eval()
        results = {}
        with torch.no_grad():
            for set in self.recorders.sets:
                sub_pred, sub_label = [], []
                for batch_id, batch_x, batch_y in self.dataset[set].get_generator(self.args.batch_size, shuffle=False):
                    tdata = torch.tensor(batch_x[1]).float()
                    tmask = torch.tensor(batch_x[2]).float()
                    if torch.cuda.is_available():
                        tdata = tdata.cuda()
                        tmask = tmask.cuda()

                    if self.args.model in ["CohortNet"]:
                        out = self.model([tdata, tmask])
                        # prediction
                        prediction = out[0]
                        # representation
                        rep = out[1]

                    pred = prediction
                    if torch.cuda.is_available():
                        pred = pred.cpu()

                    sub_pred.extend(list(pred.detach().numpy()))
                    sub_label.extend(batch_y)
                results[set] = binary_eval(y_true=sub_label, y_pred=sub_pred)
        save_flag = self.recorders.insert(results)
        return save_flag, self.recorders.get_epoch_result()

    def set_dataset(self, name, i_fold):
        assert name in ['train', 'valid', 'test'], "[x] No such dataset mode: %s" % name
        from dataset import DatasetConfig
        config = DatasetConfig(set_name=self.args.dataset)
        if self.args.dataset == "MIMIC3":
            logging.info("[*]loading the MIMIC3 dataset: %s."%name)
            from dataset import MIMIC3SetLoader
            return MIMIC3SetLoader(self.args, name, config, i_fold)
        else:
            raise NotImplementedError("[x] cannot support the dataset: %s"%self.args.dataset)

    def set_criterion(self, type="bce"):
        assert type in ['bce'], "[x] No such criterion: %s" % type
        if type == "bce":
            criterion = torch.nn.BCELoss()

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1 and len(self.args.gpu) > 1:
                criterion = criterion.cuda()
        return criterion

    def set_model(self, args):
        if self.args.dataset_mode == "regular":
            if self.args.model == "CohortNet":
                from model import CohortNet
                model = CohortNet(o_dim=self.output_dim, f_num=self.input_dim, e_dim=args.embed_dim, c_dim=args.compress_dim,
                                  h_dim=args.hidden_dim, fusion_dim=args.fusion_dim, cluster_num=args.k,
                                  clip_min=args.clip_min, clip_max=args.clip_max, active=args.active)
        else:
            raise NotImplementedError("No such mode: %s" % (args.dataset_mode))

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1 and len(args.gpu) > 1:
                model = torch.nn.DataParallel(model)

            model = model.cuda()
            cudnn.benchmark = True

        assert self.args.opt in ['adam', 'sgd'], "[x] No such optimizer: %s" % self.args.opt
        if self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=self.args.lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
            logging.info("[*] optimizer: SGD, lr: %f, momentum: %f, weight decay: %f" % (args.lr,
                                                                                         self.args.momentum,
                                                                                         self.args.weight_decay))
        elif self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            logging.info("[*] optimizer: Adam, lr: %f, weight decay: %f" % (args.lr, self.args.weight_decay))
        else:
            raise NotImplementedError()

        logging.info(model)
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info("Model param num: %d  trainable: %d" % (total_num, trainable_num))
        self.model = model
        self.optimizer = optimizer

        # load pretrained parameters
        if self.args.model_path != "#":
            assert os.path.exists(args.model_path), "[x] Model path is invalid: %s" % self.args.model_path
            self.load_model(args.model_path)

    def set_log_path(self, timestamp):
        log_path = self.log_root
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        if self.args.debug:
            log_path += "/%s_%s_debug/" % (self.args.model, timestamp)
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            return log_path

        log_path += "/%s/" % self.args.dataset
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        log_path += "/%s/" % self.args.application
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        log_path += "/%s_%s/" % (self.args.model, self.train_mode)
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        log_path += "/%s_%s/" % (self.args.model, timestamp)
        if self.args.mode == "eval":
            log_path = log_path[:-1] + "_eval"
        if self.args.random:
            log_path = log_path[:-1] + "_random"
        log_path = log_path + "/"

        if not os.path.exists(log_path):
            os.mkdir(log_path)
        return log_path

    def save_results(self, epoch, i_fold, results):
        logging.info("[*] Saving files...")
        state = {
            'args': self.args,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        save_file = os.path.join(self.log_path, 'ckpt_{i}.pth'.format(i=i_fold))
        torch.save(state, save_file)
        np.save(os.path.join(self.log_path, "best_valid_{i}".format(i=i_fold)), results)
        self.recorders.save(os.path.join(self.log_path, "recorders_{i}.npz".format(i=i_fold)))
        self.recorders.record_to_csv(self.log_path)

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        state_dict = ckpt['model']
        model_dict = self.model.state_dict()
        # print(state_dict, model_dict)
        new_state_dict = {}
        skipcount, loadedcount = 0, 0
        for k, v in state_dict.items():
            # print(k,v.shape, v[0])
            k2 = k.replace("module.", "")
            if k in model_dict.keys():
                new_state_dict[k] = v
                loadedcount += 1
            elif k2 in model_dict.keys():
                new_state_dict[k2] = v
                loadedcount += 1
            else:
                logging.info("skiped: %s"%k)
                skipcount += 1
        model_dict.update(new_state_dict)
        self.model.load_state_dict(model_dict)
        logging.info("[*] Model loaded!")
        logging.info("[*] skipped: %d  loaded: %d" % (skipcount, loadedcount))

    def eval_model(self, model_path):
        logging.info("=============== eval ===============")
        self.load_model(model_path)
        save_flag, results = self.validate()
        logging.info(self.recorders.to_string())

if __name__ == '__main__':
    args = get_parser()
    if args.debug:
        args.batch_size = 64
    trainer = ModelTrainer(args)
