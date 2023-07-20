#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from tqdm import tqdm
import logging


# this is configuration for different dataset
class DatasetConfig(object):
    def __init__(self, set_name):
        assert set_name in ['MIMIC3'], "[x] Not compatible to this dataset: %s"%set_name
        self.name = set_name

        if os.path.exists("/home/qingpeng/dataset/"):
            root_path = "/home/qingpeng/dataset/"
        self.check_file_existence(root_path)

        self.set_types = ['train', 'valid', 'test']
        self.debug_size = 240
        self.dataset_modes = ['regular']
        self.dataset_path = root_path
        self.selected_index = None
        self.labels = None

        if set_name == "MIMIC3":
            self.dataset_path = os.path.join(root_path, set_name, "data_v4/")
            self.check_file_existence(self.dataset_path)
            self.labels = ['inhos_mortality']
            self.selected_index = np.array([0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20, 22, 27, 29, 30, 31,
                                            32, 33, 35, 36, 40, 41, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60,
                                            61, 62, 64, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 80, 82, 83, 84, 86,
                                            87, 88, 90, 91, 95, 96, 100, 102, 103])


    def check_file_existence(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("The file cannot be found under this path: %s"%path)

class MIMIC3SetLoader(object):
    def __init__(self, args, set_type, config, fold_id=0, max_len=-1):
        self.data_path = config.dataset_path

        # self.args = args
        self.type = set_type.lower()
        self.label = args.application.lower()
        self.mode = args.dataset_mode
        assert self.label in config.labels, "[x]No application for MIMIC3: %s" % self.label
        assert self.type in config.set_types, "[x]No such type for dataset: %s" % self.type
        assert self.mode in config.dataset_modes, "[x]No such mode for dataset: %s" % self.mode

        self.selected_index = config.selected_index
        set_idx = config.set_types.index(self.type)

        # sample clean mode
        self.ffill = args.ffill
        self.ffill_steps = args.ffill_steps
        self.standardization = args.standardization
        self.data_clip = args.data_clip
        self.data_clip_min = args.data_clip_min
        self.data_clip_max = args.data_clip_max

        self.fold = self.process_fold(fold_id, set_idx, max_len)
        if args.debug:
            self.fold = self.fold[:config.debug_size]
            # logging.info("[*] %s is running in the debug mode." % name)
        self.get_dataset(self.fold)

        # print(self.data['data'].shape)
        self.sample_size, self.time_dim, self.input_dim = self.data['data'].shape

    def __getitem__(self, index):
        if self.mode == 'regular':
            return self.get_regular_item(index)

    def __len__(self):
        return len(self.fold)

    def _ffill_by_steps(self, input_x, input_m, ffill_steps=48):
        (tsize, fsize) = input_x.shape
        for f in range(fsize):
            last_ob = -1
            t_count = 0
            for t in range(tsize):
                if input_m[t][f] != 0:
                    last_ob = input_x[t][f]
                    t_count = 0
                elif last_ob != -1 and input_m[t][f] == 0 and t_count < ffill_steps:
                    input_x[t, f] = last_ob
                    input_m[t, f] = 1
                    t_count += 1

        return input_x, input_m

    def get_dataset(self, indices):
        self.data = {
            "time": [],
            "data": [],
            "mask": [],
            "length": [],
            "labels": []
        }
        for index in tqdm(indices):
            sample_index = index[:-4]
            sample = self.__getitem__(sample_index)

            for key in sample.keys():
                if key in self.data.keys():
                    self.data[key].append(sample[key])

        for key in self.data.keys():
            self.data[key] = np.array(self.data[key])

    def get_generator(self, batch_size, shuffle, return_whole=True):
        fold_len = self.__len__()
        fold = np.array(range(fold_len))
        dataset = self.data

        def _generator():
            batch_id = 0
            if shuffle:
                np.random.shuffle(fold)
            batch_from = 0
            while batch_from < fold_len:
                batch_fold = fold[batch_from:batch_from + batch_size]
                input_info = None
                input_x = dataset['data'][batch_fold]
                input_m = dataset['mask'][batch_fold]
                if self.mode == 'regular':
                    input_t = dataset['time'][batch_fold]
                    inputs = [input_info, input_x, input_m, input_t]
                elif self.mode == 'irregular':
                    input_t = dataset['time'][batch_fold]
                    input_l = dataset['length'][batch_fold]
                    inputs = [input_info, input_x, input_m, input_t, input_l]
                else:
                    raise NotImplementedError("[!]No such dataset mode: %s" % self.mode)
                input_y = dataset['labels'][batch_fold][:, np.newaxis]
                yield (batch_id, inputs, input_y)
                batch_from += batch_size
                batch_id += 1

        def _inputs_generator():
            for batch_id, inputs, _ in _generator():
                yield (batch_id, inputs)

        if not return_whole:
            return _inputs_generator()
        else:
            return _generator()

    def get_whole_set(self, return_label=True):
        inputs = [self.data['demo'], self.data['data'], self.data['mask'], self.data['time'], self.data['diag']]
        if return_label:
            return inputs, self.data['labels']
        else:
            return inputs

    def get_regular_item(self, index):
        data = np.load(os.path.join(self.data_path, "%s.npz" % index), allow_pickle=True)
        result = {}

        regular_data = data['regular_data'][()]
        result['data'] = regular_data['tdata']
        result['mask'] = regular_data['tmask']
        result['time'] = regular_data['stime']
        if self.selected_index is not None:
            result['data'] = result['data'][:, self.selected_index]
            result['mask'] = result['mask'][:, self.selected_index]

        if self.standardization:
            result['data'] = (result['data'] - self.t_norm['avg']) / self.t_norm['std']

            if self.data_clip:
                result['data'] = np.clip(result['data'], a_min=self.t_norm['min'], a_max=self.t_norm['max'])

        if self.ffill:
            result['data'], result['mask'] = self._ffill_by_steps(result['data'], result['mask'], self.ffill_steps)
        result['data'] = result['data'] * result['mask']

        result['labels'] = self.get_label(data)
        return result

    def get_label(self, data):
        label_data = data['labels']
        if self.label == "inhos_mortality":
            return label_data[1]
        else:
            raise NotImplementedError()

    def process_fold(self, fold_id, set_type, max_len):
        folds_info = np.load(os.path.join(self.data_path, "%s_folds.npz" % self.label), allow_pickle=True)
        fold = folds_info["fold_tvt"][fold_id][set_type]
        if max_len != -1:
            fold = fold[:max_len]
        logging.info("[*] %s: %s" % (self.type, fold[:5]))

        self.t_norm = self.process_norm(folds_info, fold_id)
        self.demo_norm = folds_info["demo_norm"][()][fold_id]

        self.output_dim = int(folds_info['out_dim'])
        return fold

    def process_norm(self, folds_info, fold_id):
        t_norm = folds_info["regular_norm"][()][fold_id]
        norm_name = ['avg', 'std', 'min', 'max']
        for name in norm_name:
            if self.selected_index is not None:
                t_norm[name] = np.array(t_norm[name])[self.selected_index]
            else:
                t_norm[name] = np.array(t_norm[name])
        return t_norm
