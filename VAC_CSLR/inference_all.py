import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import warnings
warnings.filterwarnings("ignore")

import pdb
import sys
import glob
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import OrderedDict

faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model
import csv

def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


# preprocessing
def resize_img(img_path, dsize):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img
        
        
def resize_dataset(video_path, new_video_path):
    img_list = glob.glob(f"{video_path}*.png")
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=(256,256))
        rs_img = rs_img 
        #rs_img = cv2.normalize(rs_img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F) # normalize to [0,1]
        rs_img_path = new_video_path + img_path.rsplit('/', 1)[-1]
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)

class Processor():

    def __init__(self, arg, video_path):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.video_path = video_path
        self.dataset = None
        self.data_loader = None
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model, self.optimizer = self.loading()
        self.extract_annotation()

        
        
    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)
           
                
    # loading model and data
    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer
        
        
    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        model.load_state_dict(weights, strict=True)
      
        
    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model.conv2d = nn.DataParallel(
                model.conv2d,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device)
        model = convert_model(model)
        model.cuda()
        return model
    
    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict
    
    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")
    
        
    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        self.dataset = self.feeder(video_path = self.video_path, gloss_dict=self.gloss_dict, dataset=self.arg.dataset, **self.arg.feeder_args)
        self.data_loader = self.build_dataloader(self.dataset)
        print("Loading data finished.")
        
    def update_video_path(self, video_path):
        self.video_path = video_path
        self.load_data()
        self.extract_annotation()
        
    def build_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size = self.arg.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )
    
    def extract_annotation(self):
        classes = ['dev', 'test', 'train']
        if self.arg.dataset in self.video_path.split('/'):
            aux = self.video_path.split('/')
            #video_id = aux[6]
            video_id = aux[-1]
            if not video_id:
                video_id = aux[-2]
            print(f'{video_id =}')
            for cls  in classes:
                try:
                    df = pd.read_csv(f'./datasets_srv/{self.arg.dataset}/annotations/{cls}.csv', sep='|')
                    record = df.loc[df['id'] == video_id]
                    
                    if not record.empty:
                        self.annotation = record.iloc[0]['annotation']
                        return
                except Exception:
                    print("Erro ao obter anotações")
        self.annotation = None
        return 
    
    def print_log_annotation(self, output_path='./output.txt'):
        self.recoder.print_log(f"Original Sents: {self.annotation}", output_path)
          
    
if __name__ == '__main__':
    
    video_paths = glob.glob('./datasets_srv/testes_cam_config0/*.mkv')
    
    video_paths += glob.glob('./datasets_srv/testes_cam_caua/*.mp4')
    gt = './gt.csv'
        
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    # preprocessing
    processor = Processor(args, '')
    utils.pack_code("./", args.work_dir)
    
 
    # inference
    if processor.arg.load_weights is None and processor.arg.load_checkpoints is None:
        raise ValueError('Please appoint --load-weights.')
    
    output_path = './output-{}.txt'.format('.'.join(processor.arg.load_weights.split('/')[-1].split('.')[:-1]))
    if gt:
        annotations = pd.read_csv(gt)
        
    print(video_paths)
    print(output_path)
    for video_path in video_paths:
        video = video_path.split('/')[-1].replace('.mp4', '').replace('.mkv', '')
        print(f"Processing {video}")
        
        processor.update_video_path(video_path)
        if gt:
            processor.annotation = annotations.loc[annotations['video'] == video]['gloss'].values[0]
        processor.model.eval()
        
        next_item = next(iter(processor.data_loader))
        vid = processor.device.data_to_device(next_item[0])
        vid_lgt = processor.device.data_to_device(next_item[1])
        
        with torch.no_grad():
            ret_dict = processor.model(vid, vid_lgt)
        
        # final output
        output = []
        for gloss in ret_dict['recognized_sents'][0]:
            output.append(gloss[0])

        
        processor.recoder.print_log('Model:   {}.'.format(processor.arg.model), output_path)
        processor.recoder.print_log('Weights: {}.'.format(processor.arg.load_weights), output_path)
        processor.recoder.print_log(f'Video: {video_path}', output_path)
        processor.print_log_annotation(output_path)
        processor.recoder.print_log(f"Recognized Sents: {' '.join(output)} \n", output_path)
        
            