#!/usr/bin/env python

import asyncio
import json
import websockets

import warnings
warnings.filterwarnings("ignore")

import importlib
import cv2
import glob
import os
import utils
import yaml
import numpy as np
import torch
import torch.nn as nn
from modules.sync_batchnorm import convert_model
from collections import OrderedDict

## definicoes do modelo
# classe
def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

# pre-processamento
def resize_img(img_path, dsize):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img
        
def resize_dataset(video_path, new_video_path):
    img_list = glob.glob(f"{video_path}*.png")
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=(256,256))
        rs_img_path = new_video_path + img_path.rsplit('/', 1)[-1]
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)

# processor
class Processor():

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.video_path = None
        self.dataset = None
        self.data_loader = None
        self.gloss_dict = np.load('./preprocess/phoenix2014/gloss_dict.npy', allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model = None
        
    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)
             
    # carrega o modelo e os dados
    def load_model(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        self.load_model_weights(model, self.arg.load_weights)
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.model = model
        
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
    
    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        self.dataset = self.feeder(video_path = self.video_path, gloss_dict=self.gloss_dict)
        self.data_loader = self.build_dataloader(self.dataset)
        print("Loading data finished.")
        
    def build_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size = self.arg.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )
    
    def add_video_path(self, path):
        self.video_path = path

## websockets
vid2glo_is_running = False
accepted_json = json.dumps( { "event": "message-accepted" } )
rejected_json = json.dumps( { "event": "message-rejected" } )

# chama a inferência
async def callVideo2Gloss(websocket, ws_message):
    global vid2glo_is_running

    vid2glo_is_running = True

    # pre-processamento
    message_in = json.loads(ws_message)
    video_path = message_in["input"]
    new_video_path = video_path + 'preprocessed/'
    resize_dataset(video_path, new_video_path)
    processor.add_video_path(new_video_path)

    # carrega os dados
    processor.load_data()

    # inferencia
    if processor.arg.load_weights is None and processor.arg.load_checkpoints is None:
        raise ValueError('Please appoint --load-weights.')
    processor.model.eval()
    next_vid = next(iter(processor.data_loader))
    vid = processor.device.data_to_device(next_vid[0])
    vid_lgt = processor.device.data_to_device(next_vid[1])
    ret_dict = processor.model(vid, vid_lgt)

    vid2glo_is_running = False

    # gera o string
    output = []
    for gloss in ret_dict['recognized_sents'][0]:
        output.append(gloss[0])
        gloss_string = ' '.join(output)

    # escreve no arquivo de output
    text_file = open(message_in['output'], 'w')
    n = text_file.write(gloss_string)
    text_file.close()

    # cria o json
    message_out = {
        "input": message_in['input'],
        "output": message_in['output']
    }

    # envia a resposta
    try:
        await websocket.send(json.dumps(message_out))
    except websockets.exceptions.ConnectionClosed:
        print("Connection with client closed.")

# carrega o modelo
async def loadModelOnVideo2Gloss():
    global vid2glo_is_running, processor

    # pega os argumentos (arquivo configs)
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

    # carrega o modelo
    vid2glo_is_running = True

    processor = Processor(args)
    processor.load_model()

    vid2glo_is_running = False

# handler
async def handler(websocket):
    global vid2glo_is_running, accepted_json, rejected_json

    async for ws_msg in websocket:
        if not vid2glo_is_running:
            asyncio.create_task(callVideo2Gloss(websocket, ws_msg))
            await websocket.send(accepted_json)
        else:
            await websocket.send(rejected_json)

async def main():
    async with websockets.serve(handler, "", 8001):
        await asyncio.Future() # run forever

if __name__ == "__main__":
    asyncio.run(loadModelOnVideo2Gloss()) # carrega o modelo apenas quando inicia o server
    asyncio.run(main())