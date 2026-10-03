import glob
import cv2
import yaml
import torch
import importlib
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import OrderedDict
import utils
from modules.sync_batchnorm import convert_model
import csv
import shutil
import time
import faulthandler
faulthandler.enable()
import os  # noqa: E402
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import warnings  # noqa: E402
warnings.filterwarnings("ignore")


def import_class(name):
    components = name.rsplit(".", 1)
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
        rs_img = resize_img(img_path, dsize=(256, 256))
        rs_img = rs_img
        # rs_img = cv2.normalize(rs_img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F) # normalize to [0,1]
        rs_img_path = new_video_path + img_path.rsplit("/", 1)[-1]
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)


def extract_annotation(dataset_root, classes=["dev", "test", "train"]):
    annotations = {}
    for cls in classes:
        try:
            # Carrega o arquivo CSV correspondente e armazena no cache
            df = pd.read_csv(f"{dataset_root}/annotations/{cls}.csv", sep="|")
            annotations[cls] = df.set_index("id")["annotation"].to_dict()
        except Exception as e:
            print(f"Erro ao obter anotações para a classe {cls}: {e}")
            continue  # Passa para a próxima classe se houver erro
    all_annotations = {}
    for cls_annotations in annotations.values():
        all_annotations.update(cls_annotations)

    return all_annotations


class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(
            self.arg.work_dir, self.arg.print_log, self.arg.log_interval
        )
        self.dataset = {}
        self.data_loader = {}
        self.ctrl = args.dataset_info.get("use_ctrl", False)
        self.gloss_dict = np.load(
            self.arg.dataset_info["dict_path"], allow_pickle=True
        ).item()
        self.arg.model_args["num_classes"] = len(self.gloss_dict) + 1
        self.model, self.optimizer = self.loading()

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open("{}/config.yaml".format(self.arg.work_dir), "w") as f:
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
                    print("Successfully Remove Weights: {}.".format(w))
                else:
                    print("Can Not Remove Weights: {}.".format(w))
        weights = self.modified_weights(state_dict["model_state_dict"], False)
        model.load_state_dict(weights, strict=True)

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model.conv2d = nn.DataParallel(
                model.conv2d,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device,
            )
        model = convert_model(model)
        model.cuda()
        return model

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict(
            [(k.replace(".module", ""), v) for k, v in state_dict.items()]
        )
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict["rng_state"]["cuda"]):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict["rng_state"])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(
                state_dict["scheduler_state_dict"])

        self.arg.optimizer_args["start_epoch"] = state_dict["epoch"] + 1
        self.recoder.print_log(
            "Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}"
        )

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        mode_list = ["dev", "test"]
        train_flag_list = [False, False]

        if self.ctrl:
            mode_list.append("ctrl")
            train_flag_list.append(False)

        dataset_list = zip(mode_list, train_flag_list)

        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info["dataset_root"]
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(
                self.dataset[mode], mode, train_flag
            )
        print("Loading data finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size
            if mode == "train"
            else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )


def process_args():
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", "r") as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    return args


if __name__ == "__main__":
    args = process_args()

    classes = ["dev", "test"]
    if args.dataset_info.get("use_ctrl", False):
        classes.append("ctrl")

    annotations = extract_annotation(
        args.dataset_info["dataset_root"], classes)

    processor = Processor(args)
    utils.pack_code("./", args.work_dir)

    # inference
    if processor.arg.load_weights is None and processor.arg.load_checkpoints is None:
        raise ValueError("Please appoint --load-weights.")

    processor.model.eval()

    os.makedirs(args.work_dir, exist_ok=True)
    csv_file = f"{args.work_dir}/inference_data.csv"
    if os.path.exists(csv_file):
        os.rename(csv_file, f"{args.work_dir}/inference_data_old.csv")

    for cls in reversed(classes):
        print(f"cls = {cls}")
        output_file = f"{args.work_dir}/log_{cls}.txt"
        renamed_file = f"{args.work_dir}/log_old.txt"

        # Verifica se o arquivo original existe
        if os.path.exists(output_file):
            # Renomeia o arquivo
            shutil.move(output_file, renamed_file)
            print(f"Arquivo renomeado para {renamed_file}")
        else:
            print(f"O arquivo {output_file} não existe")

        # Cria um novo arquivo log.txt
        with open(output_file, "w") as f:
            f.write("")  # Escreve um arquivo vazio

        # Print GPU memory usage
        gpu_mem = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU memory allocated for weights: {gpu_mem:.2f} GB")

        for data in processor.data_loader[cls]:
            video_id = data[-1][0].split("|")[0]

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            vid = processor.device.data_to_device(data[0])
            vid_lgt = processor.device.data_to_device(data[1])

            frames_gb = round(vid.element_size() * vid.numel() / (1024**3), 2)
            num_frames = vid_lgt[0].item()

            gpu_mem = round(torch.cuda.memory_allocated() / (1024**3), 2)

            start_time = time.time()
            with torch.no_grad():
                ret_dict = processor.model(vid, vid_lgt)
            end_time = time.time()

            inference_time_ms = (end_time - start_time) * 1000
            inference_time_per_frame_ms = inference_time_ms / num_frames

            # Collect data to input in a CSV
            csv_data = {
                "video_id": video_id,
                "num_frames": num_frames,
                "frames_size(GB)": frames_gb,
                "gpu_memory_allocated": gpu_mem,
                "inference_time(ms)": inference_time_ms,
                "inference_time_per_frame(ms)": inference_time_per_frame_ms,
            }

            file_exists = os.path.isfile(csv_file)
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=csv_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_data)

            output = []
            print(video_id)

            for gloss in ret_dict["recognized_sents"][0]:
                output.append(gloss[0])

            conv_out = []
            for sample in ret_dict["conv_sents"]:
                for word_idx, word in enumerate(sample):
                    aux = "1 {:.2f} {:.2f} {}\n".format(
                        word_idx * 1.0 /
                        100, (word_idx + 1) * 1.0 / 100, word[0]
                    ).replace("\n", "")

                    conv_out.append(aux)

            processor.recoder.print_log(
                "Model:   {}.".format(processor.arg.model), output_file
            )
            processor.recoder.print_log(
                "Weights: {}.".format(processor.arg.load_weights), output_file
            )
            processor.recoder.print_log(f"Video: {video_id}", output_file)
            processor.recoder.print_log(
                f"Original Sents: {annotations.get(video_id, 'NOT FOUND')}", output_file
            )
            # processor.recoder.print_log(f"Conv Sents: {' - '.join(conv_out)} ", output_file)
            processor.recoder.print_log(
                f"Recognized Sents: {' '.join(output)} \n", output_file
            )

            with open("./output.txt", "w") as file:
                file.write(" ".join(output))
