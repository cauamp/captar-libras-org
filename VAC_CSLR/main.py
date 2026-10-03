import yaml
import torch
import importlib
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import wandb
import utils
from modules.sync_batchnorm import convert_model
from seq_scripts import seq_train, seq_eval, seq_feature_generation
import datetime
import os
import faulthandler
import pandas as pd
import glob
from wer.calculate_wer import wer
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
faulthandler.enable()


class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info["dict_path"], allow_pickle=True).item()
        self.arg.model_args["num_classes"] = len(self.gloss_dict) + 1
        self.model, self.optimizer = self.loading()
        self.dataset_name = self.arg.dataset

        self.wandb_run = None
        if self.arg.wandb_args["active"] and self.arg.phase == "train":
            if self.arg.wandb_args["wandb_key"] is not None:
                print()
                wandb.login(key=open(self.arg.wandb_args["wandb_key"], "r").read().strip())
                wandb.init(
                    project="VAC_CSLR-2026",
                    config=vars(self.arg),
                    name=f"{self.arg.work_dir.split('/')[-2]}-{str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).strftime('%d%m%Y-%H%M'))}",
                    entity="VeRLab-captar-sign2text",
                )
                self.wandb_run = wandb.run

    def start(self):
        if self.arg.phase == "train":
            self.min_extra_wer = 999
            self.recoder.print_log("Parameters:\n{}\n".format(str(vars(self.arg))))
            seq_model_list = []
            for epoch in range(self.arg.optimizer_args["start_epoch"], self.arg.num_epoch):
                if self.wandb_run:
                    if hasattr(self.model.conv2d, "module"):
                        l4c1_weights = self.model.conv2d.module.layer4[1].conv1.weight.data.cpu().numpy()
                        l1c1_weights = self.model.conv2d.module.layer1[0].conv1.weight.data.cpu().numpy()
                    else:
                        l4c1_weights = self.model.conv2d.layer4[1].conv1.weight.data.cpu().numpy()
                        l1c1_weights = self.model.conv2d.layer1[0].conv1.weight.data.cpu().numpy()
                    wandb.log(
                        {"conv2d-l4c1_weights": l4c1_weights},
                        step=epoch,
                    )
                    wandb.log(
                        {"conv2d-l1c1_weights": l1c1_weights},
                        step=epoch,
                    )

                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                # train end2end model
                loss_value = np.mean(
                    seq_train(
                        self.data_loader["train"],
                        self.model,
                        self.optimizer,
                        self.device,
                        epoch,
                        self.recoder,
                        self.arg.grad_accumulation_steps,
                    )
                )
                if self.wandb_run:
                    wandb.log({"Loss": loss_value, "Epoch": epoch}, step=epoch)

                if eval_model:
                    dev_wer = seq_eval(
                        self.arg,
                        self.data_loader["dev"],
                        self.model,
                        self.device,
                        "dev",
                        epoch,
                        self.arg.work_dir,
                        self.recoder,
                        self.arg.evaluate_tool,
                        dataset=self.dataset_name,
                    )

                    self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                    if self.wandb_run:
                        wandb.log({"Dev WER": dev_wer, "Epoch": epoch}, step=epoch)

                    if self.arg.extra_eval:
                        if not hasattr(self, "extra_eval_data"):
                            self.extra_eval_data = []
                        extra_wer = self.seq_extra_eval(
                            epoch,
                            self.arg.work_dir,
                        )

                        if extra_wer < self.min_extra_wer:
                            self.min_extra_wer = extra_wer
                            save_model = True

                if save_model:
                    if self.arg.extra_eval:
                        model_path = "{}extra_{:.2f}_dev_{:05.2f}_e{}.pt".format(
                            self.arg.work_dir, extra_wer, dev_wer, epoch
                        )
                    else:
                        model_path = "{}dev_{:05.2f}_e{}.pt".format(self.arg.work_dir, dev_wer, epoch)
                    seq_model_list.append(model_path)
                    print("seq_model_list", seq_model_list)
                    self.save_model(epoch, model_path)

        elif self.arg.phase == "test":
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                raise ValueError("Please appoint --load-weights.")
            self.recoder.print_log("Model:   {}.".format(self.arg.model))
            self.recoder.print_log("Weights: {}.".format(self.arg.load_weights))
            train_wer = seq_eval(
                self.arg,
                self.data_loader["train_eval"],
                self.model,
                self.device,
                "train",
                6667,
                self.arg.work_dir,
                self.recoder,
                self.arg.evaluate_tool,
                dataset=self.dataset_name,
            )
            dev_wer = seq_eval(
                self.arg,
                self.data_loader["dev"],
                self.model,
                self.device,
                "dev",
                6667,
                self.arg.work_dir,
                self.recoder,
                self.arg.evaluate_tool,
                dataset=self.dataset_name,
            )
            test_wer = seq_eval(
                self.arg,
                self.data_loader["test"],
                self.model,
                self.device,
                "test",
                6667,
                self.arg.work_dir,
                self.recoder,
                self.arg.evaluate_tool,
                dataset=self.dataset_name,
            )
            self.recoder.print_log("Evaluation Done.\n")
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model,
                    self.device,
                    mode,
                    self.arg.work_dir,
                    self.recoder,
                )

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open("{}/config.yaml".format(self.arg.work_dir), "w") as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.optimizer.scheduler.state_dict(),
                "rng_state": self.rng.save_rng_state(),
            },
            save_path,
        )

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

        if self.arg.extra_eval:
            self.build_extra_eval()

        return model, optimizer

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

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print("Successfully Remove Weights: {}.".format(w))
                else:
                    print("Can Not Remove Weights: {}.".format(w))

        weights = self.modified_weights(state_dict["model_state_dict"], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        if len(self.arg.ignore_model_weights):
            for w in self.arg.ignore_model_weights:
                if weights.pop(w, None) is not None:
                    print("Successfully Remove Weights: {}.".format(w))
                    weights[w] = model.state_dict()[w]
                else:
                    print("Can Not Remove Weights: {}.".format(w))
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace(".module", ""), v) for k, v in state_dict.items()])
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
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args["start_epoch"] = state_dict["epoch"] + 1
        self.recoder.print_log(f"Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info["dataset_root"]
            print(f"prefix: {arg['prefix']}")
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )

    def build_extra_eval(self):
        self.extra_eval_feeder = import_class(self.arg.extra_eval["feeder"])
        self.extra_dataset = self.extra_eval_feeder(
            video_path=None,
            gloss_dict=self.gloss_dict,
            dataset=self.arg.dataset,
            **self.arg.extra_eval["feeder_args"],
        )

        self.extra_eval_paths = []
        for path in self.arg.extra_eval["video_path"]:
            self.extra_eval_paths += glob.glob(path)

        self.extra_eval_gt = {}
        for path in self.arg.extra_eval["gt_path"]:
            self.extra_eval_gt[path] = pd.read_csv(path)
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                self.extra_eval_gt[row["video"]] = row["gloss"]

    def update_extra_eval_data_loader_path(self, video_path):
        self.extra_dataset.video_path = video_path

        self.extra_eval_data_loader = torch.utils.data.DataLoader(
            self.extra_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=self.extra_eval_feeder.collate_fn,
            pin_memory=True,  # Improve data transfer speed to GPU
            prefetch_factor=2,  # Adjust to balance performance
            persistent_workers=True,  # Keep workers alive to avoid reloading
        )

    def seq_extra_eval(self, epoch, work_dir):
        self.model.eval()

        wers = []
        e_sum = 0
        t_sum = 0

        self.recoder.print_log(f"{'-' * 10} Epoch {epoch} {'-' * 10}", f"{work_dir}/extra_log.txt")

        infer_data = self.extra_eval_paths if not self.extra_eval_data else self.extra_eval_data

        for data in tqdm(infer_data, desc="Extra Evaluation"):
            if isinstance(data, str):
                self.update_extra_eval_data_loader_path(data)
                next_item = next(iter(self.extra_eval_data_loader))
                next_item = (
                    next_item[0],
                    next_item[1],
                    os.path.splitext(os.path.basename(data))[0],
                )
                self.extra_eval_data.append(next_item)
            else:
                next_item = data

            vid = self.device.data_to_device(next_item[0])
            vid_lgt = self.device.data_to_device(next_item[1])
            with torch.no_grad():
                ret_dict = self.model(vid, vid_lgt)

            out = []
            for gloss in ret_dict["recognized_sents"][0]:
                out.append(gloss[0])
            out = " ".join(out)
            _id = next_item[-1]
            expected = self.extra_eval_gt[_id]

            wer_value, e, t = wer(expected, out)
            wers.append(wer_value)
            self.recoder.print_log(f"{expected},{out},{wer_value: 2.2f}%", f"{work_dir}/extra_log.txt")
            e_sum += e
            t_sum += t

        wer_primary = e_sum / t_sum * 100
        wer_mean = np.mean(wers)
        self.recoder.print_log(
            f"EXTRA mean: {wer_mean: 2.2f}% primary: {wer_primary: 2.2f}%",
            f"{work_dir}/extra_log.txt",
        )

        self.recoder.print_log(
            f"Epoch {epoch}, EXTRA mean: {wer_mean: 2.2f}% primary: {wer_primary: 2.2f}%",
            f"{work_dir}/extra.txt",
        )

        if self.wandb_run:
            wandb.log({"Extra WER": wer_mean, "Epoch": epoch}, step=epoch)

        return wer_mean


def import_class(name):
    components = name.rsplit(".", 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == "__main__":
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)
    sparser = utils.get_parser()
    p = sparser.parse_args()
    # p.config = "baseline_iter.yaml"
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
    processor = Processor(args)
    utils.pack_code("./", args.work_dir)
    processor.start()
