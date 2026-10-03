import copy
import csv
import glob
import os
import warnings

import cv2
import ignite.distributed as idist
import numpy as np
import pandas as pd
import torch
import math

import torch.nn.functional as F
from absl import app
from ignite.utils import convert_tensor
from ml_collections import config_flags
from tqdm import tqdm
from transformers import AutoTokenizer

from augmentation.get_aug import get_aug
from metrics.bleu_score import BLEUScore
from metrics.wer import WER
from models.get_models import get_model

warnings.filterwarnings("ignore", message="`load_model` does not return")
warnings.filterwarnings(
    "ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="`resume_download` is deprecated")

# Set to True to calculate the entropy for each inference.
CALCULATE_ENTROPY = False


class Processor:
    def __init__(self, cfg):
        self.cfg = cfg

        self.support_bfloat = torch.cuda.is_bf16_supported()

        print("Support bfloat16:", self.support_bfloat)
        self.dtype = torch.bfloat16 if self.support_bfloat else torch.float16

        if cfg.mixup:
            self.max_length = 128
        else:
            self.max_length = 64

        if "max_length" in cfg:
            self.max_length = cfg.max_length

        self.model, self.tokenizer = self.load_model()

        self.__video_path = None
        self.transform = get_aug(
            cfg.aug_name,
            cfg.aug_params,
        )

        if 'gt_path' in cfg:
            if cfg.gt_path.endswith(".csv"):
                self.gt = pd.read_csv(
                    cfg.gt_path,
                    sep=";",
                    encoding="utf-8",
                )
            elif cfg.gt_path.endswith(".tsv"):
                self.gt = pd.read_csv(
                    cfg.gt_path,
                    sep="\t",
                    encoding="utf-8",
                )
        else:
            self.gt = None

    @property
    def video_path(self):
        """
        Get the video path.
        """
        return self.__video_path

    @video_path.setter
    def video_path(self, video_path):
        """
        Set the video frames for processing.
        """
        if not (video_path.endswith(".mp4") or video_path.endswith(".avi")):
            raise ValueError(
                "Invalid video format. Only .mp4 and .avi are supported. Currently: {}".format(
                    video_path
                )
            )
        self.__video_path = video_path
        name_parts = video_path.split("/")[-1].split('.')
        name = '.'.join(
            name_parts[:-1]) if len(name_parts) > 1 else name_parts[0]

        try:
            self.__sentence = self.gt[self.gt["name"]
                                      == name]["sentence"].values[0]
        except Exception:
            sentence_id = name.split("_s")[-1]
            try:
                self.__sentence = self.gt[self.gt["sentence_id"] ==
                                          sentence_id]["sentence"].values[0]
            except Exception:
                self.__sentence = ' '

    def load_model(self):
        """
        Load the trained model and tokenizer.
        """
        cfg = self.cfg

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.lm_name, **cfg.additional_tokens)

        if "pretext" in cfg and len(cfg.pretext) > 0:
            pretext = cfg.pretext
            pretext_tokens = tokenizer(pretext)["input_ids"]
            pretext_length = len(pretext_tokens)

            if cfg.pretext[-1] == " ":
                pretext_tokens = pretext_tokens[:-1]
                pretext_length = pretext_length - 1
        else:
            pretext = ""
            pretext_tokens = tokenizer(pretext)["input_ids"]
            pretext_length = 1

        self.pretext = pretext
        self.pretext_length = pretext_length
        self.pretext_tokens = pretext_tokens

        dict_model_params = cfg.model_params.to_dict()
        dict_model_params["pretext_length"] = pretext_length

        model = get_model(cfg.model_name, dict_model_params)
        model.load_state_dict(
            torch.load(cfg.checkpoint_path, map_location="cuda")["model"]
        )
        model = model.to(device="cuda", dtype=self.dtype)
        model.eval()

        dict_text = tokenizer(
            [
                (self.pretext + tokenizer.eos_token)
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        self.text_ids = dict_text["input_ids"][:, :-1]
        self.text_mask = dict_text["attention_mask"][:, :-1].bool()
        return model, tokenizer

    def preprocess_video(self, frame_size=(224, 224)):
        """
        Preprocess the video into frames suitable for the model.
        """
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames = np.array(frames)

        frames = self.transform.aug_video(frames, isValid=True)
        return frames

    def prep_batch(self, batch, isValid=False, cuda=True):
        idx, frames, sentence = (
            batch["index"],
            batch["frames"],
            batch["sentence"],
        )

        frame_features = frames

        frame_mask = torch.zeros(
            len(frame_features), max([len(feat) for feat in frame_features])
        )
        for i, fr in enumerate(frame_features):
            frame_mask[i, : len(fr)] = 1.0
        frame_mask = frame_mask.bool()

        dict_text = self.tokenizer(
            [
                (self.pretext + sent + self.tokenizer.eos_token)
                # (self.tokenizer.bos_token + sent + self.tokenizer.eos_token)
                for sent in sentence
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        assert torch.all(
            dict_text["input_ids"][:, : self.pretext_length]
            == torch.tensor(self.pretext_tokens).unsqueeze(0)
        )

        text_ids = dict_text["input_ids"][:, :-1]

        text_attention_mask = copy.deepcopy(dict_text["attention_mask"])
        gt_ids = dict_text["input_ids"][:, self.pretext_length:]
        gt_text_mask = text_attention_mask[:, self.pretext_length:].bool()

        res = {
            "model_input": {
                "text_mask": dict_text["attention_mask"][:, :-1].bool(),
                "text_ids": text_ids,
                "frame_features": [f.to(self.dtype) for f in frame_features],
                "frame_mask": frame_mask,
                "max_len": torch.tensor(
                    self.cfg.max_seq_len if "max_seq_len" in self.cfg else 512
                ),
            },
            "targets": {
                "gloss_ids": batch["gloss_ids"] if "gloss_ids" in batch else [],
                "pseudo_gloss_ids": batch["pseudo_gloss_ids"]
                if "pseudo_gloss_ids" in batch
                else [],
                "index": torch.stack(idx),
                "sentence": sentence,
                "gt_ids": gt_ids.to(self.dtype),
                "gt_text_mask": gt_text_mask,
            },
        }

        return (
            convert_tensor(res, device=idist.device(), non_blocking=True)
            if cuda
            else res
        )

    def run_inference(self):
        """
        Run inference on the video frames.
        """
        video_frames = self.preprocess_video(self.video_path).to(self.dtype)
        if video_frames is None or len(video_frames) == 0:
            raise ValueError("No frames extracted from the video.")

        frame_mask = torch.ones(1, len(video_frames)).bool()

        model_input = {
            "text_mask": self.text_mask,
            "text_ids": self.text_ids,
            "frame_features": (video_frames, ),
            "frame_mask": frame_mask,
            "max_len": torch.tensor(
                self.cfg.max_seq_len if "max_seq_len" in self.cfg else 512
            ),
        }
        model_input = convert_tensor(
            model_input, device=idist.device(), non_blocking=True)

        with torch.inference_mode(True):
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                generated = self.model(
                    **model_input, generate=True, get_scores=CALCULATE_ENTROPY)

        output_ids = generated["output_ids"]

        pred = output_ids.detach().cpu()[0]
        seq_idx = 0

        indices_eos = torch.where(pred == self.tokenizer.eos_token_id)[0]
        if len(indices_eos) > 0:
            # Include EOS token itself for logprob calculation
            pred = pred[:indices_eos[0]+1]
            effective_length = indices_eos[0] + 1
        else:
            effective_length = pred.shape[0]

        predicted = self.tokenizer.decode(
            pred,
            skip_special_tokens=True,
        ).strip()

        if CALCULATE_ENTROPY:
            scores = generated['scores']

            entropies = []
            margins = []

            for t in range(effective_length - 1):  # Ignore the last token (EOS)
                step_logits = scores[t][seq_idx]  # shape: [vocab_size]
                step_probs = F.softmax(step_logits, dim=-1)

                entropy = -torch.sum(step_probs * step_probs.log())
                entropies.append(entropy.item())

                top2 = torch.topk(step_logits, 2)
                margin = top2.values[0] - top2.values[1]
                margins.append(margin.item())

            # Calculate average entropy and margin
            avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
            avg_margin = sum(margins) / len(margins) if margins else 0.0

            print(
                f"Avg Entropy: {avg_entropy:.4f}, Avg Margin: {avg_margin:.4f}")

        return predicted, self.__sentence.strip()


def main(_):
    print(f"Loaded configuration: {CONFIG} \n\n")
    videos_path = CONFIG.value.videos_path

    processor = Processor(CONFIG.value)

    inference_dir = CONFIG.value.checkpoint_path.split(
        "/")[-1].split(".pt")[0]
    os.makedirs(
        f"./inferences/{inference_dir}",
        exist_ok=True,
    )

    videos = glob.glob(videos_path)
    videos = [v for v in videos if "_signal" not in v]

    infereces = []
    with open(f'./inferences/{inference_dir}/log.log', 'w') as f:
        f.write(f"Checkpoint: {CONFIG.value.checkpoint_path}\n")
        f.write(f"Videos: {videos_path}\n")
        # Preprocess video
        for video_path in tqdm(videos, desc="Processing videos"):
            processor.video_path = video_path
            name = video_path.split('/')[-1].split(videos_path[-4:])[0]
            output, expected = processor.run_inference()

            print(f"Video: {name}")
            print(f"Output: {output}")
            print(f"Expected: {expected}")

            f.write(f"{'=' * 50}\n")
            f.write(f"Video: {name}\n")
            f.write(f"{'-' * 50}\n")
            f.write(f"Output: {output}\n")
            f.write(f"{'-' * 50}\n")
            f.write(f"Expected: {expected}\n")
            f.write(f"{'=' * 50}\n")
            infereces.append(
                {
                    "video": name,
                    "output": output,
                    "expected": expected,
                }
            )

    with open(f'./inferences/{inference_dir}/log.csv', 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=["video", "output", "expected"])
        writer.writeheader()
        for inference in infereces:
            writer.writerow(inference)

    bleu = BLEUScore()
    bleu.update(
        (
            [inference["output"] for inference in infereces],
            [inference["expected"] for inference in infereces],
        )
    )
    bleu_scores = bleu.compute()

    wer = WER()
    wer.update(
        (
            [inference["output"] for inference in infereces],
            [inference["expected"] for inference in infereces],
        )
    )
    wer_results = wer.compute()

    # Save BLEU scores and inferences to a .md report file
    report_file_path = f'./inferences/{inference_dir}/report.md'
    with open(report_file_path, 'w', encoding='utf-8') as report_file:
        # Write BLEU scores
        report_file.write("# Inference Report\n\n")
        report_file.write(f"## Checkpoint: {CONFIG.value.checkpoint_path}\n")
        report_file.write(f"## Videos: {videos_path}\n")
        report_file.write("\n## BLEU Scores\n")
        for i, score in enumerate(bleu_scores.values()):
            report_file.write(f"- *BLEU-{i + 1}*: {score:.2f}\n")

        report_file.write(
            f"\n\n## WER Primary: {wer_results['wer_primary']:.2f}%\n")
        report_file.write(
            f"## Mean WER: {wer_results['wer_mean']:.2f}%\n")
        report_file.write(
            f"## Std WER: {wer_results['wer_std']:.2f}%\n\n")

        # Write inferences table
        report_file.write("## Inferences\n\n")
        report_file.write("| Video | Output | Expected | WER |\n")
        report_file.write("|-------|--------|----------|-----|\n")
        for i, inference in enumerate(infereces):
            report_file.write(
                f"| {inference['video']} | {inference['output']} | {inference['expected']} | {wer_results['wer_list'][i]:.2f}%\n"
            )

    print(f"Report saved to {report_file_path}")


if __name__ == "__main__":
    CONFIG = config_flags.DEFINE_config_file(
        "config",
        default="",
    )

    app.run(main)
