import IPython.display
from pathlib import Path

import os
import numpy as np

# try:
#     import tensorflow  # required in Colab to avoid protobuf compatibility issues
# except ImportError:
#     pass

import torch
from torch import nn
import pandas as pd
import whisper
import torchaudio

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from tqdm.notebook import tqdm
import pyopenjtalk
import evaluate

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from Dataset import *

TRAIN_PATH = "../data/aishell/train.csv"
DEV_PATH = "../data/aishell/dev.csv"
SAMPLE_RATE = 16000
# 這兩個參數目前好像沒用到
BATCH_SIZE = 2
TRAIN_RATE = 0.8
# ---------------------
language = "zh"
AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120

# SEED = 3407
# seed_everything(SEED, workers=True)

# 測試用
# woptions = whisper.DecodingOptions(language=language, without_timestamps=True)
# wmodel = whisper.load_model("base")
# 生成whisper訓練的token
# wtokenizer = whisper.tokenizer.get_tokenizer(True, language=language, task=woptions.task)
# dataset = Dataset(TRAIN_PATH, wtokenizer, SAMPLE_RATE)
# loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=WhisperDataCollatorWhithPadding())

class Config:
    learning_rate = 0.0005
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 16
    num_worker = 20
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    sample_rate = SAMPLE_RATE


class WhisperModelModule(LightningModule):
    def __init__(self, cfg: Config, model_name="base", lang=language, train_dataset=TRAIN_PATH,
                 eval_dataset=DEV_PATH) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=language, task=self.options.task)

        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("train/lr", lr, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        """创建优化器和调度器"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.cfg.learning_rate,
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        # return optimizer

    def setup(self, stage=None):
        """初始设置（加载数据集）"""
        # 這邊在計算總步數，需要在scheduler用到
        if stage == 'fit' or stage is None:
            self.t_total = (
                    (len(Dataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)) // (self.cfg.batch_size))
                    // self.cfg.gradient_accumulation_steps
                    * float(self.cfg.num_train_epochs)
            )

    def train_dataloader(self):
        """创建训练数据加载器"""
        dataset = Dataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.cfg.batch_size,
                                           drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                                           collate_fn=WhisperDataCollatorWhithPadding()
                                           )

    def val_dataloader(self):
        """创建验证数据加载器"""
        dataset = Dataset(self.__eval_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.cfg.batch_size,
                                           num_workers=self.cfg.num_worker,
                                           collate_fn=WhisperDataCollatorWhithPadding()
                                           )


if __name__ == '__main__':
    log_output_dir = "log"
    check_output_dir = "exp"

    train_name = "whisper"
    train_id = "00001"

    model_name = "base"
    lang = "zh"

    cfg = Config()
    Path(log_output_dir).mkdir(exist_ok=True)
    Path(check_output_dir).mkdir(exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1  # all model save
    )
    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = WhisperModelModule(cfg, model_name, lang, TRAIN_PATH, DEV_PATH)

    trainer = Trainer(
        precision=16,  # fp16模式進行訓練
        gpus=-1,  # 多顯卡訓練
        # accelerator=DEVICE,
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list
    )
    trainer.fit(model)
