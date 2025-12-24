import json
import os
import statistics
from pathlib import Path
from tqdm import tqdm

import torch
import yaml
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from .datasets import CaptionDataset


def train(device, loader, model, reconstruction_loss, optimizer):
    accuracies = []
    losses = []

    for imgs, units, seq_lens, padding_masks in tqdm(loader):
        imgs = imgs.to(device)
        units = units.to(device)
        seq_lens = seq_lens.to(device)
        padding_masks = padding_masks.to(device)

        logits, kl_loss = model(imgs, units, seq_lens, padding_masks)

        logits, _, _, _ = pack_padded_sequence(
            logits, (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )
        targets, _, _, _ = pack_padded_sequence(
            units[:, 1:], (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )

        loss = reconstruction_loss(logits, targets) + kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.sum(torch.argmax(logits, dim=1) == targets) / logits.size(0)
        accuracies.append(accuracy.item())
        losses.append(loss.item())

    return statistics.mean(accuracies), statistics.mean(losses)


@torch.inference_mode()
def validate(device, loader, model, reconstruction_loss):
    accuracies = []
    losses = []

    for imgs, units, seq_lens, padding_masks in tqdm(loader):
        imgs = imgs.to(device)
        units = units.to(device)
        seq_lens = seq_lens.to(device)
        padding_masks = padding_masks.to(device)

        logits, kl_loss = model(imgs, units, seq_lens, padding_masks)

        logits, _, _, _ = pack_padded_sequence(
            logits, (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )
        targets, _, _, _ = pack_padded_sequence(
            units[:, 1:], (seq_lens - 1).cpu().tolist(), batch_first=True, enforce_sorted=False
        )

        loss = reconstruction_loss(logits, targets) + kl_loss

        accuracy = torch.sum(torch.argmax(logits, dim=1) == targets) / logits.size(0)
        accuracies.append(accuracy.item())
        losses.append(loss.item())

    return statistics.mean(accuracies), statistics.mean(losses)


def train_i2u(
    model, i2u_pretrain_config, input_data_folder: Path, data_filename: str, output_model_path: Path, word_map
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data_folder = str(input_data_folder)
    output_model_path = str(output_model_path)

    train_loader = DataLoader(
        CaptionDataset(input_data_folder, data_filename, "TRAIN"),
        i2u_pretrain_config["batch_size"],
        shuffle=True,
        num_workers=i2u_pretrain_config["num_workers"],
        pin_memory=i2u_pretrain_config["pin_memory"],
    )

    val_loader = DataLoader(
        CaptionDataset(input_data_folder, data_filename, "VAL"),
        i2u_pretrain_config["batch_size"],
        num_workers=i2u_pretrain_config["num_workers"],
        pin_memory=i2u_pretrain_config["pin_memory"],
    )

    model = model.to(device)
    reconstruction_loss = nn.CrossEntropyLoss(ignore_index=word_map["<pad>"], reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=i2u_pretrain_config["lr"])

    max_accuracy = 0
    for epoch in range(1, 1 + i2u_pretrain_config["epoch"]):
        train_accuracy, train_loss = train(device, train_loader, model, reconstruction_loss, optimizer)
        val_accuracy, val_loss = validate(device, val_loader, model, reconstruction_loss)

        print(
            "epoch",
            epoch,
            "train_accuracy",
            train_accuracy,
            "train_loss",
            train_loss,
            "val_accuracy",
            val_accuracy,
            "val_loss",
            val_loss,
            flush=True,
        )

        if max_accuracy < val_accuracy:
            max_accuracy = val_accuracy
            torch.save(model.state_dict(), output_model_path)
