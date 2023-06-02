import json
import os
import statistics

import torch
import yaml
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from datasets import CaptionDataset
from models_i2u import ImageToUnit


def train(device, loader, model, reconstruction_loss, optimizer):
    accuracies = []
    losses = []

    for imgs, units, seq_lens, padding_masks in loader:
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

    for imgs, units, seq_lens, padding_masks in loader:
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


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_folder = os.path.join(os.path.dirname(__file__), "../..", config["I2U"]["data_folder"])
    word_map_path = os.path.join(os.path.dirname(__file__), "../..", config["I2U"]["word_map"])
    model_path = os.path.join(os.path.dirname(__file__), "../..", config["I2U"]["model_path"])

    train_loader = DataLoader(
        CaptionDataset(data_folder, config["I2U"]["data_name"], "TRAIN"),
        config["I2U"]["batch_size"],
        shuffle=True,
        num_workers=config["I2U"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        CaptionDataset(data_folder, config["I2U"]["data_name"], "VAL"),
        config["I2U"]["batch_size"],
        num_workers=config["I2U"]["num_workers"],
        pin_memory=True,
    )

    with open(word_map_path) as j:
        word_map = json.load(j)

    model = ImageToUnit(word_map).to(device)
    reconstruction_loss = nn.CrossEntropyLoss(ignore_index=word_map["<pad>"], reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["I2U"]["lr"])

    max_accuracy = 0
    for epoch in range(1, 1 + config["I2U"]["epoch"]):
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
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    with open("../../conf/spolacq3.yaml") as y:
        config = yaml.safe_load(y)

    main(config)
