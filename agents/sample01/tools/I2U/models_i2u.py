# inference code is copied from
# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py

"""
MIT License

Copyright (c) 2018 Sagar Vinodababu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys

import torch
import torch.nn.functional as F
from torch import nn

from .module import PositionalEncoding
from ..dino.vision_transformer import VisionTransformer


class ImageToUnit(nn.Module):
    def __init__(
        self,
        # Vocab
        word_map: dict,
        max_len: int = 102,
        # VAE
        d_model: int = 768,
        d_embed: int = 8,
        std: float = 0.1,
        kl_weight: float = 0.01,
        # Transformer
        num_layers: int = 6,
        layer_norm_eps: float = 1e-5,
        nhead: int = 8,
        activation="gelu",
        norm_first: bool = True,
    ):
        super().__init__()

        self.word_map = word_map
        self.vocab_size = len(word_map)
        self.max_len = max_len
        self.d_embed = d_embed
        self.std = std
        self.kl_weight = kl_weight

        self.embed = nn.Embedding(self.vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        self.linear1 = nn.Linear(d_model, d_embed)
        self.linear2 = nn.Linear(d_embed, d_model)
        self.head = nn.Linear(d_model, self.vocab_size)

        self.image_encoder = VisionTransformer(patch_size=8, qkv_bias=True)
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
            map_location="cpu",
        )
        self.image_encoder.load_state_dict(state_dict)

    def forward(
        self,
        imgs: torch.Tensor,
        units: torch.Tensor,
        seq_len: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ):
        x = self.embed(units)
        x = x.permute(1, 0, 2)
        x = self.pos_enc(x)
        x = x.permute(1, 0, 2)

        z = self.encoder(x, src_key_padding_mask=padding_mask)

        # Global average pooling
        z = z * padding_mask.logical_not().unsqueeze(2)
        z = z.sum(dim=1) / seq_len.unsqueeze(1)

        # Reparameterization trick
        mean = self.linear1(z)
        std = torch.full_like(mean, self.std)
        eps = torch.randn_like(mean)
        z = mean + self.std * eps

        z = self.linear2(z)

        with torch.no_grad():
            self.image_encoder.eval()
            img_features = self.image_encoder(imgs)

        memory = torch.stack([img_features, z], dim=1)

        x = self.decoder(
            x,
            memory,
            tgt_mask=self.subsequent_mask(x.size(1)).to(x.device),
            tgt_key_padding_mask=padding_mask,
        )
        logits = self.head(x)
        return logits, self.kl_weight * self.kl_loss(mean, std)

    def subsequent_mask(self, size: int):
        return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)

    def kl_loss(self, mean: torch.Tensor, std: torch.Tensor):
        var = std**2
        return torch.sum(-1 / 2 * torch.sum(1 + var.log() - mean**2 - var, dim=1), dim=0)

    @torch.inference_mode()
    def infer(self, action: torch.Tensor, beam_size: int = 50):
        image_features = action[:, : -self.d_embed]
        sentence_embed = action[:, -self.d_embed :]
        sentence_embed = self.linear2(sentence_embed)
        memory = torch.stack([image_features, sentence_embed], dim=1)
        memory = memory.expand(beam_size, 2, 768)

        k = beam_size

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.word_map["<start>"]]] * k).to(self.embed.weight.device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(self.embed.weight.device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            x = self.embed(seqs)
            x = x.permute(1, 0, 2)
            x = self.pos_enc(x)
            x = x.permute(1, 0, 2)

            x = self.decoder(
                x,
                memory,
                tgt_mask=self.subsequent_mask(x.size(1)).to(x.device),
            )
            x = x[:, -1, :]
            x = self.head(x)
            scores = F.log_softmax(x, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode="floor")  # (s)
            next_word_inds = top_k_words % self.vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [
                ind for ind, next_word in enumerate(next_word_inds) if next_word != self.word_map["<end>"]
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            memory = memory[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step == self.max_len:
                break
            step += 1

        if len(complete_seqs_scores) != 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = []
        return seq
