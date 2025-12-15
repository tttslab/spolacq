import math
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
from transformers.models.hubert.modeling_hubert import HubertModel, HubertPreTrainedModel


def fix_random_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mincut_dp_torch(W: torch.Tensor, num_syllables: int, max_duration: int = 50):
    """
    Args:
        W (`torch.FloatTensor` of shape `(sequence_length, sequence_length)`):
            Self-similarity matrix.
    """
    T = W.shape[0]

    W_pad = torch.zeros((T + 1, T + 1), dtype=W.dtype, device=W.device)
    W_pad[1:, 1:] = W
    W = W_pad

    W_cumsum = W.cumsum(dim=0).cumsum(dim=1)  # (T + 1, T + 1)
    V = W_cumsum[-1]  # (T + 1,)

    # vol[i, j] = W[i : j + 1].sum()
    vol = V[1:][None, :] - V[:-1][:, None]  # (T, T)

    arange = torch.arange(T, device=W.device)
    i, j = torch.meshgrid(arange, arange + 1, indexing="ij")

    # W_sum[i, j] = W[i : j + 1, i : j + 1].sum()
    W_sum = W_cumsum[j, j] - 2 * W_cumsum[i, j] + W_cumsum[i, i]
    cut = vol - W_sum
    ncut = cut / (cut + W_sum / 2)

    B = torch.zeros((T + 1, num_syllables + 1), dtype=torch.int)
    C = torch.full((T + 1, num_syllables + 1), torch.finfo(W.dtype).max, device=W.device)
    C[0, 0] = 0

    # dynamic programming
    for t in range(1, T + 1):
        start = max(0, t - max_duration)
        s = min(num_syllables, t)
        c = C[start:t, :s] + ncut[start:t, t - 1 : t]
        idx = torch.argmin(c, dim=0, keepdim=True)
        B[t, 1 : s + 1] = idx + start
        C[t, 1 : s + 1] = torch.gather(c, dim=0, index=idx)

    # backtrack
    prev_b = T
    boundary = [prev_b]
    for k in range(num_syllables, 0, -1):
        prev_b = B[prev_b, k]
        boundary.append(prev_b)
    boundary.reverse()
    return boundary


def mincut_torch(
    hidden_states: torch.Tensor,
    sec_per_frame: float = 0.02,
    sec_per_syllable: float = 0.2,
    merge_threshold: Optional[float] = 0.3,
    max_duration: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A computationally efficient PyTorch implementation of the exact minimum cut algorithm

    Args:
        hidden_states (`torch.FloatTensor` of shape `(sequence_length, hidden_size)`):
            Latent speech frame representations.
    """
    num_syllables = int(math.ceil(len(hidden_states) * sec_per_frame / sec_per_syllable))

    ssm = hidden_states @ hidden_states.T
    ssm = ssm - torch.min(ssm) + 1e-7  # make it non-negative
    seg_boundary_frame = mincut_dp_torch(ssm, num_syllables, max_duration)

    seg_boundary_frame_pairs = [[l, r] for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])]
    pooled_feat = torch.stack([hidden_states[l:r].mean(0) for l, r in seg_boundary_frame_pairs])

    if merge_threshold is not None and len(seg_boundary_frame_pairs) >= 3:
        all_sim = torch.nn.functional.cosine_similarity(pooled_feat[:-1], pooled_feat[1:])
        min_id = torch.argmax(all_sim)
        while all_sim[min_id] >= merge_threshold and len(seg_boundary_frame_pairs) >= 3:
            l_merge, r_merge = seg_boundary_frame_pairs[min_id], seg_boundary_frame_pairs[min_id + 1]
            seg_boundary_frame_pairs = [
                pair for i, pair in enumerate(seg_boundary_frame_pairs) if i != min_id and i != min_id + 1
            ]
            seg_boundary_frame_pairs.insert(min_id, [l_merge[0], r_merge[1]])
            pooled_feat = torch.stack([hidden_states[l:r].mean(0) for l, r in seg_boundary_frame_pairs])
            all_sim = torch.nn.functional.cosine_similarity(pooled_feat[:-1], pooled_feat[1:])
            min_id = torch.argmax(all_sim)

    boundaries = torch.tensor(seg_boundary_frame_pairs, device=hidden_states.device) * sec_per_frame
    return boundaries, pooled_feat, torch.tensor(seg_boundary_frame_pairs, device=hidden_states.device)


class S5HubertForSyllableDiscovery(HubertPreTrainedModel):
    def __init__(
        self,
        config,
        segmentation_layer: int = 8,
        n_units_step1: int = 24576,
        n_units_step2: int = 8192,
        seed: int = 0,
    ):
        super().__init__(config)
        self.segmentation_layer = segmentation_layer
        self.n_units_step1 = n_units_step1
        self.n_units_step2 = n_units_step2

        self.hubert = HubertModel(config)
        self.hubert.eval()

        self.register_buffer("quantizer1", torch.rand(n_units_step1, config.hidden_size))
        self.register_buffer("quantizer2", torch.zeros(n_units_step1, dtype=torch.int))

        fix_random_seed(seed)

    @torch.inference_mode()
    def get_hidden_states(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.hubert(input_values, output_hidden_states=True).hidden_states
        return hidden_states[self.segmentation_layer].squeeze(0)

    def forward(
        self,
        input_values: torch.Tensor,
        sec_per_frame: float = 0.02,
        sec_per_syllable: float = 0.2,
        merge_threshold: Optional[float] = 0.3,
        max_duration: int = 50,
    ) -> Dict[str, torch.Tensor]:
        hidden_states = self.get_hidden_states(input_values)

        frame_similarity = hidden_states @ hidden_states.T
        boundary, segment_features, frame_boundary = mincut_torch(
            hidden_states,
            sec_per_frame=sec_per_frame,
            sec_per_syllable=sec_per_syllable,
            merge_threshold=merge_threshold,
            max_duration=max_duration,
        )

        # deduplicated syllabic units
        units_step1 = torch.cdist(segment_features, self.quantizer1).argmin(1)
        units = self.quantizer2[units_step1]

        # duplicated syllabic units
        durations = frame_boundary[:, 1] - frame_boundary[:, 0]
        duplicated_units = torch.repeat_interleave(units, durations)
        return {
            "units": units,
            "units_step1": units_step1,
            "duplicated_units": duplicated_units,
            "boundary": boundary,
            "frame_boundary": frame_boundary,
            "hidden_states": hidden_states,
            "frame_similarity": frame_similarity,
            "durations": durations,
        }


def get_unit(path):
    wav, sr = torchaudio.load(path)
    wav = torchaudio.functional.resample(wav, sr, 16000)

    model = (
        S5HubertForSyllableDiscovery.from_pretrained("ryota-komatsu/s5-hubert", cache_dir="./cache/S2U").cuda().eval()
    )
    outputs = model(wav.cuda())

    units = outputs["units"].cpu().tolist()
    units = [str(u) for u in units]

    return units
