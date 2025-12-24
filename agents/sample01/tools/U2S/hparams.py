import os

import tensorflow as tf
import yaml


class HParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def parse(self, hparams_string: str) -> None:
        if not hparams_string:
            return

        for param in hparams_string.split(","):
            key, value = param.split("=")
            key = key.strip()
            value = value.strip()

            try:
                if "." in value:
                    value = float(value)
                elif value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    value = int(value)
            except ValueError:
                pass  # If the value cannot be converted, it remains as a string.

            setattr(self, key, value)

    def values(self):
        return self._hparams


def create_hparams(config=None, hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=2000,
        iters_per_checkpoint=2000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=["embedding.weight"],
        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files=os.path.join(os.path.dirname(__file__), "../..", config["U2S"]["filelists_train"]),
        validation_files=os.path.join(os.path.dirname(__file__), "../..", config["U2S"]["filelists_val"]),
        text_cleaners=["english_cleaners"],
        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        ################################
        # Model Parameters             #
        ################################
        n_symbols=4096,
        symbols_embedding_dim=512,
        # Encoder parameters
        encoder_kernel_size=3,
        encoder_n_convolutions=1,
        encoder_embedding_dim=512,
        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,
        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=256,
        mask_padding=True,  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.get_logger().info("Parsing command line hparams: %s", hparams_string)

    if verbose:
        tf.get_logger().info("Final parsed hparams: %s", hparams.values())

    return hparams
