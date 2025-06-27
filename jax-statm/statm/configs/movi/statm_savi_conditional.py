# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Config for a conditional STATM-SAVi++ model.

STATM-SAVi++ operates on 128x128 video frames and uses a ResNet-34 backbone.
Note: Due to resource constraints, the evaluation "config.eval_slice_size = 6" during training.
As a result, the memory buffer in STATM remains shorter than 6, which leads to lower evaluation
scores compared to those reported in the paper. In the paper, the evaluation is conducted with
a buffer length of 24. Therefore, to obtain results comparable to the paper, the model should be re-evaluated
after training with a buffer size of 24. If you wish to select model checkpoints during training
based on evaluation performance, you may set "config.eval_slice_size = 24". However, please note
that this may increase memory consumption.
"""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.seed = 42
    config.seed_data = True

    config.batch_size = 64
    config.val_batch_size = 2
    config.num_train_steps = 100000

    # Adam optimizer config.
    config.learning_rate = 2e-4
    config.warmup_steps = 2500
    config.max_grad_norm = 0.05

    config.log_loss_every_steps = 50
    config.eval_every_steps = 1000
    config.checkpoint_every_steps = 5000

    config.train_metrics_spec = {
        "loss": "loss",
        "ari": "ari",
        "ari_nobg": "ari_nobg",
    }
    config.eval_metrics_spec = {
        "eval_loss": "loss",
        "eval_ari": "ari",
        "eval_ari_nobg": "ari_nobg",
    }

    config.data = ml_collections.ConfigDict({
        "tfds_name": "movi_e/128x128:1.0.0",  # Dataset for training/eval.
        "data_dir": "mnt/",  # Path to the directory where the dataset is stored.
        "shuffle_buffer_size": config.batch_size * 8,
    })

    # NOTE: MOVi-A, MOVi-B, and MOVi-C only contain up to 10 instances (objects),
    # i.e. it is safe to reduce config.max_instances to 10 for these datasets,
    # resulting in more efficient training/evaluation. We set this default to 23,
    # since MOVi-D and MOVi-E contain up to 23 objects per video. Setting
    # config.max_instances to a smaller number than the maximum number of objects
    # in a dataset will discard objects, ultimately giving different results.
    config.max_instances = 23
    config.num_slots = config.max_instances + 1  # Only used for metrics.
    config.logging_min_n_colors = config.max_instances

    config.preproc_train = [
        "video_from_tfds",
        f"sparse_to_dense_annotation(max_instances={config.max_instances})",
        "temporal_random_strided_window(length=6)",
        "random_resized_crop" +
        "(height=128, width=128, min_object_covered=0.75)",
        "transform_depth(transform='log_plus')",
        "flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
    ]

    config.preproc_eval = [
        "video_from_tfds",
        f"sparse_to_dense_annotation(max_instances={config.max_instances})",
        "temporal_crop_or_pad(length=24)",
        "resize_small(128)",
        "transform_depth(transform='log_plus')",
        "flow_to_rgb()"  # NOTE: This only uses the first two flow dimensions.
    ]

    config.eval_slice_size = 24
    config.eval_slice_keys = [
        "video", "segmentations", "flow", "boxes", "depth"
    ]

    # Dictionary of targets and corresponding channels. Losses need to match.
    config.targets = {"flow": 3, "depth": 1}
    config.losses = ml_collections.ConfigDict({
        f"recon_{target}": {"loss_type": "recon", "key": target}
        for target in config.targets})

    config.conditioning_key = "boxes"

    config.model = ml_collections.ConfigDict({
        "module": "statm.modules.STATMSAVi",

        # Encoder.
        "encoder": ml_collections.ConfigDict({
            "module": "statm.modules.FrameEncoder",
            "reduction": "spatial_flatten",

            "backbone": ml_collections.ConfigDict({
                "module": "statm.modules.ResNet34",
                "num_classes": None,
                "axis_name": "time",
                "norm_type": "group",
                "small_inputs": True
            }),
            "pos_emb": ml_collections.ConfigDict({
                "module": "statm.modules.PositionEmbedding",
                "embedding_type": "linear",
                "update_type": "project_add",
                "output_transform": ml_collections.ConfigDict({
                    "module": "statm.modules.MLP",
                    "hidden_size": 64,
                    "layernorm": "pre"
                }),
            }),
            # Transformer.
            "output_transform": ml_collections.ConfigDict({
                "module": "statm.modules.Transformer",
                "num_layers": 4,
                "num_heads": 4,
                "qkv_size": 16 * 4,
                "mlp_size": 1024,
                "pre_norm": True,
            }),
        }),

        # Corrector.
        "corrector": ml_collections.ConfigDict({
            "module": "statm.modules.SlotAttention",
            "num_iterations": 1,
            "qkv_size": 256,
        }),

        # Predictor.
        "predictor": ml_collections.ConfigDict({
            "module": "statm.modules.TimeSpaceTransformerBlock2",
            "num_heads": 4,
            "qkv_size": 256,
            "mlp_size": 1024
        }),

        # Initializer.
        "initializer": ml_collections.ConfigDict({
            "module": "statm.modules.CoordinateEncoderStateInit",
            "prepend_background": True,
            "center_of_mass": False,
            "embedding_transform": ml_collections.ConfigDict({
                "module": "statm.modules.MLP",
                "hidden_size": 256,
                "output_size": 128,
                "layernorm": None
            }),
        }),

        # Decoder.
        "decoder": ml_collections.ConfigDict({
            "module":
                "statm.modules.SpatialBroadcastDecoder",
            "resolution": (8, 8),  # Update if data resol. or strides change.
            "early_fusion": True,
            "backbone": ml_collections.ConfigDict({
                "module": "statm.modules.CNN",
                "features": [64, 64, 64, 64],
                "kernel_size": [(5, 5), (5, 5), (5, 5), (5, 5)],
                "strides": [(2, 2), (2, 2), (2, 2), (2, 2)],
                "layer_transpose": [True, True, True, True]
            }),
            "pos_emb": ml_collections.ConfigDict({
                "module": "statm.modules.PositionEmbedding",
                "embedding_type": "linear",
                "update_type": "project_add"
            }),
            "target_readout": ml_collections.ConfigDict({
                "module": "statm.modules.Readout",
                "keys": list(config.targets),
                "readout_modules": [ml_collections.ConfigDict({
                    "module": "statm.modules.MLP",
                    "num_hidden_layers": 0,
                    "hidden_size": 0, "output_size": config.targets[k]})
                    for k in config.targets],
            }),
        }),
        "decode_corrected": True,
        "decode_predicted": False,  # Disable prediction decoder to save memory.
    })

    # Define which video-shaped variables to log/visualize.
    config.debug_var_video_paths = {
        "recon_masks": "SpatialBroadcastDecoder_0/alphas",
    }
    for k in config.targets:
        config.debug_var_video_paths.update({
            f"{k}_recon": f"SpatialBroadcastDecoder_0/{k}_combined"})

    # Define which attention matrices to log/visualize.
    config.debug_var_attn_paths = {
        "corrector_attn": "SlotAttention_0/InvertedDotProductAttention_0/GeneralizedDotProductAttention_0/attn"
    }

    # Widths of attention matrices (for reshaping to image grid).
    config.debug_var_attn_widths = {
        "corrector_attn": 16,
    }

    return config
