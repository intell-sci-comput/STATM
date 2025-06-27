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

"""Main file for running the model trainer."""
import functools
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple, Type, Union

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from statm.lib import evaluator
from statm.lib import input_pipeline
from statm.lib import losses
from statm.lib import utils
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from jax.numpy import isin
from jax import image as IM
from jax import vmap

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

from clu import platform
import jax
from ml_collections import config_flags, ConfigDict
from statm.configs.movi import statm_savi_conditional_small as small_config
from statm.configs.movi import statm_savi_conditional as savi_config
# from statm.configs.movi import savi_unconditional as unconditional_config
from statm.lib import trainer
import tensorflow as tf


Array = jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
PRNGKey = Array


def evaluate(model, state, eval_ds, loss_fn_eval, eval_metrics_cls, config, root_dir):
    """Evaluate the model."""
    eval_metrics, eval_batch, eval_preds = evaluator.evaluate(
        model,
        state,
        eval_ds,
        loss_fn_eval,
        eval_metrics_cls,
        predicted_max_num_instances=config.num_slots,
        ground_truth_max_num_instances=config.max_instances + 1,  # Incl. bg.
        slice_size= 24,  # must be 24 when evaluating on movi dataset
        slice_keys=config.get("eval_slice_keys"),  # [flow ,video...]
        conditioning_key=config.get("conditioning_key"),  # bbox
        remove_from_predictions=config.get("remove_from_predictions"),  #
        metrics_on_cpu=config.get("metrics_on_cpu", False),
        root_dir=root_dir)

    metrics_res = eval_metrics.compute()
    flatten_metrics_res = utils.flatten_named_dicttree(metrics_res)
    return flatten_metrics_res, eval_preds, eval_batch


# def evaluate_miou(model, state, eval_ds, loss_fn_eval, eval_metrics_cls, config):
#     miou = evaluator.compute_miou(
#         model,
#         state,
#         eval_ds,
#         loss_fn_eval,
#         eval_metrics_cls,
#         predicted_max_num_instances=config.num_slots,
#         ground_truth_max_num_instances=config.max_instances + 1,  # Incl. bg.
#         slice_size=24,  # 6
#         slice_keys=config.get("eval_slice_keys"),  # [flow ,video...]
#         conditioning_key=config.get("conditioning_key"),  # bbox
#         remove_from_predictions=config.get("remove_from_predictions"),  #
#         metrics_on_cpu=config.get("metrics_on_cpu", False))
#     return miou


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str,
                       target_step: int = 101):
    """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
    rng = jax.random.PRNGKey(config.seed)
    tf.io.gfile.makedirs(workdir)

    # Input pipeline.
    rng, data_rng = jax.random.split(rng)
    # Make sure each host uses a different RNG for the training data.
    if config.get("seed_data", True):  # Default to seeding data if not specified.
        data_rng = jax.random.fold_in(data_rng, jax.host_id())
    else:
        data_rng = None
    train_ds, eval_ds = input_pipeline.create_datasets(config, data_rng)

    # Initialize model
    model = utils.build_model_from_config(config.model)  # create model from config
   
    optimizer_def = flax.optim.Adam(learning_rate=config.learning_rate)  # pytype: disable=module-attr

    # Construct TrainMetrics and EvalMetrics, metrics collections.
    eval_metrics_cls = utils.make_metrics_collection("EvalMetrics",
                                                     config.eval_metrics_spec)

    def init_model(rng):
        rng, init_rng, model_rng, dropout_rng = jax.random.split(rng, num=4)

        init_conditioning = None
        if config.get("conditioning_key"):
            init_conditioning = jnp.ones(
                [1] + list(train_ds.element_spec[config.conditioning_key].shape)[2:],
                jnp.int32)
        init_inputs = jnp.ones(
            [1] + list(train_ds.element_spec["video"].shape)[2:],
            jnp.float32)  # [1, 24, 128, 128, 3]->[128,128,3]
        initial_vars = model.init(
            {"params": model_rng, "state_init": init_rng, "dropout": dropout_rng},
            video=init_inputs, conditioning=init_conditioning,
            padding_mask=jnp.ones(init_inputs.shape[:-1], jnp.int32))

        # Split into state variables (e.g. for batchnorm stats) and model params.
        # Note that `pop()` on a FrozenDict performs a deep copy.
        state_vars, initial_params = initial_vars.pop("params")  # pytype: disable=attribute-error

        # Filter out intermediates (we don't want to store these in the TrainState).
        state_vars = utils.filter_key_from_frozen_dict(
            state_vars, key="intermediates")
        return state_vars, initial_params

    state_vars, initial_params = init_model(rng)

    # show the parameter overview
    parameter_overview.log_parameter_overview(initial_params)  # pytype: disable=wrong-arg-types
    optimizer = optimizer_def.create(initial_params)

    state = utils.TrainState(
        step=1, optimizer=optimizer, rng=rng, variables=state_vars)

    loss_fn = functools.partial(
        losses.compute_full_loss, loss_config=config.losses)

    # load checkpoint if exists
    checkpoint_dir = os.path.join(workdir, "checkpoints-0")
    ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir)
    # state = ckpt.restore_or_initialize(state)
    
    target_step = target_step
    target_ckpt_path = os.path.join(checkpoint_dir, f"ckpt-{target_step}")
    state = ckpt.restore(state, target_ckpt_path)

    # Replicate our parameters.
    state = flax.jax_utils.replicate(state, devices=jax.local_devices())
    del rng  # rng is stored in the state.

    # root for visualization
    viz_dir = os.path.join(workdir, "vis")

    # Start JAX profiler to record execution trace (for TensorBoard visualization)
    # jax.profiler.start_trace("/mnt/tensorboard")
    eval_ari, prds, ev_batch = evaluate(model, state, eval_ds, loss_fn, eval_metrics_cls,
                                        config, viz_dir)
    # Ensure all computations finish before stopping the trace
    # eval_ari['eval_ari'].block_until_ready()
    # Stop the JAX profiler trace and flush data to the output directory
    # jax.profiler.stop_trace()

    # miou = evaluate_miou(model, state, eval_ds, loss_fn, eval_metrics_cls, config)
    # print("miou:", miou)
    print("eval_ari:", eval_ari)


def main(argv):
    del argv

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.

    target_step = 101
    # statm-savi small
    # config = small_config.get_config()
    # workdir = 'mnt/small_block2_movi_b'
    # statm-savi++
    config = savi_config.get_config()
    workdir = 'mnt/statm_savi++_block2_movi_a'

    # unconditional
    # config = unconditional_config.get_config()
    # workdir = 'mnt/block2_movi_c_just_video_200k'


    logging.info("JAX host: %d / %d", jax.host_id(), jax.host_count())
    logging.info("JAX devices: %r", jax.devices())

    train_and_evaluate(config, workdir,target_step)


if __name__ == "__main__":
    app.run(main)
