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

"""Model evaluation."""

import functools
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union

from absl import logging
from clu import metrics
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from statm.lib import losses
from statm.lib import utils
import tensorflow as tf
from einops import rearrange

Array = jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
PRNGKey = Array


def get_eval_metrics(
        preds: Dict[str, ArrayTree],
        batch: Dict[str, Array],
        loss_fn: losses.LossFn,
        eval_metrics_cls: Type[metrics.Collection],
        predicted_max_num_instances: int,
        ground_truth_max_num_instances: int,
) -> Union[None, metrics.Collection]:
    """Compute the metrics for the model predictions in inference mode.

  The metrics are averaged across *all* devices (of all hosts).

  Args:
    preds: Model predictions.
    batch: Inputs that should be evaluated.
    loss_fn: Loss function that takes model predictions and a batch of data.
    eval_metrics_cls: Evaluation metrics collection.
    predicted_max_num_instances: Maximum number of instances in prediction.
    ground_truth_max_num_instances: Maximum number of instances in ground truth,
      including background (which counts as a separate instance).

  Returns:
    The evaluation metrics.
  """
    loss, loss_aux = loss_fn(preds, batch)
    # 全局评估
    metrics_update = eval_metrics_cls.gather_from_model_output(
        loss=loss,
        **loss_aux,
        predicted_segmentations=utils.remove_singleton_dim(
            preds["outputs"].get("segmentations")),  # pytype: disable=attribute-error
        ground_truth_segmentations=batch.get("segmentations"),
        predicted_max_num_instances=predicted_max_num_instances,
        ground_truth_max_num_instances=ground_truth_max_num_instances,
        padding_mask=batch.get("padding_mask"),
        mask=batch.get("mask"))

    # 评估分段
    # metrics_update = eval_metrics_cls.gather_from_model_output(
    #     loss=loss,
    #     **loss_aux,
    #     predicted_segmentations=utils.remove_singleton_dim(
    #         preds["outputs"].get("segmentations")[:, 1:, :]),  # pytype: disable=attribute-error
    #     ground_truth_segmentations=batch.get("segmentations")[:, 1:, :],
    #     predicted_max_num_instances=predicted_max_num_instances,
    #     ground_truth_max_num_instances=ground_truth_max_num_instances,
    #     padding_mask=batch.get("padding_mask")[:, 1:, :],
    #     mask=batch.get("mask"))

    # 评估第一帧 jnp.expand_dims
    # metrics_update = eval_metrics_cls.gather_from_model_output(
    #     loss=loss,
    #     **loss_aux,
    #     predicted_segmentations=jnp.expand_dims(utils.remove_singleton_dim(
    #         preds["outputs"].get("segmentations")[:, 0, :]), axis=1),  # pytype: disable=attribute-error
    #     ground_truth_segmentations=jnp.expand_dims(batch.get("segmentations")[:, 0, :], axis=1),
    #     predicted_max_num_instances=predicted_max_num_instances,
    #     ground_truth_max_num_instances=ground_truth_max_num_instances,
    #     padding_mask=jnp.expand_dims(batch.get("padding_mask")[:, 0, :], axis=1),
    #     mask=batch.get("mask"))
    return metrics_update


def eval_first_step(
        model: nn.Module,
        state_variables: flax.core.FrozenDict,
        params: Dict[str, ArrayTree],
        batch: Dict[str, Array],
        rng: PRNGKey,
        conditioning_key: Optional[str] = None
) -> Dict[str, ArrayTree]:
    """Get the model predictions with a freshly initialized recurrent state.

  The model is applied to the inputs using all devices on the host.

  Args:
    model: Model used in eval step.
    state_variables: State variables for the model.
    params: Params for the model.
    batch: Inputs that should be evaluated.
    rng: PRNGKey for model forward pass.
    conditioning_key: Optional string. If provided, defines the batch key to be
      used as conditioning signal for the model. Otherwise this is inferred from
      the available keys in the batch.
  Returns:
    The model's predictions.
  """
    logging.info("eval_first_step(batch=%s)", batch)

    conditioning = None
    if conditioning_key:
        conditioning = batch[conditioning_key]
    preds, mutable_vars = model.apply(
        {"params": params, **state_variables}, video=batch["video"],
        conditioning=conditioning, mutable="intermediates",
        rngs={"state_init": rng}, train=False,
        padding_mask=batch.get("padding_mask"))  # rang

    if "intermediates" in mutable_vars:
        preds["intermediates"] = flax.core.unfreeze(mutable_vars["intermediates"])

        # Spatio-Temporal Attention
        slot_att = preds["intermediates"]["SlotAttention_0"]
        concat_fn = lambda _, *x: functools.partial(jnp.concatenate, axis=1)(*x)
        # jax.tree_map(concat_fn, slot_att,e)
        leaves = jax.tree_leaves(slot_att)
        temp_list = []
        for leaf in leaves:
            temp_list.append(leaf)
        slot_att = concat_fn(temp_list[0], temp_list)
        slot_att = jnp.expand_dims(slot_att, axis=2)
        slot_att = tuple([slot_att])
        preds["intermediates"]["SlotAttention_0"]['InvertedDotProductAttention_0']['GeneralizedDotProductAttention_0'][
            'attn'] = slot_att

    return preds


def eval_continued_step(
        model: nn.Module,
        state_variables: flax.core.FrozenDict,
        params: Dict[str, ArrayTree],
        batch: Dict[str, Array],
        rng: PRNGKey,
        recurrent_states: Array
) -> Dict[str, ArrayTree]:
    """Get the model predictions, continuing from a provided recurrent state.

  The model is applied to the inputs using all devices on the host.

  Args:
    model: Model used in eval step.
    state_variables: State variables for the model.
    params: The model parameters.
    batch: Inputs that should be evaluated.
    rng: PRNGKey for model forward pass.
    recurrent_states: Recurrent internal model state from which to continue.
  Returns:
    The model's predictions.
  """
    logging.info("eval_continued_step(batch=%s, recurrent_states=%s)", batch,
                 recurrent_states)

    preds, mutable_vars = model.apply(
        {"params": params, **state_variables}, video=batch["video"],
        conditioning=recurrent_states, continue_from_previous_state=True,
        mutable="intermediates", rngs={"state_init": rng}, train=False,
        padding_mask=batch.get("padding_mask"))

    if "intermediates" in mutable_vars:
        preds["intermediates"] = flax.core.unfreeze(mutable_vars["intermediates"])

        # Spatio-Temporal Attention
        slot_att = preds["intermediates"]["SlotAttention_0"]
        concat_fn = lambda _, *x: functools.partial(jnp.concatenate, axis=1)(*x)
        leaves = jax.tree_leaves(slot_att)
        temp_list = []
        for leaf in leaves:
            temp_list.append(leaf)
        slot_att = concat_fn(temp_list[0], temp_list)
        slot_att = jnp.expand_dims(slot_att, axis=2)
        slot_att = tuple([slot_att])
        preds["intermediates"]["SlotAttention_0"]['InvertedDotProductAttention_0']['GeneralizedDotProductAttention_0'][
            'attn'] = slot_att

    return preds


def eval_step(
        model: nn.Module,
        state: utils.TrainState,
        batch: Dict[str, Array],
        rng: PRNGKey,
        p_eval_first_step: Callable[..., Dict[str, ArrayTree]],
        p_eval_continued_step: Callable[..., Dict[str, ArrayTree]],
        slice_size: Optional[int] = None,
        slice_keys: Optional[Sequence[str]] = None,
        conditioning_key: Optional[str] = None,
        remove_from_predictions: Optional[Sequence[str]] = None
) -> Dict[str, ArrayTree]:
    """Compute the metrics for the given model in inference mode.

  The model is applied to the inputs using all devices on the host. Afterwards
  metrics are averaged across *all* devices (of all hosts).

  Args:
    model: Model used in eval step.
    state: Replicated model state.
    batch: Inputs that should be evaluated.
    rng: PRNGKey for model forward pass.
    p_eval_first_step: A parallel version of the function eval_first_step.
    p_eval_continued_step: A parallel version of the function
      eval_continued_step.
    slice_size: Optional integer, if provided, evaluate the model on temporal
      slices of this size instead of on the full sequence length at once.
    slice_keys: Optional list of strings, the keys of the tensors which will be
      sliced if slice_size is provided.
    conditioning_key: Optional string. If provided, defines the batch key to be
      used as conditioning signal for the model. Otherwise this is inferred from
      the available keys in the batch.
    remove_from_predictions: Remove the provided keys. The default None removes
      "states" and "states_pred" from model output to save memory. Disable this
      if either of these are required in the loss function or for visualization.
  Returns:
    Model predictions.
  """
    if remove_from_predictions is None:
        remove_from_predictions = ["states", "states_pred"]

    seq_len = batch["video"].shape[2]
    # Sliced evaluation (i.e. on smaller temporal slices of the video).
    if slice_size is not None and slice_size < seq_len:
        num_slices = int(np.ceil(seq_len / slice_size))

        assert slice_keys is not None, (
            "Slice keys need to be provided for sliced evaluation.")

        preds_per_slice = []
        # Get predictions for first slice (with fresh recurrent state).
        batch_slice = utils.get_slices_along_axis(
            batch, slice_keys=slice_keys, start_idx=0, end_idx=slice_size)
        preds_slice = p_eval_first_step(model, state.variables,
                                        state.optimizer.target, batch_slice, rng,
                                        conditioning_key)
        preds_slice = jax.tree_map(np.asarray, preds_slice)  # Copy to CPU.
        preds_per_slice.append(preds_slice)

        # Iterate over remaining slices (re-using the previous recurrent state).
        for slice_idx in range(1, num_slices):
            recurrent_states = preds_per_slice[-1]["states_pred"]
            batch_slice = utils.get_slices_along_axis(
                batch, slice_keys=slice_keys, start_idx=slice_idx * slice_size,
                end_idx=(slice_idx + 1) * slice_size)
            preds_slice = p_eval_continued_step(
                model, state.variables, state.optimizer.target,
                batch_slice, rng, recurrent_states)
            preds_slice = jax.tree_map(np.asarray, preds_slice)  # Copy to CPU.
            preds_per_slice.append(preds_slice)

        # Remove states from predictions before concat to save memory.
        for k in remove_from_predictions:
            for i in range(num_slices):
                _ = preds_per_slice[i].pop(k, None)

        # Join predictions along sequence dimension.
        concat_fn = lambda _, *x: functools.partial(np.concatenate, axis=2)([*x])
        preds = jax.tree_map(concat_fn, preds_per_slice[0], *preds_per_slice)

        # Truncate to original sequence length.
        # NOTE: This op assumes that all predictions have a (complete) time axis.
        preds = jax.tree_map(lambda x: x[:, :, :seq_len], preds)

    # Evaluate on full sequence if no (or too large) slice size is provided.
    else:
        preds = p_eval_first_step(model, state.variables,
                                  state.optimizer.target, batch, rng,
                                  conditioning_key)
        for k in remove_from_predictions:
            _ = preds.pop(k, None)

    return preds


def evaluate(
        model: nn.Module,
        state: utils.TrainState,
        eval_ds: tf.data.Dataset,
        loss_fn: Callable[[Dict[str, ArrayTree], Dict[str, ArrayTree]], Array],
        eval_metrics_cls: Type[metrics.Collection],
        predicted_max_num_instances: int,
        ground_truth_max_num_instances: int,
        slice_size: Optional[int] = None,
        slice_keys: Optional[Sequence[str]] = None,
        conditioning_key: Optional[str] = None,
        remove_from_predictions: Optional[Sequence[str]] = None,
        metrics_on_cpu: bool = False,
        root_dir: Optional[str] = None,
) -> Tuple[metrics.Collection, Dict[str, ArrayTree], Dict[str, ArrayTree]]:
    """Evaluate the model on the given dataset."""
    eval_metrics = None
    batch = None
    preds = None
    rng = state.rng[0]  # Get training state PRNGKey from first replica.

    if metrics_on_cpu and jax.process_count() > 1:
        raise NotImplementedError(
            "metrics_on_cpu feature cannot be used in a multi-host setup."
            " This experiment is using {} hosts.".format(jax.process_count()))
    metric_devices = jax.devices("cpu") if metrics_on_cpu else jax.devices()

    p_eval_first_step = jax.pmap(
        eval_first_step,
        axis_name="batch",
        static_broadcasted_argnums=(0, 5),
        devices=jax.devices())
    p_eval_continued_step = jax.pmap(
        eval_continued_step,
        axis_name="batch",
        static_broadcasted_argnums=(0),
        devices=jax.devices())
    p_get_eval_metrics = jax.pmap(
        get_eval_metrics,
        axis_name="batch",
        static_broadcasted_argnums=(2, 3, 4, 5),
        devices=metric_devices,
        backend="cpu" if metrics_on_cpu else None)

    def reshape_fn(x):
        """Function to reshape preds and batch before calling p_get_eval_metrics."""
        return np.reshape(x, [len(metric_devices), -1] + list(x.shape[2:]))

    for batch in eval_ds:

        rng, eval_rng = jax.random.split(rng)
        eval_rng = jax.random.fold_in(eval_rng, jax.host_id())  # Bind to host.
        eval_rngs = jax.random.split(eval_rng, jax.local_device_count())
        batch = jax.tree_map(np.asarray, batch)
        preds = eval_step(
            model=model,
            state=state,
            batch=batch,
            rng=eval_rngs,
            p_eval_first_step=p_eval_first_step,
            p_eval_continued_step=p_eval_continued_step,
            slice_size=slice_size,
            slice_keys=slice_keys,
            conditioning_key=conditioning_key,
            remove_from_predictions=remove_from_predictions)

        if metrics_on_cpu:
            # Reshape replica dim and batch-dims to work with metric_devices.
            preds = jax.tree_map(reshape_fn, preds)
            batch = jax.tree_map(reshape_fn, batch)
        # Get metric updates.
        update = p_get_eval_metrics(preds, batch, loss_fn, eval_metrics_cls,
                                    predicted_max_num_instances,
                                    ground_truth_max_num_instances)
        update = flax.jax_utils.unreplicate(update)
        eval_metrics = (
            update if eval_metrics is None else eval_metrics.merge(update))

    assert eval_metrics is not None

    return eval_metrics, batch, preds


Ndarray = Union[np.ndarray, jnp.ndarray]

