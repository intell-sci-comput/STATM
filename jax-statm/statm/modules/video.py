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

"""Video module library."""

import functools
from typing import Any, Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple, Union

from flax import linen as nn
import jax.numpy as jnp
from statm.lib import utils
from statm.modules import misc

Shape = Tuple[int]

DType = Any
Array = jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class CorrectorPredictorTuple(NamedTuple):
    corrected: ProcessorState
    predicted: ProcessorState


class STATMSAVi(nn.Module):
    """Video model consisting of encoder, recurrent processor, and decoder."""

    encoder: Callable[[], nn.Module]
    decoder: Callable[[], nn.Module]
    corrector: Callable[[], nn.Module]
    predictor: Callable[[], nn.Module]
    initializer: Callable[[], nn.Module]
    decode_corrected: bool = True
    decode_predicted: bool = True

    @nn.compact
    def __call__(self, video: Array, conditioning: Optional[Array] = None,
                 continue_from_previous_state: bool = False,
                 padding_mask: Optional[Array] = None,
                 train: bool = False) -> ArrayTree:
        """Performs a forward pass on a video.

    Args:
      video: Video of shape `[batch_size, n_frames, height, width, n_channels]`.
      conditioning: Optional jnp.ndarray used for conditioning the initial state
        of the recurrent processor.
      continue_from_previous_state: Boolean, whether to continue from a previous
        state or not. If True, the conditioning variable is used directly as
        initial state.
      padding_mask: Binary mask for padding video inputs (e.g. for videos of
        different sizes/lengths). Zero corresponds to padding.
      train: Indicating whether we're training or evaluating.

    Returns:
      A dictionary of model predictions.
    """

        if padding_mask is None:
            padding_mask = jnp.ones(video.shape[:-1], jnp.int32)

        # video.shape = (batch_size, n_frames, height, width, n_channels)
        # Vmapped over sequence dim.
        encoded_inputs = self.encoder()(video, padding_mask, train)  # pytype: disable=not-callable
        if continue_from_previous_state:
            assert conditioning is not None, (
                "When continuing from a previous state, the state has to be passed "
                "via the `conditioning` variable, which cannot be `None`.")
            init_state = conditioning[:, -1]  # We currently only use last state.
        else:
            # Same as above but without encoded inputs.
            # init_state.shape = (b,slot_num,slot_dim)
            init_state = self.initializer()(
                conditioning, batch_size=video.shape[0], train=train)  # pytype: disable=not-callable

        B, T, D, C = encoded_inputs.shape
        correct_slots = []
        predict_slots = []
        predict_state = init_state
        corrector = self.corrector()
        predictor = self.predictor()
        for t in range(T):
            c_slot = corrector(predict_state, encoded_inputs[:, t], padding_mask[:, t], train=train)
            correct_slots.append(c_slot)
            predict_buffer = correct_slots  # (b,t,h*w,c)
            # fixed buffer size
            # predict_buffer = predict_buffer[-6:]

            predict_buffer = jnp.stack(predict_buffer, axis=1)
            pre_query = predict_buffer[:, -1]  # (b,h*w,c)
            # Spatio-Temporal Attention
            p_slot = predictor(pre_query, predict_buffer, train=train)
            # baseline
            # p_slot = predictor(pre_query, train=train)
            predict_state = p_slot
            predict_slots.append(p_slot)
        states = CorrectorPredictorTuple(corrected=jnp.stack(correct_slots, axis=1)
                                         , predicted=jnp.stack(predict_slots, axis=1))

        # Decode latent states.
        decoder = self.decoder()  # Vmapped over sequence dim.
        outputs = decoder(states.corrected,
                          train) if self.decode_corrected else None  # pytype: disable=not-callable
        outputs_pred = decoder(states.predicted,
                               train) if self.decode_predicted else None  # pytype: disable=not-callable

        return {
            "states": states.corrected,
            "states_pred": states.predicted,
            "outputs": outputs,
            "outputs_pred": outputs_pred,
        }


class FrameEncoder(nn.Module):
    """Encoder for single video frame, vmapped over time axis."""

    backbone: Callable[[], nn.Module]
    pos_emb: Callable[[], nn.Module] = misc.Identity
    reduction: Optional[str] = None
    output_transform: Callable[[], nn.Module] = misc.Identity

    # Vmapped application of module, consumes time axis (axis=1).
    @functools.partial(utils.time_distributed, in_axes=(1, 1, None))
    @nn.compact
    def __call__(self, inputs: Array, padding_mask: Optional[Array] = None,
                 train: bool = False) -> Tuple[Array, Dict[str, Array]]:
        del padding_mask  # Unused.

        # inputs.shape = (batch_size, height, width, n_channels)
        x = self.backbone()(inputs, train=train)

        x = self.pos_emb()(x)

        if self.reduction == "spatial_flatten":
            batch_size, height, width, n_features = x.shape
            x = jnp.reshape(x, (batch_size, height * width, n_features))
        elif self.reduction == "spatial_average":
            x = jnp.mean(x, axis=(1, 2))
        elif self.reduction == "all_flatten":
            batch_size, height, width, n_features = x.shape
            x = jnp.reshape(x, (batch_size, height * width * n_features))
        elif self.reduction is not None:
            raise ValueError("Unknown reduction type: {}.".format(self.reduction))

        output_block = self.output_transform()

        if hasattr(output_block, "qkv_size"):
            # Project to qkv_size if used transformer.
            x = nn.relu(nn.Dense(output_block.qkv_size)(x))

        x = output_block(x, train=train)
        return x
