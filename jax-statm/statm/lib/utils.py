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

"""Common utils."""

import seaborn as sns
import functools
import importlib
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Type, Union

from absl import logging
from clu import metrics as base_metrics
import flax
from flax import linen as nn
from flax import traverse_util
import jax
from jax.example_libraries import optimizers as jax_optim
import jax.numpy as jnp
import jax.ops
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
import skimage.transform
from statm.lib import metrics
import tensorflow as tf
import os
from jax import image as IM
from jax import vmap
import optax
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

Array = Union[np.ndarray, jnp.ndarray]
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
DictTree = Dict[str, Union[Array, "DictTree"]]  # pytype: disable=not-supported-yet
PRNGKey = Array
ConfigAttr = Any
MetricSpec = Dict[str, str]


@flax.struct.dataclass
class TrainState:
    """Data structure for checkpointing the model."""
    step: int
    optimizer: flax.optim.Optimizer  # pytype: disable=module-attr
    variables: flax.core.FrozenDict
    rng: PRNGKey


METRIC_TYPE_TO_CLS = {
    "loss": base_metrics.Average.from_output(name="loss"),
    "ari": metrics.Ari,
    "ari_nobg": metrics.AriNoBg,
}


def make_metrics_collection(
        class_name: str,
        metrics_spec: Optional[MetricSpec]) -> Type[base_metrics.Collection]:
    """Create class inhering from metrics.Collection based on spec."""
    metrics_dict = {}
    if metrics_spec:
        for m_name, m_type in metrics_spec.items():
            metrics_dict[m_name] = METRIC_TYPE_TO_CLS[m_type]

    return flax.struct.dataclass(
        type(class_name,
             (base_metrics.Collection,),
             {"__annotations__": metrics_dict}))


def flatten_named_dicttree(metrics_res: DictTree, sep: str = "/"):
    """Flatten dictionary."""
    metrics_res_flat = {}
    for k, v in traverse_util.flatten_dict(metrics_res).items():
        metrics_res_flat[(sep.join(k)).strip(sep)] = v
    return metrics_res_flat


def clip_grads(grad_tree: ArrayTree, max_norm: float, epsilon: float = 1e-6):
    """Gradient clipping with epsilon.

  Adapted from jax.example_libraries.optimizers.clip_grads.

  Args:
    grad_tree: ArrayTree of gradients.
    max_norm: A float, clip gradients above this maximum norm.
    epsilon: A float, offset to avoid division by zero.

  Returns:
    ArrayTree of clipped gradients.
  """
    norm = jax_optim.l2_norm(grad_tree)
    clip_coef = max_norm / (norm + epsilon)
    normalize = lambda g: jnp.where(clip_coef < 1., g * clip_coef, g)
    return jax.tree_map(normalize, grad_tree)


def spatial_broadcast(x: Array, resolution: Sequence[int]) -> Array:
    """Broadcast flat inputs to a 2D grid of a given resolution."""
    # x.shape = (batch_size, features).
    x = x[:, jnp.newaxis, jnp.newaxis, :]
    return jnp.tile (x, [1, resolution[0], resolution[1], 1])


def time_distributed(cls, in_axes=1, axis=1):
    """Wrapper for time-distributed (vmapped) application of a module."""
    return nn.vmap(
        cls, in_axes=in_axes, out_axes=axis, axis_name="time",
        # Stack debug vars along sequence dim and broadcast params.
        variable_axes={
            "params": None, "intermediates": axis, "batch_stats": None},
        split_rngs={"params": False, "dropout": True, "state_init": True})


def broadcast_across_batch(inputs: Array, batch_size: int) -> Array:
    """Broadcasts inputs across a batch of examples (creates new axis)."""
    return jnp.broadcast_to(
        arr=jnp.expand_dims(inputs, axis=0),
        shape=(batch_size,) + inputs.shape)


def create_gradient_grid(
        samples_per_dim: Sequence[int], value_range: Sequence[float] = (-1.0, 1.0)
) -> Array:
    """Creates a tensor with equidistant entries from -1 to +1 in each dim.

  Args:
    samples_per_dim: Number of points to have along each dimension.
    value_range: In each dimension, points will go from range[0] to range[1]

  Returns:
    A tensor of shape [samples_per_dim] + [len(samples_per_dim)].
  """
    s = [jnp.linspace(value_range[0], value_range[1], n) for n in samples_per_dim]
    pe = jnp.stack(jnp.meshgrid(*s, sparse=False, indexing="ij"), axis=-1)
    return jnp.array(pe)


def convert_to_fourier_features(inputs: Array, basis_degree: int) -> Array:
    """Convert inputs to Fourier features, e.g. for positional encoding."""

    # inputs.shape = (..., n_dims).
    # inputs should be in range [-pi, pi] or [0, 2pi].
    n_dims = inputs.shape[-1]

    # Generate frequency basis.
    freq_basis = jnp.concatenate(  # shape = (n_dims, n_dims * basis_degree)
        [2 ** i * jnp.eye(n_dims) for i in range(basis_degree)], 1)

    # x.shape = (..., n_dims * basis_degree)
    x = inputs @ freq_basis  # Project inputs onto frequency basis.

    # Obtain Fourier features as [sin(x), cos(x)] = [sin(x), sin(x + 0.5 * pi)].
    return jnp.sin(jnp.concatenate([x, x + 0.5 * jnp.pi], axis=-1))


def prepare_images_for_logging(
        config: ml_collections.ConfigDict,
        batch: Optional[Dict[str, ArrayTree]] = None,
        preds: Optional[Dict[str, ArrayTree]] = None,
        n_samples: int = 5,
        n_frames: int = 5,
        min_n_colors: int = 1,
        epsilon: float = 1e-6,
        first_replica_only: bool = False) -> Dict[str, Array]:
    """Prepare images from batch and/or model predictions for logging."""

    images = dict()
    # Converts all tensors to numpy arrays to run everything on CPU as JAX
    # eager mode is inefficient and because memory usage from these ops may
    # lead to OOM errors.
    batch = jax.tree_map(np.array, batch)
    preds = jax.tree_map(np.array, preds)

    if n_samples <= 0:
        return images

    if not first_replica_only:
        # Move the two leading batch dimensions into a single dimension. We do this
        # to plot the same number of examples regardless of the data parallelism.
        batch = jax.tree_map(lambda x: np.reshape(x, (-1,) + x.shape[2:]), batch)
        preds = jax.tree_map(lambda x: np.reshape(x, (-1,) + x.shape[2:]), preds)
    else:
        batch = jax.tree_map(lambda x: x[0], batch)
        preds = jax.tree_map(lambda x: x[0], preds)

    # Limit the tensors to n_samples and n_frames.
    batch = jax.tree_map(
        lambda x: x[:n_samples, :n_frames] if x.ndim > 2 else x[:n_samples],
        batch)
    preds = jax.tree_map(
        lambda x: x[:n_samples, :n_frames] if x.ndim > 2 else x[:n_samples],
        preds)

    # Log input data.
    if batch is not None:
        images["video"] = video_to_image_grid(batch["video"])
        if "segmentations" in batch:
            images["mask"] = video_to_image_grid(convert_categories_to_color(
                batch["segmentations"], min_n_colors=min_n_colors))
        if "flow" in batch:
            images["flow"] = video_to_image_grid(batch["flow"])
        if "boxes" in batch:
            images["boxes"] = draw_bounding_boxes(
                batch["video"],
                batch["boxes"],
                min_n_colors=min_n_colors)

    # Log model predictions.
    if preds is not None and preds.get("outputs") is not None:
        if "segmentations" in preds["outputs"]:  # pytype: disable=attribute-error
            images["segmentations"] = video_to_image_grid(
                convert_categories_to_color(
                    preds["outputs"]["segmentations"], min_n_colors=min_n_colors))

    def shape_fn(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            return x.shape

    # Log intermediate variables.
    if preds is not None and "intermediates" in preds:

        logging.info("intermediates: %s",
                     jax.tree_map(shape_fn, preds["intermediates"]))

        for key, path in config.debug_var_video_paths.items():
            log_vars = retrieve_from_collection(preds["intermediates"], path)
            if log_vars is not None:
                if not isinstance(log_vars, Sequence):
                    log_vars = [log_vars]
                for i, log_var in enumerate(log_vars):
                    log_var = np.array(log_var)  # Moves log_var to CPU.
                    images[key + "_" + str(i)] = video_to_image_grid(log_var)
            else:
                logging.warning("%s not found in intermediates", path)

        # Log attention weights.
        for key, path in config.debug_var_attn_paths.items():
            log_vars = retrieve_from_collection(preds["intermediates"], path)
            if log_vars is not None:
                if not isinstance(log_vars, Sequence):
                    log_vars = [log_vars]
                for i, log_var in enumerate(log_vars):
                    log_var = np.array(log_var)  # Moves log_var to CPU.
                    images.update(
                        prepare_attention_maps_for_logging(
                            attn_maps=log_var,
                            key=key + "_" + str(i),
                            map_width=config.debug_var_attn_widths.get(key),
                            video=batch["video"],
                            epsilon=epsilon,
                            n_samples=n_samples,
                            n_frames=n_frames))
            else:
                logging.warning("%s not found in intermediates", path)

    # Crop each image to a maximum of 3 channels for RGB visualization.
    for key, image in images.items():
        if image.shape[-1] > 3:
            logging.warning("Truncating channels of %s for visualization.", key)
            images[key] = image[..., :3]

    return images


def prepare_attention_maps_for_logging(attn_maps: Array, key: str,
                                       map_width: int, epsilon: float,
                                       n_samples: int, n_frames: int,
                                       video: Array):
    """Visualize (overlayed) attention maps as an image grid."""
    images = {}  # Results dictionary.
    attn_maps = unflatten_image(attn_maps[..., None], width=map_width)

    num_heads = attn_maps.shape[2]
    for head_idx in range(num_heads):
        attn = attn_maps[:n_samples, :n_frames, head_idx]
        attn /= attn.max() + epsilon  # Standardizes scale for visualization.
        # attn.shape: [bs, seq_len, 11, h', w', 1]

        bs, seq_len, _, h_attn, w_attn, _ = attn.shape
        images[f"{key}_head_{head_idx}"] = video_to_image_grid(attn)

        # Attention maps are interpretable when they align with object boundaries.
        # However, if they are overly smooth then the following visualization which
        # overlays attention maps on video is helpful.
        video = video[:n_samples, :n_frames]
        # video.shape: [bs, seq_len, h, w, 3]
        video_resized = []
        for i in range(n_samples):
            for j in range(n_frames):
                video_resized.append(
                    skimage.transform.resize(video[i, j], (h_attn, w_attn), order=1))
        video_resized = np.array(video_resized).reshape(
            (bs, seq_len, h_attn, w_attn, 3))
        attn_overlayed = attn * np.expand_dims(video_resized, 2)
        images[f"{key}_head_{head_idx}_overlayed"] = video_to_image_grid(
            attn_overlayed)

    return images


def convert_categories_to_color(
        inputs: Array, min_n_colors: int = 1, include_black: bool = True) -> Array:
    """Converts int-valued categories to color in last axis of input tensor.

  Args:
    inputs: `np.ndarray` of arbitrary shape with integer entries, encoding the
      categories.
    min_n_colors: Minimum number of colors (excl. black) to encode categories.
    include_black: Include black as 0-th entry in the color palette. Increases
      `min_n_colors` by 1 if True.

  Returns:
    `np.ndarray` with RGB colors in last axis.
  """
    if inputs.shape[-1] == 1:  # Strip category axis.
        inputs = np.squeeze(inputs, axis=-1)
    inputs = np.array(inputs, dtype=np.int32)  # Convert to int.

    # Infer number of colors from inputs.
    n_colors = int(inputs.max()) + 1  # One color per category incl. 0.
    if include_black:
        n_colors -= 1  # If we include black, we need one color less.

    if min_n_colors > n_colors:  # Use more colors in color palette if requested.
        n_colors = min_n_colors

    rgb_colors = get_uniform_colors(n_colors)

    if include_black:  # Add black as color for zero-th index.
        rgb_colors = np.concatenate((np.zeros((1, 3)), rgb_colors), axis=0)
    return rgb_colors[inputs]


def get_uniform_colors(n_colors: int) -> Array:
    """Get n_colors with uniformly spaced hues."""
    hues = np.linspace(0, 1, n_colors, endpoint=False)
    hsv_colors = np.concatenate(
        (np.expand_dims(hues, axis=1), np.ones((n_colors, 2))), axis=1)
    rgb_colors = matplotlib.colors.hsv_to_rgb(hsv_colors)
    return rgb_colors  # rgb_colors.shape = (n_colors, 3)


def unflatten_image(image: Array, width: Optional[int] = None) -> Array:
    """Unflatten image array of shape [batch_dims..., height*width, channels]."""
    n_channels = image.shape[-1]
    # If width is not provided, we assume that the image is square.
    if width is None:
        width = int(np.floor(np.sqrt(image.shape[-2])))
        height = width
        assert width * height == image.shape[-2], "Image is not square."
    else:
        height = image.shape[-2] // width
    return image.reshape(image.shape[:-2] + (height, width, n_channels))


def video_to_image_grid(video: Array) -> Array:
    """Transform video to image grid by folding sequence dim along width."""
    if len(video.shape) == 5:
        n_samples, n_frames, height, width, n_channels = video.shape
        video = np.transpose(video, (0, 2, 1, 3, 4))  # Swap n_frames and height.
        image_grid = np.reshape(
            video, (n_samples, height, n_frames * width, n_channels))
    elif len(video.shape) == 6:
        n_samples, n_frames, n_slots, height, width, n_channels = video.shape
        # Put n_frames next to width.
        video = np.transpose(video, (0, 2, 3, 1, 4, 5))
        image_grid = np.reshape(
            video, (n_samples, n_slots * height, n_frames * width, n_channels))
    else:
        raise ValueError("Unsupported video shape for visualization.")
    return image_grid


def draw_bounding_boxes(video: Array,
                        boxes: Array,
                        min_n_colors: int = 1,
                        include_black: bool = True) -> Array:
    """Draw bounding boxes in videos."""
    colors = get_uniform_colors(min_n_colors - include_black)

    b, t, h, w, c = video.shape
    n = boxes.shape[2]
    image_grid = tf.image.draw_bounding_boxes(
        np.reshape(video, (b * t, h, w, c)),
        np.reshape(boxes, (b * t, n, 4)),
        colors).numpy()
    image_grid = np.reshape(
        np.transpose(np.reshape(image_grid, (b, t, h, w, c)),
                     (0, 2, 1, 3, 4)),
        (b, h, t * w, c))
    return image_grid


def plot_image(ax: plt.Axes, image: Array):
    """Add an image visualization to a provided `plt.Axes` instance."""
    num_channels = image.shape[-1]
    if num_channels == 1:
        image = image.reshape(image.shape[:2])
    ax.imshow(image, cmap="viridis")
    ax.grid(False)
    plt.axis("off")


def visualize_image_dict(images: Dict[str, Array], plot_scale: int = 10):
    """Visualize a dictionary of images in colab using maptlotlib."""

    for key in images.keys():
        logging.info("Visualizing key: %s", key)
        n_images = len(images[key])
        fig = plt.figure(figsize=(n_images * plot_scale, plot_scale))
        for idx, image in enumerate(images[key]):
            ax = fig.add_subplot(1, n_images, idx + 1)
            plot_image(ax, image)
        plt.show()


def filter_key_from_frozen_dict(
        frozen_dict: flax.core.FrozenDict, key: str) -> flax.core.FrozenDict:
    """Filters (removes) an item by key from a flax.core.FrozenDict."""
    if key in frozen_dict:
        frozen_dict, _ = frozen_dict.pop(key)
    return frozen_dict


def prepare_dict_for_logging(nested_dict: Dict[str, Any], parent_key: str = "",
                             sep: str = "_") -> Dict[str, Any]:
    """Prepare a nested dictionary for logging with `clu.metric_writers`.

  Args:
    nested_dict: A nested dictionary, e.g. obtained from a
      `ml_collections.ConfigDict` via `.to_dict()`.
    parent_key: String used in recursion.
    sep: String used to separate parent and child keys.

  Returns:
    Flattened dict.
  """
    items = []
    for k, v in nested_dict.items():
        # Flatten keys of nested elements.
        new_key = parent_key + sep + k if parent_key else k

        # Convert None values, lists and tuples to strings.
        if v is None:
            v = "None"
        if isinstance(v, list) or isinstance(v, tuple):
            v = str(v)

        # Recursively flatten the dict.
        if isinstance(v, dict):
            items.extend(prepare_dict_for_logging(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def retrieve_from_collection(
        variable_collection: ArrayTree, path: str) -> ArrayTree:
    """Finds variables by their path by recursively searching the collection.

  Args:
    variable_collection: Nested dict containing the variables (or tuples/lists
      of variables).
    path: Path to variable in module tree, similar to Unix file names (e.g.
      '/module/dense/0/bias').

  Returns:
    The requested variable, variable collection or None (in case the variable
      could not be found).
  """
    key, _, rpath = path.strip("/").partition("/")

    # In case the variable is not found, we return None.
    if (key.isdigit() and not isinstance(variable_collection, Sequence)) or (
            key.isdigit() and int(key) >= len(variable_collection)) or (
            not key.isdigit() and key not in variable_collection):
        return None

    if key.isdigit():
        key = int(key)

    if not rpath:
        return variable_collection[key]
    else:
        return retrieve_from_collection(variable_collection[key], rpath)


def build_model_from_config(config: ml_collections.ConfigDict) -> nn.Module:
    """Build a Flax model from a (nested) ConfigDict."""
    model_constructor = _parse_config(config)
    if callable(model_constructor):
        return model_constructor()
    else:
        raise ValueError("Provided config does not contain module constructors.")


def _parse_config(config: Union[ml_collections.ConfigDict, ConfigAttr]
                  ) -> Union[Callable[..., nn.Module], ConfigAttr]:
    """Recursively parses a nested ConfigDict and resolves module constructors."""

    if isinstance(config, list):
        return [_parse_config(c) for c in config]
    elif isinstance(config, tuple):
        return tuple([_parse_config(c) for c in config])
    elif not isinstance(config, ml_collections.ConfigDict):
        return config
    elif "module" in config:
        module_constructor = _resolve_module_constructor(config.module)
        kwargs = {k: _parse_config(v) for k, v in config.items() if k != "module"}
        return functools.partial(module_constructor, **kwargs)
    else:
        return {k: _parse_config(v) for k, v in config.items()}


def _resolve_module_constructor(
        constructor_str: str) -> Callable[..., nn.Module]:
    import_str, _, module_name = constructor_str.rpartition(".")
    py_module = importlib.import_module(import_str)
    return getattr(py_module, module_name)


def get_slices_along_axis(
        inputs: Dict[str, Array],
        slice_keys: Sequence[str],
        start_idx: int = 0,
        end_idx: int = -1,
        axis: int = 2,
        pad_value: int = 0) -> Dict[str, Array]:
    """Extracts slices from a dictionary of tensors along the specified axis.

  The slice operation is only applied to `slice_keys` dictionary keys. If
  `end_idx` is larger than the actual size of the specified axis, padding is
  added (with values provided in `pad_value`).

  Args:
    inputs: Dictionary of tensors.
    slice_keys: Iterable of strings, the keys for the inputs dictionary for
      which to apply the slice operation.
    start_idx: Integer, defining the first index to be part of the slice.
    end_idx: Integer, defining the end of the slice interval (exclusive). If set
      to `-1`, the end index is set to the size of the axis. If a value is
      provided that is larger than the size of the axis, zero-padding is added
      for the remaining elements.
    axis: Integer, the axis along which to slice.
    pad_value: Integer, value to be used in padding.

  Returns:
    Dictionary of tensors where elements described in `slice_keys` are sliced,
      and all other elements are returned as original.
  """

    max_size = None
    pad_size = 0

    # Check shapes and get maximum size of requested axis.
    for key in slice_keys:
        curr_size = inputs[key].shape[axis]
        if max_size is None:
            max_size = curr_size
        elif max_size != curr_size:
            raise ValueError(
                "For specified tensors the requested axis needs to be of equal size.")

    # Infer end index if not provided.
    if end_idx == -1:
        end_idx = max_size

    # Set padding size if end index is larger than maximum size of requested axis.
    elif end_idx > max_size:
        pad_size = end_idx - max_size
        end_idx = max_size

    outputs = {}
    for key in slice_keys:
        outputs[key] = np.take(
            inputs[key], indices=np.arange(start_idx, end_idx), axis=axis)

        # Add padding if necessary.
        if pad_size > 0:
            pad_shape = np.array(outputs[key].shape)
            np.put(pad_shape, axis, pad_size)  # In-place op.
            padding = pad_value * np.ones(pad_shape, dtype=outputs[key].dtype)
            outputs[key] = np.concatenate((outputs[key], padding), axis=axis)

    return outputs


def get_element_by_str(
        dictionary: Dict[str, Any], multilevel_key: str, separator: str = "/"
) -> Any:
    """Gets element in a dictionary with multilevel key (e.g., "key1/key2")."""
    keys = multilevel_key.split(separator)
    if len(keys) == 1:
        return dictionary[keys[0]]
    return get_element_by_str(
        dictionary[keys[0]], separator.join(keys[1:]), separator=separator)


def set_element_by_str(
        dictionary: Dict[str, Any], multilevel_key: str, new_value: Any,
        separator: str = "/"):
    """Sets element in a dictionary with multilevel key (e.g., "key1/key2")."""
    keys = multilevel_key.split(separator)
    if len(keys) == 1:
        if keys[0] not in dictionary:
            key_error = (
                "Pretrained {key} was not found in trained model. "
                "Make sure you are loading the correct pretrained model "
                "or consider adding {key} to exceptions.")
            raise KeyError(key_error.format(type="parameter", key=keys[0]))
        dictionary[keys[0]] = new_value
    else:
        set_element_by_str(
            dictionary[keys[0]],
            separator.join(keys[1:]),
            new_value,
            separator=separator)


def remove_singleton_dim(inputs: Optional[Array]) -> Optional[Array]:
    """Removes the final dimension if it is singleton (i.e. of size 1)."""
    if inputs is None:
        return None
    if inputs.shape[-1] != 1:
        logging.warning("Expected final dimension of inputs to be 1, "
                        "received inputs of shape %s: ", str(inputs.shape))
        return inputs
    return inputs[..., 0]


def subplot_image(ax, img, label=None):
    ax.imshow(img)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    if label:
        # ax.set_title(label, fontsize=3, y=-21)
        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.set_xlabel(label, fontsize=3)
        ax.axis('on')


def viz_slots_flow(
        vid: Array,
        gt_flow: Array,
        gt_seg: Array,
        att: Array,
        output_fn: Dict[str, Array],
        num_objects=24,
        trunk=None):
    """
    Plot the video and slots
    Args:
        vid,(L H W C)
        vid:
        gt_flow:
        gt_seg:
        att:
        output_fn:
        num_objects:
        trunk: truncate temporal dim for viz clarity
    """
    if trunk is None:
        trunk = len(vid)
    T = min(len(vid), trunk)
    n_objs = att.shape[1]
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)

    # The attention mask highlights the visible part of the image.
    slots = vid[:, np.newaxis, :, :, :] * att[:, :, :, :, np.newaxis] + (1 - att[:, :, :, :, np.newaxis])
    slots = jnp.clip(slots, 0, 1)
    plt.close()

    # set the figure size and dpi
    plt.rcParams['axes.linewidth'] = 0.1
    if n_objs > 12:
        fig, ax = plt.subplots(T, n_objs + 3, dpi=400)
    else:
        fig, ax = plt.subplots(T, n_objs + 3, dpi=400, figsize=(3, 5))

    plt.subplots_adjust(wspace=0.01, hspace=0.1)
    for t in range(T):
        subplot_image(ax[t, 0], vid[t], 'frame')
        subplot_image(ax[t, 1], gt_flow[t], 'gt_fl')
        rgb_seg = np.expand_dims(gt_seg[t], axis=-1)
        rgb_seg = segmentation_to_rgb(rgb_seg, num_objects=num_objects)
        subplot_image(ax[t, 2], rgb_seg, 'gt_sg')
        for obj in range(3, n_objs + 3):
            subplot_image(ax[t, obj], slots[t, obj - 3], f'slot {obj - 2}')

    plt.savefig(output_fn)


def resize_channel(channel: Array, new_shape):
    # channel has shape [height, width]
    # Resize the channel using image.resize
    return IM.resize(channel, new_shape, method='bilinear')


# 定义一个函数来处理多通道图像
def resize_image(img, new_shape):
    # img has shape [channels, height, width]
    # Apply resize_channel to each channel using vmap
    # vmap maps over the 0th axis (channels) of img
    return vmap(resize_channel, in_axes=(0, None))(img, new_shape)


def viz_slot_gif(vid, att, output_fn, trunk=None, send_to_wandb=False):
    # Assume data is an array of shape [T, subimage_nums, h, w, c]
    # Here we use random data as a placeholder for real input
    if trunk is None:
        trunk = len(vid)
    T = min(len(vid), trunk)

    # data = att
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)

    # for movi-e
    data = vid[:, np.newaxis, :, :, :] * att[:, :, :, :, np.newaxis] + (1 - att[:, :, :, :, np.newaxis])
    data = jnp.clip(data, 0, 1)
    plt.close()
    subimage_nums = data.shape[1] # Number of sub-images (slots)
    h, w, c = data.shape[-3:]

    # Determine the number of rows and columns based on subimage_nums
    num_columns = 3
    num_rows = 4
    if subimage_nums < 15:
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 9))
    else:
        num_columns = 6
        num_rows = 4
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 10))


    for i in range(num_rows):
        for j in range(num_columns):
            if i * num_columns + j >= subimage_nums:
                axes[i, j].axis('off')
            else:
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

    #  Initialize an image object for each subimage
    images = [axes[i, j].imshow(np.zeros((h, w, c))) for i in range(num_rows) for j in range(num_columns) if
              i * num_columns + j < subimage_nums]

    # Set the title for each subplot
    for i, ax in enumerate(axes.flatten()):
        if i < subimage_nums:
            ax.set_title(f'slot {i + 1}')

    # Initialization function for displaying empty images
    def init():
        for img in images:
            img.set_array(np.zeros((h, w, c)))
        return images

    # Update function to refresh image content
    def update(frame):
        for i, img in enumerate(images):
            img.set_array(data[frame, i])
        return images

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1, wspace=0.1)
    # Create the animation object
    ani = FuncAnimation(fig, update, frames=range(T), init_func=init, blit=True)

    # Save as GIF
    ani.save(output_fn, writer='pillow', fps=12)


def viz_video(vid, output_fn, trunk=None, send_to_wandb=False):
    # Assume video is an array of shape [T, h, w, c]
    # Here we use random data as a placeholder for real video input

    T = len(vid)
    h, w, c = vid.shape[1:]
    video = vid
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 9))

    img = ax.imshow(np.zeros((h, w, c)))

    ax.set_xticks([])
    ax.set_yticks([])

    def init():
        img.set_array(np.zeros((h, w, c)))
        return [img]

    def update(frame):
        img.set_array(video[frame])
        return [img]

    ani = FuncAnimation(fig, update, frames=range(T), init_func=init, blit=True)

    ani.save(output_fn, writer='pillow', fps=12)


def viz_segs(prds, output_fn, index=3, max_num_instances=24):
    def plot_images(prds, output_fn, index, max_num_instances=24):
        # Assume you have an array named video_data with shape (24, 128, 128, 3)
        plt.close()
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        rows = 1
        columns = len(prds) // index
        if columns == 6:
            fig, axes = plt.subplots(rows, columns, figsize=(10, 2))
        elif columns == 8:
            fig, axes = plt.subplots(rows, columns, figsize=(15, 2))
        else:
            fig, axes = plt.subplots(rows, columns, figsize=(20, 2))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1, wspace=0.02)

        video_data = prds
        for i in range(video_data.shape[0]):
            if i % index == 0:
                frame = video_data[i]  # get the current frame
                frame = np.expand_dims(frame, axis=-1)
                frame = segmentation_to_rgb(frame, num_objects=max_num_instances)
                # ax = axes[i // 6, i % 6]  # select the subplot
                ax = axes[i // index]
                ax.imshow(frame)
                ax.axis('off')

        plt.savefig(output_fn)

    plot_images(prds, output_fn, index, max_num_instances)


def segmentation_to_rgb(seg, palette=None, num_objects=None, bg_color=(0, 0, 0)):
    # If the number of objects is not provided,
    # assume they are sequentially labeled and find the maximum label
    if num_objects is None:
        num_objects = np.max(seg)  # find the maximum label in the segmentation

    # If no color palette is provided, use the 'hls' palette from the seaborn library
    if palette is None:
        palette = [bg_color] + sns.color_palette('hls', num_objects - 1)

    # Create a zero-initialized array of shape (height, width, 3) to store the colored image
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.float32)

    # Iterate over each object label and assign the corresponding color to the color image
    for i in range(num_objects):
        seg_img[np.nonzero(seg[:, :, 0] == i)] = palette[i]

    return seg_img

