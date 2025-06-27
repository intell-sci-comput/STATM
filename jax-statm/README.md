# **STATM-SAVi**
**Reasoning-Enhanced Object-Centric Learning for Videos**

**[📄 Paper Link ](https://dl.acm.org/doi/10.1145/3690624.3709168)**  <!-- TODO: Replace with actual paper link -->
||**[📄 arXiv ](https://arxiv.org/abs/2403.15245v2)**

---

## **INTRODUCTION**

This repository contains the **JAX implementation** of **STATM-SAVi**, primarily demonstrating training results on the [MOVi datasets](https://console.cloud.google.com/storage/browser/kubric-public/tfds?pli=1&inv=1&invt=Abyp-w).

---

### **Installation & Training**

> ⚠️ **Note:** This project provides **two environments**:
>
> - **Environment 1:** Compatible with **RTX 3090**, **RTX 4090**, and **A100**  
> - **Environment 2:** Compatible with **RTX 3090**, **RTX 4090**, **A100**, and **H800**


Use `conda` to create the environment from `environment1.yml`:

```bash
conda env create -f environment1.yml
conda activate statm_a100
```
---

To train the smallest **STATM-SAVi** model on the [MOVi-A dataset](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md) using A100 GPUs.:

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m statm.main --config statm/configs/movi/statm_savi_conditional_small.py --workdir test/
```

We recommend using **Environment 2** if you plan to train your own models, as it is compatible with a wider range of GPUs and CUDA versions.


### **Pre-trained Weight**

We provide all pre-trained weights for **STATM-SAVi (small)** and **STATM-SAVi++ (batch size 32)** on on the MOVi datasets trained under **Environment 1** [here](https://huggingface.co/Kang-na/STATM).

We also provide pre-trained weights for **STATM-SAVi++ (batch size 64)**  trained under **Environment 2** [here](https://huggingface.co/Kang-na/STATM).

> ⚠️ **Important Note:**  
> The weights trained under **Environment 1** and **Environment 2** are **not interchangeable**, due to differences in **JAX versions** required by different GPUs.  
> For example, loading weights across environments may cause errors such as `flax.errors.ScopeParamShapeError: Inconsistent shapes between value and initializer for parameter "scale" in "/vmap(FrameEncoder_0)/ResNet_0/init_bn": (64,), (1, 1, 1, 64).`.  
> Please ensure you switch to the correct environment before loading and evaluating pre-trained weights.

###  Performance Tip: Replace For-Loop with `nn.scan`

In our current implementation, temporal steps `t` are processed using a standard Python `for` loop.  
To achieve significantly faster training and inference speeds, we recommend replacing the loop in `video.py` (lines 90 to 110) with a Flax `nn.scan`-based recurrent module.

You can refer to the [`slot-attention-video`](https://github.com/google-research/slot-attention-video/blob/main/savi/modules/video.py) implementation for a practical example of how to wrap recurrent modules using `nn.scan`.

### **Acknowledgement**
We sincerely thank [slot-attention-video](https://github.com/google-research/slot-attention-video) for open-sourcing their codebase. Our work is primarily built upon their implementation with several key improvements.

