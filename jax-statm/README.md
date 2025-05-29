# **STATM-SAVi**
**Reasoning-Enhanced Object-Centric Learning for Videos**

**[📄 Paper Link ](https://dl.acm.org/doi/10.1145/3690624.3709168)**  <!-- TODO: Replace with actual paper link -->
||**[📄 arXiv ](https://arxiv.org/abs/2403.15245v2)**

---

## **INTRODUCTION**

This repository contains the **JAX implementation** of **STATM-SAVi**, primarily demonstrating training results on the [MOVi dataset](https://console.cloud.google.com/storage/browser/kubric-public/tfds?pli=1&inv=1&invt=Abyp-w).

---

## **INSTALLATION & TRAINING**

> ⚠️ **Note:** This project provides **two environments**:
>
> - **Environment 1:** Compatible with **RTX 3090**, **RTX 4090**, and **A100**  
> - **Environment 2:** Compatible with **RTX 3090**, **RTX 4090**, **A100**, and **H800**


Use `conda` to create the environment from `environment1.yml`:

```bash
conda env create -f environment1.yml
conda activate jax_statm
```
---

To train the smallest **STATM-SAVi** model on the [MOVi-A dataset](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md):

```bash
python -m savi.main --config savi/configs/movi/savi_conditional_small.py --workdir samll_temp/
```
