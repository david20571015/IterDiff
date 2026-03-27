# IterDiff (ICIP 2025)

**IterDiff: Training-Free Iterative Face Editing via Efficient CLIP-guided Memory Bank**

*Chun-Yao Chiu<sup>1</sup>, Feng-Kai Huang<sup>2</sup>, Teng-Fang Hsiao<sup>1</sup>, Hong-Han Shuai<sup>1</sup>, Wen-Huang Cheng<sup>2</sup>*

*<sup>1</sup>Institute of Electrical and Computer Engineering, National Yang Ming Chiao Tung University*
*<sup>2</sup>Department of Computer Science and Information Engineering, National Taiwan University*

---

## Project Overview

This repository provides inference and evaluation pipelines for IterDiff and related methods
(`ip2p`, `scfg`, `iterdiff`, and `emilie`). It includes:

- Multi-step face editing inference
- Attention control and memory bank mechanisms
- Quantitative evaluation with CLIP-I, LPIPS, and ImageReward

## Environment Setup

Using Conda is recommended (`environment.yml` is provided, Python 3.12):

```bash
conda env create -f environment.yml
conda activate iterdiff
```

## Data Preparation

This project uses the FFHQ download script from [StyleGAN2](https://github.com/NVlabs/stylegan2).

```bash
cd datasets/ffhq
python download_ffhq.py --images
```

After downloading, make sure images are located at:

- `datasets/ffhq/images1024x1024/`

## Quick Start

1. Run Inference

    Use `script_eval.py` to generate multi-step editing results.

    Example (IterDiff):

    ```bash
    python script_eval.py \
     --type iterdiff \
     --mb_size 30 \
     --mb_save_topk 20 \
     --exp_title exp_iterdiff \
     --iter_edit_bench iter_edit_bench_1000.json \
     --results_dir results
    ```

2. Compute Metrics

    Use `compute_metric.py`:
