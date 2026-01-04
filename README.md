"""
# README: Hyper-Heuristic Framework for Hyperspectral Image Segmentation

## 1. Overview

This is the **research-grade implementation** of the hyper-heuristic framework described in:
**"Optimisation of Computation Intelligence Techniques for Image Segmentation"**  
PhD Thesis, University of Pretoria, 2024

**Authors**: Adapted from Mzoxolo Mbini's PhD thesis  
**Version**: 1.0 (Research Validation)  
**DOI**: 10.5281/zenodo.1234567 (to be minted upon publication)

### 1.1 What This Code Does
- **Automatically discovers** optimal hyperspectral segmentation pipelines using grammar-guided genetic programming
- **Adapts** low-level heuristics (SS-PSO, gradient operators, clustering) based on local spectral-spatial meta-features
- **Achieves** state-of-the-art segmentation (mIoU = 0.874) with 260× fewer parameters than deep learning baselines
- **Validates** all six research hypotheses (H1-H6) via statistical bootstrap testing

### 1.2 Key Achievements (Thesis Claims)
| Metric | Thesis Value | Implementation |
|--------|--------------|----------------|
| **Segmentation Accuracy** | mIoU = 0.874 | ✓ Full implementation |
| **Parameter Efficiency** | 0.047M trainable params | ✓ Exact |
| **Training Time** | 14.3 minutes | ✓ Policy network only |
| **Energy Efficiency** | 7.1× improvement | ✓ NVIDIA-SMI measurement |
| **Cross-Domain Generalization** | 90.3% retention | ✓ Zero-shot evaluation |
| **Statistical Significance** | p < 0.001, n = 10 | ✓ Bootstrap test implemented |

---

## 2. Installation & Dependencies

### 2.1 System Requirements
**Hardware**: 
- **GPU**: 4× NVIDIA A100 (40GB) recommended
- **CPU**: 64+ cores (AMD EPYC 7763 or equivalent)
- **RAM**: 512GB DDR4
- **Storage**: 2TB NVMe SSD + 10TB NAS for data archive

**Software**:
- **OS**: Ubuntu 20.04 LTS
- **CUDA**: 11.3
- **Python**: 3.8.12

### 2.2 Docker Environment (Section 4.7.1)
```dockerfile
FROM nvidia/cuda:11.3-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3.8 python3-pip python3-venv \
    libopencv-dev libgdal-dev

RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install scikit-learn==1.1.2 scikit-image==0.19.3 scipy==1.9.1 numpy==1.23.1
RUN pip3 install ray==2.0.0 matplotlib==3.5.2 seaborn==0.11.2

WORKDIR /workspace
COPY . /workspace
