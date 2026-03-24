# [ICLR'26] Loc<sup>2</sup>: Interpretable Cross-View Localization via Depth-Lifted Local Feature Matching
[[`Arxiv`](https://arxiv.org/pdf/2509.09792)][[`BibTeX`](#citation)]

![](overview.png)

## 📝 Abstract
We propose an accurate and interpretable fine-grained cross-view localization method that estimates the 3 Degrees of Freedom (DoF) pose of a ground-level image by matching its local features with a reference aerial image. Unlike prior approaches that rely on global descriptors or bird's-eye-view (BEV) transformations, our method directly learns ground-aerial image-plane correspondences using weak supervision from camera poses. The matched ground points are lifted into BEV space with monocular depth predictions, and scale-aware Procrustes alignment is then applied to estimate camera rotation, translation, and optionally the scale between relative depth and the aerial metric space. This formulation is lightweight, end-to-end trainable, and requires no pixel-level annotations. Experiments show state-of-the-art accuracy in challenging scenarios such as cross-area testing and unknown orientation. Furthermore, our method offers strong interpretability: correspondence quality directly reflects localization accuracy and enables outlier rejection via RANSAC, while overlaying the re-scaled ground layout on the aerial image provides an intuitive visual cue of localization performance.

## 📦 Checkpoints
📁 [**Download pretrained models**](https://drive.google.com/drive/folders/1JQHSxN-IRViKdFri2m9JtLMR_JdFLBIO)

## 🗂️ Data Preparation

### VIGOR
Please download and prepare the VIGOR dataset by following the instructions in the [official repository](https://github.com/Jeff-Zilence/VIGOR/blob/main/data/DATASET.md).

### KITTI
Please download and organize the KITTI dataset according to the directory structure used in [HighlyAccurate](https://github.com/YujiaoShi/HighlyAccurate).

## 📊 Evaluation

## 🚀 Training

## ✅ To-Do

- [x] Initial repo structure
- [x] Evaluation pipeline
- [ ] Pretrained checkpoints
- [ ] Training scripts
- [ ] Visualization tools

## Citation
```bibtex
@inproceedings{xia2026loc,
  title={{Loc}$^2$: Interpretable Cross-View Localization via Depth-Lifted Local Feature Matching},
  author={Xia, Zimin and Xu, Chenghao and Alahi, Alexandre},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
