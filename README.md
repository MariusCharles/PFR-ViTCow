# ViTCow Project  
**Self-Supervised Cow Behavior Classification from Video**

## Overview

ViTCow explores **self-supervised learning** for cow behavior analysis, addressing major challenges in livestock AI:

- High cost of manual annotation  
- Limited labeled data  
- Poor cross-farm generalization  

**Approach:**  
We pre-train a **VideoMAE Vision Transformer** on large-scale *unlabeled farm videos* to learn robust spatiotemporal representations of cow movements. The pretrained model can then be **fine-tuned with minimal labeled data** for downstream behavior classification.

**Goal:**  
Develop a foundation model that generalizes across farms and camera setups.

**Dataset:**  
288 hours of RGB video from **5 farms**, **6 cameras per farm**.

---

## Project Components

### dataset_creation/

Automated pipeline to generate cow-centered training clips:

- Download raw SFTP videos  
- Split recordings into short clips  
- Detect & track cows using **SAM3** (zero-shot text prompting)  
- Produce **224×224 cropped videos** centered on individual cows  
- Upload processed clips  


---

### videomae_training/

Self-supervised pretraining using VideoMAE:

- Prepare dataset splits  
- Configure VideoMAE v1  
- Distributed GPU training  
- Output = pretrained encoder  


---

## Pipeline

Raw Videos → Cow Segmentation & Tracking → Cow-Centered Clips → VideoMAE Pretraining → Pretrained Model → Fine-Tuning → Behavior Classifier → Model Evaluation

---

## Team

Marius Charles - Mafalda Frere - Damien Glaizal - Marie Meier - Charlotte Tibi  

**Supervisors:** Joseph Allyndrée · Antoine Cornuéjols · Christine Martin

---

## References

- [SAM3 (Meta)](https://github.com/facebookresearch/sam3)
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE) 