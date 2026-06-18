# Image Retrieval with Metric Learning and CNNs  
(2024-2 Machine Learning Term Project)

This project was conducted as part of a university machine learning course.

* * *

## 1. Introduction

This project compares different sampling strategies and loss functions for image retrieval based on metric learning.

We evaluate how embedding learning with CNN-based feature extractors performs under different training settings.

---

## 2. Method

![process](https://github.com/user-attachments/assets/a45c99b4-4aa6-4aed-87e8-20f74cf043af)

Metric learning is a method that projects input features into an embedding space where similarity or distance can be measured to learn relationships between data points.

In this experiment, features are extracted using a ResNet backbone, followed by metric learning in the embedding space.

Cosine similarity is used to measure distances between embeddings.

We evaluate two loss functions:
1. Triplet Loss  
2. Margin-based Loss  

---

## 3. Experiments

### 3.1 Dataset

We use the Stanford Online Products (SOP) dataset, which consists of 12 product categories and 22,634 product images.

The dataset is split into:

- Training set: 51,085 images  
- Validation set: 8,466 images  
- Query set: 11,317 images  
- Gallery set: 49,186 images  

To construct valid positive and negative pairs for Triplet Loss, each mini-batch contains at least 2–4 images per class.

We further filter classes with at least 6 images per subclass for stable sampling.

The remaining images are split into training, validation, query, and gallery sets accordingly.

---

### 3.2 Experimental Setup

- Backbone: ResNet-50 pretrained on ImageNet-21k  
- Optimizer: Adam  
- Augmentation: RandomResizedCrop, RandomHorizontalFlip  
- Learning rate: 1e-5 for first 30 epochs, then decayed to 3e-6  
- Weight decay (L2 regularization): 4e-5  
- Output layer learning rate: 2× base learning rate  
- Training epochs: 40  

---

### 3.3 Results

We evaluate four combinations of loss functions and sampling strategies:

| Method | d = 128 | d = 256 | d = 512 |
|--------|--------:|--------:|--------:|
| Triplet loss + random sampling | 72.29 | 74.45 | 76.03 |
| Triplet loss + semi-hard sampling | 72.78 | 74.50 | 75.89 |
| Margin loss + random sampling | 68.43 | 69.22 | 70.13 |
| Margin loss + distance weighted sampling | 78.07 | 79.58 | 79.93 |

---

## 4. Conclusion

This study demonstrates that metric learning performance in image retrieval is highly influenced by the choice of loss function and sampling strategy.

In particular, margin-based loss with distance-weighted sampling achieves the best overall performance.

---

## 5. References

[1] He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.  
[2] Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR 2015.  
[3] Wu et al., *Sampling Matters in Deep Embedding Learning*, ICCV 2017.  
[4] https://github.com/rksltnl/Deep-Metric-Learning-CVPR16  
[5] https://github.com/Confusezius/Deep-Metric-Learning-Baselines  

---

## Note

This project was conducted as a 2024-2 Machine Learning term project at university.
