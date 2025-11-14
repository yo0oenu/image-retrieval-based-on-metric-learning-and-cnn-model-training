# [24-2학기 기계학습 텀프로젝트]
#### 주제: Image Retrieval Based on Metric Learning and CNN Model Training
###### 전기정보공학과 조원호      
###### 글로벌테크노경영학과 신연우
* * *

## 1. Introduction
Metric Learning을 기반 다양한 sampling 및 Loss 조합에서의 Image Retrieval 비교.  

  
## 2. Method

![process](https://github.com/user-attachments/assets/a45c99b4-4aa6-4aed-87e8-20f74cf043af)


* Metric Learning: Input Feature를 특정 임베딩 공간으로 projection 하여 유사성(Similarity) 혹은, 거리(Distance) 기반으로 데이터의 관계를 학습시키는 방법이다.
* 해당 실험은 RestNet backbone을 통해 Feature 추출 이후, Metric Learning을 진행하였다.
* 임베딩 공간에서 유사도 측정은 Cosine Similarity를 사용하였다.
* 두 가지 Loss 함수를 사용하여 실험을 진행하였다.    
  1: Triplet Loss  
  2: Margin Based Loss 


## 3. Experiments

### 1. 데이터셋 구성

   * Stanford Online Products(SOP) 데이터셋을 사용하였다.
   * 쇼핑몰의 12가지 상품 카테고리와 각 카테고리에 해당하는 22634개의 제품 이미지로 구성되어 있다.
   *  59,551장의 학습 이미지 / 60,502장의 테스트 이미지를 포함하고 있다.
   * 학습 데이터셋: 51,085 / 검증 데이터셋: 8,466 / 쿼리 데이터셋: 11,317 /   
   * 데이터베이스 데이터셋: 49,186
     
> &nbsp; Triplet loss 함수에 필요한 양성 샘플과 음성 샘플을 얻기 위해, 매 미니배치마다 클래스별 이미지가 적어도 2~4장 이상 필요하다. 따라서 기존의 학습 이미지에서 각 하위 클래스별로 6장 이상의 이미지가 있는 클래스들만 필터링 하고, 필터링 된 클래스의 이미지 중에서 2장씩 추출하여 검증 데이터셋을 구성하였다. 이후 남은 이미지들로 학습 데이터셋을 구성하였다. <br>
 테스트 데이터셋 분할의 경우, 각 하위 클래스별로 이미지가 2장 있는 클래스에 대해서만 이미지를 한 장씩 추출하여 쿼리 데이터셋을 구성 하였고, 나머지 이미지들로 데이터베이스 데이터셋을 구성하였다.
 
### 2. 실험 환경
   * 하드웨어: 2개의 NVIDIA 2080Ti 11GB GPU
   * 소프트웨어: CUDA 11.3, PyTorch 1.12.1
   * 모델: PyTorch에서 제공하는 ImageNet-21k 데이터셋으로 pre-trained 된 ResNet50 모델의 가중치를 사용하였다.  

### 3. 실험 결과
>  &nbsp;&nbsp; [2]와 [3]의 논문 결과를 확인하고자 1. Triplet Loss + Semi-hard Negative Sampling 2. Triplet Loss + Random Sampling 3. Margin Based Loss + Distance  weighted Sampling 4. Margin Based Loss + Random Sampling, 총 4가지 조합의 실험을 진행하였다.  <br>
&nbsp;&nbsp; 모든 세팅은 총 40 epoch동안 학습
![그래프](https://github.com/user-attachments/assets/390818a8-9d11-4155-bb94-14f186ac4f82)
* Augmentation:  RandomResizedCrop, RandomHorizontalFlip
* Optimizer: Adam  
* Learning Late: 30 epoch 이전까지는 lr = 1e-5, 이후 lr은 70% 감소하여 3e-6을 유지하도록 설정하였으며, λ = 4e-5의 L2-regulization. (출력 레이어의 학습률은 base lr의 2배로, λ는 기존의 0.5배)

       
| Method                                  | d = 128 | d = 256 | d = 512 |
|-----------------------------------------|---------|---------|---------|
| triplet loss + random sampling         | 72.29   | 74.45   | 76.03   |
| triplet loss + semi-hard sampling      | 72.78   | 74.50   | 75.89   |
| margin loss + random sampling          | 68.43   | 69.22   | 70.13   |
| margin loss + distance weighted sampling | 78.07   | 79.58   | 79.93   |




**4. 참고문헌 및 사이트**  <br>
[1] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. Shen, Shufan, et al. <br>
[2] Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015. <br>
[3] Wu, Chao-Yuan, et al. "Sampling matters in deep embedding learning." Proceedings of the IEEE international conference on computer vision. 2017.  <br>
[4] [Deep Metric Learning CVPR16](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16)  
[5] [Deep Metric Learning Baselines](https://github.com/Confusezius/Deep-Metric-Learning-Baselines)  
