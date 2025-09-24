# [24-2학기 기계학습 텀프로젝트]
#### 주제: Image Retrieval Based on Metric Learning and CNN Model Training
###### 전기정보공학과 조원호      
###### 글로벌테크노경영학과 신연우
* * *

**1. Introduction**

&nbsp;&nbsp;본 프로젝트에서는 Metric Learning을 기반으로 CNN모델을 학습하여 이미지 기반 이미지 검색을 구현하였다.  

  
**2. Method**

![process](https://github.com/user-attachments/assets/a45c99b4-4aa6-4aed-87e8-20f74cf043af)


&nbsp;Metric Learning은 사용해 데이터를 특정 임베딩 공간으로 변환하여 유사성(Similarity) 혹은, 거리(Distance) 기반으로 데이터의 관계를 학습시키는 방법이다.

* 본 프로젝트에서는 RestNet  모델을 사용하여 입력 이미지로부터 Feature을 추출한 이후, 출력 벡터를 임베딩 벡터로 변환하여 Metric Learning을 진행하였다.

* 임베딩 공간에서 벡터 간의 유사도를 측정하기 위해 Cosine Similarity를 사용하였다.

* 본 프로젝트에서는 두 가지 Loss 함수를 사용하여 실험을 진행하였다.    
[ 1 ]: Triplet Loss  <br>
[ 2 ]: Margin Based Loss 


**3. Experiments**

1. 데이터셋 구성

   * Stanford Online Products(SOP) 데이터셋을 사용하였다.
   * 쇼핑몰의 12가지 상품 카테고리와 각 카테고리에 해당하는 22634개의 제품 이미지로 구성되어 있다.
   *  59,551장의 학습 이미지 / 60,502장의 테스트 이미지를 포함하고 있다.
   * 학습 데이터셋: 51,085 / 검증 데이터셋: 8,466 / 쿼리 데이터셋: 11,317 /   
데이터베이스 데이터셋: 49,186

  > &nbsp;&nbsp;Triplet loss 함수에 필요한 양성 샘플과 음성 샘플을 얻기 위해, 매 미니배치마다 클래스별 이미지가 적어도 2~4장 이상 필요하다. 따라서 기존의 학습 이미지에서 각 하위 클래스별로 6장 이상의 이미지가 있는 클래스들만 필터링 하고, 필터링 된 클래스의 이미지 중에서 2장씩 추출하여 검증 데이터셋을 구성하였다. 이후 남은 이미지들로 학습 데이터셋을 구성하였다. <br>
 &nbsp; 테스트 데이터셋 분할의 경우, 각 하위 클래스별로 이미지가 2장 있는 클래스에 대해서만 이미지를 한 장씩 추출하여 쿼리 데이터셋을 구성 하였고, 나머지 이미지들로 데이터베이스 데이터셋을 구성하였다. <br>
 
2. 실험 환경

   * 하드웨어: 2개의 NVIDIA 2080Ti 11GB GPU
   * 소프트웨어: CUDA 11.3, PyTorch 1.12.1
   * 모델: PyTorch에서 제공하는 ImageNet-21k 데이터셋으로 pre-trained 된 ResNet50 모델의 가중치를 사용하였다.  
> &nbsp; 이때, 마지막 출력 레이어를 제거하고, 선형 레이어를 추가하였다. 해당 레이어는 편향항이 없으며, He uniform 방식으로 초기화 하였다. <br>
마지막 출력에 L2 norm을 적용하여 출력 벡터의 각 원소들이 [0, 1]의 범위에 속하도록 하였다.

3. 실험 결과
>  &nbsp;&nbsp; [2]와 [3]의 논문 결과를 확인하고자 1. Triplet Loss + Semi-hard Negative Sampling 2. Triplet Loss + Random Sampling 3. Margin Based Loss + Distance  weighted Sampling 4. Margin Based Loss + Random Sampling, 총 4가지 조합의 실험을 진행하였다.  <br>
&nbsp;&nbsp; epoch 수는 모든 세팅에서 동일하게 40으로 설정하였고, 훈련 데이터에서의 Loss는 매 epoch마다, 검증 데이터에서의 Loss는 5 epoch마다 측정하였다. 추가로, 검증 Loss의 값이 5번 이상 감소되지 않는다면 학습이 종료되도록 설정하였다.  
![그래프](https://github.com/user-attachments/assets/390818a8-9d11-4155-bb94-14f186ac4f82)
* 과적합 방지를 위해 50% 확률로 RandomResizedCrop, RandomHorizontalFlip 두 가지 Data Augmentation을 사용하였다.
* Optimizer는 Adam optimimer을 사용  
* 30 epoch 이전까지는 1e-5 학습률을 유지, 이후는 학습률을 70% 감소하여 3e-6을 유지하도록 설정하였으며, λ = 4e-5의 L2 규제항을 적용하였다.
* 사전 훈련된 모델의 가중치보다 새로 추가한 출력 레이어의 가중치를 더 많이 업데이트 해야 하기 때문에 가중치 학습률은 기존의 2배로, λ는 기존의 0.5배로 설정하였다.  
               
| Method                                  | d = 128 | d = 256 | d = 512 |
|-----------------------------------------|---------|---------|---------|
| triplet loss + random sampling         | 72.29   | 74.45   | 76.03   |
| triplet loss + semi-hard sampling      | 72.78   | 74.50   | 75.89   |
| margin loss + random sampling          | 68.43   | 69.22   | 70.13   |
| margin loss + distance weighted sampling | 78.07   | 79.58   | 79.93   |
> 성능 평가 지표로는 Recall@1을 사용하였다. <br>
Recall@k 지표는 쿼리 이미지와 데이터베이스의 모든 이미지들 간의 코사인 유사도를 계산하여, 가장 유사한 이미지 k장을 추출했을 때 해당 이미지가 쿼리 이미지와 동일한 클래스라면 1, 다른 클래스라면 0으로 결과를 낸다. 이때 모든 쿼리 이미지에 대한 해당 결과의 평균값이 Recall@가 된다.
<br>


**4. 참고문헌 및 사이트**  <br>
[1] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. Shen, Shufan, et al. <br>
[2] Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015. <br>
[3] Wu, Chao-Yuan, et al. "Sampling matters in deep embedding learning." Proceedings of the IEEE international conference on computer vision. 2017.  <br>

[4] [Deep Metric Learning CVPR16](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16)  
[5] [Deep Metric Learning Baselines](https://github.com/Confusezius/Deep-Metric-Learning-Baselines)  



