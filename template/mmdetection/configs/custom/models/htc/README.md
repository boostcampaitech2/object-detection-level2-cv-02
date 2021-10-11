# Results and Models
## Base
| Exp num | Backbone  | RoI Head   | Epoch |initial lr |Lr schd | Optimizer | val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_l | val/bbox_mAP_m | val/bbox_mAP_s | config | checkpoint |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | swin-b | htc | 13 | 1e-4 | step | AdamW | 0.449 | 0.619 | 0.473 | 0.532 | 0.077 | 0.10773 | path | path |
| 2 | swin-b | htc | 14 | 1e-4 | step | AdamW | 0.406 | 0.597 | 0.438 | 0.479 | 0.086 | 0.013 | path | path |
| 3 | swin-b | htc | 17 | 1e-4 | step | AdamW | 0.455 | 0.622 | 0.485 | 0.538 | 0.08 | 0.043 | path | path

## Experiment
inference 에 사용할 가장 좋은 checkpoint 기준 score 작성  
<span style="color:green">1. backbone(모델 크기)에 따른 전체 metric 변화 관찰</span>  
- 실험 가설 : 모델이 커질수록 더 잘 예측할 것이다
- 실험 방법 : backbone 변경
- 결과 : swin-t, swin-b를 비교햇을 때 모델의 크기가 클수록 val/mAP50이 높게 나옴 
- 결과 원인 분석 : 모델의 크기가 커질수록 복잡한 표현을 할 수 있기 때문이다

<span style="color:green">2. FPN을 BiFPN으로 변경</span>  
- 실험 가설 : BiFPN을 통해 low/high level feauter을 잘 섞으면 val/bbox_mAP_S, val/bbox_mAP_M이 개선될 것이다
- 실험 방법 : FPN 구조 변경  
- 결과 : val/mAP50이 오히려 떨어졌음  
- 결과 원인 분석 : htc 모델의 특징으로 여러번 feature을 추출하기 때문에 BiFPN을 사용해도 개선이 없는 것으로 보임

<span style="color:green">3. Mosaic augmentation을 활용하여 1번 모델을 finetuning</span>  
- 실험 가설 : Mosaic Augmentation을 데이터에 적용하여 모델을 학습 시키면 통해 데이터의 수를 증가시키고, 작은 bounding box 이미지를 학습하는 효과를 얻을 수 있을 것이다.
- 실험 방법 : Yolo-v4의 Mosaic augmentation을 데이터셋에 적용  
- 결과 : val/mAP50이 약간 상승했고, test/mAP50이 0.644로 크게 증가했다.
- 결과 원인 분석 : Mosaic augmentation을 통해 모델이 Robust 해진 것으로 보인다.
- 참고 자료 : [Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)


<span style="color:green">3. backbone parameter freeze 에 따른 전체 metric 변화 관찰</span>  
- 실험 가설 : 데이터 갯수가 적으므로 어쩌고 하면 저쩌고 할 것이다. 
- 실험 방법 : backbone parameter freeze 하고 학습   
- 결과 : ~~
- 결과 원인 분석 : ~ 
- 참고 자료 : ~~   

| Exp num | Backbone  | RoI Head   | Epoch |initial lr |Lr schd | Optimizer | Inf time (fps) | val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_l | val/bbox_mAP_m | val/bbox_mAP_s |train/loss_rpn_cls | train/s1.loss_bbox | train/s2.loss_cls | train/s1.acc |train/s2.acc | train/loss |
|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:---------------:|:--------------:|:--------------:|:---------------:|:------------:|:----------:|
| 1      | swin | htc      | 1x  |1.000e-06     |step|  sgd(momentum = 0.9 , ~)       | 42.3   | 42.3   | 42.3   | 42.3   | 42.3   | 37.4    | 37.4   | 37.4 | 37.4  | 37.4  | 37.4  |37.4   | 37.4  | 37.4  |
| 2      | pvt | htc      | 1x  |1.000e-06     |step|  AdamW       | 42.3   | 0.487   | 0.643  |  0.521  | 0.561   | 0.187    | 0.017  | 37.4 | 37.4  | 37.4  | 37.4  |37.4   | 37.4  | 37.4  |
## Leader board 결과(제출했을 시)
| Exp num | Public LB map  | 
|:-------:|:---------:|
| 1     | 54.5 | 
| 2     | 0.598 | 