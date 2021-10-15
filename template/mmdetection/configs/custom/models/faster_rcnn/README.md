# Results and Models
실험 해본 모델 (backbone 말고 detector)
1. vfnet : loss 계산 안됨 
원인예측 : bbox 꼭지점 9개로 예측하는 모델임

2. detr : map 엄청 낮음
원인예측 : encoder-decorder구조라 학습하기엔 데이터가 너무 부족함

# Results and Models
## Base : Faster R-CNN
| Exp num | Backbone  | RoI Head   | Epoch |Best Epoch |initial lr |Lr schd | Optimizer | Image size| val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_s | val/bbox_mAP_m | val/bbox_mAP_l |val/loss_rpn_cls | val/loss_rpn_bbox | val/loss_cls | val/acc |val/loss_bbox | val/loss |
|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:---------------:|:--------------:|:--------------:|:---------------:|:------------:|:----------:|
| 1      | R50 | Faster R-CNN      | 1x  |10 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001) | 1024x1024   |0.231   | 0.4240    | 0.225  | 0.000  |  0.046  |  0.281  | 37.4 | 0.0524 |0.0421 | 0.2682  |92.1514|  0.2525| 0.6152|

workflow = [('train', 1), ('val', 1)]
-> wandb: WARNING Step must only increase in log calls.  Step 13189 < 13190; dropping {'val/loss_rpn_cls': 0.05315568791115581, 'val/loss_rpn_bbox': 0.04183597948659605, 'val/loss_cls': 0.27059912833014155, 'val/acc': 92.09659235264228, 'val/loss_bbox': 0.25402145996326353, 'val/loss': 0.6196122553532686, 'learning_rate': 2.0000000000000005e-05, 'momentum': 0.9}.

workflow = [('train', 1)]  
Epoch(val) [11][489] bbox_mAP: 0.2190, bbox_mAP_50: 0.4130, bbox_mAP_75: 0.2080, bbox_mAP_s: 0.0030, bbox_mAP_m: 0.0480, bbox_mAP_l: 0.2640

loss 없음


## Experiment
inference 에 사용할 가장 좋은 checkpoint 기준 score 작성  
<span style="color:blue">1. backbone에 따른 전체 metric 변화 관찰</span>  
- 실험 가설 : 백본에 따라 meatirc 변화가 있을 것이다. 
- 실험 방법 : backbone 변경
- 결과 : 파라미터가 큰 모델일수록 더 잘됨 - ex ) pvt b0 보다 b5가 더 잘됨
- 결과 원인 분석 : 모델의 크기가 커질수록 복잡한 표현을 할 수 있기 때문이다
- 참고 자료 : 참고한 글이나 논문이나 블로그 있으면 link 달기

| Exp num | Backbone  | RoI Head   | Epoch |Best Epoch |initial lr |Lr schd | Optimizer | Image size| val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_s | val/bbox_mAP_m | val/bbox_mAP_l |
|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 1-1      | swin_t | Faster R-CNN      | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001) | 1024x1024   |0.2110   |0.4190   | 0.1920  | 0.0040 | 0.0430 | 0.2520  |
| 1-2      | pvtv2_b0 | Faster R-CNN      | 1x  |12 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001) | 1024x1024   |0.164  |0.337  | 0.145  |0.000 | 0.031 |0.197  |
| 1-3      | pvtv2_b5 | Faster R-CNN      | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001) | 1024x1024   |0.345 |0.565 |  0.365 |0.011 | 0.080  | 0.409 |
