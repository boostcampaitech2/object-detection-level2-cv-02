# Results and Models
실험 해본 모델 (backbone 말고 detector)
1. double_heads_faster_rcnn 

# Results and Models
## Base : Double Head Faster R-CNN
| Exp num | Backbone  | RoI Head   | Epoch |Best Epoch |initial lr |Lr schd | Optimizer | Image size| val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_s | val/bbox_mAP_m | val/bbox_mAP_l |val/loss_rpn_cls | val/loss_rpn_bbox | val/loss_cls | val/acc |val/loss_bbox | val/loss |
|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:---------------:|:--------------:|:--------------:|:---------------:|:------------:|:----------:|
| 1      | ResNet | RPNHead      | 40  |38 |0.000001   |dict(policy="step", warmup="linear", warmup_ratio=0.001, warmup_iters=1000, step=[7, 12])|dict(type="AdamW", lr=0.000001, weight_decay=0.0001, paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1, decay_mult=1.0)}) | 1024x1024   |0.349   | 0.514    | 0.364  | 0.002  |  0.071  |  0.41  |  |  | |   | |  | |


## Experiment
inference 에 사용할 가장 좋은 checkpoint 기준 score 작성  
<span style="color:blue">1. faster-rcnn과 비교하여 double head를 사용하여 더 높은 성능을 기대</span>  
- 실험 목적 : two stage detector의 two header를 사용하여 더 많은 bbox를 기대
- 실험 방법 : double head와 일반 rcnn의 성능 비교  
- 결과 : double head를 사용했을 때 첫 에포크부터 더 높은 성능을 나타냄
- 결과 원인 분석 : bbox를 예측하는 head를 2개 사용함으로 더 많은 box를 보게된다 



## Leader board 결과(제출했을 시)
> 너무 낮은 mAP스코어로 제출하지못했음

| Exp num | Public LB map  | 
|:-------:|:---------:|
| 1     | 0.444 | 