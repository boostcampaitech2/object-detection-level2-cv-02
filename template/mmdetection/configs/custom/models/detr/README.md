# Results and Models
실험 해본 모델 (backbone 말고 detector)
1. detr : map 엄청 낮음
원인예측 : encoder-decorder구조라 학습하기엔 데이터가 너무 부족함

# Results and Models
## Base : Faster R-CNN
| Exp num | Backbone  | RoI Head   | Epoch |Best Epoch |initial lr |Lr schd | Optimizer | Image size| val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_s | val/bbox_mAP_m | val/bbox_mAP_l |val/loss_rpn_cls | val/loss_rpn_bbox | val/loss_cls | val/acc |val/loss_bbox | val/loss |
|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:---------------:|:--------------:|:--------------:|:---------------:|:------------:|:----------:|
| 1      | ResNet | DETRHead      | 45  |40 |lr=0.0001   |dict(policy="step", step=[100])|dict(type="AdamW", lr=0.0001, weight_decay=0.0001, paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1, decay_mult=1.0)}) | 1024x1024   |0.006   | 0.022    | 0.003  | 0.000  |  0.001  |  0.007  |  |  | |   | |  | |


## Experiment
inference 에 사용할 가장 좋은 checkpoint 기준 score 작성  
<span style="color:blue">1. backbone에 따른 전체 metric 변화 관찰</span>  
- 실험 목적 : one stage detector를 활용하여 ensemble효과를 기대 
- 실험 방법 : transformer를 이용한 detection 
- 결과 : 엄청 느린 학습속도로 인해 어느정도의 성능을 보기전에 중단
- 결과 원인 분석 : Encoder Decoder구조로 이루어져서 학습하는데에 많은 속도가 걸렸던 것이 아닐까 추측 
- 참고 자료 : https://velog.io/@sjinu/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0End-to-End-Object-Detection-with-Transformers


## Leader board 결과(제출했을 시)
> 너무 낮은 mAP스코어로 제출하지못했음

| Exp num | Public LB map  | 
|:-------:|:---------:|
| 1     | 0 | 