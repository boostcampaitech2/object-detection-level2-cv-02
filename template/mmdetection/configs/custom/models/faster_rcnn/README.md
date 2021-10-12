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
- 결과 원인 분석 : ~ 
- 참고 자료 : 참고한 글이나 논문이나 블로그 있으면 link 달기

<span style="color:red">2. 이미지 scale에 따른 bbox_small metric 변화 관찰</span>  
- 실험 가설 : 이미지 크기가 클수록 작은객체를 더 잘 탐지할 것이다. 
- 실험 방법 : input image scale 변경  
- 결과 : input image scale이 클수록 작은 object 탐지율 올라감  
- 결과 원인 분석 : ~ 
- 참고 자료 : 참고한 글이나 논문이나 블로그 있으면 link 달기   

<span style="color:green">3. backbone parameter freeze 에 따른 전체 metric 변화 관찰</span>  
- 실험 가설 : 데이터 갯수가 적으므로 어쩌고 하면 저쩌고 할 것이다. 
- 실험 방법 : backbone parameter freeze 하고 학습   
- 결과 : ~~
- 결과 원인 분석 : ~ 
- 참고 자료 : ~~   

| Exp num | Backbone  | RoI Head   | Epoch |Best Epoch |initial lr |Lr schd | Optimizer | Image size| val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_s | val/bbox_mAP_m | val/bbox_mAP_l |val/loss_rpn_cls | val/loss_rpn_bbox | val/loss_cls | val/acc |val/loss_bbox | val/loss |
|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:---------------:|:--------------:|:--------------:|:---------------:|:------------:|:----------:|
| 1-1      | swin_t | Faster R-CNN      | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001) | 1024x1024   |0.2110   |0.4190   | 0.1920  | 0.0040 | 0.0430 | 0.2520  | 0.0399| 0.0434 |0.2599 |92.1923| 0.2660| 0.6092|
| 1-2      | pvtv2_b0 | Faster R-CNN      | 1x  |12 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001) | 1024x1024   |0.164  |0.337  | 0.145  |0.000 | 0.031 |0.197  | 0.0406|0.0432 | 0.2779 |91.9211| 0.2630| 0.6248|
| 1-3      | pvtv2_b5 | Faster R-CNN      | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001) | 1024x1024   |0.345 |0.565 |  0.365 |0.011 | 0.080  | 0.409 |0.0547|0.0373 | 0.2471 |92.9791| 0.2188|0.5580|

  bbox_mAP: 0.3450, bbox_mAP_50: 0.5650, bbox_mAP_75: 0.3650, bbox_mAP_s: 0.0110, bbox_mAP_m: 0.0800, bbox_mAP_l: 0.4090, bbox_mAP_copypaste: 0.345 0.565 0.365 0.011 0.080 0.409
2021-10-11 21:07:55,372 - mmdet - INFO - Exp name: config.py
2021-10-11 21:07:55,372 - mmdet - INFO - Epoch(val) [11][489]   loss_rpn_cls: 0.0547, loss_rpn_bbox: 0.0373, loss_cls: 0.2471, acc: 92.9791, loss_bbox: 0.2188, loss: 0.5580

## Leader board 결과(제출했을 시)
| Exp num | Public LB map  | 
|:-------:|:---------:|
| 1     | 54.5 | 
| 2     | 0.598 | 