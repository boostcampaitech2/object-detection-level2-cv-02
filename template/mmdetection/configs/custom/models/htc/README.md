# Results and Models
## Base
| Exp num | Backbone  | RoI Head   | Epoch |initial lr |Lr schd | Optimizer | Inf time (fps) | val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_l | val/bbox_mAP_m | val/bbox_mAP_s |train/loss_rpn_cls | train/s1.loss_bbox | train/s2.loss_cls | train/s1.acc |train/s2.acc | train/loss |
|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:---------------:|:--------------:|:--------------:|:---------------:|:------------:|:----------:|
| 0      | htc | 1x      | 1.000e-06   | step     | sgd(momentum = 0.9 , ~)   | 5.8            | 42.3   | 42.3   | 42.3   | 42.3   | 42.3   | 37.4    | 37.4   | 37.4 | 37.4  | 37.4  | 37.4  |37.4   | 37.4  | 37.4  |
## Experiment
inference 에 사용할 가장 좋은 checkpoint 기준 score 작성  
<span style="color:blue">1. backbone에 따른 전체 metric 변화 관찰</span>  
- 실험 가설 : 데이터 갯수가 적으므로 어쩌고 하면 저쩌고 할 것이다. 
- 실험 방법 : backbone 변경
- 결과 : 파라미터가 큰 모델일수록 더 잘됨, swin이 더 잘됨 등등
- 결과 원인 분석 : ~ 
- 참고 자료 : 참고한 글이나 논문이나 블로그 있으면 link 달기

<span style="color:red">2. 이미지 scale에 따른 bbox_small metric 변화 관찰</span>  
- 실험 가설 : 데이터 갯수가 적으므로 어쩌고 하면 저쩌고 할 것이다. 
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

| Exp num | Backbone  | RoI Head   | Epoch |initial lr |Lr schd | Optimizer | Inf time (fps) | val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_l | val/bbox_mAP_m | val/bbox_mAP_s |train/loss_rpn_cls | train/s1.loss_bbox | train/s2.loss_cls | train/s1.acc |train/s2.acc | train/loss |
|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:---------------:|:--------------:|:--------------:|:---------------:|:------------:|:----------:|
| 1-1       | htc | 1x      | 1.000e-06   | step     | sgd(momentum = 0.9 , ~)   | 5.8            | 42.3   | 42.3   | 42.3   | 42.3   | 42.3   | 37.4    | 37.4   | 37.4 | 37.4  | 37.4  | 37.4  |37.4   | 37.4  | 37.4  |
| 1-2     | htc | 20e     | 1.000e-06   | cosine anealing    | sgd(momentum = 0.9 , ~)   | -              | 42.3   | 42.3   | 42.3   | 42.3   | 43.3   | 38.3    | 37.4  | 37.4  | 37.4  | 37.4  | 37.4  | 37.4  | 37.4  |  37.4 |

## Leader board 결과(제출했을 시)
| Exp num | Public LB map  | 
|:-------:|:---------:|
| 1-2     | 54.5 | 
