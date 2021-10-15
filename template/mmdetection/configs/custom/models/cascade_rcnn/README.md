# Results and Models
## Base
| Exp num | Backbone  | RoI Head   | Epoch |initial lr |Lr schd | Optimizer | val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_l | val/bbox_mAP_m | val/bbox_mAP_s | config | checkpoint |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | pvt-v2-b0 | cascadeRCNN | - | 1e-4 | step | AdamW | 0.36 | 0.499 | 0.39 | 0.423 | 0.112 | 0.005 |
| 2 | pvt-v2-b5 | cascadeRCNN | - | 1e-4 | step | AdamW | 0.459 | 0.593 | 0.498 | 0.534 | 0.164 | 0.003 |
| 3 | pvt-v2-b5 | cascadeRCNN | - | 1e-4 | step | AdamW | 0.446 | 0.585 | 0.478 | 0.522 | 0.136 | 0.003 |

## Experiment
inference 에 사용할 가장 좋은 checkpoint 기준 score 작성  
<span style="color:green">1. backbone(모델 크기)에 따른 전체 metric 변화 관찰</span>  
- 실험 가설 : 모델이 커질수록 더 잘 예측할 것이다
- 실험 방법 : backbone resnet에서 pvt-v2-b0으로 변경
- 결과 : 기존의 resnet 백본과 비교햇을 때 val/mAP50이 높게 나옴 
- 결과 원인 분석 : 모델의 크기가 커질수록 복잡한 표현을 할 수 있기 때문이다

<span style="color:green">2. backbone(모델 크기)에 따른 전체 metric 변화 관찰</span>  
- 실험 가설 : 모델이 커질수록 더 잘 예측할 것이다
- 실험 방법 : backbone pvt-v2-b0에서 pvt-v2-b5으로 변경
- 결과 : 기존의 b0과 비교햇을 때 val/mAP50이 0.1 높아짐.
- 결과 원인 분석 : 모델의 크기가 커질수록 복잡한 표현을 할 수 있기 때문이다

<span style="color:green">3. FPN을 PAFPN으로 변경</span>  
- 실험 가설 :  PAFPN을 통해 low/high level feauter을 잘 섞으면 val/bbox_mAP_S, val/bbox_mAP_M이 개선될 것이다
- 실험 방법 : Neck FPN을 PAFPN으로 변경
- 결과 : 기존의 FPN과 비교햇을 때 val/mAP50이 낮게 나옴. 
- 결과 원인 분석 :  cascadeRCNN 모델의 특징으로 여러번 feature을 추출하기 때문에 PAFPN을 사용해도 개선이 없는 것으로 보임

## Leader board 결과(제출했을 시)
| Exp num | mAP50  | 
|:-------:|:---------:|
| 1 | 0.457 | 
| 2 | 0.553 |
| 3 | - |
