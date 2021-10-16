# Results and Models
## Base : Faster R-CNN
| Exp num | Backbone  | RoI Head   | Epoch |Best Epoch |initial lr |Lr schd | Optimizer | Image size| val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_s | val/bbox_mAP_m | val/bbox_mAP_l |
|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 0      | R50 | retinanet   | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|AdamW| 1024x1024   |0.231   | 0.4240    | 0.225  | 0.000  |  0.046  |  0.281  |

+---------------+-------+-------------+-------+------------+-------+  
| category      | AP    | category    | AP    | category   | AP    |  
+---------------+-------+-------------+-------+------------+-------+  
| General trash | 0.114 | Paper       | 0.220 | Paper pack | 0.156 |  
| Metal         | 0.096 | Glass       | 0.155 | Plastic    | 0.117 |  
| Styrofoam     | 0.124 | Plastic bag | 0.469 | Battery    | 0.034 |  
| Clothing      | 0.126 | None        | None  | None       | None  |  
+---------------+-------+-------------+-------+------------+-------+  
epoch 12 기준 class별 AP. 
Battery, Metal 이 잘 안나옴. 


## Experiment
| Exp num | Backbone  | RoI Head   | Epoch |Best Epoch |initial lr |Lr schd | Optimizer | Image size| val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_s | val/bbox_mAP_m | val/bbox_mAP_l |
|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 1-1     | R50 | retinanet   | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|AdamW | 1024x1024   | 0.162   | 0.121 |  0.059  | 0.006  | 0.013  |  0.077  |
| 1-2     | R50 | retinanet   | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|AdamW | 1024x1024   | 0.135   | 0.212 |  0.135  | 0.000 | 0.033  |  0.162  |
| 2-1    | R50 | retinanet   | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|AdamW | 1024x1024   | 0.157  | 0.253  |  0.155  |  0.001| 0.041 |  0.188  |
| 2-2    | R50 | retinanet   | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|AdamW | 1024x1024   | 0.147  |  0.247  |  0.151   |  0.018| 0.031 |  0.177 |
| 3-1    | R50 | retinanet   | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|AdamW | 1024x1024   | 0.147  |  0.247  |  0.151   |  0.018| 0.031 |  0.177 |
| 3-2    | R50 | retinanet   | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|AdamW | 1024x1024   | 0.147  |  0.247  |  0.151   |  0.018| 0.031 |  0.177 |
| 3-3    | R50 | retinanet   | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|AdamW | 1024x1024   | 0.147  |  0.247  |  0.151   |  0.018| 0.031 |  0.177 |

inference 에 사용할 가장 좋은 checkpoint 기준 score 작성  
1. retinanet map_small 점수 늘리고자 anchor 사이즈 줄임/늘림
- 실험 가설 : anchor box사이즈 줄면 map small 증가할 것
- 실험 방법 : anchor_generator의 octave_base_scale 4->2 / 4->8
- 결과 : 성능이 모두 떨어짐
- bbox_mAP_copypaste: 0.162 0.264 0.162 0.010 0.034 0.196 / base
- 1-1. bbox_mAP_copypaste: 0.064 0.121 0.059 0.006 0.013 0.077 / test_small
- 1-2. bbox_mAP_copypaste: 0.135 0.212 0.135 0.000 0.033 0.162 / test_big 


2. mosaic fine tune 
- 실험 가설 : 데이터 증강과 비슷한 효과로 성능이 오를 것이다.
- 실험 방법 : mosaic transfrom 
- 결과 : 
- 2-1. map50은 base와 비교했을 때 오히려 더 떨어짐
- 2-2.  그냥 모자이크만 한거 - bbox small 이 좋아짐


3. base k-fold
base 모델 Stratified Group k-fold 진행
3-1 ~ 3-3 : fold1 ~ fold3
