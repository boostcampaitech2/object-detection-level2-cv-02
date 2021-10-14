# Results and Models
## Base : Faster R-CNN
| Exp num | Backbone  | RoI Head   | Epoch |Best Epoch |initial lr |Lr schd | Optimizer | Image size| val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_s | val/bbox_mAP_m | val/bbox_mAP_l |val/loss_rpn_cls | val/loss_rpn_bbox | val/loss_cls | val/acc |val/loss_bbox | val/loss |
|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|:--------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:---------------:|:--------------:|:--------------:|:---------------:|:------------:|:----------:|
| 1      | R50 | retinanet   | 1x  |11 |lr=0.002   |dict(_delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11])|dict(_delete_=True, type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001) | 1024x1024   |0.231   | 0.4240    | 0.225  | 0.000  |  0.046  |  0.281  | 37.4 | 0.0524 |0.0421 | 0.2682  |92.1514|  0.2525| 0.6152|

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
inference 에 사용할 가장 좋은 checkpoint 기준 score 작성  
1. retinanet map_small 점수 늘리고자 anchor 사이즈 줄임
- 실험 가설 : anchor box사이즈 줄면 map small 증가할 것
- 실험 방법 : anchor_generator의 octave_base_scale 4->2
- 결과 : 성능이 모두 떨어짐
- bbox_mAP_copypaste: 0.162 0.264 0.162 0.010 0.034 0.196 / base
- bbox_mAP_copypaste: 0.064 0.121 0.059 0.006 0.013 0.077 / test_smll
- bbox_mAP_copypaste: 0.135 0.212 0.135 0.000 0.033 0.162 / etst_big 

- 결과 원인 분석 : 
- 참고 자료 : 참고한 글이나 논문이나 블로그 있으면 link 달기

2. mosaic fine tune 
- 실험 가설 : 이미지 크기가 클수록 작은객체를 더 잘 탐지할 것이다. 
- 실험 방법 : input image scale 변경  
- 결과 : 
- bbox_mAP_copypaste: 0.157 0.253 0.155 0.001 0.041 0.188 | map50은 0.253 base와 비교했을 때 오히려 더 떨어짐
- bbox_mAP_copypaste: 0.147 0.247 0.151 0.018 0.031 0.177 | 그냥 모자이크만 한거 - bbox small 이 좋아짐. 
- 결과 원인 분석 : ~ 
- 참고 자료 : 참고한 글이나 논문이나 블로그 있으면 link 달기   


## Leader board 결과(제출했을 시)
| Exp num | Public LB map  | 
|:-------:|:---------:|
| 1     | . | 
| 2     |.| 