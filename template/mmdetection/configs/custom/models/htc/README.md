# Results and Models
## Base
| Exp num | Backbone  | RoI Head   | Epoch |initial lr |Lr schd | Optimizer | val/bbox_mAP| val/bbox_mAP_50 |  val/bbox_mAP_75 | val/bbox_mAP_l | val/bbox_mAP_m | val/bbox_mAP_s | config | checkpoint |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | swin-b | htc | 13 | 1e-4 | step | AdamW | 0.449 | 0.619 | 0.473 | 0.532 | 0.077 | 0.10773 | [config](https://github.com/boostcampaitech2/object-detection-level2-cv-02/blob/master/template/mmdetection/configs/custom/models/htc/htc_swin_b.py) | [Google](https://drive.google.com/file/d/1AKzqWlWGRL3D1WP6i6zBEhj40-VKLJeS/view?usp=sharing) |
| 2 | swin-b | htc | 14 | 1e-4 | step | AdamW | 0.406 | 0.597 | 0.438 | 0.479 | 0.086 | 0.013 | [config](https://github.com/boostcampaitech2/object-detection-level2-cv-02/blob/master/template/mmdetection/configs/custom/models/htc/htc_swin_b_bifpn.py) | [Google](https://drive.google.com/file/d/1qwZXqeQ6NV3k7aUFM2gOzHijnbby20Hm/view?usp=sharing) |
| 3 | swin-b | htc | 17 | 1e-4 | step | AdamW | 0.455 | 0.622 | 0.485 | 0.538 | 0.08 | 0.043 | [config](https://github.com/boostcampaitech2/object-detection-level2-cv-02/blob/master/template/mmdetection/configs/custom/models/htc/htc_swin_b_finetune_mosaic.py) | [Google](https://drive.google.com/file/d/1-vQtS_ekP70gcHJmpfUpmmsiDD377l_x/view?usp=sharing) |
| 4 | swin-t | htc | 15 | 1e-4 | step | AdamW | 0.26 | 0.453 | 0.277 | 0.31 | 0.042 | 0.036 | [config](https://github.com/boostcampaitech2/object-detection-level2-cv-02/blob/master/template/mmdetection/configs/custom/models/htc/htc_swin_t_frozen.py) | |
| 5 | swin-l | htc | 15 | 1e-4 | step | AdamW | 0.303 | 0.531 | 0.3 | 0.361 | 0.048 | 0.043 | [config](https://github.com/boostcampaitech2/object-detection-level2-cv-02/blob/master/template/mmdetection/configs/custom/models/htc/htc_swin_l_frozen.py) | | 
| 6 | swin-b | htc | 15 | 1e-4 | step | AdamW | 0.43 | 0.612 | 0.46 | 0.508 | 0.083 | 0.032 | [config](https://github.com/boostcampaitech2/object-detection-level2-cv-02/blob/master/template/mmdetection/configs/custom/models/htc/htc_swin_b_384.py) | |
| 7 | swin-b | htc | - | 1e-4 | step | AdamW | 0.420 | 0.606 | 0.453 | 0.495 | 0.120 | 0.008 | [config](https://github.com/boostcampaitech2/object-detection-level2-cv-02/blob/master/template/mmdetection/configs/custom/models/htc/htc_swin_b.py) | [Google](https://drive.google.com/drive/folders/1WYRmSmzs4cQpSX09r8IsPNUpt7M9DJKJ?usp=sharing) |
| 8 | pvt-v2-b5 | htc | - | 1e-4 | step | AdamW | 0.487 | 0.643 | 0.521 | 0.561 | 0.187 | 0.017 |
| 9 | pvt-v2-b5 | htc | - | 1e-4 | step | AdamW | - | - | - | - | - | - |


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

<span style="color:green">4. Swin-T 에 대한 Transfer Learning 적용</span>  
- 실험 가설 : Coco dataset에 pretrained 된 Swin-T 모델은 feautre을 잘 추출할 것이다. Swin-L과의 비교를 위하여 실험.
- 실험 방법 : backbone freezing   
- 결과 : mAP의 개선은 존재하지 않았다
- 결과 원인 분석 : Coco dataset과 Custom dataset의 차이 때문인 것으로 보인다.

<span style="color:green">5. Swin-L 에 대한 Transfer Learning 적용</span>  
- 실험 가설 : Coco dataset에 pretrained 된 Swin-L 모델을 freezing하고, 전체 모델을 학습하면 모델의 크기가 커지므로 성능에 개선이 있을 것이다
- 실험 방법 : backbone freezing   
- 결과 : 실험4 보다는 val/mAP가 상승하였다. 그러나 모델 전체를 training 한 것보다는 mAP가 떨어졌다.
- 결과 원인 분석 : Coco dataset과 Custom dataset의 차이 때문인 것으로 보인다.

<span style="color:green">6. Swin-B 384, pretrain input size를 확대하여 모델의 크기를 크게함</span>  
- 실험 가설 : 모델의 크기가 커질수록, pretrained input size가 클수록, 작은 이미지를 더 잘 검출할 수 있을 것이다.
- 실험 방법 : backbone 모델을 Swin-B 384로 변경
- 결과 : Swin-B 224을 사용한 실험1보다 mAP가 떨어졌다.
- 결과 원인 분석 : 원인불명

<span style="color:green">7. SOTA 모델인 실험 1에 k-fold 적용</span>  
- 실험 가설 : 학습 데이터가 부족하므로 k-fold를 통해 모델을 학습 시키면 성능이 향상할 것이다
- 실험 방법 : k-fold training, k=3
- 결과 : mVal/mAP50=0.606, test/mAP50=0.629로 LB에서 점수가 약간 상승했다.
- 결과 원인 분석 : 데이터 수의 증가. fold가 충분하지 못해 val score가 낮게 나온 것으로 추정된다.


<span style="color:green">8. backbone pvt-v2-b5로 변경</span>  
- 실험 가설 : 기존 HTC에서 보다 큰 backbone 모델을 사용하면 성능이 오를것이다.
- 실험 방법 : backbone 모델을 pvt-v2-b5로 변경
- 결과 : val에서 성능향상을 보였지만 LB에서 점수가 낮아졌다. 
- 결과 원인 분석 : 실험 환경이 달라 validation dataset에서 차이가 생겼다.

<span style="color:green">9. backbone pvt-v2-b5 Mosaic augmentation을 활용한 전체 데이터셋 학습</span>  
- 실험 가설 : 마지막 학습으로 val을 나누지 않고 전체 데이터를 학습하여 보다 Robust한 성능 기대
- 실험 방법 : 성능이 크게 향상됐던 Mosaic augmentation을 활용하여 전체 데이터셋 학습.
- 결과 : 성능을 확인할 수 없음.
- 결과 원인 분석 : 다른 모델들과 ensemble한 결과를 제출했기에 이번 실험의 결과를 따로 확인할 수 없었음.


## Leader board 결과(제출했을 시)
| Exp num | mAP50  | 
|:-------:|:---------:|
| 1 | 0.626 | 
| 2 | 0.604 |
| 3 | 0.644 |
| 4 | |
| 5 | |
| 6 | 0.612 | 
| 7 | 0.629 |
| 8 | 0.598 |
