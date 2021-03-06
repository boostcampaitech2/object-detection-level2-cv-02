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
inference ??? ????????? ?????? ?????? checkpoint ?????? score ??????  
<span style="color:green">1. backbone(?????? ??????)??? ?????? ?????? metric ?????? ??????</span>  
- ?????? ?????? : ????????? ???????????? ??? ??? ????????? ?????????
- ?????? ?????? : backbone ??????
- ?????? : swin-t, swin-b??? ???????????? ??? ????????? ????????? ????????? val/mAP50??? ?????? ?????? 
- ?????? ?????? ?????? : ????????? ????????? ???????????? ????????? ????????? ??? ??? ?????? ????????????

<span style="color:green">2. FPN??? BiFPN?????? ??????</span>  
- ?????? ?????? : BiFPN??? ?????? low/high level feauter??? ??? ????????? val/bbox_mAP_S, val/bbox_mAP_M??? ????????? ?????????
- ?????? ?????? : FPN ?????? ??????  
- ?????? : val/mAP50??? ????????? ????????????  
- ?????? ?????? ?????? : htc ????????? ???????????? ????????? feature??? ???????????? ????????? BiFPN??? ???????????? ????????? ?????? ????????? ??????

<span style="color:green">3. Mosaic augmentation??? ???????????? 1??? ????????? finetuning</span>  
- ?????? ?????? : Mosaic Augmentation??? ???????????? ???????????? ????????? ?????? ????????? ?????? ???????????? ?????? ???????????????, ?????? bounding box ???????????? ???????????? ????????? ?????? ??? ?????? ?????????.
- ?????? ?????? : Yolo-v4??? Mosaic augmentation??? ??????????????? ??????  
- ?????? : val/mAP50??? ?????? ????????????, test/mAP50??? 0.644??? ?????? ????????????.
- ?????? ?????? ?????? : Mosaic augmentation??? ?????? ????????? Robust ?????? ????????? ?????????.
- ?????? ?????? : [Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)

<span style="color:green">4. Swin-T ??? ?????? Transfer Learning ??????</span>  
- ?????? ?????? : Coco dataset??? pretrained ??? Swin-T ????????? feautre??? ??? ????????? ?????????. Swin-L?????? ????????? ????????? ??????.
- ?????? ?????? : backbone freezing   
- ?????? : mAP??? ????????? ???????????? ?????????
- ?????? ?????? ?????? : Coco dataset??? Custom dataset??? ?????? ????????? ????????? ?????????.

<span style="color:green">5. Swin-L ??? ?????? Transfer Learning ??????</span>  
- ?????? ?????? : Coco dataset??? pretrained ??? Swin-L ????????? freezing??????, ?????? ????????? ???????????? ????????? ????????? ???????????? ????????? ????????? ?????? ?????????
- ?????? ?????? : backbone freezing   
- ?????? : ??????4 ????????? val/mAP??? ???????????????. ????????? ?????? ????????? training ??? ???????????? mAP??? ????????????.
- ?????? ?????? ?????? : Coco dataset??? Custom dataset??? ?????? ????????? ????????? ?????????.

<span style="color:green">6. Swin-B 384, pretrain input size??? ???????????? ????????? ????????? ?????????</span>  
- ?????? ?????? : ????????? ????????? ????????????, pretrained input size??? ?????????, ?????? ???????????? ??? ??? ????????? ??? ?????? ?????????.
- ?????? ?????? : backbone ????????? Swin-B 384??? ??????
- ?????? : Swin-B 224??? ????????? ??????1?????? mAP??? ????????????.
- ?????? ?????? ?????? : ????????????

<span style="color:green">7. SOTA ????????? ?????? 1??? k-fold ??????</span>  
- ?????? ?????? : ?????? ???????????? ??????????????? k-fold??? ?????? ????????? ?????? ????????? ????????? ????????? ?????????
- ?????? ?????? : k-fold training, k=3
- ?????? : mVal/mAP50=0.606, test/mAP50=0.629??? LB?????? ????????? ?????? ????????????.
- ?????? ?????? ?????? : ????????? ?????? ??????. fold??? ???????????? ?????? val score??? ?????? ?????? ????????? ????????????.


<span style="color:green">8. backbone pvt-v2-b5??? ??????</span>  
- ?????? ?????? : ?????? HTC?????? ?????? ??? backbone ????????? ???????????? ????????? ???????????????.
- ?????? ?????? : backbone ????????? pvt-v2-b5??? ??????
- ?????? : val?????? ??????????????? ???????????? LB?????? ????????? ????????????. 
- ?????? ?????? ?????? : ?????? ????????? ?????? validation dataset?????? ????????? ?????????.

<span style="color:green">9. backbone pvt-v2-b5 Mosaic augmentation??? ????????? ?????? ???????????? ??????</span>  
- ?????? ?????? : ????????? ???????????? val??? ????????? ?????? ?????? ???????????? ???????????? ?????? Robust??? ?????? ??????
- ?????? ?????? : ????????? ?????? ???????????? Mosaic augmentation??? ???????????? ?????? ???????????? ??????.
- ?????? : ????????? ????????? ??? ??????.
- ?????? ?????? ?????? : ?????? ???????????? ensemble??? ????????? ??????????????? ?????? ????????? ????????? ?????? ????????? ??? ?????????.


## Leader board ??????(???????????? ???)
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
