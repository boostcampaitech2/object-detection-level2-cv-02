# Boostcamp Recycle Trash Object Detection Challenge
Code for 4th place solution in Boostcamp AI Tech Recycle Trash Object detection Challenge.

대량 생산, 대량 소비의 시대에 살며 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있다.  
분리수거는 이러한 환경부담을 줄일 수 있는 방법이다. 해당 대회는 쓰레기를 detection하는 모델을 만들어 정확한 분리수거를 돕는 것에 기여한다. 

Contributors  
[김서원_T2036](https://github.com/swkim-sm), [이유진_T2167](https://github.com/Yiujin), [이한빈_T2176](https://github.com/binlee52), [정세종_T2201](https://github.com/sejongjeong), [조현동_T2215](https://github.com/JODONG2), [허지훈_T2241](https://github.com/hojihun5516), [허정훈_T2240](https://github.com/herjh0405)

# Archive contents
```
detection
├── dataset
├── template
│   ├──mmdetection
│   │  ├──configs
│   │  │  └──custom
│   │  │     ├──helper
│   │  │     │  ├──dateset.py
│   │  │     │  ├──runtime.py
│   │  │     │  └──schedule.py
│   │  │     └──models
│   │  │        ├──cascade_rcnn
│   │  │        ├──faster_rcnn
│   │  │        └──htc
│   │  ├──tools
│   │  │  ├──train.py
│   │  │  ├──inference.py
│   │  │  ├──ensemble.py
│   │  │  ├──make_fold_annotation.py
│   │  │  └──vis_submission.ipynb
│   │  └──submission
│   │     ├──ensemble_inference.py
│   │     └──ensemble_inf_cfg.json
│   ├──live
└── └──detectron

```

# Train
```
cd mmdetection
```
1. vanilla train   
```
python tools/train.py [config path]
```
2. k-fold train  
```
python tools/make_fold_annotation.py [original_train_json_path]
python tools/train_cv.py [config path]
```
3. pseudo labeling train & inference   
```
python tools/inference_with_pseudo_labeling.py [config path]
```
4. optimization with wandb sweeps  
- Setting
    - change sweep.yaml file
- Command
    - create sweep graph 
        ```
        wandb sweep sweep.yaml
        ```
    - then you can get url
    - you have to change **sweepID**

        ```
        wandb agent ProjectName/sweepID
        ```
    - train



# inference
```
cd mmdetection
```
1. vanilla inference  
```
python tools/inference.py [config path]
```
2. k-fold inference  
```
python tools/inference_cv.py [config names] [work_dir]
```

# visualization result
- Make submission csv file after training
- Change `PRED_CSV` in vis_submission.ipynb  
- Run cells  

# ensemble
```
cd submission
```
1. Modify ensemble_inf_cfg.json
2. Run ensemble_inference.py
```
python ensemble_inference.py ensemble_inf_cfg.json
```

## yolo
https://github.com/ultralytics/yolov5

## EfficeintDet
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch


# final ensemble model
ensemble_method : Weighted Boxes Fusion
## Model and Experiments
- "htc_swin_b_384.csv"
- "hsbfm_with_psudolabilng_8.csv"
- "htc_pvt_finetune_mosaic_final.csv"
- "htc_swin_b_finetune_mosaic.csv"
- "htc_swin_b.csv"
- "htc_swin_b_kfold.csv"
- "cascadercnn_pvt.csv"
- "faster_rcnn_pvtv2_b5_final.csv"

| Model(detector) | Exepriments Result |
|:---------------:|:------------------:|
|Hybrid Task Cascade| [README](https://github.com/boostcampaitech2/object-detection-level2-cv-02/tree/develop/template/mmdetection/configs/custom/models/htc)
| CasCade R-CNN | [README](https://github.com/boostcampaitech2/object-detection-level2-cv-02/tree/develop/template/mmdetection/configs/custom/models/cascade_rcnn) |
| Faster R-CNN | [READEM](https://github.com/boostcampaitech2/object-detection-level2-cv-02/tree/develop/template/mmdetection/configs/custom/models/faster_rcnn)


