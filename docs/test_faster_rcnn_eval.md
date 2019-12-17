# Test Faster R-CNN and COCO Eval

1. get result and save to json file
   
   ```python3.5 tools/test.py configs/faster_rcnn_r50_fpn_1x.py work_dirs/faster_rcnn_r50_fpn_2x/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth --json_out work_dirs/faster_rcnn_r50_fpn_2x/results.json```
2. calculate the mAP and class_wise result

    ```python3.5 tools/coco_eval.py work_dirs/faster_rcnn_r50_fpn_2x/results.bbox.json --class_wise```

    the result:
     ```
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.377
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.592
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.411
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.219
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.414
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.487
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.310
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.496
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.519
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.323
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.557
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.659
    ```
    
    | category      | AP     | category     | AP     | category       | AP     |
    |:--------------|:-------|:-------------|:-------|:---------------|:-------|
    | person        | 51.927 | bicycle      | 26.985 | car            | 41.377 |
    | motorcycle    | 40.385 | airplane     | 59.298 | bus            | 60.641 |
    | train         | 56.722 | truck        | 31.628 | boat           | 26.124 |
    | traffic light | 26.112 | fire hydrant | 62.744 | stop sign      | 64.214 |
    | parking meter | 45.815 | bench        | 21.200 | bird           | 32.572 |
    | cat           | 58.103 | dog          | 55.649 | horse          | 53.589 |
    | sheep         | 47.060 | cow          | 52.172 | elephant       | 58.757 |
    | bear          | 62.244 | zebra        | 62.788 | giraffe        | 63.022 |
    | backpack      | 15.719 | umbrella     | 34.946 | handbag        | 12.683 |
    | tie           | 30.923 | suitcase     | 34.823 | frisbee        | 62.801 |
    | skis          | 21.522 | snowboard    | 31.179 | sports ball    | 42.212 |
    | kite          | 37.412 | baseball bat | 24.157 | baseball glove | 32.252 |
    | skateboard    | 45.148 | surfboard    | 32.991 | tennis racket  | 44.265 |
    | bottle        | 36.711 | wine glass   | 33.435 | cup            | 39.698 |
    | fork          | 29.191 | knife        | 14.624 | spoon          | 11.916 |
    | bowl          | 40.408 | banana       | 21.608 | apple          | 17.993 |
    | sandwich      | 29.823 | orange       | 28.630 | broccoli       | 21.762 |
    | carrot        | 18.814 | hot dog      | 29.431 | pizza          | 48.275 |
    | donut         | 41.719 | cake         | 32.236 | chair          | 24.418 |
    | couch         | 36.430 | potted plant | 24.976 | bed            | 36.289 |
    | dining table  | 24.641 | toilet       | 54.775 | tv             | 52.635 |
    | laptop        | 55.670 | mouse        | 58.818 | remote         | 25.370 |
    | keyboard      | 49.128 | cell phone   | 31.551 | microwave      | 51.033 |
    | oven          | 29.406 | toaster      | 32.145 | sink           | 33.601 |
    | refrigerator  | 50.185 | book         | 13.916 | clock          | 49.260 |
    | vase          | 35.269 | scissors     | 20.931 | teddy bear     | 42.508 |
    | hair drier    | 6.081  | toothbrush   | 14.554 |                |        |
    
