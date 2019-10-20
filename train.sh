# train mobilenetv2_ssd cosine
# CUDA_VISIBLE_DEVICES=0 python3 ./tools/train.py configs/ssdmnv2_coco_cosine.py  --gpus 1 --validate

# train faster_rcnn_r50_fpn_1x.py
# CUDA_VISIBLE_DEVICES=0 python3 ./tools/train.py configs/faster_rcnn_r50_fpn_1x.py --gpus 1 --validate

# train mobilenetv2_ssd cosine
CUDA_VISIBLE_DEVICES=0 python3 ./tools/train.py configs/pascal_voc/ssd300_mobilenetv2_voc.py  --gpus 1 --validate