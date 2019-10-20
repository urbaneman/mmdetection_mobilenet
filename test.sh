CUDA_VISIBLE_DEVICES=0 python3 tools/test.py configs/pascal_voc/ssd300_mobilenetv2_voc.py\
 work_dirs/ssd300_mobilenet_v2_helmet/latest.pth --out work_dirs/ssd300_mobilenet_v2_helmet/test_result.pkl

# than run python3 ./tools/voc_eval.py ./work_dirs/ssd300_mobilenet_v2_helmet/test_result.pkl ./configs/pascal_voc/ssd300_mobilenetv2_voc.py
# to get the mAP of voc eval.
# +-----------------+------+-------+--------+-----------+-------+
# | class           | gts  | dets  | recall | precision | ap    |
# +-----------------+------+-------+--------+-----------+-------+
# | person          | 3938 | 16930 | 0.856  | 0.199     | 0.761 |
# | blue            | 531  | 1384  | 0.836  | 0.321     | 0.757 |
# | white           | 1037 | 2278  | 0.699  | 0.318     | 0.610 |
# | yellow          | 817  | 2498  | 0.800  | 0.262     | 0.710 |
# | red             | 773  | 1934  | 0.745  | 0.298     | 0.656 |
# | none            | 1139 | 3305  | 0.622  | 0.214     | 0.521 |
# | light_jacket    | 737  | 2717  | 0.843  | 0.229     | 0.706 |
# | red_life_jacket | 100  | 184   | 0.880  | 0.478     | 0.836 |
# +-----------------+------+-------+--------+-----------+-------+
# | mAP             |      |       |        |           | 0.695 |
# +-----------------+------+-------+--------+-----------+-------+