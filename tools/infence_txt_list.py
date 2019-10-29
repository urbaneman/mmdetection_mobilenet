from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result
import mmcv
import cv2
import numpy as np
import torch
import os
import time

def bboxSelect(result, select_cls, CLASSES, score_thr=0.3):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)# draw bounding boxes
    # print(bboxes)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # print(labels)
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        # print(bboxes)
        labels = labels[inds]
        # print(labels)
        labels_select = np.full(labels.shape[0], select_cls)
        inds_l = (labels == labels_select)
        bboxes = bboxes[inds_l, :]
        # print(bboxes)
        labels = labels[inds_l]
        # print(labels)
    return bboxes, labels

# config_file = '../configs/faster_rcnn_r50_fpn_1x.py'
# # download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../work_dirs/faster_rcnn_r50_fpn_1x/epoch_12.pth'

config_file = '../configs/pascal_voc/ssd300_mobilenetv2_voc_1.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../work_dirs/ssd300_mobilenet_v2_helmet_2/latest.pth'


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img_path = '../demo/demo.jpg'

image_dir = r"/media/gzzn/WITAI/Dataset/COCO/COCO_baby/JPEGImages_person"
image_list_txt = r"/media/gzzn/WITAI/Dataset/COCO/COCO_baby/name_list_person/have_person_name_part.txt"
image_list = open(image_list_txt).readlines()
txt_save_dir = r"/media/gzzn/WITAI/Dataset/COCO/COCO_baby/none_hat"

CLASSES = ('person', 'blue', 'white', 'yellow', 'red', 'none', 'light_jacket', 'red_life_jacket')

if not os.path.exists(txt_save_dir):
    os.mkdir(txt_save_dir)
for image_name in image_list:
    image_name = image_name.strip()
    image_path = os.path.join(image_dir, image_name+'.jpg')
    # video_name_idx = os.path.splitext(video_name)[0].split('_')
    # video_class = "%s_%s_%s_%s"%(video_name_idx[1], video_name_idx[2], video_name_idx[3], video_name_idx[4])
    txt_name = image_name + ".txt"
    txt_path = os.path.join(txt_save_dir, txt_name)
    txt_w = open(txt_path, 'w')
    image = cv2.imread(image_path)
    result = inference_detector(model, image)
    bboxes, labels = bboxSelect(result, 5, model.CLASSES, 0.5)
    image_draw = image
    for idx, label in enumerate(labels):
        bbox = bboxes[idx, :-1]
        # img_crop = rotate90_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        image_draw = cv2.rectangle(image_draw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)
        person_line = "none %d %d %d %d\n" % (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])) # video_name_idx[0]
        txt_w.write(person_line)
        image_draw = cv2.putText(image_draw, CLASSES[label], (int(bbox[0]), int(bbox[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                 (0,255,0))
        # cv2.imwrite(frame_save_path, img_crop)
    txt_w.close()
    cv2.imshow('image_draw', image_draw)
    if cv2.waitKey(2) & 0xFF == ord('q'):  # ?q???
        break

cv2.destroyAllWindows()
