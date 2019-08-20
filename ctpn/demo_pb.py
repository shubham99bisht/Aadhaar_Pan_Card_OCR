from __future__ import print_function

import glob
import os
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

sys.path.append(os.getcwd())
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import _get_blobs
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    new_boxes = ["date", "pay", "amount_wrd1", "amount_wrd2", "amount_dig", "account_no", "micr"]
    base_name = image_name.split('/')[-1]
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    print("shape",img.shape)
    color = (0,255,0)
    for i in range(len(boxes)):
        if i <7:
            label = new_boxes[i]
        else:
            label = "none"
        min_x, min_y, max_x, max_y = map(lambda x:x, boxes[i])
        if boxes[i]==[0,0,0,0]:
            continue
        cv2.line(img, (min_x, min_y), (min_x, max_y), color, 2)
        cv2.line(img, (min_x, min_y), (max_x, min_y), color, 2)
        cv2.line(img, (min_x, max_y), (max_x, max_y), color, 2)
        cv2.line(img, (max_x, min_y), (max_x, max_y), color, 2)
        x,y = max_x, min_y+ 10
        if i in [0,4,5]:
            x,y = min_x,max_y+16
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(os.path.join("data/results", base_name), img)
    # line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
    # f.write(line)


def clean_and_label(boxes, img, scale):
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    img_height, img_width = img.shape[0], img.shape[1]
    micr_box, max_y_global = [],0
    amt_wrd2, left_bottom = [], 0

    date, right_top = [], 10000
    amt_dig, right_bottom = [], 0

    remaining_boxes = []
    right_boxes = []

    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue

        min_x = int(min(int(box[0] // scale), int(box[2] // scale), int(box[4] // scale), int(box[6] // scale)))
        min_y = int(min(int(box[1] // scale), int(box[3] // scale), int(box[5] // scale), int(box[7] // scale)))
        max_x = int(max(int(box[0] // scale), int(box[2] // scale), int(box[4] // scale), int(box[6] // scale)))
        max_y = int(max(int(box[1] // scale), int(box[3] // scale), int(box[5] // scale), int(box[7] // scale)))

        if max_y > max_y_global:
            max_y_global = max_y
            micr_box = [min_x, min_y, max_x, max_y]

        # remove small boxes
        height = max_y - min_y
        if height/img_height < 0.045:
            continue

        # remove upper left boxes
        if min_y/img_height < 0.16 and min_x/img_width<0.6:
            continue

        # remove upper right box
        # if min_y/img_height < 0.065:
        #     if height/img_height > 0.085:
        #         pass
        #     else:
        #         continue

        # remove lower right boxes
        if min_y/img_height > 0.47:
            continue

        # remove lower left boxes
        if min_x/img_width<0.08:
            if max_x/img_width > 0.2:
                min_x = int(0.08*img_width)
            else:
                continue

        if min_x > img_width//2:
            right_boxes.append([min_x, min_y, max_x, max_y])

        if min_x<img_width//2:
            if max_y > left_bottom:
                left_bottom = max_y
                amt_wrd2 = [min_x, min_y, max_x, max_y]

        # if min_x>img_width//2:
        #     if max_y < right_top:
        #         right_top = max_y
        #         date = [min_x, min_y, max_x, max_y]
        #
        # if min_x > img_width//2:
        #     if max_y > right_bottom:
        #         right_bottom = max_y
        #         amt_dig = [min_x, min_y, max_x, max_y]

        if min_x < img_width//2:
            print("Adding to remaining_boxes")
            remaining_boxes += [[min_x, min_y, max_x, max_y]]

    print(amt_wrd2, micr_box)

    remaining_boxes.sort(key=lambda x: x[1], reverse = True)
    try:
        amt_wrd1 = remaining_boxes[1]
    except:
        amt_wrd1 = [0,0,0,0]
    try:
        pay = remaining_boxes[2]
    except:
        pay = [0,0,0,0]

    acc_minx = int(0.15*img_width)
    acc_miny = int(amt_wrd2[3]+ 0.03*img_height)
    acc_maxx= int(0.33*img_width)
    acc_maxy = int(acc_miny + 0.07*img_height)
    accountno = [acc_minx, acc_miny, acc_maxx, acc_maxy]

    right_boxes.sort(key = lambda x: (x[3]-x[1]), reverse= True)
    try:
        right_boxes = right_boxes[:2]
        right_boxes.sort(key = lambda x: x[1], reverse= True)
    except:
        pass
    try:
        date = right_boxes[1]
    except:
        date = [0,0,0,0]
    try:
        amt_dig = right_boxes[0]
    except:
        amt_dig = [0,0,0,0]

    new_boxes = [date, pay, amt_wrd1, amt_wrd2, amt_dig, accountno, micr_box]
    return new_boxes


if __name__ == '__main__':

    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with gfile.FastGFile('data/ctpn.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.tif'))

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        img = cv2.imread(im_name)
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        blobs, im_scales = _get_blobs(img, None)
        if cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                dtype=np.float32)
        cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        boxes = clean_and_label(boxes, img, scale)
        draw_boxes(img, im_name, boxes, scale)
