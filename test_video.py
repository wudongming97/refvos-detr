# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import time
import math
import os
from pathlib import Path

import numpy as np
from skimage.morphology import disk
import cv2

from dataset_utils import *
import util.misc as utils

from dataloader_davis import DAVIS17V2
from models.detr_vos import build

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--lr_drop', default=80, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=20, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--name', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

# # all boxes are [num, height, width] binary array
def compute_mean_IoU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    if U == 0:
        mean_IoU = 1
    else:
        mean_IoU = float(I) / U

    return mean_IoU

def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels
    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(np.bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))



    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F

def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

def read_video_part(video_part, device):
    images = video_part['images'].to(device)
    segannos = video_part['segannos'] if video_part.get('segannos') is not None else None
    sentence = torch.LongTensor(video_part['sentence']).to(device)
    object_id = video_part['object_id']
    fnames = video_part['fnames']
    return images, segannos, sentence, object_id, fnames

def main(args):

    print("Starting initial Testing")

    batch_size = 1
    nframes = 20
    nframes_val = 2

    num_steps = 20
    # im_h = 320
    # im_w = 320
    im_h = 240
    im_w = 432

    # size = (480, 864)
    size = (im_h, im_w)

    def image_read(path):
        pic = Image.open(path)
        transform = tv.transforms.Compose(
            [tv.transforms.Resize(size, interpolation=Image.BILINEAR),
             tv.transforms.ToTensor(),
             tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        return transform(pic)

    def label_read(path):
        if os.path.exists(path):
            pic = Image.open(path)
            transform = tv.transforms.Compose(
                [tv.transforms.Resize(size, interpolation=Image.NEAREST),
                 LabelToLongTensor()])
            label = transform(pic)
        else:
            label = torch.LongTensor(1, *size).fill_(255)  # Put label that will be ignored
        return label

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_folder = '../data/DAVIS/'

    val_set = DAVIS17V2(data_folder, '2017', 'val', image_read, label_read, None, nframes_val)

    print("Start testing")

    device = torch.device(args.device)

    model = build(args)
    model.to(device)
    # model = torch.nn.parallel.DataParallel(model)

    output_dir = './model/' + args.name
    model_file = output_dir + '/epoch_80.pth'
    # model = torch.nn.parallel.DataParallel(model)

    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_file)
    #
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in pretrained_dict.items():
    #     name = k[7:]  # remove `module.` ## 多gpu 训练带moudule默认参数名字,预训练删除
    #     new_state_dict[name] = v
    #
    # model_dict.update(new_state_dict)
    # model.load_state_dict(model_dict)

    model.load_state_dict(torch.load(model_file))

    model.eval()

    start_time = time.time()


    seg_total = 0
    score_thresh = [0.1, 0.2, 0.3, 0.4, 0.5]
    J_value = np.zeros(len(score_thresh), dtype=np.float)
    F_value = np.zeros(len(score_thresh), dtype=np.float)


    with torch.no_grad():

        for seqname, video_parts in val_set.get_video_generator():

            video_frames = 0

            for video_part in video_parts:

                images, segannos, sentence, object_id, fnames = read_video_part(video_part, device)

                # Read data
                vos_images = images.view([-1, 3, im_h, im_w])
                vos_segannos = segannos.float().view([-1, 1, im_h, im_w]).numpy()
                vos_sentence = sentence.view(-1, num_steps)

                outputs = model(vos_images, vos_sentence, int(vos_images.shape[0]))
                outputs = outputs[-1].sigmoid().cpu().detach().numpy()

                # predicts = outputs[3].softmax(dim=1).max(1)[1].cpu().detach().numpy()

                # predicts = ((outputs >= 0.5) * 255).astype(np.uint8)
                # save_file = os.path.join('Mask', seqname)
                # if not os.path.isdir(save_file):
                #     os.mkdir(save_file)
                # for i in range(predicts.shape[0]):
                #     img = predicts[i][0]
                #     img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                #     cv2.imwrite('./' + save_file + '/' + str(object_id) + '_' + str(i + video_frames) + '.jpg', img)

                for n_score in range(len(score_thresh)):

                    predicts = (outputs >= score_thresh[n_score]).astype(np.float32)

                    num_frames = predicts.shape[0]

                    for n_frame in range(num_frames):

                        mean_IoU = compute_mean_IoU(predicts[n_frame].squeeze(), vos_segannos[n_frame].squeeze())
                        J_value[n_score] += mean_IoU

                        mean_F = f_measure(predicts[n_frame].squeeze(), vos_segannos[n_frame].squeeze())
                        F_value[n_score] += mean_F

                        if score_thresh[n_score] == 0.5:
                            print('Seqname: {}, object_id: {}, frame: {}, IoU : {}, F : {}'.format(seqname, object_id, video_frames + n_frame + 1, mean_IoU, mean_F))

                # num_frames = predicts.shape[0]
                # for n_frame in range(num_frames):
                #
                #     mean_IoU = compute_mean_IoU(predicts[n_frame].squeeze(), vos_segannos[n_frame].squeeze())
                #     J_value[0] += mean_IoU
                #
                #     mean_F = f_measure(predicts[n_frame].squeeze(), vos_segannos[n_frame].squeeze())
                #     F_value[0] += mean_F
                #
                #
                #     print('Seqname: {}, object_id: {}, frame: {}, IoU : {}, F : {}'.format(seqname, object_id,
                #                                                                                video_frames + n_frame + 1,
                #                                                                                mean_IoU, mean_F))

                seg_total += num_frames

                video_frames += num_frames
                # msg = 'cumulative IoU = %f' % (cum_I / cum_U)
                # for n_eval_iou in range(len(eval_seg_iou_list)): # eval iou > .5, .6, .7, .8, .9
                #     eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                #     seg_correct[n_eval_iou] += (I / U >= eval_seg_iou)



        # Print results
        print('Segmentation evaluation')
        for n_score in range(len(score_thresh)):
            result_str = ''
            # for n_eval_iou in range(len(eval_seg_iou_list)):
            #     result_str += 'precision@%s = %f\n' % \
            #                   (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
            result_str += 'threshold %.2f: J = %.4f, F = %.4f' % (score_thresh[n_score], J_value[n_score] / seg_total, F_value[n_score] / seg_total)
            print(result_str)

        # result_str = ''
        # # for n_eval_iou in range(len(eval_seg_iou_list)):
        # #     result_str += 'precision@%s = %f\n' % \
        # #                   (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
        # result_str += 'J = %.4f, F = %.4f' % (J_value[0] / seg_total, F_value[0] / seg_total)
        # print(result_str)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
