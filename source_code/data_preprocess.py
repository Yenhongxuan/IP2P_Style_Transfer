import os
import argparse



import imageio
from tqdm import tqdm
import cv2
import numpy as np

from style_transfer.test.framework import Stylization
from utils import *

from ONNX_YOLOv8_Instance_Segmentation.yoloseg import YOLOSeg




def main(opt):
    yoloseg = YOLOSeg(opt.model_seg, conf_thres=0.3, iou_thres=0.3)


    if opt.style_bg == 'None':
        print('No styleed background')
    else:
        use_Global = False
        rshape = ReshapeTool()
        style = cv2.imread(opt.style_bg)
        style = cv2.resize(style, (style.shape[1]//opt.style_img_resize_ratio, style.shape[0]//opt.style_img_resize_ratio), cv2.INTER_AREA)
        style_fname = os.path.split(opt.style_bg)[1]
        print('Opened style image "{}"'.format(style_fname))
        style_framework = Stylization(opt.style_cp, cuda=True, use_Global=use_Global)
        style_framework.prepare_style(style)

        print('Loaded style framework "{}"'.format(opt.style_cp))

    video_id = count_folder(opt.output)
    video_dir = os.path.join(opt.output, f'video_{video_id}')
    os.makedirs(video_dir)

    video = imageio.get_reader(opt.input)
    try:
        for i, frame in tqdm(enumerate(video)):
            frame_id = count_folder(video_dir)
            frame_id = str(frame_id).zfill(4)
            frame_dir = os.path.join(video_dir, f'frame_{frame_id}')
            os.makedirs(frame_dir)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if opt.style_bg == 'None':
                frame_processing(frame, frame_dir, yoloseg)
            else:
                frame_processing(frame, frame_dir, yoloseg, style_framework, rshape)
    except Exception as e:
        print(f'An error occurred: {e}')

def frame_processing(frame, frame_dir, yoloseg, style_framework=None, reshape=None):
    cv2.imwrite(os.path.join(frame_dir, f'frame.jpg'), frame)
    boxes, scores, class_ids, masks = yoloseg(frame)
    all_masks = []
    for i, (box, score, class_id, mask) in enumerate(zip(boxes, scores, class_ids, masks)):
        if class_id == 0:
            mask = mask.astype(np.uint8)
            cv2.imwrite(os.path.join(frame_dir, f'person_mask_{i}.jpg'), mask)
            person_frame = np.zeros_like(frame)
            for j in range(person_frame.shape[2]):
                person_frame[:, :, j] = frame[:, :, j] * mask
            cv2.imwrite(os.path.join(frame_dir, f'person_frame_{i}.jpg'), person_frame)

            mask_gray_scale = 255 * mask
            cv2.imwrite(os.path.join(frame_dir, f'person_mask_{i}_gray_scale.jpg'), mask_gray_scale)
            all_masks.append(mask)
    all_masks = np.array(all_masks)
    all_mask = np.any(all_masks, axis=0).astype(np.uint8)
    cv2.imwrite(os.path.join(frame_dir, f'all_mask.jpg'), all_mask)
    all_mask_gray_scale = 255 * all_mask
    cv2.imwrite(os.path.join(frame_dir, f'all_mask_gray_scale.jpg'), all_mask_gray_scale)

    if style_framework is not None:
        background = np.zeros_like(frame)
        for j in range(background.shape[2]):
            background[:, :, j] = frame[:, :, j] * (1 - all_mask)
        background_styled = bg_style_transfer(background, style_framework, reshape)
        cv2.imwrite(os.path.join(frame_dir, f'background.jpg'), background)
        cv2.imwrite(os.path.join(frame_dir, f'background_styled.jpg'), background_styled)

def count_folder(folder_path):
    try:
        items = os.listdir(folder_path)
        subfolders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
        return len(subfolders)
    except Exception as e:
        print(f'An error occurred: {e}')
        return None

def bg_style_transfer(frame, style_framework, reshape):
    frame = resize(frame[..., ::-1])

    # Crop the image
    H,W,C = frame.shape
    new_input_frame = reshape.process(frame)

    # Stylization
    styled_input_frame = style_framework.transfer(new_input_frame)

    # Crop the image back
    styled_input_frame = styled_input_frame[64:64+H,64:64+W,:]

    # cast as unsigned 8-bit integer (not necessarily needed)
    styled_input_frame = styled_input_frame.astype('uint8')

    return styled_input_frame[..., ::-1]




def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str, default='../results/')
    parser.add_argument('--model_seg', type=str, default='./ONNX_YOLOv8_Instance_Segmentation/models/yolov8m-seg.onnx')
    parser.add_argument('--style_bg', type=str, default='./style_transfer/inputs/styles/starry_night.jpg')
    parser.add_argument('--style_cp', type=str, default="./style_transfer/test/Model/style_net-TIP-final.pth")
    parser.add_argument('--style_img_resize_ratio', type=float, default=1)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = opt_parser()
    main(opt)