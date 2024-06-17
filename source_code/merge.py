
import os
import argparse
import glob


import cv2
import numpy as np



def merge(frame_dir):
    prefix = "person_mask_*"
    search_pattern = os.path.join(frame_dir, prefix)
    person_masks = glob.glob(search_pattern)
    person_masks.sort()

    bg_styled = cv2.imread(os.path.join(frame_dir, 'background_styled.jpg'))
    all_mask = cv2.imread(os.path.join(frame_dir, 'all_mask.jpg'))
    height, width, channels = bg_styled.shape
    result = np.zeros_like(bg_styled)
    result += bg_styled * (1 - all_mask)
    
    for person_mask in person_masks:
        if 'gray_scale' in person_mask:
            continue
        mask_id = str(person_mask).split('/')[-1].split('.')[0].split('_')[-1]
        mask = cv2.imread(person_mask)
        print(frame_dir)
        print('mask_id', mask_id)
        fg_styled = cv2.imread(os.path.join(frame_dir, f'person_frame_{mask_id}_styled.jpg'))
        fg_resized = cv2.resize(fg_styled, (width, height), interpolation=cv2.INTER_CUBIC)
        result += fg_resized * mask
        
    cv2.imwrite(os.path.join(frame_dir, f'result.jpg'), result)







def main(opt):
    items = os.listdir(opt.video)
    frame_dirs = [item for item in items if os.path.isdir(os.path.join(opt.video, item))]
    frame_dirs.sort()
    for frame_dir in frame_dirs:
        merge(os.path.join(opt.video, frame_dir))


    codec = cv2.VideoWriter_fourcc(*opt.output_type)
    frames = []
    frame_size = None
    for i in range(len(frame_dirs)):
        img_path = os.path.join(opt.video, frame_dirs[i], 'result.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                if frame_size is None:
                    frame_size = (img.shape[1], img.shape[0])
                frames.append(img)
            else:
                raise Exception(f'Cound not read image: {img_path}')
        else:
            raise Exception(f'Path does not exist: {img_path}')
        
    if frames and frame_size:
        # Create a VideoWriter object
        output_path = os.path.join(opt.video, opt.output_file)
        out = cv2.VideoWriter(output_path, codec, opt.fps, frame_size)
        
        # Write each frame to the video
        for frame in frames:
            out.write(frame)
        
        # Release the VideoWriter object
        out.release()
        print(f"Video saved as {output_path}")
    else:
        print("No frames to write to the video")

def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='../results/video_0')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--output_file', type=str, default='result.mp4')
    parser.add_argument('--output_type', type=str, default='mp4v')
    opt = parser.parse_args()
    return opt



if __name__ == '__main__':
    opt = opt_parser()
    main(opt)
