import os
import argparse
import glob





def opt_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='CP_final_result/video_1')
    parser.add_argument('--prompt', type=str, default='"Turn them into clowns"')
    parser.add_argument('--cuda', type=str, default='1')
    opt = parser.parse_args()
    return opt


def main(opt):
    video_name = str(opt.video).split('/')[-1]
    items = os.listdir(opt.video)
    frame_dirs = [item for item in items if os.path.isdir(os.path.join(opt.video, item))]
    command_list = []
    prefix = f'person_frame*'
    for frame_dir in frame_dirs:
        search_pattern = os.path.join(opt.video, frame_dir, prefix)
        person_frames = glob.glob(search_pattern)
        person_frames.sort()

        for person_frame in person_frames:
            input_path = person_frame
            output_path = input_path[:-4] + '_styled.jpg'
            command_list.append(f'CUDA_VISIBLE_DEVICES={opt.cuda} python edit_cli.py --input {input_path} --output {output_path} --edit {opt.prompt}')
    bash_name = f'{video_name}.sh'
    with open(bash_name, 'w') as f:
        for command in command_list:
            f.write(f'{command}\n')


if __name__ == '__main__':
    opt = opt_parser()
    main(opt)