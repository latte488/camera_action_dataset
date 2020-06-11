from camera import Camera
import camera
import cv2
import toml
from tqdm import tqdm
import os
import shutil

class ImagePathLoader:

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

def make(root, loader, actions, video_per_action, frame_number, frame_size, fps, fmt):
    fmt_list = list(fmt)
    if len(fmt_list) != 4:
        raise ValueError('fmt requires 4 characters')
    if os.path.exists(root):
        shutil.rmtree(root)
        os.makedirs(root)
    else:
        os.makedirs(root)
    fmt = cv2.VideoWriter_fourcc(fmt_list[0], fmt_list[1], fmt_list[2], fmt_list[3])
    camera = Camera(frame_number, frame_size)
    video_i = 0
    label_dict = { 'Action label' : {}, 'Video label' : {} }
    for action_label, action in enumerate(actions):
        action_name = action(camera).__class__.__name__
        label_dict['Action label'][action_name] = action_label
    total = len(loader) * len(actions) * video_per_action
    progress = tqdm(total=total, desc='Creating dataset')
    for loader_i in range(len(loader)):
        image = cv2.imread(loader[loader_i + 1])
        for action_label, action in enumerate(actions):
            action = action(camera)
            action_name = action.__class__.__name__
            for _ in range(video_per_action):
                video_i += 1
                video_name = f'{video_i}.mp4'
                video_path = f'{root}/{video_name}'
                action.start(image)
                writer = cv2.VideoWriter(video_path, fmt, fps, frame_size)
                for _ in range(camera.frame_number):
                    action.update()
                    writer.write(camera.capture(image))
                writer.release()
                label_dict['Video label'][video_name] = action_name
                progress.update(1)
    with open(f'{root}/0.toml', mode='w') as f:
        toml.dump(label_dict, f)

