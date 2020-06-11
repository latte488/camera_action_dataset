import cv2
import torchvision
import toml
import numpy as np

class Dataset(torchvision.datasets.VisionDataset):


    def __init__(self, root, train=True, transforms=None, transform=None, target_transform=None):
        super(Dataset, self).__init__(root, transforms, transform, target_transform)
        self.root = root
        label_path = f'{self.root}/0.toml'
        label_dist = toml.load(label_path)
        self.action_label_dist = label_dist['Action label']
        self.path_and_label = list(label_dist['Video label'].items())
        data_number = len(self.path_and_label)
        train_number = int(data_number * 5 / 6)
        if train:
            self.number = train_number
            self.path_and_label = self.path_and_label[:train_number]
        else:
            self.path_and_label = self.path_and_label[train_number:]
            self.number = len(self.path_and_label)

    def __getitem__(self, index):
        video_name, label = self.path_and_label[index]
        video_path = f'{self.root}/{video_name}'
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError('No file in path. Check the `root` path.')
        frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video = []
        for _ in range(int(frame_number)):
            ok, image = cap.read()
            if not ok: 
                raise ValueError('Failed read video.')
            video.append(image.T)
        return np.array(video).astype(np.float32), self.action_label_dist[label]



    def __len__(self):
        return self.number

if __name__ == '__main__':
    root = 'data/uiuc_T01_camera_action_dataset'
    train_t01 = Dataset(root, train=True)
    test_t01 = Dataset(root, train=False)
    print(f'Train : {train_t01}')
    print(f'Test  : {test_t01}')
