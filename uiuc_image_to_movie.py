import camera
import maker

class uiuc_texture(maker.ImagePathLoader):

    def __init__(self, number):
        self.root = 'data/uiuc_texture_dataset'
        self.type = f'T{number:02}'
        self.number = number

    def __getitem__(self, index):
        return f'{self.root}/{self.type}/{self.type}_{index:02}.jpg'

    def __len__(self):
       return 40

if __name__ == '__main__':
    actions = [
        camera.MoveUp,
        camera.MoveDown,
        camera.MoveLeft,
        camera.MoveRight,
        camera.RotateLeft,
        camera.RotateRight,
        camera.ZoomIn,
        camera.ZoomOut,
    ]
    for i in range(25):
        maker.make(
            root=f'data/uiuc_T{i + 1:02}_camera_action_dataset', 
            loader=uiuc_texture(i + 1), 
            actions=actions,
            video_per_action=4,
            frame_number=10,
            frame_size=(32, 32),
            fps=5,
            fmt='mp4v',
        )

