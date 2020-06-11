import cv2
import random
import math
import os
import shutil

class Camera:

    def __init__(self, frame_number, frame_size):
        self.frame_number = frame_number
        self.frame_width, self.frame_height = frame_size
        self.reset()

    def reset(self):
        self.x = 0
        self.y = 0
        self.angle = 0
        self.scale = 1

    def capture(self, image):
        center_x = self.x + self.frame_width / 2
        center_y = self.y + self.frame_height / 2
        center = (center_x, center_y)
        mat = cv2.getRotationMatrix2D(center, self.angle, self.scale)
        image_height, image_width, _ = image.shape
        image_size = (image_width, image_height)
        top = self.y
        bottom = self.y + self.frame_height
        left = self.x
        right = self.x + self.frame_width
        return cv2.warpAffine(image, mat, image_size)[top:bottom, left:right]

class Action:

    def __init__(self, camera):
        self.camera = camera
    
    @property
    def frame_number(self):
        return self.camera.frame_number

    @property
    def frame_width(self):
        return self.camera.frame_width

    @property
    def frame_height(self):
        return self.camera.frame_height

    @property
    def x(self):
        return self.camera.x

    @x.setter
    def x(self, value):
        self.camera.x = value

    @property
    def y(self):
        return self.camera.y

    @y.setter
    def y(self, value):
        self.camera.y = value

    @property
    def angle(self):
        return self.camera.angle

    @angle.setter
    def angle(self, value):
        self.camera.angle = value

    @property
    def scale(self):
        return self.camera.scale

    @scale.setter
    def scale(self, value):
        self.camera.scale = value

    def reset(self):
        self.camera.reset()

    def start(self, image):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

class MoveUp(Action):

    def __init__(self, camera):
        super(MoveUp, self).__init__(camera)

    def start(self, image):
        self.reset()
        image_height, image_width, _ = image.shape
        self.x = random.randint(0, image_width - self.frame_width)
        self.y = random.randint(self.frame_number, image_height - self.frame_height)

    def update(self):
        self.y -= 1

class MoveDown(Action):

    def __init__(self, camera):
        super(MoveDown, self).__init__(camera)

    def start(self, image):
        self.reset()
        image_height, image_width, _ = image.shape
        self.x = random.randint(0, image_width - self.frame_width)
        self.y = random.randint(0, image_height - self.frame_height - self.frame_number)

    def update(self):
        self.y += 1

class MoveLeft(Action):

    def __init__(self, camera):
        super(MoveLeft, self).__init__(camera)

    def start(self, image):
        self.reset()
        image_height, image_width, _ = image.shape
        self.x = random.randint(0 + self.frame_number, image_width - self.frame_width)
        self.y = random.randint(0, image_height - self.frame_height)

    def update(self):
        self.x -= 1

class MoveRight(Action):

    def __init__(self, camera):
        super(MoveRight, self).__init__(camera)

    def start(self, image):
        self.reset()
        image_height, image_width, _ = image.shape
        self.x = random.randint(0, image_width - self.frame_width - self.frame_number)
        self.y = random.randint(0, image_height - self.frame_height)

    def update(self):
        self.x += 1

class Rotate(Action):

    def __init__(self, camera, right):
        super(Rotate, self).__init__(camera)
        self.direction =1 if right else -1

    def start(self, image):
        self.reset()
        half_diagonal = math.sqrt(self.frame_width ** 2 + self.frame_height ** 2) // 2 + 1
        image_height, image_width, _ = image.shape
        self.x = random.randint(0 + half_diagonal, image_width - half_diagonal - self.frame_width)
        self.y = random.randint(0 + half_diagonal, image_height - half_diagonal - self.frame_height)

    def update(self):
        self.angle += 180 / (self.frame_number - 1) * self.direction

class RotateLeft(Rotate):

    def __init__(self, camera):
        super(RotateLeft, self).__init__(camera, False)

class RotateRight(Rotate):

    def __init__(self, camera):
        super(RotateRight, self).__init__(camera, True)

class ZoomIn(Action):

    def __init__(self, camera):
        super(ZoomIn, self).__init__(camera)

    def start(self, image):
        self.reset()
        image_height, image_width, _ = image.shape
        self.x = random.randint(0, image_width - self.frame_width)
        self.y = random.randint(0, image_height - self.frame_height)
    
    def update(self):
        self.scale += 1 / (self.frame_number - 1)

class ZoomOut(Action):

    def __init__(self, camera):
        super(ZoomOut, self).__init__(camera)

    def start(self, image):
        self.reset()
        half_frame_width = self.frame_width // 2 + 1
        half_frame_height = self.frame_height // 2 + 1
        image_height, image_width, _ = image.shape
        self.x = random.randint(0 + half_frame_width, image_width - self.frame_width - half_frame_width)
        self.y = random.randint(0 + half_frame_height, image_height - self.frame_height - half_frame_height)

    def update(self):
        self.scale -= 0.5 / (self.frame_number - 1)

class MoveUpLeft(Action):
    pass

class MoveUpRight(Action):
    pass

class MoveDownLeft(Action):
    pass

class MoveDownRight(Action):
    pass

if __name__ == '__main__':
    root = 'data/sample'
    if os.path.exists(root):
        shutil.rmtree(root)
        os.makedirs(root)
    else:
        os.makedirs(root)
    image_path = 'data/uiuc_texture_dataset/T01/T01_01.jpg'
    fps = 60
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    camera = Camera(frame_number=100, frame_size=(128, 128))
    image = cv2.imread(image_path)
    actions = [
        MoveUp(camera),
        MoveDown(camera),
        MoveLeft(camera),
        MoveRight(camera),
        RotateLeft(camera),
        RotateRight(camera),
        ZoomIn(camera),
        ZoomOut(camera),
    ]
    frame_size = (camera.frame_width, camera.frame_height)
    for action in actions:
        action.start(image)
        writer = cv2.VideoWriter(f'data/sample/{action.__class__.__name__}.mp4', fmt, fps, frame_size)
        for i in range(camera.frame_number):
            action.update()
            writer.write(camera.capture(image))
        writer.release()
