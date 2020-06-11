from camera_action_dataset import Dataset
import torch
import torchvision

def prepare_camera_action(root, batch_size):
    transform = torchvision.transforms.ToTensor()
    train_dataset = Dataset(
        root=root, 
        train=True,
        transform=transform
    )   
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )   
    test_dataset = Dataset(
        root=root,
        train=False,
        transform=transform
    )   
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )   
    return train_loader, test_loader

def prepare(batch_size):
    return prepare_camera_action(root='data/uiuc_T01_camera_action_dataset', batch_size=batch_size)
