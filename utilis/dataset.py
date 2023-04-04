from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import torch
import pickle
import numpy as np
import os
import torch.nn.functional as F
from einops import rearrange

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()

class ASD(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []
        for root, dirs, files in os.walk("/root/Desktop/video_classification/asd_224x224_same_step"):
            for file in files:
                with open(os.path.join(root, file), 'rb') as video_file:
                    video = pickle.load(video_file)
                    self.data.append(video)

                    self.labels.append(int(file.split("_")[1].split(".")[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        img = torch.as_tensor(img, dtype=torch.float32).permute(3, 0, 1, 2) / 255
        # img = F.interpolate(img, size=(224, 224), mode='nearest')

        label = torch.as_tensor(label, dtype=torch.long)

        return img, label

class ASD_random_select_train(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.data = []
        self.labels = []
        for root, dirs, files in os.walk("/root/Desktop/video_classification/asd_224x224_all_frames"):
            for file in files:
                with open(os.path.join(root, file), 'rb') as video_file:
                    video = pickle.load(video_file)
                    self.data.append(video)

                    self.labels.append(int(file.split("_")[1].split(".")[0]))

    def split_video_random(self, frames):
        index = []
        for i in range(16):
            index.append(np.random.randint(frames))
        return np.array(index)

    def split_video_same_stpes(self, frames, step):
        index = []
        for i in range(frames):
            if i % int(frames / step) == 0 and len(index) != step:
                index.append(i)

        return np.array(index) 

    def block_random_split(self, frames):
        num_block = int(frames / 16)
        index = []
        for i in range(16):
            index.append(np.random.randint(i*num_block, (i+1)*num_block))
        return index        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        if self.mode == "train":
            img = img[self.split_video_random(img.shape[0]), :]
            img = torch.as_tensor(img, dtype=torch.float32).permute(3, 0, 1, 2) / 255
        elif self.mode == "test":
            img = img[self.split_video_same_stpes(img.shape[0], step=16), :]
            img = torch.as_tensor(img, dtype=torch.float32).permute(3, 0, 1, 2) / 255
        elif self.mode == "inference":
            img = torch.as_tensor(img, dtype=torch.float32).permute(3, 0, 1, 2) / 255
            index_ = self.split_video_same_stpes(img.shape[1], step=16)
            imgs = []
            for i in range((index_[1] - index_[0])):
                imgs.append(img[:, (index_ + i), :])
        label = self.labels[index]

        label = torch.as_tensor(label, dtype=torch.long)

        if self.mode =="inference":
            return imgs, label
        else:
            return img, label


if __name__ == "__main__":
    data_ds = ASD_random_select_train(mode="train")
    train_loader = DataLoader(data_ds, batch_size=32)
    for idx, (x, y) in enumerate(train_loader):
        print(x.shape)
