
import torch
import torch.nn as nn
from utilis.dataset import ASD_random_select_train
# from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from loguru import logger
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
from einops import rearrange
import os

# tensorboard --logdir=runs
# writer = SummaryWriter("runs/asd")

####### setting
EPOCH = 200
LR = 1e-4
BATCH_SIZE = 16
device = "cuda"
model_name = "resNet_mean_pool_base"
logger.add(f"experiment/logs/{model_name}.log")
######

class resNetForFrames(nn.Module):
    def __init__(self):
        super(resNetForFrames,self).__init__()
        self.resNet = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.fc = nn.Linear(in_features=512, out_features=2)
    
    def forward(self,x):
        out = 0
        for i in range(16):
            out += self.resNet(x[:,:,i,:]).squeeze()
        out = out / 16
        out = self.fc(out)
        return out

data_ds = ASD_random_select_train(mode="train")
data_ds_test = ASD_random_select_train(mode="test")
k = 5
splits=KFold(n_splits=k, shuffle=True, random_state=42)
model = resNetForFrames().to(device)
model = nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), LR)
criterion = nn.CrossEntropyLoss()

for fold, (train_idx,val_idx) in enumerate(splits.split(range(data_ds.__len__()))):

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(data_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(data_ds_test, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=4)

    best_test_value = 0
    for epoch in tqdm(range(EPOCH)):
        # train
        logger.info(f"training at epoch : {epoch}")
        model.train()
        train_loss = []
        train_acc = []
        for idx, (x, y) in enumerate(train_loader):

            if len(x) != BATCH_SIZE:
                continue

            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            out = model(x).squeeze()
            loss = criterion(out, y)
            train_loss.append(loss.item())
            train_acc.append(
                (torch.sum(torch.argmax(torch.sigmoid(out), dim=1) == y) / len(y)).cpu().detach().numpy()
            )

            loss.backward()
            optimizer.step()

        # writer.add_scalars("Base/train_acc", {f"{model_name}" : np.mean(train_acc)}, global_step=epoch)
        # writer.add_scalars("Base/train_loss", {f"{model_name}" : np.mean(train_loss)}, global_step=epoch)
        logger.info(f"train acc : {np.mean(train_acc)}")
        logger.info(f"train loss : {np.mean(train_loss)}")

        # eval
        model.eval()
        test_loss = []
        test_acc = []
        for idx, (x, y) in enumerate(test_loader):

            if len(x) != BATCH_SIZE:
                continue

            x = x.to(device)
            y = y.to(device)

            out = model(x).squeeze()
            loss = criterion(out, y)
            test_loss.append(loss.item())
            test_acc.append(
                (torch.sum(torch.argmax(torch.sigmoid(out), dim=1) == y) / len(y)).cpu().detach().numpy()
            )
        # writer.add_scalars("Base/test_acc", {f"{model_name}" : np.mean(test_acc)}, global_step=epoch)
        # writer.add_scalars("Base/test_loss", {f"{model_name}" : np.mean(test_loss)}, global_step=epoch)
        logger.info(f"test acc : {np.mean(test_acc)}")
        logger.info(f"test loss : {np.mean(test_loss)}")        

        if np.mean(test_acc) > best_test_value:
            best_test_value = np.mean(test_acc)
            logger.critical(f"bets test accuracy : {best_test_value}") 
            if not os.path.exists(f"experiment/pths/{model_name}"):
                os.makedirs(f"experiment/pths/{model_name}")
            torch.save(model.state_dict(), f"experiment/pths/{model_name}/fold_{fold}_{np.mean(test_acc)}.pth")

    break

