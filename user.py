#user.py
import torch
import copy
import torch.optim as optim
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from config import (
    device,
    batch_size,
    base_lr,
    backbone_base_lr,
    min_lr,
    backbone_min_lr,
    momentum,
    weight_decay,
    num_epochs,
)
from model import MobileNetV3

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (
            x - self.mean.type_as(x)[None,:,None,None]
        ) / self.std.type_as(x)[None,:,None,None]
    
norm = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class User:
    def __init__(self, user_id, dataloader):
        self.user_id = user_id
        self.dataloader = dataloader
        self.model = MobileNetV3()
        self.model = self.model.to(device)
        
        self.loss_fn = nn.CrossEntropyLoss()

        #fully finetuning implementation
        #The backbone and head have different initial learning rates, but they share the same cosine decay schedule.
        #The ratio between them stays constant.

        head_params = self.model.head.parameters()
        backbone_params = [
            p for name, p in self.model.named_parameters()
            if not name.startswith("head")
        ]

        self.opt = optim.SGD(
            [
                {"params": head_params, "lr": base_lr},
                {"params": backbone_params, "lr": backbone_base_lr},
            ],
            momentum=momentum,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=num_epochs,
            eta_min=min_lr,
        )

    def train(self):
        self.model.train()
        correct = 0
        total = 0
        losses = []


        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if batch_idx == batch_size:
                break

            images = images.to(device)#send to gpu
            labels = labels.to(device)

            pred = self.model(norm(images))
            pred_labels = pred.argmax(dim=1)

            loss = self.loss_fn(pred, labels)

            #back propagation
            self.opt.zero_grad()
            loss.backward() 
            self.opt.step()
            losses.append(loss.item())

            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

            acc = round(correct/total, 2)
            avg_loss = round(sum(losses)/len(losses), 2)

        print(
        f"{self.user_id} | Loss: {avg_loss} | Acc: {acc} | "
        f"lr: {self.opt.param_groups[0]['lr']:.6f} | "
        f"backbone_lr: {self.opt.param_groups[1]['lr']:.6f}"
        )
        self.scheduler.step()
        return avg_loss, acc

    def FedProx(self):
        self.model.train()
        mu = 0.05
        correct = 0
        total = 0
        losses = []

        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if batch_idx == batch_size:
                break

            images = images.to(device)#send to gpu
            labels = labels.to(device)

            pred = self.model(norm(images))
            pred_labels = pred.argmax(dim=1)

            loss = self.loss_fn(pred, labels)

            prox = 0.0
            for name, param in self.model.named_parameters():
                if param.dtype.is_floating_point:
                    prox += (param - self.global_weight[name]).pow(2).sum()

            
            loss += prox * (mu/2)
            

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            losses.append(loss.item())

            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

            acc = round(correct/total, 2)
            avg_loss = round(sum(losses)/len(losses), 2)

        prox_drift = round(prox.item() * (mu/2), 2)
        self.scheduler.step()
        return avg_loss, acc, prox_drift



    def get_weight(self, global_weight=None):
        return self.model.state_dict()

    def set_weight(self, global_weight):

        self.model.load_state_dict(global_weight)
        self.global_weight = copy.deepcopy(global_weight)

