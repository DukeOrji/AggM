#server.py
import torch
import torch.nn as nn
import copy
from collections import Counter
from model import MobileNetV3
from statistics import mean
from config import device
from user import norm


class Server:
    def __init__(self):
        
        self.global_model = MobileNetV3()
        self.global_model = self.global_model.to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        

        
    def broadcast_weight(self):
        return self.global_model.state_dict()


    def evaluate(self, dataloader):

        total = 0
        correct = 0

        losses = []

        with torch.no_grad():
            self.global_model.eval()

            for images, labels in dataloader:

                images = images.to(device)
                labels = labels.to(device)
                pred = self.global_model(norm(images))
                pred_labels = pred.argmax(dim=1)

                loss = self.loss_fn(pred, labels)
                losses.append(loss.item())
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)

            acc = round(correct/total, 2)
            avg_loss = round(mean(losses), 2)
        return avg_loss, acc
