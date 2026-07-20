import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn


class MobileNetV3(nn.Module):
    
    def __init__(
        self,
        num_classes:int= 10,
        pretrained_weights:bool=True,
        freeze_backbone:bool=False
    ):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained_weights else None
        backbone = mobilenet_v3_small(weights=weights)

        in_features = backbone.classifier[0].in_features

        for p in backbone.parameters():
            p.requires_grad = not freeze_backbone

        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.head = nn.Linear(in_features, num_classes)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
