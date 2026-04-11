import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if param.use_adapter:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels=22, out_channels=22, kernel_size=1),
                nn.BatchNorm2d(22),
            )
        self.use_adapter = param.use_adapter
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()
        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(22 * 4 * 200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(22 * 4 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(22 * 4 * 200, 4 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(4 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'convo':
            self.classifier = nn.Sequential(
                # (b, 22, 4, 200)

                # Khối Conv 1
                nn.Conv2d(22, 32, kernel_size=(1, 25), stride=(1, 25), bias=False), 
                nn.BatchNorm2d(32),
                nn.ELU(),
                # Output: (b, 32, 4, 8)

                # Khối Conv 2
                nn.Conv2d(32, 64, kernel_size=(4, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                # Output: (b, 64, 1, 8)  <-- Lúc này nén về 1 sẽ đỡ "thốn" hơn

                # Global Pooling
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(), # Output: 64

                # Fully Connected Layers
                nn.Linear(64, 16),
                nn.ELU(),
                nn.Dropout(param.dropout), 
                
                nn.Linear(16, param.num_of_classes)
            )
        elif param.classifier == 'linear':
            self.classifier = nn.Sequential(
                # B: Batch, C: 22, S: 4, D: 200
                Rearrange('b c s d -> b c (s d)'), # (b, 22, 800)
                
                # Nén từng kênh một (Shared weights across channels)
                nn.Linear(800, 16),
                nn.LayerNorm(16), # Giúp ổn định tín hiệu giữa các kênh
                nn.ELU(),
                
                # Trộn thông tin giữa các kênh
                Rearrange('b c d -> b (c d)'), # (b, 22 * 16 = 352)
                nn.Linear(22 * 16, 16),
                nn.ELU(),
                nn.Dropout(param.dropout),
                
                nn.Linear(16, param.num_of_classes),
            )

    def forward(self, x):
        # x = x / 100
        if self.use_adapter:
            x = self.adapter(x)
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out
