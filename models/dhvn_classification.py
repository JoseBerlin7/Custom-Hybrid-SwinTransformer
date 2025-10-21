import torch.nn as nn
from .encoder import DHVNEncoder
from .decoder import DHVNDecoder

# # 6. Classification Model
class DHVNClassification(nn.Module):
    def __init__(self, in_ch=3, base_dim=32, norm="group", num_classes=10, dropout=0.0):
        super().__init__()
        self.encoder = DHVNEncoder(in_ch, base_dim, norm)
        self.decoder = DHVNDecoder(base_dim, norm)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_dim*4, num_classes)
        )


    def forward(self, x, get_gates=False):
        down_feats, down_gates = self.encoder(x)
        up_feats, up_gates = self.decoder(down_feats, classification=True)

        logits = self.fc(up_feats['u_s2'])

        if get_gates:
            return logits, down_gates, up_gates
        return logits