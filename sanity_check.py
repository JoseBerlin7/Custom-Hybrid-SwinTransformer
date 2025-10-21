import torch
from models import DHVNClassification

model = DHVNClassification(num_classes=10)
x = torch.randn(2, 3, 32, 32)
with torch.no_grad():
    y, d_g, u_g = model(x, get_gates=True)
print(y.shape)
print(d_g)
print(u_g)
