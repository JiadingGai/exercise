import torch, torch.nn.functional as F
from torch import nn

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
    def forward(self, x):
        return F.relu(self.conv(x))

model = torch.compile(Toy().eval(), backend="openxla")   # 1st call will trace
out   = model(torch.randn(1, 3, 224, 224))

# Print the captured graph
gm = model._torchdynamo_orig_callable.__compiled_fn      # grab GraphModule
print(gm.graph.print_tabular())

