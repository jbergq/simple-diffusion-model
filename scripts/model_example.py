# %%

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent.absolute()))

import torch

from src.model.network.unet import UNet

# %%

t = torch.tensor([100])
inp = torch.normal(0, 1, (1, 3, 256, 256))

model = UNet(3, 3, 5)

out = model(inp, t)

# %%
