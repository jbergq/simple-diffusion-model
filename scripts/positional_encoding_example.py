# %%

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent.absolute()))

import torch
import matplotlib.pyplot as plt

from src.model.network.layers.positional_encoding import PositionalEncoding

x = torch.arange(100)
print(x)

# %%

pos_enc = PositionalEncoding(1000, 28 ** 2)

x_ = pos_enc(x)

print(x_)

# %%

num_plots = 10

fig, axs = plt.subplots(num_plots, 1)

for i in range(num_plots):
    axs[i].plot(x_[1::2, 50+i])
# %%
