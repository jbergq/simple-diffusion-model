# %%

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent.absolute()))

import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.model.network.layers.positional_encoding import PositionalEncoding

# Get time steps
num_t_steps = 100
ts = torch.arange(num_t_steps)
print(ts)

# %% Create positional encoding vector for each time step

emb_size = 512
pos_enc = PositionalEncoding(1000, emb_size)

emb = pos_enc(ts)

print(emb)

# %% Plot sine and cosines over time

num_plots = 10
fig, axs = plt.subplots(num_plots, 1)

for i in range(num_plots):
    axs[i].plot(emb[1::2, 50 + i])

# %% Draw positional encoding vectors along x-axis

# Index all elements and create img coords
i = torch.arange(emb.shape[0] * emb.shape[1])
xs = torch.div(i, emb.shape[1], rounding_mode="trunc")
ys = torch.remainder(i, emb.shape[1])

# Draw in image according to magnitude
arr = torch.zeros((num_t_steps, emb_size))
arr[xs, ys] = emb.flatten()

# Show image with colorbar
plt.figure(figsize=(10, 10))
ax = plt.gca()
im = plt.imshow(arr, cmap="gnuplot")

plt.xlabel("Positional encoding vector")
plt.ylabel("Time step")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

# %%
