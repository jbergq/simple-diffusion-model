# %%
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%

data_dir = Path("../data")

img_path = data_dir / "apple-touch-icon-144x144-precomposed.png"
img = cv2.imread(str(img_path))[:, :, :1]
img = img / 255

x = img
x_t = x

# %%

num_iters = 20
beta = 0.05

x_ts = [x_t]

for i in range(num_iters):
    x_t = np.random.normal(np.sqrt(1 - beta) * x_t, beta)
    x_ts.append(x_t)

# %%

for x_t in x_ts:
    fig, axs = plt.subplots(2)

    axs[0].imshow(x_t, cmap="gray")
    axs[1].hist(x_t.flatten() * 255, bins=255)
