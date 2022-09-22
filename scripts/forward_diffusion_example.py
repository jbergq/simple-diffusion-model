# %%
import re
from pathlib import Path

import imageio
import cv2
import numpy as np
import matplotlib
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
beta_t = np.ones((num_iters)) * 0.05

x_ts = [x_t]

for i in range(num_iters):
    x_t = np.random.normal(np.sqrt(1 - beta_t[i]) * x_t, np.sqrt(beta_t[i]))
    x_ts.append(x_t)

# %%

out_dir = Path("../output")
plot_dir = out_dir / "plots"

# matplotlib.use("Agg")

for i, x_t in enumerate(x_ts):
    fig, axs = plt.subplots(2)

    axs[0].imshow(x_t, cmap="gray")
    axs[1].hist(x_t.flatten() * 255, bins=255)

    fig.savefig(plot_dir / f"plot_{i}.png")

# %% Create animated GIF

plot_paths = plot_dir.glob("*.png")
plot_paths = sorted(
    plot_paths, key=lambda path: int(re.search(r"(?<=plot_).*", path.stem).group(0))
)

plot_imgs = []

for plot_path in plot_paths:
    plot_imgs.append(cv2.imread(str(plot_path)))

imageio.mimsave(out_dir / "forward_diffusion_hist.gif", plot_imgs, duration=0.5)

# %%

imageio.mimsave(
    out_dir / "forward_diffusion.gif",
    [(np.clip(x_t, 0, 1) * 255).astype(dtype=np.uint8) for x_t in x_ts],
    duration=0.5,
)

# %% Reparametrization
# A nice property of forward diffusion process is that we can sample at any arbitrary time
# step  in a closed form using a reparameterization trick.
#
# Because sum of normal distr. variables is also norm dist., can express x_t in terms of x_0.

alpha_t = 1 - beta_t

alpha_hat = np.prod(alpha_t)

x_t_hat = np.random.normal(
    np.sqrt(alpha_hat) * x, np.sqrt(1 - alpha_hat) * np.identity(x.shape[-1])
)

fig, axs = plt.subplots(2)

axs[0].imshow(x_t_hat, cmap="gray")
axs[1].hist(x_t_hat.flatten() * 255, bins=255)

plt.show()

# %%
