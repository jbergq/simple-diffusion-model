# %%

import re
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent.absolute()))

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import PILToTensor

from src.data.datasets.mnist import MNIST
from src.utils.diffusion import forward_diffusion


dataset = MNIST("../data", train=True, download=True, transform=PILToTensor())

# %% Load MNIST image

img = dataset[0]["img"][0]

# %%
x = np.array(img)
x_t = x

# %%

num_iters = 300
beta_t = np.ones((num_iters)) * 0.005

# Forward diffusion process.
x_ts = [x_t]

for i in range(num_iters):
    x_t = np.random.normal(np.sqrt(1 - beta_t[i]) * x_t, np.sqrt(beta_t[i]))
    x_ts.append(x_t)

# %%

out_dir = Path("../output")
plot_dir = out_dir / "plots"
plot_dir.mkdir(exist_ok=True)

# matplotlib.use("Agg")

for t_step, x_t in enumerate(x_ts):
    fig, axs = plt.subplots(2)

    fig.suptitle(f"Timestep: {t_step}", fontsize=12)

    axs[0].imshow(x_t[None, :, :].transpose(1, 2, 0), cmap="gray")
    axs[1].hist(x_t.flatten(), bins=255, range=(-3, 3))
    axs[1].set_ylim([0, 30])
    axs[1].set_xlabel("Pixel value")
    axs[1].set_ylabel("Counts")

    fig.savefig(plot_dir / f"plot_{t_step}.png")

# %% Create animated GIF

plot_paths = list(plot_dir.glob("*.png"))
plot_paths = sorted(plot_paths, key=lambda path: int(re.search(r"(?<=plot_).*", path.stem).group(0)))

plot_imgs = []

for i, plot_path in enumerate(plot_paths[:200]):
    plot_img = cv2.imread(str(plot_path))

    if i == 0:  # Use 30 copies of the first frame
        plot_imgs.extend([plot_img for _ in range(30)])
    else:
        plot_imgs.append(plot_img)

imageio.mimsave(out_dir / "forward_diffusion_hist.gif", plot_imgs, duration=0.05)

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

x_t_hat = forward_diffusion(x, beta_t)

fig, axs = plt.subplots(2)

axs[0].imshow(x_t_hat, cmap="gray")
axs[1].hist(x_t_hat.flatten() * 255, bins=255)

plt.show()

# %%
