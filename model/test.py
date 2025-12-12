from model.glimpsing_process import SaliencyMapBasedGlimpsing
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# --------------------------
# Utilities
# --------------------------
def create_saliency_map(image_size, centers, radii, device="cpu"):
    """
    Returns a saliency map of shape [B, H, W] with values in [0,1].
    We intentionally return [B,H,W] (no channel dim) because
    SaliencyMapBasedGlimpsing.forward() will unsqueeze channel by itself.
    """
    sal = torch.zeros((1, image_size, image_size), dtype=torch.float32, device=device)
    # OpenCV draws in-place on a numpy array; use a CPU view
    sal_np = sal[0].cpu().numpy()
    for (cx, cy), r in zip(centers, radii):
        # cv.circle takes (x, y) == (col, row)
        cv.circle(sal_np, (cx, cy), r, (1.0,), thickness=-1)
    sal[0] = torch.from_numpy(sal_np)
    return sal.to(device)

def norm_to_pixels(locs, H, W):
    """
    Convert normalized coords in [-1,1] (x,y) to pixel coords (x,y),
    where x=column in [0..W-1], y=row in [0..H-1].
    Accepts locs of shape [..., 2].
    """
    x = (locs[..., 0] + 1.0) * 0.5 * (W - 1)  # columns
    y = (locs[..., 1] + 1.0) * 0.5 * (H - 1)  # rows
    return torch.stack([x, y], dim=-1)

# --------------------------
# Config
# --------------------------
device = "cpu"
image_size = 128
centers = [(32, 32), (96, 96)]
radii = [20, 15]
n_glimpses = 100

# --------------------------
# Build synthetic saliency
# --------------------------
saliency_map = create_saliency_map(image_size, centers, radii, device=device)  # [B,H,W]

# Visualize the map
plt.figure()
plt.imshow(saliency_map[0].cpu().numpy(), cmap='gray', origin='upper')
plt.title('Saliency Map')
plt.axis('off')
plt.show()

# --------------------------
# Glimpsing policy
# --------------------------
glimpsing_process = SaliencyMapBasedGlimpsing(
    image_size=image_size,      # must match the saliency spatial size
    ior_mask_size=100,           # IoR radius in "saliency pixels"
    error_neurons=None,         # not used here
    soft_ior=False,
    glimpse_size_to_mask_size_ratio=1,
)

B = saliency_map.shape[0]
H = W = image_size

# IoR history lives in saliency space: [B,1,H,W]
ior_mask_hist = torch.zeros(B, 1, H, W, device=device)

# Start location (normalized). (1,1) is the top-right/bottom-right corner depending on your convention.
# If you prefer center start, use torch.zeros(B, 2).
locations = torch.ones(B, 2, device=device)

loc_hist = []

for t in range(n_glimpses):
    # saliency_map is [B,H,W]; the module will unsqueeze channel internally
    locations, ior_mask_hist = glimpsing_process.forward(
        saliency_map, ior_mask_hist, locations
    )
    loc_hist.append(locations.detach().clone())

# Stack into [B, T, 2]
loc_hist = torch.stack(loc_hist, dim=1)
print("Saccade locations (normalized [-1,1], x then y):")
print(loc_hist)

# --------------------------
# Plot saccades on the map
# --------------------------
px_locs = norm_to_pixels(loc_hist[0], H, W).cpu().numpy()  # [T,2] -> (x,y)
xs, ys = px_locs[:, 0], px_locs[:, 1]

plt.figure(figsize=(5,5))
plt.imshow(saliency_map[0].cpu().numpy(), cmap='gray', origin='upper')
plt.title('Saccade sequence')
plt.axis('off')

# Red crosses for each fixation, with tiny order labels
for i, (x, y) in enumerate(zip(xs, ys), start=1):
    plt.plot(x, y, 'rx', markersize=8, mew=2)      # red "×"
    plt.text(x + 2, y - 2, str(i), color='r', fontsize=8)

# Optionally connect them with a thin line
plt.plot(xs, ys, '-', linewidth=1)
plt.savefig('/home/caterina/Code/glimpse-based-active-perception/saccade_sequence.png', bbox_inches='tight')
plt.show()
