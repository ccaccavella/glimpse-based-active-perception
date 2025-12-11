from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF

from thop import profile


# -------------------------------------------------------------------------
# Helper functions (as in your original code)
# -------------------------------------------------------------------------

def gather_related_locations(locations_grid, loc_inds, invalid=0.):
    """
    locations_grid: [B, N, D]  (N = H*W)
    loc_inds:       [B, T, S]
    returns:        [B, T, S, D]
    """
    _, N, D = locations_grid.shape  # N = H*W, D - feature dim
    B, T, S = loc_inds.shape

    if loc_inds.max() == N:
        # special case: indices where idx == N are invalid and should map to zeros
        locations_grid = torch.cat(
            [locations_grid, invalid * torch.ones([B, 1, D], device=locations_grid.device)],
            dim=1
        )

    indices = loc_inds.reshape(B, T * S).unsqueeze(-1).expand(-1, -1, D)
    output = torch.gather(locations_grid, 1, indices).reshape([B, T, S, D])

    return output


def generate_local_indices(img_size, K, padding='constant', dilation=1, device='cpu'):
    """
    img_size: int or (H,W)
    K: neighbourhood (window) size
    returns: [1, H*W, K*K] index grid for each pixel
    """
    if isinstance(img_size, (tuple, list)):
        H, W = img_size
    else:
        H, W = img_size, img_size

    indice_maps = torch.arange(H * W, device=device).reshape([1, 1, H, W]).float()

    assert K % 2 == 1
    half_K = int((K - 1) / 2) * dilation

    assert padding in ['reflection', 'constant'], "unsupported padding mode"
    if padding == 'reflection':
        pad_fn = torch.nn.ReflectionPad2d(half_K)
    else:
        pad_fn = torch.nn.ConstantPad2d(half_K, H * W)

    indice_maps = pad_fn(indice_maps)
    local_inds = F.unfold(indice_maps, kernel_size=K, stride=1, dilation=dilation)  # [B, K*K, H*W]
    local_inds = local_inds.permute(0, 2, 1)  # [B, H*W, K*K]
    return local_inds



# -------------------------------------------------------------------------
# New profile-friendly class with a core submodule and custom FLOP counter
# -------------------------------------------------------------------------

class ErrorNeuronsCore(nn.Module):
    """
    Core saliency computation, separated as a submodule so we can attach
    a custom FLOP counter for thop.
    """
    def __init__(
            self,
            center_size,
            neighbourhood_size,
            dilation=1,
            similarity_fn_name: Literal['cos', 'mse'] = 'cos',
            wta_surround=False,
            blurr_image=False,
            normalize_patches=True,
            blurr_size=None,
            saliency_threshold=None,
            replication_padding=False,
            resize=None,
    ):
        super().__init__()
        self.center_size = center_size
        self.neighbourhood_size = neighbourhood_size
        self.dilation = dilation
        self.similarity_fn_name = similarity_fn_name
        self.wta_surround = wta_surround
        self.blurr_image = blurr_image
        self.normalize_patches = normalize_patches
        self.blurr_size = blurr_size
        self.saliency_threshold = saliency_threshold
        self.replication_padding = replication_padding
        self.resize = resize

        # Use standard modules where possible
        if replication_padding:
            self.center_pad = nn.ReplicationPad2d(center_size // 2)
        else:
            self.center_pad = None

        if normalize_patches:
            # We'll still use F.layer_norm for exact match; this is optional.
            # You could replace this with nn.LayerNorm if you want module-based stats.
            self.use_layer_norm_module = False
            if self.use_layer_norm_module:
                self.patch_norm = nn.LayerNorm(center_size * center_size)
        else:
            self.use_layer_norm_module = False

    def forward(self, image):
        # image: [B, C, H, W]
        if self.blurr_image:
            blurr_size = self.blurr_size or 7
            image = VF.gaussian_blur(image, [blurr_size, blurr_size])
            image = image / image.flatten(2).max(-1).values[..., None, None]

        if self.resize is not None:
            image = VF.resize(image, [self.resize, self.resize])

        # extract center patches
        if self.replication_padding and self.center_pad is not None:
            padded = self.center_pad(image)
            bu_preds = F.unfold(padded, self.center_size, stride=1)
        else:
            bu_preds = F.unfold(image, self.center_size, stride=1,
                                padding=self.center_size // 2)
        bu_preds = bu_preds.permute(0, 2, 1)  # [B, H*W, C*center_size^2]

        if self.normalize_patches:
            B, T, D = bu_preds.shape
            C = image.shape[1]
            bu = bu_preds.reshape(B, T, C, -1)
            # exact match with original: F.layer_norm over last dim (=center_size^2)
            bu = F.layer_norm(bu, [self.center_size ** 2])
            bu_preds = bu.reshape(B, T, D)
            bu_preds = bu_preds + 1.
            bu_preds = bu_preds + 1.

        B, C, H, W = image.shape
        T = H * W

        neighbour_inds = generate_local_indices(
            img_size=[H, W], K=self.neighbourhood_size,
            device=image.device, dilation=self.dilation
        ).expand(B, -1, -1).long()  # [B, H*W, S]

        node_inds = torch.arange(H * W, device=image.device).reshape([1, H * W, 1]).expand(
            B, -1, self.neighbourhood_size ** 2)

        center_idx = (self.neighbourhood_size ** 2) // 2
        neighbour_inds = neighbour_inds[..., torch.arange(self.neighbourhood_size ** 2) != center_idx]
        node_inds = node_inds[..., :-1]

        gathered_nodes = gather_related_locations(bu_preds, node_inds)
        gathered_surround = gather_related_locations(bu_preds, neighbour_inds)

        if self.similarity_fn_name == 'cos':
            cos_sim = F.cosine_similarity(gathered_nodes, gathered_surround, dim=-1)
            similarity_distance = 1 - ((1 + cos_sim) / 2)
        elif self.similarity_fn_name == 'mse':
            similarity_distance = torch.sum(
                (gathered_nodes - gathered_surround) ** 2, dim=-1
            )
        else:
            raise NotImplementedError

        if self.wta_surround:
            similarity_distance = torch.where(
                neighbour_inds == H * W,
                torch.inf, similarity_distance
            ).min(-1).values.reshape(-1, H, W)
        else:
            invalid_mask = torch.where(neighbour_inds == H * W, 0., 1.)
            similarity_distance = (similarity_distance * invalid_mask).sum(-1) / \
                                  invalid_mask.sum(-1)
            similarity_distance = similarity_distance.reshape(-1, H, W)

        saliency_map = similarity_distance

        if self.saliency_threshold is not None:
            saliency_map = torch.where(
                saliency_map >= self.saliency_threshold, saliency_map, 0.
            )

        return saliency_map


class ErrorNeuronsModule(nn.Module):
    """
    New version that wraps ErrorNeuronsCore as a submodule so we can attach
    a custom FLOP counter with thop.
    """
    def __init__(
            self,
            center_size,
            neighbourhood_size,
            dilation=1,
            similarity_fn_name: Literal['cos', 'mse'] = 'cos',
            wta_surround=False,
            blurr_image=False,
            normalize_patches=True,
            blurr_size=None,
            saliency_threshold=None,
            replication_padding=False,
            resize=None,
    ):
        super().__init__()
        self.core = ErrorNeuronsCore(
            center_size=center_size,
            neighbourhood_size=neighbourhood_size,
            dilation=dilation,
            similarity_fn_name=similarity_fn_name,
            wta_surround=wta_surround,
            blurr_image=blurr_image,
            normalize_patches=normalize_patches,
            blurr_size=blurr_size,
            saliency_threshold=saliency_threshold,
            replication_padding=replication_padding,
            resize=resize,
        )

    def forward(self, image):
        return self.core(image)


# -------------------------------------------------------------------------
# Custom FLOP counter for ErrorNeuronsCore (for thop)
# -------------------------------------------------------------------------

def count_error_neurons_core(m: ErrorNeuronsCore, inputs, output):
    """
    Custom FLOP counter for ErrorNeuronsCore.

    We approximate FLOPs by counting the patch similarity computations:

        - For MSE: ~3 * D flops per comparison
        - For Cosine: ~8 * D flops per comparison

    where:
        D = C * center_size^2
        comparisons per image = H * W * (neighbourhood_size^2 - 1)
    """
    x = inputs[0]  # image: [B, C, H, W]
    B, C, H, W = x.shape
    center_size = m.center_size
    neighbourhood_size = m.neighbourhood_size

    D = C * (center_size ** 2)
    T = H * W
    S = neighbourhood_size ** 2 - 1

    if m.similarity_fn_name == 'mse':
        flops_per = 3 * D
    elif m.similarity_fn_name == 'cos':
        flops_per = 8 * D
    else:
        flops_per = 0

    total_flops = B * T * S * flops_per

    # thop expects to add to m.total_ops
    m.total_ops += torch.DoubleTensor([float(total_flops)])


# # -------------------------------------------------------------------------
# # Small example: run both versions on an image and compare
# # -------------------------------------------------------------------------

# if __name__ == "__main__":
#     import cv2
#     import numpy as np

#     # --- Load image (BGR) and convert to tensor [1,3,H,W], float in [0,1] ---
#     img = cv2.imread("/home/caterina/Code/pySaliencyMap/test3.jpg")   # change path as needed
#     if img is None:
#         raise FileNotFoundError("Could not load test3.jpg")

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1,3,H,W]

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     img_t = img_t.to(device)

#     # --- Model hyperparameters ---
#     center_size = 5
#     neighbourhood_size = 3
    
#     from error_neurons import ErrorNeurons
#     # --- Instantiate old/plain and new/module versions ---
#     plain = ErrorNeurons(
#         center_size=center_size,
#         neighbourhood_size=neighbourhood_size,
#         similarity_fn_name='mse',
#         wta_surround=False,
#         blurr_image=False,
#         normalize_patches=True,
#         replication_padding=True,
#     ).to(device)

#     new = ErrorNeuronsModule(
#         center_size=center_size,
#         neighbourhood_size=neighbourhood_size,
#         similarity_fn_name='mse',
#         wta_surround=False,
#         blurr_image=False,
#         normalize_patches=True,
#         replication_padding=True,
#     ).to(device)

#     # --- Forward both ---
#     with torch.no_grad():
#         saliency_plain = plain(img_t)      # [1,H,W]
#         saliency_new = new(img_t)          # [1,H,W]

#     # --- Compare outputs ---
#     diff = (saliency_plain - saliency_new).abs()
#     print("Max diff between plain and new ErrorNeurons:", diff.max().item())
#     print("Mean diff:", diff.mean().item())

#     # --- Profile MACs of the new module with thop ---
#     macs, params = profile(
#         new, inputs=(img_t,),
#         custom_ops={ErrorNeuronsCore: count_error_neurons_core},
#         verbose=False
#     )
#     print("ErrorNeuronsModule MACs (approx):", macs)
#     print("ErrorNeuronsModule params:", params)

#     # plot in one fig salnecy old, new, and diference
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     plt.imshow(saliency_plain[0].cpu().numpy(), cmap='gray', origin='upper')
#     plt.title('Saliency Plain')
#     plt.axis('off')
#     plt.subplot(1, 3, 2)
#     plt.imshow(saliency_new[0].cpu().numpy(), cmap='gray', origin='upper')
#     plt.title('Saliency New')
#     plt.axis('off')
#     plt.subplot(1, 3, 3)
#     plt.imshow(diff[0].cpu().numpy(), cmap='hot', origin='upper')
#     plt.title('Difference')
#     plt.axis('off')
#     plt.show()
