import cv2
import torch
import random
import torchvision
import numpy as np
from config import cfg


@torch.jit.script
def apply_mask_and_noise(mask: torch.Tensor, noise: torch.Tensor, img: torch.Tensor, n_noisy: int, n_channels: int):  # , translation: torch.Tensor
    imgs = img[None].repeat(n_noisy + 1, 1, 1, 1)

    if n_channels == 3:
        # apply noise and mask on all RGB color channels equally
        noise = noise.repeat(1, 3, 1, 1)
        mask = mask[:, None].repeat(1, 3, 1, 1)
    else:
        mask = mask[:, None]

    imgs[0:n_noisy] *= mask  # apply noise mask
    imgs[0:n_noisy] += noise  # apply additive (Gaussian) noise
    imgs[0:n_noisy] = imgs[0:n_noisy].clamp_(min=0, max=1)
    return imgs


@torch.jit.script
def create_elliptic_mask(size: int, center: torch.Tensor, radius: torch.Tensor, ellip: torch.Tensor):
    x = torch.arange(size, dtype=torch.float32)[:, None]
    y = torch.arange(size, dtype=torch.float32)[None]

    # distance of each pixel to the ellipsis' center
    dist_from_center = torch.sqrt(ellip[:, :, None, None]*(x - center[:, :, 0:1, None])**2
                                  + (y - center[:, :, 1:2, None])**2/ellip[:, :, None, None])
    masks = dist_from_center <= radius[:, :, None, None]
    mask, _ = torch.max(masks, dim=0)
    return mask


class EllipseNoiseTransform:
    def __init__(self, seed=None):
        if seed:
            self.gen = torch.Generator()
            self.gen.manual_seed(seed)
        else:
            self.gen = None

    def __call__(self, img):
        n_noisy = cfg.args.n_noisy

        radius = torch.randint(low=cfg.noise.radius.low, high=cfg.noise.radius.high, size=(cfg.noise.n_ellipses, n_noisy), generator=self.gen)
        center = torch.randint(low=1, high=cfg.ds.res-2, size=(cfg.noise.n_ellipses, n_noisy, 2), generator=self.gen)
        ellip = torch.rand(size=(cfg.noise.n_ellipses, n_noisy), generator=self.gen) + 0.5

        gaussian_noise = cfg.noise.gaussian_var * torch.randn(size=(n_noisy, 1, img.shape[1], img.shape[2]), generator=self.gen) if cfg.noise.gaussian_var else torch.tensor(0)

        mask = create_elliptic_mask(size=img.shape[2], center=center, radius=radius, ellip=ellip)

        return apply_mask_and_noise(mask, gaussian_noise, img, n_noisy, n_channels=cfg.ds.n_channels)


def collate_fn_noisy_imgs(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels, dtype=torch.uint8)

    # reduce number of noisy images according to a random number
    n_noisy = torch.randint(low=0, high=cfg.n_noisy + 1, size=())

    imgs = imgs[:, cfg.args.n_noisy - n_noisy:cfg.args.n_noisy + 1]
    return imgs, labels


def _data_transforms_mnist():
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Pad(padding=2),
        torchvision.transforms.ToTensor(),
        EllipseNoiseTransform(),
    ])

    valid_transform = torchvision.transforms.Compose([
        torchvision.transforms.Pad(padding=2),
        torchvision.transforms.ToTensor(),
        EllipseNoiseTransform(seed=42),
    ])

    return train_transform, valid_transform


def _data_transforms_tless():
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        TlessReshapeTransform()
    ])

    return train_transform, train_transform


class ToTensor:
    def __call__(self, img):
        # convert cv2 image (uint8) to torch image (float32)
        return torch.from_numpy(np.float32(np.swapaxes(img, -1, -2)) / 255.)


class RandomHorizontalFlip:
    def __call__(self, img):
        if random.random() < 0.5:
            return cv2.flip(img, 1)
        return img


def _data_transforms_celeba64():
    train_transform = torchvision.transforms.Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        EllipseNoiseTransform()
    ])
    valid_transform = torchvision.transforms.Compose([
        ToTensor(),
    ])

    return train_transform, valid_transform


class TlessReshapeTransform:
    def __call__(self, img):
        img = torch.split(img, (cfg.ds.multi_image_encoding + 1) * [cfg.ds.res], dim=2)
        img = torch.stack(img, dim=0)
        return img
