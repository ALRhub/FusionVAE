import io
import os
import cv2
import sys
import lmdb
import torch
import argparse
import numpy as np
import torchvision

sys.path.append(os.path.join(sys.path[0], '..'))
from utilities import EllipseNoiseTransform
from config import cfg, update_and_save_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True, choices=["train", "valid"])
    parser.add_argument('--celeba_dir', type=str, required=True,
                        help='directory of the CelebA dataset. The dataset will be downloaded to this folder if it '
                             'does not exist yet')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory for the LMDB folders')
    args = parser.parse_args()

    cfg.args.dataset = 'celeba_64'
    cfg.args.n_noisy = 3  # number of noisy images per target image
    update_and_save_config()

    # args.celeba_dir = "/home/duf7rng/data/celeba/celeba_torch"
    # args.output_dir = os.path.join(os.getenv('DIR_DATA'), 'celeba')  # target location for storing lmdb files
    output_lmdb = os.path.join(args.output_dir, f'{args.split}.lmdb')

    os.makedirs(output_lmdb, exist_ok=True)

    # download the data if necessary
    ds = torchvision.datasets.CelebA(root=args.celeba_dir, split=args.split, download=True)

    if args.split == "valid":
        ellipse_noise_transform = EllipseNoiseTransform()

    # create lmdb
    env = lmdb.open(output_lmdb, map_size=1e12)
    with env.begin(write=True) as txn:
        for i in range(len(ds)):
            file_path = os.path.join(ds.root, ds.base_folder, "img_align_celeba", ds.filename[i])

            img = cv2.imread(file_path)[40:218 - 30, 15:178 - 15]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
            img = cv2.resize(src=img, dsize=(64, 64), interpolation=cv2.INTER_AREA)
            if args.split == "valid":
                img = np.transpose(img, axes=(2, 1, 0))
                im = ellipse_noise_transform(torch.tensor(img.astype(np.float32)/255.)).numpy()
                im = (im * 255.0).astype(np.uint8)
                img_split = np.split(im, indices_or_sections=(cfg.args.n_noisy + 1), axis=0)
                img_array = np.concatenate(img_split, axis=2)[0]
                img = np.transpose(img_array, axes=(2, 1, 0))
            is_success, buffer = cv2.imencode(ext=".png", img=img, params=[cv2.IMWRITE_PNG_COMPRESSION, 9])
            io_buf = io.BytesIO(buffer)
            txn.put(str(i).encode(), io_buf.getvalue())

            if i % 200 == 0 or (i == len(ds) - 1):
                print(i)
