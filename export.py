"""
This script exports detection/ description using pretrained model.

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

## basic
import argparse
import cv2
import logging
import os
import yaml
from pathlib import Path

import numpy as np
## torch
import torch
import torch.optim
import torch.utils.data


torch.set_default_tensor_type(torch.FloatTensor)



def _read_image(path):
    input_image = cv2.imread(path)
    input_image = cv2.resize(input_image, (1248, 384),interpolation=cv2.INTER_AREA)

    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    input_image = input_image.astype('float32') / 255.0

    H, W = input_image.shape[0], input_image.shape[1]

    img_aug = torch.tensor(input_image, dtype=torch.float32).view(1, -1, H, W)

    return img_aug


@torch.no_grad()
def inference_superpoint(config, output_dir, img_path, args):

    # basic settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Inference on device: %s", device)

    # model loading
    from Val_model_heatmap import Val_model_heatmap
    ## load pretrained
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    # Run inference on dataloader
    img_0 = _read_image(img_path)

    # first image, no matches
    def get_pts_desc_from_agent(val_agent, img, device="cpu"):
        """
        pts: list [numpy (3, N)]
        desc: list [numpy (256, N)]
        """
        _ = val_agent.run(
            img.to(device)
        )  # heatmap: numpy [batch, 1, H, W]
        # heatmap to pts
        pts = val_agent.heatmap_to_pts()

        # heatmap, pts to desc
        desc_sparse = val_agent.desc_to_sparseDesc()

        outs = {"pts": pts[0], "desc": desc_sparse[0]}
        return outs

    outs = get_pts_desc_from_agent(val_agent, img_0, device=device)
    pts, desc = outs["pts"], outs["desc"]  # pts: np [3, N]

    # save keypoints
    pred = dict({"prob": pts.transpose(), "desc": desc.transpose()})
    filename = os.path.basename(img_path).split(".")[0]

    save_output = os.path.join(output_dir,  "predictions")
    os.makedirs(save_output, exist_ok=True)

    path = Path(save_output, "{}.npz".format(filename))
    np.savez_compressed(path, **pred)


if __name__ == "__main__":
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # inference using superpoint
    p_train = subparsers.add_parser("inference")
    p_train.add_argument("config", type=str)
    p_train.add_argument("exper_name", type=str)
    p_train.add_argument("input_img", type=str)
    p_train.set_defaults(func=inference_superpoint)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f)
    print("check config!! ", config)
    EXPER_PATH = "./logs"
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Running command {}".format(args.command.upper()))
    args.func(config, output_dir, args.input_img, args)
