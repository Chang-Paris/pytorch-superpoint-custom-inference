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
import random
import yaml
from pathlib import Path

import numpy as np
## torch
import torch
import torch.optim
import torch.utils.data
from tqdm import tqdm
import torch.utils.data as data
import torch.nn.functional as F


torch.set_default_tensor_type(torch.FloatTensor)

# todo needs to be cleaned
import collections


def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# define custom dataset
class customDataset(data.Dataset):
    default_config = {
        "cache_in_memory": False,
        "validation_size": 100,
        "truncate": None,
        "preprocessing": {"resize": [240, 320]},
        "num_parallel_calls": 10,
        "homography_adaptation": {"enable": False},
    }

    def __init__(
        self,
        config,
    ):
        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.root = Path(self.config["root"])

        self.crawl_folders()
        if self.config['preprocessing']['resize']:
            self.sizer = self.config['preprocessing']['resize']

    def crawl_folders(self):
        sequence_set = []

        for img_url in os.listdir(self.root):
            # intrinsics and imu_pose_matrixs are redundant for superpoint training
            intrinsics = np.eye(3)
            full_url = os.path.join(self.root, img_url)

            sample = {
                "intrinsics": intrinsics,
                "imgs": [full_url],
                "scene_name": "",
                "name": [""],
                "frame_ids": [0]
            }

            sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set
        logging.info("Finished crawl_folders for KITTI.")

    def get_img_from_sample(self, sample):
        imgs_path = sample["imgs"]
        return str(imgs_path[0])

    def get_from_sample(self, entry, sample):
        return str(sample[entry][0])

    def format_sample(self, sample):
        sample_fix = {}
        sample_fix["image"] = str(sample["imgs"][0])
        sample_fix["name"] = str(sample["scene_name"] + "/" + sample["name"][0])
        sample_fix["scene_name"] = str(sample["scene_name"])

        return sample_fix

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        '''

        :param index:
        :return:
            image: tensor (H, W, channel=1)
        '''
        def _read_image(path):
            cell = 8
            input_image = cv2.imread(path)
            input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            H, W = input_image.shape[0], input_image.shape[1]

            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            input_image = input_image.astype('float32') / 255.0
            return input_image

        sample = self.samples[index]
        sample = self.format_sample(sample)
        input = {}
        input.update(sample)

        img_o = _read_image(sample['image'])
        H, W = img_o.shape[0], img_o.shape[1]
        img_aug = img_o.copy()

        img_aug = torch.tensor(img_aug, dtype=torch.float32).view(-1, H, W)

        input.update({'image': img_aug})

        name = sample['name']

        input.update({'name': name, 'scene_name': "./"})  # dummy scene name
        return input


# define data loader here
def dataLoader(config, dataset='', export_task='train'):
    logging.info(f"load dataset from : {dataset}")
    test_set = customDataset(
        config['data'],
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        pin_memory=True
    )
    return {'test_set': test_set, 'test_loader': test_loader}


@torch.no_grad()
def inference_superpoint(config, output_dir, args):
    from utils.loader import get_save_path
    from utils.var_dim import squeezeToNumpy

    # basic settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Inference on device: %s", device)

    save_path = get_save_path(output_dir)
    save_output = save_path / "../predictions"
    os.makedirs(save_output, exist_ok=True)

    # data loading
    task = config["data"]["dataset"]
    data = dataLoader(config, dataset=task)
    test_set, test_loader = data["test_set"], data["test_loader"]

    # model loading
    from Val_model_heatmap import Val_model_heatmap
    ## load pretrained
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    # Run inference on dataloader
    count = 0
    for i, sample in tqdm(enumerate(test_loader)):
        img_0 = sample['image']

        # first image, no matches
        def get_pts_desc_from_agent(val_agent, img, device="cpu"):
            """
            pts: list [numpy (3, N)]
            desc: list [numpy (256, N)]
            """
            heatmap_batch = val_agent.run(
                img.to(device)
            )  # heatmap: numpy [batch, 1, H, W]
            # heatmap to pts
            pts = val_agent.heatmap_to_pts()

            # heatmap, pts to desc
            desc_sparse = val_agent.desc_to_sparseDesc()

            outs = {"pts": pts[0], "desc": desc_sparse[0]}
            return outs
        print(img_0)
        outs = get_pts_desc_from_agent(val_agent, img_0, device=device)
        pts, desc = outs["pts"], outs["desc"]  # pts: np [3, N]

        # save keypoints
        pred = {"image": squeezeToNumpy(img_0)}
        pred.update({"prob": pts.transpose(), "desc": desc.transpose()})

        filename = str(count)
        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)
        count += 1


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
    p_train.set_defaults(func=inference_superpoint)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f)
    print("check config!! ", config)
    EXPER_PATH = "./logs"
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Running command {}".format(args.command.upper()))
    args.func(config, output_dir, args)
