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

def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    # homographies = homographies.unsqueeze(0) if len(homographies.shape) == 2 else homographies
    batch_size = homographies.shape[0]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    homographies = homographies.view(batch_size*3,3)
    # warped_points = homographies*points
    # points = points.double()
    warped_points = homographies@points.transpose(0,1)
    # warped_points = np.tensordot(homographies, points.transpose(), axes=([2], [0]))
    # normalize the points
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0,:,:] if no_batches else warped_points

def inv_warp_image_batch(img, mat_homo_inv, device='cpu', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    '''
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)

    Batch, channel, H, W = img.shape
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H)), dim=2)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv, device)
    src_pixel_coords = src_pixel_coords.view([Batch, H, W, 2])
    src_pixel_coords = src_pixel_coords.float()

    warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
    return warped_img


def compute_valid_mask(image_shape, inv_homography, device='cpu', erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        `erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """
    # mask = H_transform(tf.ones(image_shape), homography, interpolation='NEAREST')
    # mask = H_transform(tf.ones(image_shape), homography, interpolation='NEAREST')
    if inv_homography.dim() == 2:
        inv_homography = inv_homography.view(-1, 3, 3)
    batch_size = inv_homography.shape[0]
    mask = torch.ones(batch_size, 1, image_shape[0], image_shape[1]).to(device)
    mask = inv_warp_image_batch(mask, inv_homography, device=device, mode='nearest')
    mask = mask.view(batch_size, image_shape[0], image_shape[1])
    mask = mask.cpu().numpy()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)

    return torch.tensor(mask).to(device)


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
        export=False,
        transform=None,
        task="train",
        seed=0,
        sequence_length=1,
        **config,
    ):
        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)

        self.transforms = transform
        self.action = "val"

        # get files
        self.root = Path(self.config["root"])  # Path(KITTI_DATA_PATH)
        """
        root_split_txt = self.config.get("root_split_txt", None)
        self.root_split_txt = Path(
            self.root if root_split_txt is None else root_split_txt
        )
        scene_list_path = (
            self.root_split_txt / "val.txt"
        )
        self.scenes = [
            # (label folder, raw image path)
            (Path(self.root / folder), Path(self.root / folder ) ) \
                for folder in open(scene_list_path)
        ]
        """
        self.crawl_folders(sequence_length)
        self.compute_valid_mask = compute_valid_mask
        if self.config['preprocessing']['resize']:
            self.sizer = self.config['preprocessing']['resize']

    def crawl_folders(self, sequence_length):
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

        valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))
        input.update({'image': img_aug})
        input.update({'valid_mask': valid_mask})

        name = sample['name']

        input.update({'name': name, 'scene_name': "./"})  # dummy scene name
        return input


# define data loader here
def dataLoader(config, dataset='', export_task='train'):
    logging.info(f"load dataset from : {dataset}")
    test_set = customDataset(
        export=True,
        task=export_task,
        **config['data'],
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
