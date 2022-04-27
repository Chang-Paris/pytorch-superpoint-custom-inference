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

# import from other modules todo needs to be cleaned
from utils.tools import dict_update
from utils.homographies import sample_homography_np as sample_homography
from utils.utils import compute_valid_mask
from utils.photometric import ImgAugTransform, customizedTransform
from utils.utils import inv_warp_image, inv_warp_image_batch, warp_points


# define custom dataset
class customDataset(data.Dataset):
    default_config = {
        "labels": None,
        "cache_in_memory": False,
        "validation_size": 100,
        "truncate": None,
        "preprocessing": {"resize": [240, 320]},
        "num_parallel_calls": 10,
        "augmentation": {
            "photometric": {
                "enable": False,
                "primitives": "all",
                "params": {},
                "random_order": True,
            },
            "homographic": {"enable": False, "params": {}, "valid_border_margin": 0},
        },
        "warped_pair": {"enable": False, "params": {}, "valid_border_margin": 0},
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
        self.action = "train" if task == "train" else "val"

        # get files
        self.root = Path(self.config["root"])  # Path(KITTI_DATA_PATH)

        root_split_txt = self.config.get("root_split_txt", None)
        self.root_split_txt = Path(
            self.root if root_split_txt is None else root_split_txt
        )
        scene_list_path = (
            self.root_split_txt / "train.txt"
            if task == "train"
            else self.root_split_txt / "val.txt"
        )
        self.scenes = [
            # (label folder, raw image path)
            (Path(self.root / folder), Path(self.root / folder ) ) \
                for folder in open(scene_list_path)
        ]
        # self.scenes_imgs = [
        #     Path(self.root / folder[:-4] / 'image_02' / 'data') for folder in open(scene_list_path)
        # ]

        ## only for export??

        if self.config["labels"]:
            self.labels = True
            self.labels_path = Path(self.config["labels"], task)
            print("load labels from: ", self.config["labels"] + "/" + task)
        else:
            self.labels = False

        self.crawl_folders(sequence_length)

        # other variables
        self.init_var()

    def init_var(self):
        torch.set_default_tensor_type(torch.FloatTensor)

        self.sample_homography = sample_homography
        self.inv_warp_image = inv_warp_image
        self.inv_warp_image_batch = inv_warp_image_batch
        self.compute_valid_mask = compute_valid_mask
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform
        self.warp_points = warp_points

        self.enable_photo_train = self.config['augmentation']['photometric']['enable']
        self.enable_homo_train = self.config['augmentation']['homographic']['enable']

        self.enable_homo_val = False
        self.enable_photo_val = False

        self.cell_size = 8
        if self.config['preprocessing']['resize']:
            self.sizer = self.config['preprocessing']['resize']

        self.gaussian_label = False
        if self.config['gaussian_label']['enable']:
            self.gaussian_label = True
            y, x = self.sizer
            # self.params_transform = {'crop_size_y': y, 'crop_size_x': x, 'stride': 1, 'sigma': self.config['gaussian_label']['sigma']}
        pass

    def crawl_folders(self, sequence_length):
        sequence_set = []
        # demi_length = (sequence_length-1)//2
        demi_length = sequence_length - 1
        # shifts = list(range(-demi_length, demi_length + 1))
        # shifts.pop(demi_length)
        for (scene, scene_img_folder) in self.scenes:
            # intrinsics and imu_pose_matrixs are redundant for superpoint training
            intrinsics = np.eye(3)
            # imu_pose_matrixs = np.eye(4)
            # intrinsics = (
            #     np.genfromtxt(scene / "cam.txt").astype(np.float32).reshape((3, 3))
            # )
            # imu_pose_matrixs = (
            #     np.genfromtxt(scene / "imu_pose_matrixs.txt")
            #     .astype(np.float64)
            #     .reshape(-1, 4, 4)
            # )
            # imgs = sorted(scene.files('*.jpg'))

            ##### get images
            # print(f"scene_img_folder: {scene_img_folder}")
            image_paths = list(scene_img_folder.iterdir())
            names = [p.stem for p in image_paths]
            imgs = [str(p) for p in image_paths]
            # files = {"image_paths": image_paths, "names": names}
            #####


            # X_files = sorted(scene.glob('*.npy'))
            if len(imgs) < sequence_length:
                continue
            # for i in range(demi_length, len(imgs)-demi_length):
            for i in range(0, len(imgs) - demi_length):
                sample = None
                # sample = {'intrinsics': intrinsics, 'imu_pose_matrixs': [imu_pose_matrixs[i]], 'imgs': [imgs[i]], 'Xs': [load_as_array(X_files[i])], 'scene_name': scene.name, 'frame_ids': [i]}
                if self.labels:
                    p = Path(self.labels_path, scene.name, "{}.npz".format(names[i]))
                    # print(f"label Path: {p}")
                    if p.exists():
                        sample = {
                            "intrinsics": intrinsics,
                            # "imu_pose_matrixs": [imu_pose_matrixs[i]],
                            "image": [imgs[i]],
                            "scene_name": scene.name,
                            "frame_ids": [i],
                        }
                        sample.update({"name": [names[i]], "points": [str(p)]})
                else:
                    sample = {
                        "intrinsics": intrinsics,
                        # "imu_pose_matrixs": [imu_pose_matrixs[i]],
                        "imgs": [imgs[i]],
                        "scene_name": scene.name,
                        "frame_ids": [i],
                        "name": [names[i]],
                    }

                if sample is not None:
                    # for j in shifts:
                    for j in range(1, demi_length + 1):
                        sample["image"].append(imgs[i + j])
                        # sample["imu_pose_matrixs"].append(imu_pose_matrixs[i + j])
                        # sample['Xs'].append(load_as_array(X_files[i])) # [3, N]
                        sample["frame_ids"].append(i + j)
                    sequence_set.append(sample)
                # print(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set
        logging.info("Finished crawl_folders for KITTI.")

    def get_img_from_sample(self, sample):
        # imgs = [_preprocess(_read_image(img)[:, :, np.newaxis]) for img in sample['imgs']]
        # imgs = [_read_image(img) for img in sample['imgs']]
        imgs_path = sample["imgs"]
        # print(len(imgs_path))
        # print(str(imgs_path[0]))

        # assert len(imgs) == self.sequence_length
        return str(imgs_path[0])

    def get_from_sample(self, entry, sample):
        # print(f"sample: {sample}")
        return str(sample[entry][0])

    def format_sample(self, sample):
        sample_fix = {}
        if self.labels:
            entries = ["image", "points", "name"]
            # sample_fix['image'] = get_img_from_sample(sample)
            for entry in entries:
                sample_fix[entry] = self.get_from_sample(entry, sample)
        else:
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
            # print(f"path: {path}, image: {image}")
            # print(f"path: {path}, image: {input_image.shape}")
            input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            H, W = input_image.shape[0], input_image.shape[1]
            # H = H//cell*cell
            # W = W//cell*cell
            # input_image = input_image[:H,:W,:]
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

            input_image = input_image.astype('float32') / 255.0
            return input_image

        def _preprocess(image):
            if self.transforms is not None:
                image = self.transforms(image)
            return image

        def get_labels_gaussian(pnts, subpixel=False):
            heatmaps = np.zeros((H, W))
            if subpixel:
                print("pnt: ", pnts.shape)
                for center in pnts:
                    heatmaps = self.putGaussianMaps(center, heatmaps)
            else:
                aug_par = {'photometric': {}}
                aug_par['photometric']['enable'] = True
                aug_par['photometric']['params'] = self.config['gaussian_label']['params']
                augmentation = self.ImgAugTransform(**aug_par)
                # get label_2D
                labels = points_to_2D(pnts, H, W)
                labels = labels[:, :, np.newaxis]
                heatmaps = augmentation(labels)

            # warped_labels_gaussian = torch.tensor(heatmaps).float().view(-1, H, W)
            warped_labels_gaussian = torch.tensor(heatmaps).type(torch.FloatTensor).view(-1, H, W)
            warped_labels_gaussian[warped_labels_gaussian > 1.] = 1.
            return warped_labels_gaussian

        from datasets.data_tools import np_to_tensor

        # def np_to_tensor(img, H, W):
        #     img = torch.tensor(img).type(torch.FloatTensor).view(-1, H, W)
        #     return img

        from datasets.data_tools import warpLabels

        def imgPhotometric(img):
            """

            :param img:
                numpy (H, W)
            :return:
            """
            augmentation = self.ImgAugTransform(**self.config['augmentation'])
            img = img[:, :, np.newaxis]
            img = augmentation(img)
            cusAug = self.customizedTransform()
            img = cusAug(img, **self.config['augmentation'])
            return img

        def points_to_2D(pnts, H, W):
            labels = np.zeros((H, W))
            pnts = pnts.astype(int)
            labels[pnts[:, 1], pnts[:, 0]] = 1
            return labels

        to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor)

        from numpy.linalg import inv
        sample = self.samples[index]
        sample = self.format_sample(sample)
        input = {}
        input.update(sample)
        # image
        # img_o = _read_image(self.get_img_from_sample(sample))
        img_o = _read_image(sample['image'])
        H, W = img_o.shape[0], img_o.shape[1]
        # print(f"image: {image.shape}")
        img_aug = img_o.copy()
        if (self.enable_photo_train == True and self.action == 'train') or (
                self.enable_photo_val and self.action == 'val'):
            img_aug = imgPhotometric(img_o)  # numpy array (H, W, 1)

        # img_aug = _preprocess(img_aug[:,:,np.newaxis])
        img_aug = torch.tensor(img_aug, dtype=torch.float32).view(-1, H, W)

        valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))
        input.update({'image': img_aug})
        input.update({'valid_mask': valid_mask})

        if self.config['homography_adaptation']['enable']:
            # img_aug = torch.tensor(img_aug)
            homoAdapt_iter = self.config['homography_adaptation']['num']
            homographies = np.stack([self.sample_homography(np.array([2, 2]), shift=-1,
                                                            **self.config['homography_adaptation']['homographies'][
                                                                'params'])
                                     for i in range(homoAdapt_iter)])
            ##### use inverse from the sample homography
            homographies = np.stack([inv(homography) for homography in homographies])
            homographies[0, :, :] = np.identity(3)
            # homographies_id = np.stack([homographies_id, homographies])[:-1,...]

            ######

            homographies = torch.tensor(homographies, dtype=torch.float32)
            inv_homographies = torch.stack([torch.inverse(homographies[i, :, :]) for i in range(homoAdapt_iter)])

            # images
            warped_img = self.inv_warp_image_batch(img_aug.squeeze().repeat(homoAdapt_iter, 1, 1, 1), inv_homographies,
                                                   mode='bilinear').unsqueeze(0)
            warped_img = warped_img.squeeze()
            # masks
            valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homographies,
                                                 erosion_radius=self.config['augmentation']['homographic'][
                                                     'valid_border_margin'])
            input.update({'image': warped_img, 'valid_mask': valid_mask, 'image_2D': img_aug})
            input.update({'homographies': homographies, 'inv_homographies': inv_homographies})

        # laebls
        if self.labels:
            pnts = np.load(sample['points'])['pts']
            # pnts = pnts.astype(int)
            # labels = np.zeros_like(img_o)
            # labels[pnts[:, 1], pnts[:, 0]] = 1
            labels = points_to_2D(pnts, H, W)
            labels_2D = to_floatTensor(labels[np.newaxis, :, :])
            input.update({'labels_2D': labels_2D})

            ## residual
            labels_res = torch.zeros((2, H, W)).type(torch.FloatTensor)
            input.update({'labels_res': labels_res})

            if (self.enable_homo_train == True and self.action == 'train') or (
                    self.enable_homo_val and self.action == 'val'):
                homography = self.sample_homography(np.array([2, 2]), shift=-1,
                                                    **self.config['augmentation']['homographic']['params'])

                ##### use inverse from the sample homography
                homography = inv(homography)
                ######

                inv_homography = inv(homography)
                inv_homography = torch.tensor(inv_homography).to(torch.float32)
                homography = torch.tensor(homography).to(torch.float32)
                #                 img = torch.from_numpy(img)
                warped_img = self.inv_warp_image(img_aug.squeeze(), inv_homography, mode='bilinear').unsqueeze(0)
                # warped_img = warped_img.squeeze().numpy()
                # warped_img = warped_img[:,:,np.newaxis]

                ##### check: add photometric #####

                # labels = torch.from_numpy(labels)
                # warped_labels = self.inv_warp_image(labels.squeeze(), inv_homography, mode='nearest').unsqueeze(0)
                ##### check #####
                warped_set = warpLabels(pnts, H, W, homography)
                warped_labels = warped_set['labels']
                # if self.transform is not None:
                # warped_img = self.transform(warped_img)
                valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homography,
                                                     erosion_radius=self.config['augmentation']['homographic'][
                                                         'valid_border_margin'])

                input.update({'image': warped_img, 'labels_2D': warped_labels, 'valid_mask': valid_mask})

            if self.config['warped_pair']['enable']:
                homography = self.sample_homography(np.array([2, 2]), shift=-1,
                                                    **self.config['warped_pair']['params'])

                ##### use inverse from the sample homography
                homography = np.linalg.inv(homography)
                #####
                inv_homography = np.linalg.inv(homography)

                homography = torch.tensor(homography).type(torch.FloatTensor)
                inv_homography = torch.tensor(inv_homography).type(torch.FloatTensor)

                # photometric augmentation from original image

                # warp original image
                warped_img = torch.tensor(img_o, dtype=torch.float32)
                warped_img = self.inv_warp_image(warped_img.squeeze(), inv_homography, mode='bilinear').unsqueeze(0)
                if (self.enable_photo_train == True and self.action == 'train') or (
                        self.enable_photo_val and self.action == 'val'):
                    warped_img = imgPhotometric(warped_img.numpy().squeeze())  # numpy array (H, W, 1)
                    warped_img = torch.tensor(warped_img, dtype=torch.float32)
                    pass
                warped_img = warped_img.view(-1, H, W)

                # warped_labels = warpLabels(pnts, H, W, homography)
                warped_set = warpLabels(pnts, H, W, homography, bilinear=True)
                warped_labels = warped_set['labels']
                warped_res = warped_set['res']
                warped_res = warped_res.transpose(1, 2).transpose(0, 1)
                # print("warped_res: ", warped_res.shape)
                if self.gaussian_label:
                    # print("do gaussian labels!")
                    # warped_labels_gaussian = get_labels_gaussian(warped_set['warped_pnts'].numpy())
                    from utils.var_dim import squeezeToNumpy
                    # warped_labels_bi = self.inv_warp_image(labels_2D.squeeze(), inv_homography, mode='nearest').unsqueeze(0) # bilinear, nearest
                    warped_labels_bi = warped_set['labels_bi']
                    warped_labels_gaussian = self.gaussian_blur(squeezeToNumpy(warped_labels_bi))
                    warped_labels_gaussian = np_to_tensor(warped_labels_gaussian, H, W)
                    input['warped_labels_gaussian'] = warped_labels_gaussian
                    input.update({'warped_labels_bi': warped_labels_bi})

                input.update({'warped_img': warped_img, 'warped_labels': warped_labels, 'warped_res': warped_res})

                # print('erosion_radius', self.config['warped_pair']['valid_border_margin'])
                valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homography,
                                                     erosion_radius=self.config['warped_pair'][
                                                         'valid_border_margin'])  # can set to other value
                input.update({'warped_valid_mask': valid_mask})
                input.update({'homographies': homography, 'inv_homographies': inv_homography})

            # labels = self.labels2Dto3D(self.cell_size, labels)
            # labels = torch.from_numpy(labels[np.newaxis,:,:])
            # input.update({'labels': labels})

            if self.gaussian_label:
                # warped_labels_gaussian = get_labels_gaussian(pnts)
                labels_gaussian = self.gaussian_blur(squeezeToNumpy(labels_2D))
                labels_gaussian = np_to_tensor(labels_gaussian, H, W)
                input['labels_2D_gaussian'] = labels_gaussian

        name = sample['name']
        to_numpy = False
        if to_numpy:
            image = np.array(img)

        input.update({'name': name, 'scene_name': "./"})  # dummy scene name
        return input


# define data loader here
def dataLoader(config, dataset='', export_task='train'):
    logging.info(f"load dataset from : {dataset}")
    from datasets.Kitti_inh import Kitti_inh as Dataset
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
