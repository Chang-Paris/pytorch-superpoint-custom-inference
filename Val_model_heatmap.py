"""This is the main validation interface using heatmap trick

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""


import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data

from utils.utils import flattenDetection


@torch.no_grad()
class Val_model_heatmap():
    def __init__(self, config, device='cpu', verbose=False):
        self.config = config
        self.model = self.config['name']
        self.params = self.config['params']
        self.weights_path = self.config['pretrained']
        self.device=device

        ## other parameters

        # self.name = 'SuperPoint'
        # self.cuda = cuda
        self.nms_dist = self.config['nms']
        self.conf_thresh = self.config['detection_threshold']
        self.nn_thresh = 0.7#self.config['nn_thresh']  # L2 descriptor distance for good match.
        self.cell = 8  # deprecated
        self.cell_size = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.sparsemap = None
        self.heatmap = None # np[batch, 1, H, W]
        self.pts = None
        self.pts_subpixel = None
        ## new variables
        self.pts_nms_batch = None
        self.desc_sparse_batch = None
        self.patches = None
        pass

    def loadModel(self):
        # model = 'SuperPointNet'
        # params = self.config['model']['subpixel']['params']
        from utils.loader import modelLoader
        self.net = modelLoader(model=self.model, **self.params)

        checkpoint = torch.load(self.weights_path,
                                map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['model_state_dict'])

        self.net = self.net.to(self.device)
        logging.info('successfully load pretrained model from: %s', self.weights_path)
        pass

    def run(self, images):
        """
        input: 
            images: tensor[batch(1), 1, H, W]

        """
        from utils.var_dim import toNumpy

        with torch.no_grad():
            outs = self.net(images)
        semi = outs['semi']
        self.outs = outs

        heatmap = flattenDetection(semi, tensor=True)
            
        heatmap_np = toNumpy(heatmap)
        self.heatmap = heatmap_np
        return self.heatmap
        pass

    def heatmap_to_pts(self):
        heatmap_np = self.heatmap

        pts_nms_batch = [self.getPtsFromHeatmap(h) for h in heatmap_np] # [batch, H, W]
        self.pts_nms_batch = pts_nms_batch
        return pts_nms_batch

    def desc_to_sparseDesc(self):
        # pts_nms_batch = [self.getPtsFromHeatmap(h) for h in heatmap_np]
        desc_sparse_batch = [self.sample_desc_from_points(self.outs['desc'], pts) for pts in self.pts_nms_batch]
        self.desc_sparse_batch = desc_sparse_batch
        return desc_sparse_batch

    def getPtsFromHeatmap(self, heatmap):
        '''
        :param self:
        :param heatmap:
            np (H, W)
        :return:
        '''
        heatmap = heatmap.squeeze()
        # print("heatmap sq:", heatmap.shape)
        H, W = heatmap.shape[0], heatmap.shape[1]
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        self.sparsemap = (heatmap >= self.conf_thresh)
        if len(xs) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys # abuse of ys, xs
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]  # check the (x, y) here
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        return pts

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def sample_desc_from_points(self, coarse_desc, pts):
        # --- Process descriptor.
        H, W = coarse_desc.shape[2]*self.cell, coarse_desc.shape[3]*self.cell
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.to(self.device)
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return desc
