# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from lib.net.voxelize import Voxelization
from lib.dataset.mesh_util import feat_select, read_smpl_constants
from lib.net.NormalNet import NormalNet
from lib.net.MLP import MLP
from lib.net.spatial import SpatialEncoder
from lib.dataset.PointFeat import PointFeat
from lib.dataset.mesh_util import SMPLX
from lib.net.VE import VolumeEncoder
from lib.net.HighVE import HighVolumeEncoder
# from lib.net.HighVE_late import HighVolumeEncoder
from lib.net.HGFilters import *
from termcolor import colored
from lib.net.BasePIFuNet import BasePIFuNet
import torch.nn as nn
import torch
import ipdb

class HGPIFuNet(BasePIFuNet):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """
    def __init__(self, cfg, projection_mode="orthogonal", error_term=nn.MSELoss()):

        super(HGPIFuNet, self).__init__(projection_mode=projection_mode, error_term=error_term)

        self.l1_loss = nn.SmoothL1Loss()
        self.opt = cfg.net
        self.root = cfg.root
        self.overfit = cfg.overfit

        channels_IF = self.opt.mlp_dim

        self.use_filter = self.opt.use_filter
        self.prior_type = self.opt.prior_type
        self.smpl_feats = self.opt.smpl_feats

        self.smpl_dim = self.opt.smpl_dim
        self.voxel_dim = self.opt.voxel_dim
        self.hourglass_dim = self.opt.hourglass_dim
        
        self.high_res_voxel = self.opt.high_res_voxel
        self.use_normals = self.opt.use_normals

        self.in_geo = [item[0] for item in self.opt.in_geo]
        self.in_nml = [item[0] for item in self.opt.in_nml]

        self.in_geo_dim = sum([item[1] for item in self.opt.in_geo])
        self.in_nml_dim = sum([item[1] for item in self.opt.in_nml])

        self.in_total = self.in_geo + self.in_nml
        self.smpl_feat_dict = None
        self.smplx_data = SMPLX()

        image_lst = [0, 1, 2]
        normal_F_lst = [0, 1, 2] if "image" not in self.in_geo else [3, 4, 5]
        normal_B_lst = [3, 4, 5] if "image" not in self.in_geo else [6, 7, 8]

        # only ICON or ICON-Keypoint use visibility

        if self.prior_type in ["icon", "keypoint"]:
            if "image" in self.in_geo:   
                self.channels_filter = [     # [[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8]]
                    image_lst + normal_F_lst,
                    image_lst + normal_B_lst,
                ]
            else:
                self.channels_filter = [normal_F_lst, normal_B_lst] # [[0, 1, 2], [3, 4, 5]]

        else:
            if self.use_normals:
                if "image" in self.in_geo:
                    self.channels_filter = [image_lst + normal_F_lst + normal_B_lst] # [0, 1, 2, 3, 4, 5, 6, 7, 8]
                else:
                    self.channels_filter = [normal_F_lst + normal_B_lst] # [0, 1, 2, 3, 4, 5]
            else:
                if "image" in self.in_geo:
                    self.channels_filter = [image_lst] # [0, 1, 2]
                else:
                    self.channels_filter = [] # []

        use_vis = (self.prior_type in ["icon", "keypoint"]) and ("vis" in self.smpl_feats)
        if self.prior_type in ["pamir", "pifu", "refinepifu"]:
            use_vis = 1

        if self.use_filter:
            channels_IF[0] = (self.hourglass_dim) * (2 - use_vis)
        else:
            channels_IF[0] = len(self.channels_filter[0]) * (2 - use_vis)

        if self.prior_type in ["icon", "keypoint"]:
            channels_IF[0] += self.smpl_dim
            
        elif self.prior_type == "pamir":
            channels_IF[0] += self.voxel_dim
            (
                smpl_vertex_code,
                smpl_face_code,
                smpl_faces,
                smpl_tetras,
            ) = read_smpl_constants(self.smplx_data.tedra_dir)
            
            self.voxelization = Voxelization(
                smpl_vertex_code,
                smpl_face_code,
                smpl_faces,
                smpl_tetras,
                volume_res=128,
                sigma=0.05,
                smooth_kernel_size=7,
                batch_size=cfg.batch_size,
                device=torch.device(f"cuda:{cfg.gpus[0]}"),
            )
            self.ve = VolumeEncoder(3, self.voxel_dim, self.opt.num_stack)

        elif self.prior_type == "pifu":
            channels_IF[0] += 1
            
        elif self.prior_type == "refinepifu":
            channels_IF[0] += self.smpl_dim
            channels_IF[0] += self.voxel_dim
            (
                smpl_vertex_code,
                smpl_face_code,
                smpl_faces,
                smpl_tetras,
            ) = read_smpl_constants(self.smplx_data.tedra_dir)
            self.voxelization = Voxelization(
                smpl_vertex_code,
                smpl_face_code,
                smpl_faces,
                smpl_tetras,
                volume_res=128,
                sigma=0.05,
                smooth_kernel_size=7,
                batch_size=cfg.batch_size,
                device=torch.device(f"cuda:{cfg.gpus[0]}"),
            )
            self.ve = VolumeEncoder(3, self.voxel_dim, self.opt.num_stack)
            
            if self.high_res_voxel:
                self.highvoxelization = Voxelization(
                    smpl_vertex_code,
                    smpl_face_code,
                    smpl_faces,
                    smpl_tetras,
                    volume_res=256,
                    sigma=0.05,
                    smooth_kernel_size=7,
                    batch_size=cfg.batch_size,
                    device=torch.device(f"cuda:{cfg.gpus[0]}"),
                )
                self.highve = HighVolumeEncoder(3+self.voxel_dim, 7, 1)
                # self.highve = HighVolumeEncoder(3, 7, 1)

            # ipdb.set_trace()
            
        else:
            print(f"don't support {self.prior_type}!")

        self.base_keys = ["smpl_verts", "smpl_faces"]

        self.icon_keys = self.base_keys + [f"smpl_{feat_name}" for feat_name in self.smpl_feats]
        self.keypoint_keys = self.base_keys + [f"smpl_{feat_name}" for feat_name in self.smpl_feats]

        self.pamir_keys = ["voxel_verts", "voxel_faces", "pad_v_num", "pad_f_num"]
        self.pifu_keys = []
        self.refinepifu_keys = self.base_keys + [f"smpl_{feat_name}" for feat_name in self.smpl_feats] + ["voxel_verts", "voxel_faces", "pad_v_num", "pad_f_num"]

        self.if_regressor = MLP(
            filter_channels=channels_IF,
            name="if",
            res_layers=self.opt.res_layers,
            norm=self.opt.norm_mlp,
            last_op=nn.Sigmoid() if not cfg.test_mode else None,
        )

        self.sp_encoder = SpatialEncoder()

        # network
        if self.use_filter:
            if self.opt.gtype == "HGPIFuNet":
                self.F_filter = HGFilter(self.opt, self.opt.num_stack, len(self.channels_filter[0]))
            else:
                print(colored(f"Backbone {self.opt.gtype} is unimplemented", "green"))

        summary_log = (
            f"{self.prior_type.upper()}:\n" + f"w/ Global Image Encoder: {self.use_filter}\n" +
            f"Image Features used by MLP: {self.in_geo}\n"
        )

        if self.prior_type == "icon":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += f"Dim of Image Features (local): {3 if (use_vis and not self.use_filter) else 6}\n"
            summary_log += f"Dim of Geometry Features (ICON): {self.smpl_dim}\n"
        elif self.prior_type == "keypoint":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += f"Dim of Image Features (local): {3 if (use_vis and not self.use_filter) else 6}\n"
            summary_log += f"Dim of Geometry Features (Keypoint): {self.smpl_dim}\n"
        elif self.prior_type == "pamir":
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PaMIR): {self.voxel_dim}\n"
        elif self.prior_type == "refinepifu":
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (refinepifu): {self.voxel_dim+self.smpl_dim}\n"
        else:
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PIFu): 1 (z-value)\n"

        summary_log += f"Dim of MLP's first layer: {channels_IF[0]}\n"

        print(colored(summary_log, "yellow"))

        self.normal_filter = NormalNet(cfg)

        init_net(self)

    def get_normal(self, in_tensor_dict):
        # ipdb.set_trace()
        # insert normal features
        if (not self.training) and (not self.overfit):
            # print(colored("infer normal","blue"))
            # ipdb.set_trace()
            with torch.no_grad():
                
                feat_lst = []
                if "image" in self.in_geo:
                    feat_lst.append(in_tensor_dict["image"])    # [1, 3, 512, 512]
                    
                if "normal_F" in self.in_geo and "normal_B" in self.in_geo:
                    if ("normal_F" not in in_tensor_dict.keys() or
                        "normal_B" not in in_tensor_dict.keys()):
                        # 如果in_tensor中没有normal，那就要预测一下
                        (nmlF, nmlB) = self.normal_filter(in_tensor_dict)
                        # ipdb.set_trace()
                    else:
                        # ipdb.set_trace()
                        nmlF = in_tensor_dict["normal_F"]
                        nmlB = in_tensor_dict["normal_B"]
                        # ipdb.set_trace()
                    # ipdb.set_trace()
                    feat_lst.append(nmlF)    # [1, 3, 512, 512]
                    feat_lst.append(nmlB)    # [1, 3, 512, 512]
            # ipdb.set_trace()
            in_filter = torch.cat(feat_lst, dim=1) # [1, 6, 512, 512]

        else:
            in_filter = torch.cat([in_tensor_dict[key] for key in self.in_geo], dim=1) # [1, 6, 512, 512]

        return in_filter

    def get_mask(self, in_filter, size=128):

        mask = (
            F.interpolate(
                in_filter[:, self.channels_filter[0]],
                size=(size, size),
                mode="bilinear",
                align_corners=True,
            ).abs().sum(dim=1, keepdim=True) != 0.0
        )

        return mask

    def filter(self, in_tensor_dict, return_inter=False):
        """
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        """
        # ipdb.set_trace()
        
        in_filter = self.get_normal(in_tensor_dict) # [1, 6, 512, 512] 正反cloth法向图

        features_G = []

        if self.prior_type in ["icon", "keypoint"]:
            if self.use_filter:
                # ipdb.set_trace()
                
                features_F = self.F_filter(in_filter[:, self.channels_filter[0]]) # (1, 3, 512, 512) | [(1, 6, 128, 128), (1, 6, 128, 128)]
                features_B = self.F_filter(in_filter[:, self.channels_filter[1]]) # (1, 3, 512, 512) | [(1, 6, 128, 128), (1, 6, 128, 128)]
                # ipdb.set_trace()
            else:
                # ipdb.set_trace()
                
                features_F = [in_filter[:, self.channels_filter[0]]]
                features_B = [in_filter[:, self.channels_filter[1]]]
                
            # ipdb.set_trace()
            for idx in range(len(features_F)):
                features_G.append(torch.cat([features_F[idx], features_B[idx]], dim=1)) # [(1, 12, 128, 128), (1, 12, 128, 128)]
                
        else:
            # ipdb.set_trace()
            
            if self.use_filter:
                # ipdb.set_trace()
                # image、normal_F、normal_B
                features_G = self.F_filter(in_filter[:, self.channels_filter[0]]) # [(1, 6, 128, 128), (1, 6, 128, 128)]
                # ipdb.set_trace()
            else:
                
                features_G = [in_filter[:, self.channels_filter[0]]]
                
        # ICON: ['smpl_verts', 'smpl_faces', 'smpl_sdf', 'smpl_cmap', 'smpl_norm', 'smpl_vis']
        # pamir: ['voxel_verts', 'voxel_faces', 'pad_v_num', 'pad_f_num']
        # refinepifu: ['smpl_verts', 'smpl_faces', 'smpl_sdf', 'voxel_verts', 'voxel_faces', 'pad_v_num', 'pad_f_num']
        self.smpl_feat_dict = {
            k: in_tensor_dict[k] if k in in_tensor_dict.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        }
        
        # ipdb.set_trace()

        # If it is not in training, only produce the last im_feat
        if not self.training:
            features_out = [features_G[-1]]
        else:
            features_out = features_G

        if return_inter:
            return features_out, in_filter # 特征的输出， 卷积前的输入
        else:
            return features_out

    def query(self, features, points, calibs, transforms=None, regressor=None):

        xyz = self.projection(points, calibs, transforms)
        (xy, z) = xyz.split([2, 1], dim=1)

        in_cube = (xyz > -1.0) & (xyz < 1.0)
        in_cube = in_cube.all(dim=1, keepdim=True).detach().float()
        
        # ipdb.set_trace()

        preds_list = []
        vol_feats = features

        if self.prior_type in ["icon", "keypoint"]:

            # smpl_verts [B, N_vert, 3]
            # smpl_faces [B, N_face, 3]
            # xyz [B, 3, N]  --> points [B, N, 3]

            point_feat_extractor = PointFeat(self.smpl_feat_dict["smpl_verts"], self.smpl_feat_dict["smpl_faces"])

            # ipdb.set_trace()
            point_feat_out = point_feat_extractor.query(xyz.permute(0, 2, 1).contiguous(), self.smpl_feat_dict)
            
            # ipdb.set_trace()
            feat_lst = [point_feat_out[key] for key in self.smpl_feats if key in point_feat_out.keys()] # ['sdf', 'cmap', 'norm', 'vis']
            # [1, 35937, 1] [1, 35937, 3] [1, 35937, 3] [1, 35937, 1])
            # ipdb.set_trace()
            smpl_feat = torch.cat(feat_lst, dim=2).permute(0, 2, 1)
            
            # ipdb.set_trace()
            if self.prior_type == "keypoint":
                kpt_feat = self.sp_encoder.forward(
                    cxyz=xyz.permute(0, 2, 1).contiguous(),
                    kptxyz=self.smpl_feat_dict["smpl_joint"],
                )
                
        elif self.prior_type == "refinepifu":
            
            # smpl_verts [B, N_vert, 3]
            # smpl_faces [B, N_face, 3]
            # xyz [B, 3, N]  --> points [B, N, 3]
            
            #--------SDF--------#

            point_feat_extractor = PointFeat(self.smpl_feat_dict["smpl_verts"], self.smpl_feat_dict["smpl_faces"])

            # ipdb.set_trace()
            point_feat_out = point_feat_extractor.query(xyz.permute(0, 2, 1).contiguous(), {"smpl_sdf":self.smpl_feat_dict["smpl_sdf"]})
            
            # ipdb.set_trace()
            feat_lst = [point_feat_out[key] for key in self.smpl_feats if key in point_feat_out.keys()] # ['sdf'] [1, 35937, 1]
            smpl_feat = torch.cat(feat_lst, dim=2).permute(0, 2, 1) # [1, 1, 8000]
            
            # ipdb.set_trace()
            
            #-------voxel-------#
            
            # ipdb.set_trace()
            voxel_verts = self.smpl_feat_dict["voxel_verts"][:, :-self.smpl_feat_dict["pad_v_num"][0], :] # [1, 7894, 3]
            voxel_faces = self.smpl_feat_dict["voxel_faces"][:, :-self.smpl_feat_dict["pad_f_num"][0], :] # [1, 24957, 4]

            # ipdb.set_trace()
            self.voxelization.update_param(
                batch_size=voxel_faces.shape[0],
                smpl_tetra=voxel_faces[0].detach().cpu().numpy(),
            )
            vol = self.voxelization(voxel_verts)    # vol ~ [0,1]
            # ipdb.set_trace()
            vol_feats = self.ve(vol, intermediate_output=self.training)
            
            # ipdb.set_trace()
            if self.high_res_voxel:
                # ipdb.set_trace()
                self.highvoxelization.update_param(
                    batch_size=voxel_faces.shape[0],
                    smpl_tetra=voxel_faces[0].detach().cpu().numpy(),
                )
                vol = self.highvoxelization(voxel_verts)
                # ipdb.set_trace()
                vol_feats = self.highve(vol, low_vol_feats=vol_feats[-1])
            
            # ipdb.set_trace()
            
        elif self.prior_type == "pamir":

            # ipdb.set_trace()
            voxel_verts = self.smpl_feat_dict["voxel_verts"][:, :-self.smpl_feat_dict["pad_v_num"][0], :]
            voxel_faces = self.smpl_feat_dict["voxel_faces"][:, :-self.smpl_feat_dict["pad_f_num"][0], :]
            
            # ipdb.set_trace()
            self.voxelization.update_param(
                batch_size=voxel_faces.shape[0],
                smpl_tetra=voxel_faces[0].detach().cpu().numpy(),
            )
            vol = self.voxelization(voxel_verts)    # vol: (1, 3, 128, 128, 128) ~ [0,1]
            
            # ipdb.set_trace()
            vol_feats = self.ve(vol, intermediate_output=self.training) # [(1, 7, 32, 32, 32)]
            
            # ipdb.set_trace()

        for im_feat, vol_feat in zip(features, vol_feats):

            # normal feature choice by smpl_vis

            if self.prior_type == "icon":
                if "vis" in self.smpl_feats:
                    # 根据vis分别选取点的特征 im_feat: [1, 12, 128, 128](正反特征图)
                    point_local_feat = feat_select(self.index(im_feat, xy), smpl_feat[:, [-1], :]) # [1, 6, 35937]
                    
                    # ipdb.set_trace()
                    point_feat_list = [point_local_feat, smpl_feat[:, :-1, :]] # [ [1, 6, 35937], [1, 7, 35937] ]
                    # 正面：point_local_feat[:3] + smpl_feat['norm'] + smpl_feat['sdf'] 反面 point_local_feat[3:] + smpl_feat['norm'] + smpl_feat['sdf']
                    # ipdb.set_trace()
                    
                else:
                    
                    point_local_feat = self.index(im_feat, xy)
                    
                    # ipdb.set_trace()
                    point_feat_list = [point_local_feat, smpl_feat[:, :, :]]
                    
                    # ipdb.set_trace()
                    
            elif self.prior_type == "refinepifu":

                    # ipdb.set_trace()
                    # [1, 6, 8000] [1, 7, 8000] [1, 1, 8000]
                    point_feat_list = [self.index(im_feat, xy), self.index(vol_feat, xyz), smpl_feat[:, :, :]]
                    
                    # ipdb.set_trace()

            elif self.prior_type == "keypoint":

                if "vis" in self.smpl_feats:
                    point_local_feat = feat_select(self.index(im_feat, xy), smpl_feat[:, [-1], :])
                    point_feat_list = [point_local_feat, kpt_feat, smpl_feat[:, :-1, :]]
                else:
                    point_local_feat = self.index(im_feat, xy)
                    point_feat_list = [point_local_feat, kpt_feat, smpl_feat[:, :, :]]

            elif self.prior_type == "pamir":

                # im_feat [B, hg_dim, 128, 128]
                # vol_feat [B, vol_dim, 32, 32, 32]
                
                point_feat_list = [self.index(im_feat, xy), self.index(vol_feat, xyz)]

            elif self.prior_type == "pifu":
                point_feat_list = [self.index(im_feat, xy), z]

            point_feat = torch.cat(point_feat_list, 1)

            # out of image plane is always set to 0
            preds = regressor(point_feat)
            preds = in_cube * preds

            preds_list.append(preds)

        return preds_list

    def get_error(self, preds_if_list, labels):
        """calcaulate error

        Args:
            preds_list (list): list of torch.tensor(B, 3, N)
            labels (torch.tensor): (B, N_knn, N)

        Returns:
            torch.tensor: error
        """
        error_if = 0

        for pred_id in range(len(preds_if_list)):
            pred_if = preds_if_list[pred_id]
            error_if += self.error_term(pred_if, labels)

        error_if /= len(preds_if_list)

        return error_if

    def forward(self, in_tensor_dict):
        """
        sample_tensor [B, 3, N]
        calib_tensor [B, 4, 4]
        label_tensor [B, 1, N]
        smpl_feat_tensor [B, 59, N]
        """

        sample_tensor = in_tensor_dict["sample"]
        calib_tensor = in_tensor_dict["calib"]
        label_tensor = in_tensor_dict["label"]

        in_feat = self.filter(in_tensor_dict)

        preds_if_list = self.query(
            in_feat, sample_tensor, calib_tensor, regressor=self.if_regressor
        )

        error = self.get_error(preds_if_list, label_tensor)

        return preds_if_list[-1], error