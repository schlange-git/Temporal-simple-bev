import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")
import nuscenesdataset 
import utils.geom
import utils.vox
import utils.misc
import utils.basic

import random

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端, 避免 show() 卡住
import matplotlib.pyplot as plt



from torchvision.models.resnet import resnet18
from efficientnet_pytorch import EfficientNet

EPS = 1e-4

from functools import partial

def set_bn_momentum(model, momentum=0.1):

            # 
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip

class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.predict_future_flow = predict_future_flow

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2]) # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        feat_output = self.feat_head(x)
        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None

        return {
            'raw_feat': x,
            'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }

import torchvision
class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        # print('input_1', input_1.shape)
        # print('input_2', input_2.shape)
        x = self.upsampling_layer(input_1, input_2)
        # print('x', x.shape)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x

class Segnet(nn.Module):
    def __init__(self, Z, Y, X, vox_util=None, 
                 use_radar=False,
                 use_lidar=False,
                 seqlen = 0,
                 use_metaradar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 encoder_type="res101"):
        super(Segnet, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])

        self.Z, self.Y, self.X = Z, Y, X
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.seqlen = seqlen
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress   
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()
        
        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEV compressor
        if self.use_radar:
            if self.use_metaradar:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y + 16*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y+1, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
        elif self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim*Y+Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                # use simple sum
                pass

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim*seqlen,
            n_classes=1,
            predict_future_flow=False
        )

        # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            
        # set_bn_momentum(self, 0.1)

        if vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None
        
    def forward(self, rgb_camXs, T_bev, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,T,S,C,H,W)
        pix_T_cams: (B,T,S,4,4)
        cam0_T_camXs: (B,T,S,4,4)
        vox_util: vox util object
        rad_occ_mem0: here is (NONE)
            - None when use_radar = False, use_lidar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        '''
        #T==2
        B, T, S, C, H, W = rgb_camXs.shape   
        assert(C==3)
        

        assert(not self.rand_flip)
        # 沿着T进行分割,然后去除这个维度
        # 按时间帧进行分割
        rgb_camXs_split = torch.split(rgb_camXs, 1, dim=1)  # 切分为 [帧1, 帧2]，形状为 (B, 1, S, C, H, W
        feat_mems_all = []  # 用于存储每帧的 3D 特征
        for t in range(T):
            rgb_camXs = rgb_camXs_split[t].squeeze(1)  # (B, S, C, H, W)

            # 调用现有逻辑处理每帧
            # reshape tensors
            __p = lambda x: utils.basic.pack_seqdim(x, B)
            __u = lambda x: utils.basic.unpack_seqdim(x, B)
            rgb_camXs_ = __p(rgb_camXs)
            #按照T维度对当前t的一组进行处理
            pix_T_cams_ = __p(pix_T_cams[:, t])       # (B*S, 4,4)
            cam0_T_camXs_ = __p(cam0_T_camXs[:, t])   # (B*S, 4,4)
            camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)

            # rgb encoder
            device = rgb_camXs_.device
            rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)
            if self.rand_flip:
                B0, _, _, _ = rgb_camXs_.shape
                self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
                rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])
            feat_camXs_ = self.encoder(rgb_camXs_)
            if self.rand_flip:
                feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
            _, C, Hf, Wf = feat_camXs_.shape

            sy = Hf/float(H)
            sx = Wf/float(W)
            Z, Y, X = self.Z, self.Y, self.X

            # unproject image feature to 3d grid
            featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
            if self.xyz_camA is not None:
                xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*S,1,1)
            else:
                xyz_camA = None
            feat_mems_ = vox_util.unproject_image_to_mem(
                feat_camXs_,
                utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
                camXs_T_cam0_, Z, Y, X,
                xyz_camA=xyz_camA)
            feat_mems = __u(feat_mems_) # B, S, C, Z, Y, X

            mask_mems = (torch.abs(feat_mems) > 0).float()
            feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X
            '''
            新加入逻辑,存储两帧的特征
            '''        
                      
            #循环停止
            
            #这里应当解决一个问题,就是这两帧图片应该同时flip或者不flip
            #并且参数相同才能进行空间匹配
            
            if self.rand_flip:
                self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
                self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
                #思考后我认为不要使用翻转策略,因为会导致后面的对齐无法进行
                feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
                feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])           
                #只有使用雷达才有occ参数                           
                if rad_occ_mem0 is not None:
                    rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                    rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])
                
                
            # bev compressing
            if self.use_radar:
                assert(rad_occ_mem0 is not None)
                if not self.use_metaradar:
                    feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                    rad_bev = torch.sum(rad_occ_mem0, 3).clamp(0,1) # squish the vertical dim
                    feat_bev_ = torch.cat([feat_bev_, rad_bev], dim=1)
                    feat_bev = self.bev_compressor(feat_bev_)
                else:
                    feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                    rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16*Y, Z, X)
                    feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
                    feat_bev = self.bev_compressor(feat_bev_)
            elif self.use_lidar:
                assert(rad_occ_mem0 is not None)
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
                feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
                feat_bev = self.bev_compressor(feat_bev_)
                
                
            else: # rgb only
                if self.do_rgbcompress:
                    feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                    feat_bev = self.bev_compressor(feat_bev_)
                else:
                    feat_bev = torch.sum(feat_mem, dim=3)
                
            feat_mems_all.append(feat_bev)  # 存储每帧的 bev 特征: 2x (B,C,H,W)
            
        bev_features = torch.stack(feat_mems_all, dim=1)  # 将列表中的[feat_mems_all]堆叠成, 形状 (B, T, C, H, W)
        B, T, C, H, W = bev_features.shape
        fused_bev = bev_features[:, -1, :, :, :]  # 从最后一帧开始融合
        
        bev_list = []
        bev_list.append(fused_bev)
        
        curr_features = []
        original_features = []
        warped_features = []
                           
        # 在 bev 中:
        #   H = z 方向 => ZMIN..ZMAX => -50..50 => total 100m => 200 px => 0.5 m/px
        #   W = x 方向 => XMIN..XMAX => -50..50 => total 100m => 200 px => 0.5 m/px

        res_z = 100 / float(Z)   # 0.5 m/px
        res_x = 100 / float(X)   # 0.5 m/px

        '''
        #这一大段的作用是验证T_bev是否按照正确的方向工作;

        identity_matrix = torch.eye(3)

        # 调整单位矩阵的形状为 (1, 1, 3, 3)
        identity_matrix = identity_matrix.view(1, 1, 3, 3)

        # 沿批量维度和时间维度复制单位矩阵，得到形状为 (B, T, 3, 3) 的张量
        T_bev = identity_matrix
        T_bev = T_bev.to(device)
        half_pz = 100   # 半个特征图对应的像素数
        dz_meter_target = half_pz * res_x  # 例如:100 * 0.5 = 50 米（单位取决于您的 res_x 定义）
        dx_meter_target = 0.0  # z方向不平移

        # 修改 T_bev 的平移部分,T_bev 的平移位于 [:, :, :2, 2]
        T_bev[..., 0, 2] = dx_meter_target  # x方向平移
        T_bev[..., 1, 2] = dz_meter_target  # z方向平移
        '''

        for t in range(T - 1):
            # 获取当前帧的 BEV 特征
            bev_feature_t = bev_features[:, t, :, :, :]

            #print(bev_feature_t.shape)
                              
            # 提取 BEV 转换矩阵
            affine_matrix = T_bev[:, t, :2, :2].clone()  # B x 2 x 2
            translation = T_bev[:, t, :2, 2].clone()     # B x 2
            
            
                        # --------------------------------------------------------------
            #  (1) 先把 "世界米" -> "像素" (在 z,x 平面)
            #      translation[:,0] 对应 x(左右),    translation[:,1] 对应 z(前后)
            #      => 要将 translation[:,1] / res_z, translation[:,0] / res_x
            #      但注意: 'affine_matrix' 若包含旋转, 可能不需要额外 scale
            #               仅对 pure translation 进行米->像素的scale
            # --------------------------------------------------------------
            dx_meter = translation[:, 0]  # x
            dz_meter = translation[:, 1]  # z

            dx_pixel = dx_meter / res_x   # => x in px
            dz_pixel = dz_meter / res_z   # => z in px

            # 这里 "affine_matrix" 负责旋转( yaw ), 不额外乘 scale_x/scale_z,
            # 假设 T_bev 里 (a,b; c,d) 已对应 yaw => 不需要再乘 pixel ratio
            # (如果 T_bev 里还有 scale(米->px), 你就要仔细核对)

            # 重新写回 translation
            translation_px = torch.stack([dx_pixel, dz_pixel], dim=-1)  # (B,2)

            # --------------------------------------------------------------
            # (2) 再把像素平移 => [-1,1] 归一化
            #    x(px)  =>   2*x(px)/W_
            #    z(px)  =>   2*z(px)/H_
            # --------------------------------------------------------------
            #  其中  x(px) 对应 "width" => W_ = 200
            #        z(px) 对应 "height" => H_ = 200
                             
            # 1) 将"像素级"平移转为 [-1, 1] 范围的归一化坐标
            translation_norm = translation_px.clone()
            
            translation_norm[:, 0] = 2* translation_norm[:, 0] / float(W) 
            translation_norm[:, 1] = 2* translation_norm[:, 1] / float(H) 


            # 2) 构造 (B,2,3) 的仿射矩阵 theta，用于 affine_grid
            theta = torch.cat([affine_matrix, translation_norm.unsqueeze(-1)], dim=-1)  # (B,2,3)

            # 3) 构建仿射变换网格
            grid = torch.nn.functional.affine_grid(
            theta,                 # (B,2,3)
            bev_feature_t.size(),  # (B, C, H, W)
            align_corners=True
            )

            # 应用仿射变换对齐 BEV 特征
            warped_bev = torch.nn.functional.grid_sample(
                bev_feature_t, grid, mode='bilinear', padding_mode='zeros', align_corners=True
            )

            bev_feature_curr = bev_features[:, -1, :, :, :]
            
            #curr_features.append(bev_feature_curr[0, 0].cpu().detach().numpy())
            #original_features.append(bev_feature_t[0, 0].cpu().detach().numpy())
            #warped_features.append(warped_bev[0, 0].cpu().detach().numpy())
            

            # 融合特征
            #使用通道维度
            bev_list.append(warped_bev)

        #拼接后C=CxT   
        feat_bev = torch.cat(bev_list, dim=1) 
        
        '''       
        
        # **显示特定帧的对比**
        fig = plt.figure(figsize=(5, 5 * len(original_features)))
        for idx, (original, warped) in enumerate(zip(original_features, warped_features)):
            plt.subplot(len(original_features), 2, idx * 2 + 1)
            plt.imshow(original, cmap='gray')
            plt.title(f"Original BEV Feature (t={idx})")
            plt.axis('off')

            plt.subplot(len(original_features), 2, idx * 2 + 2)
            plt.imshow(warped, cmap='gray')
            plt.title(f"Warped BEV Feature (t={idx})")
            plt.axis('off')

        plt.tight_layout()
        random_number = random.randint(1000, 9999)
        output_filename = f"./align/bev_visual_{random_number}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"BEV 特征可视化结果已保存至 {output_filename}")
              
        '''
        # bev decoder
        out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        raw_e = out_dict['raw_feat']
        feat_e = out_dict['feat']
        seg_e = out_dict['segmentation']
        center_e = out_dict['instance_center']
        offset_e = out_dict['instance_offset']

        return raw_e, feat_e, seg_e, center_e, offset_e

