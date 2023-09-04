from cv2 import absdiff
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class GradReverse(torch.autograd.Function):
    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    # @staticmethod
    # def backward(ctx, *grad_output):
    #     return grad_output[0] * -ctx.lambd, None

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None

gradient_scalar = GradReverse.apply


class Naive_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, (1, 2, 3))
        return x




class Up_New(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.lateral(x2)
        x1 = x1 + x2
        return self.conv(x1)



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        x1 = self.up4(x1)
        x2 = self.up2(x2)
        x3 = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x3)


class Up3(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear',
                              align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3, x4):
        x1 = self.up8(x1)
        x2 = self.up4(x2)
        x3 = self.up2(x3)
        x4 = torch.cat([x4, x3, x2, x1], dim=1)
        return self.conv(x4)


class CamEncode_New(nn.Module):
    def __init__(self, D, C, upsample, sep_dpft=False):
        super(CamEncode_New, self).__init__()
        self.D = D
        self.C = C
        self.upsample = upsample
        self.sep_dpft = sep_dpft

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.toplayer = nn.Conv2d(320, 128, kernel_size=1, padding=0)
        self.up1 = Up_New(112, 128)
        self.up2 = Up_New(40, 128)
        self.up3 = Up_New(24, 128)
        self.up4 = Up_New(16, 128)

        if self.upsample == 2:
            self.uplayer1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.reducechannel = nn.Conv2d(256, 256, kernel_size=1, padding=0)
        elif self.upsample == 4:
            self.uplayer1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.uplayer2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.reducechannel = nn.Conv2d(384, 256, kernel_size=1, padding=0)
        elif self.upsample == 8:
            self.uplayer1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            self.uplayer2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.uplayer3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.reducechannel = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        elif self.upsample == 16:
            self.uplayer1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
            self.uplayer2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            self.uplayer3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.uplayer4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.reducechannel = nn.Conv2d(640, 256, kernel_size=1, padding=0)
        else:
            assert False, "Wrong value for self.upsample"

        self.depthnet = nn.Sequential(
            nn.Conv2d(256, self.D, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.D),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.D, self.D, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.D),
            nn.ReLU(inplace=True),
        )

        self.featurenet = nn.Sequential(
            nn.Conv2d(256, self.C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
        )

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth

        # [B*N, D+C, final_dim[0], final_dim[1]]
        depth = self.depthnet(x)
        feature = self.featurenet(x)
        depth = self.get_depth_dist(depth)
        new_x = depth.unsqueeze(1) * feature.unsqueeze(2)

        # depth: [B*N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        # new_x: [B*N, C, D, final_dim[0]//downsample, final_dim[1]//downsample]
        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        # print(x.shape)

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            # print(x.shape)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x

        upscale1 = self.toplayer(endpoints['reduction_5'])
        upscale2 = self.up1(upscale1, endpoints['reduction_4'])
        if self.upsample == 2: 
            upscale1 = self.uplayer1(upscale1)
            out = torch.cat([upscale1, upscale2], dim=1)
        elif self.upsample == 4: 
            upscale4 = self.up2(upscale2, endpoints['reduction_3'])
            upscale1 = self.uplayer1(upscale1)
            upscale2 = self.uplayer2(upscale2)
            out = torch.cat([upscale1, upscale2, upscale4], dim=1)
        elif self.upsample == 8: 
            upscale4 = self.up2(upscale2, endpoints['reduction_3'])
            upscale8 = self.up3(upscale4, endpoints['reduction_2'])
            upscale1 = self.uplayer1(upscale1)
            upscale2 = self.uplayer2(upscale2)
            upscale4 = self.uplayer3(upscale4)
            out = torch.cat([upscale1, upscale2, upscale4, upscale8], dim=1)
        elif self.upsample == 16: 
            upscale4 = self.up2(upscale2, endpoints['reduction_3'])
            upscale8 = self.up3(upscale4, endpoints['reduction_2'])
            upscale16 = self.up4(upscale8, endpoints['reduction_1'])
            upscale1 = self.uplayer1(upscale1)
            upscale2 = self.uplayer2(upscale2)
            upscale4 = self.uplayer3(upscale4)
            upscale8 = self.uplayer4(upscale8)
            out = torch.cat([upscale1, upscale2, upscale4, upscale8, upscale16], dim=1)
        out = self.reducechannel(out)

        return out
        

    def forward(self, x, lidars):
        depth, x = self.get_depth_feat(x)

        return depth, x




class CamEncode(nn.Module):
    def __init__(self, D, C, upsample, sep_dpft):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.upsample = upsample
        self.sep_dpft = sep_dpft

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        if sep_dpft:
            self.up1_dp = Up(320+112, 512)
            self.up1_ft = Up(320+112, 512)

            self.up2_dp = Up2(320+112+40, 512)
            self.up2_ft = Up2(320+112+40, 512)

            self.depth_out = nn.Conv2d(512, self.D, kernel_size=1, padding=0)
            self.feature_out = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        else:
            self.up1 = Up(320+112, 512)
            if self.upsample >= 4:
                self.up2 = Up2(320+112+40, 512)
            if self.upsample >= 8:
                self.up3 = Up3(320+112+40+24, 512)
            self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x, ft = self.get_eff_depth(x)
        # Depth

        if self.sep_dpft:
            x_ = self.depth_out(x)
            ft = self.feature_out(ft)
            depth = self.get_depth_dist(x_)
            new_x = depth.unsqueeze(1) * ft.unsqueeze(2)
        else:
            x_ = self.depthnet(x)
            depth = self.get_depth_dist(x_[:, :self.D])
            new_x = depth.unsqueeze(1) * x_[:, self.D:(self.D + self.C)].unsqueeze(2)

        # depth: [B*N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        # new_x: [B*N, C, D, final_dim[0]//downsample, final_dim[1]//downsample]
        return depth, new_x, x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        # print(x.shape)

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            # print(x.shape)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x

        if self.sep_dpft:
            if self.upsample == 2:
                x_dp = self.up1_dp(endpoints['reduction_5'], endpoints['reduction_4'])
                x_ft = self.up1_ft(endpoints['reduction_5'], endpoints['reduction_4'])
            elif self.upsample == 4:
                x_dp = self.up2_dp(endpoints['reduction_5'], endpoints['reduction_4'], endpoints['reduction_3'])
                x_ft = self.up2_ft(endpoints['reduction_5'], endpoints['reduction_4'], endpoints['reduction_3'])
            else:
                assert False, "Error in value up_sample!"      
            return x_dp, x_ft
        else:
            if self.upsample == 2:
                x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
            elif self.upsample == 4:
                x = self.up2(endpoints['reduction_5'], endpoints['reduction_4'], endpoints['reduction_3'])
            elif self.upsample == 8:
                x = self.up3(endpoints['reduction_5'], endpoints['reduction_4'], endpoints['reduction_3'], endpoints['reduction_2'])

            else:
                assert False, "Error in value up_sample!"
            return x, 0
        

    def forward(self, x, lidars):
        depth, new_x, x_mid = self.get_depth_feat(x)

        return depth, new_x, x_mid


class CamEncode_DepthBin_GT(nn.Module):
    def __init__(self, D, C, upsample):
        super(CamEncode_DepthBin_GT, self).__init__()
        self.D = D
        self.C = C
        self.upsample = upsample

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        if self.upsample >= 4:
            self.up2 = Up2(320+112+40, 512)
        if self.upsample >= 8:
            self.up3 = Up3(320+112+40+24, 512)

        self.depthnet = nn.Conv2d(512, self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x, lidars):
        x_feature = self.get_eff_depth(x)

        # [B*N, C, final_dim[0], final_dim[1]]
        x_feature = self.depthnet(x_feature)

        B, N, C, imH, imW = lidars.shape
        lidars = lidars.view(B*N, C, imH, imW)

        depth = lidars
        new_x = lidars.unsqueeze(1) * x_feature.unsqueeze(2)

        # print("In side model, x_feature has nan is: ", torch.any(torch.isnan(x_feature)))
        # print("In side model, lidars has nan is: ", torch.any(torch.isnan(lidars)))

        # depth: [B*N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        # new_x: [B*N, C, D, final_dim[0]//downsample, final_dim[1]//downsample]
        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        # print(x.shape)

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            # print(x.shape)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        # print(endpoints['reduction_3'].shape)
        # print(endpoints['reduction_4'].shape)
        # print(endpoints['reduction_5'].shape)
        if self.upsample == 2:
            x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        elif self.upsample == 4:
            x = self.up2(endpoints['reduction_5'], endpoints['reduction_4'], endpoints['reduction_3'])
        elif self.upsample == 8:
            x = self.up3(endpoints['reduction_5'], endpoints['reduction_4'], endpoints['reduction_3'], endpoints['reduction_2'])
        else:
            assert False, "Error in value up_sample!"
        
        return x

    def forward(self, x, lidars):
        depth, x = self.get_depth_feat(x, lidars)

        return depth, x, 0


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1 
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.up1(x3, x1)
        x_new = self.up2(x)

        return x_new, x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, parser_name, adv_training, teacher_student, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.parser_name = parser_name
        self.adv_training = adv_training
        self.teacher_student = teacher_student
        self.new_model = self.data_aug_conf['new_model']

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 32 // (int)(self.data_aug_conf['up_scale'])
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape

        if parser_name=='lidarinputdata':  ## Use ground truth supervision as depth input
            self.camencode = CamEncode_DepthBin_GT(self.D, self.camC, self.data_aug_conf['up_scale'])
        elif self.new_model:  ## Predict depth and image features simultaneously
            self.camencode = CamEncode_New(self.D, self.camC, self.data_aug_conf['up_scale'], 
                                       self.data_aug_conf['sep_dpft'])
        else:
            self.camencode = CamEncode(self.D, self.camC, self.data_aug_conf['up_scale'], 
                                       self.data_aug_conf['sep_dpft'])
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        if self.adv_training:
            self.mid_discriminator = Naive_Discriminator(512, 1)
            self.final_discriminator = Naive_Discriminator(256, 1)
        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x, lidars):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        # depth: [B*N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        # new_x: [B*N, C, D, final_dim[0]//downsample, final_dim[1]//downsample]
        depth, x, x_mid = self.camencode(x, lidars)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, imH//self.downsample, imW//self.downsample)
        # x_mid = x_mid.view(B, N, 512, imH//self.downsample, imW//self.downsample)

        return depth, x, x_mid

    def voxel_pooling(self, geom_feats, x):
        # x: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample, C]
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans, lidars):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        depth, x, x_mid = self.get_cam_feats(x, lidars)

        # x: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample, C]
        x = self.voxel_pooling(geom, x)

        return depth, x, x_mid

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, lidars):
        depth, x, x_mid = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans, lidars)
        x, x_final = self.bevencode(x)
        if not self.teacher_student:
            if self.adv_training:
                # [B*N, 512, final_dim[0]//downsample, final_dim[1]//downsample]
                # [B, 256, final_dim[0]//downsample, final_dim[1]//downsample]
                x_mid = gradient_scalar(x_mid, 1.0)
                x_final = gradient_scalar(x_final, 1.0)
                x_mid = self.mid_discriminator(x_mid)
                x_final = self.final_discriminator(x_final)
                return depth, x, x_mid, x_final
            else:
                return depth, x, 0, 0
        else:
            if self.adv_training:
                # [B*N, 512, final_dim[0]//downsample, final_dim[1]//downsample]
                # [B, 256, final_dim[0]//downsample, final_dim[1]//downsample]
                x_mid_reverse = gradient_scalar(x_mid, 1.0)
                x_final_reverse = gradient_scalar(x_final, 1.0)
                x_mid_reverse_discriminator_out = self.mid_discriminator(x_mid_reverse)
                x_final_reverse_discriminator_out = self.final_discriminator(x_final_reverse)
                return depth, x, x_mid_reverse_discriminator_out, x_final_reverse_discriminator_out, x_final
            else:
                return depth, x, 0, 0, x_final
            


def compile_model(grid_conf, data_aug_conf, parser_name, adv_training, teacher_student, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, parser_name, adv_training, teacher_student, outC)
