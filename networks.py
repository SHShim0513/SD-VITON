import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import os
from torch.nn.utils import spectral_norm
import numpy as np

import functools


class ConditionGenerator(nn.Module):
    def __init__(self, opt, input1_nc, input2_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, num_layers=5):
        super(ConditionGenerator, self).__init__()
        self.warp_feature = opt.warp_feature
        self.out_layer_opt = opt.out_layer
        
        if num_layers == 5:
            self.ClothEncoder = nn.Sequential(
                ResBlock(input1_nc, ngf, norm_layer=norm_layer, scale='down'),  # 256
                ResBlock(ngf, ngf*2, norm_layer=norm_layer, scale='down'),  # 128
                ResBlock(ngf*2, ngf*4, norm_layer=norm_layer, scale='down'),  # 64
                ResBlock(ngf*4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 32
                ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 16
            )

            self.PoseEncoder = nn.Sequential(
                ResBlock(input2_nc, ngf, norm_layer=norm_layer, scale='down'),
                ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),
                ResBlock(ngf * 2, ngf*4, norm_layer=norm_layer, scale='down'),
                ResBlock(ngf*4, ngf * 4, norm_layer=norm_layer, scale='down'),
                ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),
            )
            
            if opt.warp_feature == 'T1':
                # in_nc -> skip connection + T1, T2 channel
                self.SegDecoder = nn.Sequential(
                    ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
                    ResBlock(ngf * 4 * 2 + ngf * 4 , ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
                    ResBlock(ngf * 4 * 2 + ngf * 4, ngf * 2, norm_layer=norm_layer, scale='up'),  # 64
                    ResBlock(ngf * 2 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up'),  # 128
                    ResBlock(ngf * 1 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up'),  # 256
            )

            # Cloth Conv 1x1
            self.conv1 = nn.Sequential(
                nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            )

            # Person Conv 1x1
            self.conv2 = nn.Sequential(
                nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            )
            
            self.flow_conv = nn.ModuleList([
                nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            ]
            )
            
            self.bottleneck = nn.Sequential(
                nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
                nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
                nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True) , nn.ReLU()),
                nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            )

        if num_layers == 6:
            self.ClothEncoder = nn.Sequential(
                ResBlock(input1_nc, ngf, norm_layer=norm_layer, scale='down'),  # 512
                ResBlock(ngf, ngf*2, norm_layer=norm_layer, scale='down'),  # 256
                ResBlock(ngf*2, ngf*4, norm_layer=norm_layer, scale='down'),  # 128
                ResBlock(ngf*4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 64
                ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 32
                ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 16
            )

            self.PoseEncoder = nn.Sequential(
                ResBlock(input2_nc, ngf, norm_layer=norm_layer, scale='down'),
                ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),
                ResBlock(ngf * 2, ngf*4, norm_layer=norm_layer, scale='down'),
                ResBlock(ngf*4, ngf * 4, norm_layer=norm_layer, scale='down'),
                ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),
                ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),
            )
        
            if opt.warp_feature == 'T1':
                # in_nc -> skip connection + T1, T2 channel
                self.SegDecoder = nn.Sequential(
                    ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
                    ResBlock(ngf * 4 * 2 + ngf * 4 , ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
                    ResBlock(ngf * 4 * 2 + ngf * 4 , ngf * 4, norm_layer=norm_layer, scale='up'),  # 64
                    ResBlock(ngf * 4 * 2 + ngf * 4, ngf * 2, norm_layer=norm_layer, scale='up'),  # 128
                    ResBlock(ngf * 2 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up'),  # 256
                    ResBlock(ngf * 1 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up'),  # 512
                )

            # Cloth Conv 1x1
            self.conv1 = nn.Sequential(
                nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            )

            # Person Conv 1x1
            self.conv2 = nn.Sequential(
                nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            )
            
            self.flow_conv = nn.ModuleList([
                nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            ]
            )
            
            self.bottleneck = nn.Sequential(
                nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
                nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
                nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
                nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True) , nn.ReLU()),
                nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            )

        
        self.conv = ResBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, scale='same')
        
        if opt.out_layer == 'relu':
            self.out_layer = ResBlock(ngf + ngf, output_nc, norm_layer=norm_layer, scale='same')

        self.residual_sequential_flow_list = nn.Sequential(
            nn.Sequential(nn.Conv3d(ngf * 8, 3, kernel_size=3, stride=1, padding=1, bias=True)),
        )
        self.out_layer_input1_resblk = ResBlock(input1_nc + input2_nc, ngf, norm_layer=norm_layer, scale='same')

        self.num_layers = num_layers

    def normalize(self, x):
        return x
    
    def forward(self, input1, input2, upsample='bilinear'):
        E1_list = []
        E2_list = []
        flow_list_tvob = []
        flow_list_taco = []
        layers_max_idx = self.num_layers - 1
        
        # Feature Pyramid Network
        for i in range(self.num_layers):
            if i == 0:
                E1_list.append(self.ClothEncoder[i](input1))
                E2_list.append(self.PoseEncoder[i](input2))
            else:
                E1_list.append(self.ClothEncoder[i](E1_list[i - 1]))
                E2_list.append(self.PoseEncoder[i](E2_list[i - 1]))

        # Compute Clothflow
        for i in range(self.num_layers):
            N, _, iH, iW = E1_list[layers_max_idx - i].size()
            grid = make_grid(N, iH, iW)

            if i == 0:
                T1 = E1_list[layers_max_idx - i]  # (ngf * 4) x 8 x 6
                T2 = E2_list[layers_max_idx - i]
                E4 = torch.cat([T1, T2], 1)
                
                flow = self.flow_conv[i](self.normalize(E4)).permute(0, 2, 3, 1)
                flow_list_tvob.append(flow)

                x = self.conv(T2)
                x = self.SegDecoder[i](x)
                
            else:
                T1 = F.interpolate(T1, scale_factor=2, mode=upsample) + self.conv1[layers_max_idx - i](E1_list[layers_max_idx - i])
                T2 = F.interpolate(T2, scale_factor=2, mode=upsample) + self.conv2[layers_max_idx - i](E2_list[layers_max_idx - i]) 
                
                flow = F.interpolate(flow_list_tvob[i - 1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)  # upsample n-1 flow
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
                warped_T1 = F.grid_sample(T1, flow_norm + grid, padding_mode='border')
                
                flow = flow + self.flow_conv[i](self.normalize(torch.cat([warped_T1, self.bottleneck[i-1](x)], 1))).permute(0, 2, 3, 1)  # F(n)
                flow_list_tvob.append(flow)

                # TACO layer of SD-VITON
                if i == layers_max_idx:
                    ## Eq.10 of SD-VITON
                    flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH - 1.0) / 2.0)], 3)
                    warped_T1 = F.grid_sample(T1, flow_norm + grid, padding_mode='border')
                    input_3d_flow_out = self.normalize(torch.cat([warped_T1, T2], 1)).unsqueeze(2)
                    flow_out = self.residual_sequential_flow_list[0](torch.cat((input_3d_flow_out, torch.zeros_like(input_3d_flow_out).cuda()), dim=2)).permute(0,2,3,4,1)
                    flow_list_taco.append(flow_out)

                if self.warp_feature == 'T1':
                    x = self.SegDecoder[i](torch.cat([x, E2_list[layers_max_idx-i], warped_T1], 1))
        
        ## Eq.11 of SD-VITON 
        N, _, iH, iW = input1.size()
        grid = make_grid(N, iH, iW)
        grid_3d = make_grid_3d(N, iH, iW)

        flow_tvob = F.interpolate(flow_list_tvob[-1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
        flow_tvob_norm = torch.cat([flow_tvob[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow_tvob[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
        warped_input1_tvob = F.grid_sample(input1, flow_tvob_norm + grid, padding_mode='border')
        
        flow_taco = F.interpolate(flow_list_taco[-1].permute(0, 4, 1, 2, 3), scale_factor=(1,2,2), mode='trilinear').permute(0, 2, 3, 4, 1)
        flow_taco_norm = torch.cat([flow_taco[:, :, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow_taco[:, :, :, :, 1:2] / ((iH/2 - 1.0) / 2.0), flow_taco[:, :, :, :, 2:3]], 4)
        warped_input1_tvob = warped_input1_tvob.unsqueeze(2)
        warped_input1_taco = F.grid_sample(torch.cat((warped_input1_tvob, torch.zeros_like(warped_input1_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
        warped_input1_taco_non_roi = warped_input1_taco[:,:,1,:,:]
        warped_input1_taco_roi = warped_input1_taco[:,:,0,:,:]
        warped_input1_tvob = warped_input1_tvob[:,:,0,:,:]

        out_inputs_resblk = self.out_layer_input1_resblk(torch.cat([input2, warped_input1_taco_roi], 1))
        
        x = self.out_layer(torch.cat([x, out_inputs_resblk], 1))

        warped_c_tvob = warped_input1_tvob[:, :-1, :, :]
        warped_cm_tvob = warped_input1_tvob[:, -1:, :, :]

        warped_c_taco_roi = warped_input1_taco_roi[:, :-1, :, :]
        warped_cm_taco_roi = warped_input1_taco_roi[:, -1:, :, :]

        warped_c_taco_non_roi = warped_input1_taco_non_roi[:,:-1,:,:]
        warped_cm_taco_non_roi = warped_input1_taco_non_roi[:,-1:,:,:]

        return flow_list_taco, x, warped_c_taco_roi, warped_cm_taco_roi, flow_list_tvob, warped_c_tvob, warped_cm_tvob

def make_grid_3d(N, iH, iW):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, 1, iW, 1).expand(N, 2, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, 1, iH, 1, 1).expand(N, 2, -1, iW, -1)
    grid_z = torch.linspace(-1.0, 1.0, 2).view(1, 2, 1, 1, 1).expand(N, -1, iH, iW, -1)
    grid = torch.cat([grid_x, grid_y, grid_z], 4).cuda()
    return grid

def make_grid(N, iH, iW):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
    grid = torch.cat([grid_x, grid_y], 3).cuda()
    return grid

class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc, scale='down', norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        assert scale in ['up', 'down', 'same'], "ResBlock scale must be in 'up' 'down' 'same'"

        if scale == 'same':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
        if scale == 'up':
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_nc, out_nc, kernel_size=1,bias=True)
            )
        if scale == 'down':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
            
        self.block = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.scale(x)
        return self.relu(residual + self.block(residual))


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False, Ddownx2=False, Ddropout=False, spectral=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.Ddownx2 = Ddownx2


        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, Ddropout, spectral=spectral)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        
        result = []
        if self.Ddownx2:
            input_downsampled = self.downsample(input)
        else:
            input_downsampled = input
        for i in range(num_D):
            
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, Ddropout=False, spectral=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.spectral_norm = spectral_norm if spectral else lambda x: x

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            if Ddropout:
                sequence += [[
                self.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                norm_layer(nf), nn.LeakyReLU(0.2, True), nn.Dropout(0.5)
            ]]
            else:

                sequence += [[
                    self.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                    norm_layer(nf), nn.LeakyReLU(0.2, True)
                ]]
                
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('no checkpoint')
        raise
    log = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.cuda()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_D(input_nc, ndf=64, n_layers_D=3, norm='instance', use_sigmoid=False, num_D=2, getIntermFeat=False, gpu_ids=[], Ddownx2=False, Ddropout=False, spectral=False):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat, Ddownx2, Ddropout, spectral=spectral)
    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda()
    netD.apply(weights_init)
    return netD
