import torch
import torch.nn as nn

from networks import make_grid as mkgrid
from networks import make_grid_3d as mkgrid_3d

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import ConditionGenerator, VGGLoss, GANLoss, load_checkpoint, save_checkpoint, define_D
from tqdm import tqdm
from utils import *


def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="test")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, default='', help='tocg checkpoint')

    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=300000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    
    # network
    parser.add_argument('--cond_G_ngf', type=int, default=96)
    parser.add_argument('--cond_G_num_layers', type=int, default=5)
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    parser.add_argument('--Ddownx2', action='store_true', help="Downsample D's input to increase the receptive field")  
    parser.add_argument('--Ddropout', action='store_true', help="Apply dropout to D")
    parser.add_argument('--num_D', type=int, default=2, help='Generator ngf')
    # training
    parser.add_argument("--lasttvonly", action='store_true')
    parser.add_argument("--interflowloss", action='store_true', help="Intermediate flow loss")
    
    # Hyper-parameters
    parser.add_argument('--G_lr', type=float, default=0.0002, help='Generator initial learning rate for adam')
    parser.add_argument('--D_lr', type=float, default=0.0002, help='Discriminator initial learning rate for adam')
    parser.add_argument('--CElamda', type=float, default=10, help='initial learning rate for adam')
    parser.add_argument('--GANlambda', type=float, default=1)
    parser.add_argument('--tvlambda_tvob', type=float, default=2)
    parser.add_argument('--tvlambda_taco', type=float, default=2)
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--spectral', action='store_true', help="Apply spectral normalization to D")
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")

    opt = parser.parse_args()
    return opt


def train(opt, train_loader, tocg, D):
    # Model
    tocg.cuda()
    tocg.train()
    D.cuda()
    D.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    if opt.fp16:
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.HalfTensor)
    else :
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor)

    # optimizer
    optimizer_G = torch.optim.Adam(tocg.parameters(), lr=opt.G_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.D_lr, betas=(0.5, 0.999))


    for step in tqdm(range(opt.load_step, opt.keep_step)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # input1
        c_paired = inputs['cloth']['paired'].cuda()
        cm_paired = inputs['cloth_mask']['paired'].cuda()
        cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        # input2
        parse_agnostic = inputs['parse_agnostic'].cuda()
        densepose = inputs['densepose'].cuda()
        openpose = inputs['pose'].cuda()
        # GT
        label_onehot = inputs['parse_onehot'].cuda()  # CE
        label = inputs['parse'].cuda()  # GAN loss
        parse_cloth_mask = inputs['pcm'].cuda()  # L1
        im_c = inputs['parse_cloth'].cuda()  # VGG
        # visualization
        im = inputs['image']
        # tucked-out shirts style
        lower_clothes_mask = inputs['lower_clothes_mask'].cuda()
        clothes_no_loss_mask = inputs['clothes_no_loss_mask'].cuda()

        # inputs
        input1 = torch.cat([c_paired, cm_paired], 1)
        input2 = torch.cat([parse_agnostic, densepose], 1)

        # forward
        flow_list_taco, fake_segmap, warped_cloth_paired_taco, warped_clothmask_paired_taco, flow_list_tvob, warped_cloth_paired_tvob, warped_clothmask_paired_tvob = tocg(input1, input2)

        # warped cloth mask one hot         
        warped_clothmask_paired_taco_onehot = torch.FloatTensor((warped_clothmask_paired_taco.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()

        # fake segmap cloth channel * warped clothmask
        cloth_mask = torch.ones_like(fake_segmap.detach())
        cloth_mask[:, 3:4, :, :] = warped_clothmask_paired_taco
        fake_segmap = fake_segmap * cloth_mask

        if opt.occlusion:
            warped_clothmask_paired_taco = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired_taco)
            warped_cloth_paired_taco = warped_cloth_paired_taco * warped_clothmask_paired_taco + torch.ones_like(warped_cloth_paired_taco) * (1-warped_clothmask_paired_taco)

            warped_clothmask_paired_tvob = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired_tvob)
            warped_cloth_paired_tvob = warped_cloth_paired_tvob * warped_clothmask_paired_tvob + torch.ones_like(warped_cloth_paired_tvob) * (1-warped_clothmask_paired_tvob)            
        
        # generated fake cloth mask & misalign mask
        fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
        misalign = fake_clothmask - warped_clothmask_paired_taco_onehot
        misalign[misalign < 0.0] = 0.0
        
        # loss warping
        loss_l1_cloth = criterionL1(warped_clothmask_paired_taco, parse_cloth_mask)
        loss_vgg = criterionVGG(warped_cloth_paired_taco, im_c)

        ## Eq.8 & Eq.9 of SD-VITON
        inv_lower_clothes_mask = lower_clothes_mask * clothes_no_loss_mask
        inv_lower_clothes_mask = 1. - inv_lower_clothes_mask
        loss_l1_cloth += criterionL1(warped_clothmask_paired_tvob*inv_lower_clothes_mask, parse_cloth_mask*inv_lower_clothes_mask)
        loss_vgg += criterionVGG(warped_cloth_paired_tvob*inv_lower_clothes_mask, im_c*inv_lower_clothes_mask)

        ## Eq.12 of SD-VITON
        roi_mask = torch.nn.functional.interpolate(parse_cloth_mask, scale_factor=0.5, mode='nearest')
        non_roi_mask = 1. - roi_mask

        flow_taco = flow_list_taco[-1]
        z_gt_non_roi = -1
        z_gt_roi = 1
        z_src_coordinate = -1
        z_dist_loss_non_roi = (torch.abs(z_src_coordinate + flow_taco[:,0:1,:,:,2] + z_gt_non_roi) * non_roi_mask).mean()
        z_dist_loss_roi = (torch.abs(z_src_coordinate + flow_taco[:,0:1,:,:,2] + z_gt_roi) * roi_mask).mean()
        
        loss_tv_tvob = 0
        loss_tv_taco = 0
        if not opt.lasttvonly:
            for flow in flow_list_taco:
                y_tv = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).mean()
                x_tv = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).mean()
                loss_tv_taco = loss_tv_taco + y_tv + x_tv 

            for flow in flow_list_tvob:
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                loss_tv_tvob = loss_tv_tvob + y_tv + x_tv
        else:
            for flow in flow_list_taco[-1:]:
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                loss_tv_taco = loss_tv_taco + y_tv + x_tv

            for flow in flow_list_tvob[-1:]:
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                loss_tv_tvob = loss_tv_tvob + y_tv + x_tv


        N, _, iH, iW = c_paired.size()
        # Intermediate flow loss
        if opt.interflowloss:
            layers_max_idx = len(flow_list_tvob)-1
            for i in range(len(flow_list_tvob)-1):
                flow = flow_list_tvob[i]
                N, fH, fW, _ = flow.size()
                grid = mkgrid(N, iH, iW)
                grid_3d = mkgrid_3d(N, iH, iW)
                
                flow = F.interpolate(flow.permute(0, 3, 1, 2), size = c_paired.shape[2:], mode=opt.upsample).permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((fW - 1.0) / 2.0), flow[:, :, :, 1:2] / ((fH - 1.0) / 2.0)], 3)
                warped_c = F.grid_sample(c_paired, flow_norm + grid, padding_mode='border')
                warped_cm = F.grid_sample(cm_paired, flow_norm + grid, padding_mode='border')
                warped_cm = remove_overlap(F.softmax(fake_segmap, dim=1), warped_cm)

                ## Eq.8 & Eq.9 of SD-VITON
                loss_l1_cloth += criterionL1(warped_cm*inv_lower_clothes_mask, parse_cloth_mask*inv_lower_clothes_mask) / (2 ** (layers_max_idx-i))
                loss_vgg += criterionVGG(warped_c*inv_lower_clothes_mask, im_c*inv_lower_clothes_mask) / (2 ** (layers_max_idx-i))


        # loss segmentation
        # generator
        CE_loss = cross_entropy2d(fake_segmap, label_onehot.transpose(0, 1)[0].long())
        fake_segmap_softmax = torch.softmax(fake_segmap, 1)
        pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
        loss_G_GAN = criterionGAN(pred_segmap, True)
        
        # discriminator
        fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()),dim=1))
        real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
        loss_D_fake = criterionGAN(fake_segmap_pred, False)
        loss_D_real = criterionGAN(real_segmap_pred, True)

        # loss sum
        loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda_tvob * loss_tv_tvob + opt.tvlambda_taco * loss_tv_taco) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda) + z_dist_loss_non_roi + z_dist_loss_roi
        loss_D = loss_D_fake + loss_D_real

        # step
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()    
            
        # display
        if (step) % 100 == 0:

            a_0 = c_paired[0].cuda()
            b_0 = im[0].cuda()
            c_0 = warped_cloth_paired_tvob[0]
            d_0 = warped_cloth_paired_taco[0]
            
            e_0 = lower_clothes_mask
            e_0 = torch.cat((e_0[0],e_0[0],e_0[0]), dim=0) 

            f_0 = densepose[0].cuda()

            g_0 = clothes_no_loss_mask
            g_0 = torch.cat((g_0[0],g_0[0],g_0[0]), dim=0) 

            h_0 = lower_clothes_mask*clothes_no_loss_mask
            h_0 = torch.cat((h_0[0],h_0[0],h_0[0]), dim=0) 

            i_0 = inv_lower_clothes_mask
            i_0 = torch.cat((i_0[0],i_0[0],i_0[0]), dim=0) 

            combine = torch.cat((a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, i_0), dim=2)

            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite('sample_fs_3/'+str(step)+'.jpg',bgr)
        

        if (step + 1) % opt.display_count == 0:
            t = time.time() - iter_start_time            
            print("step: %8d, time: %.3f\nloss G: %.4f, L1_cloth loss: %.4f, VGG loss: %.4f, TV_tvob loss: %.4f, TV_taco loss: %.4f, CE: %.4f, G GAN: %.4f\nloss D: %.4f, D real: %.4f, D fake: %.4f, z_non_roi: %.4f, z_roi: %.4f"
                % (step + 1, t, loss_G.item(), loss_l1_cloth.item(), loss_vgg.item(), loss_tv_tvob.item(), loss_tv_taco.item(), CE_loss.item(), loss_G_GAN.item(), loss_D.item(), loss_D_real.item(), loss_D_fake.item(), z_dist_loss_non_roi, z_dist_loss_roi), flush=True)

        # save
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(tocg, os.path.join(opt.checkpoint_dir, opt.name, 'tocg_step_%06d.pth' % (step + 1)))
            save_checkpoint(D, os.path.join(opt.checkpoint_dir, opt.name, 'D_step_%06d.pth' % (step + 1)))

def main():
    opt = get_opt()
    print(opt)
    print("Start to train %s!" % opt.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    
    os.makedirs('sample_fs_3', exist_ok=True)

    # create train dataset & loader
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)
    
    # Model
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=opt.cond_G_ngf, norm_layer=nn.BatchNorm2d, num_layers=opt.cond_G_num_layers) # num_layers: training condition network w/ fine_height 256 -> 5, - w/ fine_height 512 -> 6, - w/ fine_height 1024 -> 7
    
    D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2 = opt.Ddownx2, Ddropout = opt.Ddropout, n_layers_D=(opt.cond_G_num_layers-2), spectral = opt.spectral, num_D = opt.num_D) # n_layers_D: training condition network w/ fine_height 256 -> 3, - w/ fine_height 512 -> 4, - w/ fine_height 1024 -> 5

    # Load Checkpoint
    if not opt.tocg_checkpoint == '' and os.path.exists(opt.tocg_checkpoint):
        load_checkpoint(tocg, opt.tocg_checkpoint)

    # Train
    train(opt, train_loader, tocg, D)

    # Save Checkpoint
    save_checkpoint(tocg, os.path.join(opt.checkpoint_dir, opt.name, 'tocg_final.pth'))
    save_checkpoint(D, os.path.join(opt.checkpoint_dir, opt.name, 'D_final.pth'))
    print("Finished training %s!" % opt.name)


if __name__ == "__main__":
    main()
