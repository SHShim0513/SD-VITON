import torch
import torch.nn as nn

from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
import argparse
import os
import time
from cp_dataset_test import CPDatasetTest, CPDataLoader

from networks import ConditionGenerator, load_checkpoint, make_grid, make_grid_3d
from network_generator import SPADEGenerator
from tensorboardX import SummaryWriter
from utils import *

import torchgeometry as tgm
from collections import OrderedDict

from torch.nn.modules.utils import _pair, _quadruple

def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    parser.add_argument('--test_name', type=str, default='test', help='test name')
    parser.add_argument("--dataroot", default="./data/zalando-hd-resize")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--data_list", default="./data/zalando-hd-resize/test_pairs.txt")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--datasetting", default="paired")
    parser.add_argument("--fine_width", type=int, default=768)
    parser.add_argument("--fine_height", type=int, default=1024)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, default='', help='tocg checkpoint')
    parser.add_argument('--gen_checkpoint', type=str, default='./gen_step_110000.pth', help='G checkpoint')  

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    parser.add_argument('--gen_semantic_nc', type=int, default=7, help='# of input label classes without unknown class')
    
    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
        
    # Hyper-parameters
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")

    # condition generator
    parser.add_argument('--cond_G_ngf', type=int, default=96)
    parser.add_argument("--cond_G_input_width", type=int, default=192)
    parser.add_argument("--cond_G_input_height", type=int, default=256)
    parser.add_argument('--cond_G_num_layers', type=int, default=5)

    # generator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='instance normalization or batch normalization')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='most', # normal: 256, more: 512
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
    parser.add_argument("--composition_mask", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def load_checkpoint_G(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("Invalid path!")
        return
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()])
    new_state_dict._metadata = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()])
    model.load_state_dict(new_state_dict, strict=True)
    model.cuda()


def test(opt, test_loader, board, tocg, generator):
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss = gauss.cuda()
    
    # Model
    tocg.cuda()
    tocg.eval()
    generator.eval()
    
    if opt.output_dir is not None:
        output_dir = opt.output_dir
    else:
        output_dir = os.path.join('./output', opt.test_name,
                            opt.datamode, opt.datasetting, 'generator', 'output')
    grid_dir = os.path.join('./output', opt.test_name,
                             opt.datamode, opt.datasetting, 'generator', 'grid')
    
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    num = 0
    with torch.no_grad():
        for inputs in test_loader.data_loader:
            
            pose_map = inputs['pose'].cuda()
            pre_clothes_mask = inputs['cloth_mask'][opt.datasetting].cuda()
            label = inputs['parse']
            parse_agnostic = inputs['parse_agnostic']
            agnostic = inputs['agnostic'].cuda()
            clothes = inputs['cloth'][opt.datasetting].cuda() # target cloth
            densepose = inputs['densepose'].cuda()
            im = inputs['image']
            input_label, input_parse_agnostic = label.cuda(), parse_agnostic.cuda()
            pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()

            # down
            pose_map_down = F.interpolate(pose_map, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
            pre_clothes_mask_down = F.interpolate(pre_clothes_mask, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
            input_label_down = F.interpolate(input_label, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
            input_parse_agnostic_down = F.interpolate(input_parse_agnostic, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
            agnostic_down = F.interpolate(agnostic, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='nearest')
            clothes_down = F.interpolate(clothes, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')
            densepose_down = F.interpolate(densepose, size=(opt.cond_G_input_height, opt.cond_G_input_width), mode='bilinear')

            shape = pre_clothes_mask.shape
            
            # multi-task inputs
            input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
            input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

            # forward
            flow_list_taco, fake_segmap, _, warped_clothmask_taco, flow_list_tvob, _, _, = tocg(input1, input2)
            
            # warped cloth mask one hot 
            warped_cm_onehot = torch.FloatTensor((warped_clothmask_taco.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
            
            cloth_mask = torch.ones_like(fake_segmap)
            cloth_mask[:,3:4, :, :] = warped_clothmask_taco
            fake_segmap = fake_segmap * cloth_mask
                    
            # make generator input parse map
            fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(opt.fine_height, opt.fine_width), mode='bilinear'))
            fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]
            
            old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
            old_parse.scatter_(1, fake_parse, 1.0)

            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    parse[:, i] += old_parse[:, label]
                    
            # warped cloth
            N, _, iH, iW = clothes.shape
            N, flow_iH, flow_iW, _ = flow_list_tvob[-1].shape

            flow_tvob = F.interpolate(flow_list_tvob[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
            flow_tvob_norm = torch.cat([flow_tvob[:, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_tvob[:, :, :, 1:2] / ((flow_iH - 1.0) / 2.0)], 3)
            
            grid = make_grid(N, iH, iW)
            grid_3d = make_grid_3d(N, iH, iW)

            warped_grid_tvob = grid + flow_tvob_norm
            warped_cloth_tvob = F.grid_sample(clothes, warped_grid_tvob, padding_mode='border')
            warped_clothmask_tvob = F.grid_sample(pre_clothes_mask, warped_grid_tvob, padding_mode='border')

            flow_taco = F.interpolate(flow_list_taco[-1].permute(0, 4, 1, 2, 3), size=(2, iH, iW), mode='trilinear').permute(0, 2, 3, 4, 1)
            flow_taco_norm = torch.cat([flow_taco[:, :, :, :, 0:1] / ((flow_iW - 1.0) / 2.0), flow_taco[:, :, :, :, 1:2] / ((flow_iH - 1.0) / 2.0), flow_taco[:, :, :, :, 2:3]], 4)
            warped_cloth_tvob = warped_cloth_tvob.unsqueeze(2)
            warped_cloth_taco = F.grid_sample(torch.cat((warped_cloth_tvob, torch.zeros_like(warped_cloth_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
            warped_cloth_taco = warped_cloth_taco[:,:,0,:,:]

            warped_clothmask_tvob = warped_clothmask_tvob.unsqueeze(2)
            warped_clothmask_taco = F.grid_sample(torch.cat((warped_clothmask_tvob, torch.zeros_like(warped_clothmask_tvob).cuda()), dim=2), flow_taco_norm + grid_3d, padding_mode='border')
            warped_clothmask_taco = warped_clothmask_taco[:,:,0,:,:]
            
            if opt.occlusion:
                warped_clothmask_taco = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask_taco)
                warped_cloth_taco = warped_cloth_taco * warped_clothmask_taco + torch.ones_like(warped_cloth_taco) * (1 - warped_clothmask_taco)
            
            if opt.composition_mask:            
                output, comp_mask = generator(torch.cat((agnostic, densepose, warped_cloth_taco), dim=1), parse)
                comp_mask1 = comp_mask * warped_clothmask_taco
                comp_mask = parse[:,2:3,:,:] * comp_mask1
                output = warped_cloth_taco * comp_mask + output * (1 - comp_mask)
            else:
                output = generator(torch.cat((agnostic, densepose, warped_cloth_taco), dim=1), parse)

            # visualize
            unpaired_names = []
            for i in range(shape[0]):
                grid = make_image_grid([(clothes[i].cpu() / 2 + 0.5), (pre_clothes_mask[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i]+1)/2),
                                        (warped_cloth_taco[i].cpu().detach() / 2 + 0.5), (warped_clothmask_taco[i].cpu().detach()).expand(3, -1, -1), visualize_segmap(fake_parse_gauss.cpu(), batch=i),
                                        (pose_map[i].cpu()/2 +0.5), (warped_cloth_taco[i].cpu()/2 + 0.5), (agnostic[i].cpu()/2 + 0.5),
                                        (im[i]/2 +0.5), (output[i].cpu()/2 +0.5)],
                                        nrow=4)
                unpaired_name = (inputs['c_name']['paired'][i].split('.')[0] + '_' + inputs['c_name'][opt.datasetting][i].split('.')[0] + '.png')
                save_image(grid, os.path.join(grid_dir, unpaired_name))
                unpaired_names.append(unpaired_name)
                
            # save output
            save_images(output, unpaired_names, output_dir)

            num += shape[0]
            print(num)
                

def main():
    opt = get_opt()
    print(opt)
    print("Start to test %s!")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    
    # create test dataset & loader
    test_dataset = CPDatasetTest(opt)
    test_loader = CPDataLoader(opt, test_dataset)
    
    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.test_name, opt.datamode, opt.datasetting))

    ## Model
    # tocg
    input1_nc = 4
    input2_nc = opt.semantic_nc + 3
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=opt.cond_G_ngf, norm_layer=nn.BatchNorm2d, num_layers=opt.cond_G_num_layers) # num_layers: training condition network w/ fine_height 256 -> 5, - w/ fine_height 512 -> 6, - w/ fine_height 1024 -> 7
       
    # generator
    opt.semantic_nc = 7
    generator = SPADEGenerator(opt, 3+3+3)
    generator.print_network()
       
    # Load Checkpoint
    load_checkpoint(tocg, opt.tocg_checkpoint)
    load_checkpoint_G(generator, opt.gen_checkpoint)

    # Test
    test(opt, test_loader, board, tocg, generator)

    print("Finished testing!")


if __name__ == "__main__":
    main()