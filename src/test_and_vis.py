import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, DepthLoss, get_batch_iou, get_val_info, denormalize_img


def test_and_vis(version='trainval',
            dataroot='./nuScenes/data',
            nepochs=10000,
            gpuid=1,
            nsweeps=3,
            manual_seed=1,
            weighted_depth=True,

            H=900, W=1600,
            resize_lim=(0.22, 0.235),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4), # rot_lim=(-30.4, 30.4),
            up_scale=2,
            rand_flip=True,
            color_jitter=True,
            color_jitter_conf=[0.2, 0.2, 0.2, 0.1],
            rand_resize=True,
            ncams=5,
            max_grad_norm=5.0,
            pos_weight=2.13,
            depthloss_weight=0.1,
            logdir='./runs',
            parser_name='imglidardata',
            class_name='vehicle',

            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=4,
            nworkers=10,
            lr=1e-3,
            weight_decay=1e-7,

            domain_gap=True,
            source='boston',
            target='singapore'
            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'rand_resize': rand_resize,
                    'color_jitter': color_jitter,
                    'color_jitter_conf': color_jitter_conf,
                    'bot_pct_lim': bot_pct_lim,
                    'up_scale': up_scale,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }

    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(1)

    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser_name, nsweeps=nsweeps, class_name=class_name, 
                                          domain_gap=domain_gap, source=source, target=target)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if weighted_depth is True:
        depth_weights = torch.Tensor([ 2.4,  1.2,  1.0,  1.1,  1.2,  1.4,  1.8,  2.3,  2.7,  3.5,
                                      3.6,  3.9,  4.8,  5.8,  5.4,  5.3,  5.0,  5.4,  5.3,  5.9,
                                      6.5,  7.0,  7.5,  7.5,  8.5, 10.3, 10.9,  9.8, 11.5, 13.1,
                                     15.1, 15.1, 16.3, 16.3, 17.8, 19.6, 21.8, 24.5, 24.5, 28.0,
                                     28.0]).to(device)
    else:
        depth_weights = None

    loss_final = SimpleLoss(pos_weight).cuda(gpuid)
    loss_depth = DepthLoss(depth_weights).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 100 if version == 'mini' else 10000

    model.train()
    counter = 0
    for epoch in range(10000):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            # lidars: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample]
            # depth: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample]

            depth, preds = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )


            # c_count = 0
            # print(lidars.shape)
            # downsample = 32 // (int)(data_aug_conf['up_scale'])
            
            # fig, ax = plt.subplots(1, 1, figsize=(9, 16))
            # ax.imshow(denormalize_img(imgs[0][c_count]).resize([352//downsample, 128//downsample]))
            # lidars = lidars.permute(0, 1, 3, 4, 2)
            # BB, NN, HH, WW, D = lidars.shape
            # lidars = lidars.reshape(-1, D)

            # sum_lidar = np.sum(lidars.numpy(), axis=0)
            # sum_lidar = np.rint(sum_lidar/1000)
            # np.set_printoptions(precision=3)
            # print(max(sum_lidar)/sum_lidar)

            # assert False


            # # filter pixels without points
            # mask = torch.sum(lidars, 1) > 0
            # # print(np.sum(mask.numpy().astype(int)))
            # lidars = lidars.reshape(BB, NN, HH, WW, D)

            # mask = mask.reshape(BB, NN, HH, WW)
            # print(np.sum(mask.numpy().astype(int)))
            # pts = []
            # for hh in range(HH):
            #     for ww in range(WW):
            #         if mask[0, c_count, hh, ww].item() is True:
            #             for kk in range(D):
            #                 if lidars[0, c_count, hh, ww, kk].item() != 0:
            #                     pts.append([ww, hh, kk])
            #                     # ax.scatter(ww, hh, c=(kk+4.0)/45.0, s=1)
            #                     # print('here')
            #                     break
            # pts = np.array(pts)
            # ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='viridis', s=1)
            # ax.axis('off')

            # plt.savefig('/home/yunze/project/mmda/vis/pc_on_img.png', bbox_inches='tight', pad_inches=0, dpi=200)
            # assert False, 'out here.'

            # print(depth.shape)
            # np.set_printoptions(precision=3, suppress=True)
            # print(depth[0, 0, :, 4, 10].cpu().detach().numpy())
            # print(lidars[0, 0, :, 4, 10].cpu().detach().numpy())
            # # lidars = torch.clone(depth)
            
            binimgs = binimgs.to(device)
            
            if parser_name=='imglidardata':
                lidars = lidars.to(device)
                loss_f = loss_final(preds, binimgs)
                loss_d, d_isnan = loss_depth(depth, lidars)
                if d_isnan:
                    loss = loss_f
                else:
                    loss = loss_f + depthloss_weight*loss_d
            elif parser_name=='segmentationdata':
                loss = loss_final(preds, binimgs)
            else:
                assert False, "Error: parser_name incorrect."

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()


            # print(counter, loss_f.item(), loss_d.item(), loss.item())
            # assert False, 'out here.'

            if counter % 10 == 0:
                if parser_name=='imglidardata':
                    print(counter, loss_f.item(), loss_d.item(), loss.item())
                    writer.add_scalar('train/loss_f', loss_f, counter)
                    writer.add_scalar('train/loss_d', loss_d, counter)
                    writer.add_scalar('train/loss', loss, counter)
                else:
                    print(counter, loss.item())
                    writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, parser_name, loss_final, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
