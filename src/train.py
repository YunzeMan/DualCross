import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, DepthLoss, DomainLoss, L2Loss, get_batch_iou, get_val_info


def train(version='trainval',
            dataroot='/nuScenes/data',
            modelf=None,
            nepochs=10000,
            gpuid=1,
            nsweeps=3,
            manual_seed=1,
            weighted_depth=False,
            adv_training=False,
            strict_import=True,
            teacher_student=True,
            train_student=True,
            modelf_teacher=None,
            teacher_lambda=1.0,
            use_gt=True,
            gt_lambda=1.0,
            use_depth=False,
            depth_lambda=0.05,

            H=900, W=1600,
            H_lyft=1080, W_lyft=1920,
            resize_lim=(0.20, 0.235),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            up_scale=4,
            rand_flip=True,
            color_jitter=False,
            color_jitter_conf=[0.2, 0.2, 0.2, 0.1],
            rand_resize=False,
            ncams=5,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',
            parser_name='imglidardata',
            class_name='vehicle',
            sep_dpft=False,
            new_model=False,
            domainloss_lambda=0.1,
            align_place='final',
            middomain_weight=1.0,
            

            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=8,
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
                    'H_lyft': H_lyft, 'W_lyft': W_lyft,
                    'rand_flip': rand_flip,
                    'rand_resize': rand_resize,
                    'color_jitter': color_jitter,
                    'color_jitter_conf': color_jitter_conf,
                    'bot_pct_lim': bot_pct_lim,
                    'up_scale': up_scale,
                    'sep_dpft': sep_dpft,
                    'new_model': new_model,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                    }
    
    map_folder = dataroot
    strainloader, ttrainloader, tvalloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                                            grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                                            parser_name=parser_name, nsweeps=nsweeps, 
                                                            class_name=class_name, domain_gap=domain_gap, 
                                                            source=source, target=target, adv_training=adv_training)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, parser_name, adv_training, teacher_student, outC=1)
    model.to(device)
    if modelf is not None:
        print('loading existing model from: ', modelf, '...')
        model.load_state_dict(torch.load(modelf, map_location=device), strict=strict_import)
        
    if teacher_student and train_student:
        teacher_model = compile_model(grid_conf, data_aug_conf, parser_name='lidarinputdata', 
                                      adv_training=False, teacher_student=True, outC=1)
        teacher_model.to(device)
        if modelf_teacher is not None:
            print('loading existing model for my teacher model from: ', modelf_teacher, '...')
            teacher_model.load_state_dict(torch.load(modelf_teacher, map_location=device), strict=strict_import)
        teacher_model.eval()

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
    loss_domain_mid = DomainLoss().cuda(gpuid)
    loss_domain_final = DomainLoss().cuda(gpuid)
    loss_teacher = L2Loss().cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 100 if version == 'mini' else 10000

    model.train()
    max_steps=int(2e6)
    source_epoch = 0
    loss_nan_counter = 0
    np.random.seed()
    straingenerator = iter(strainloader)

    sample_domain = 0
    if adv_training: 
        ttraingenerator = iter(ttrainloader)
        target_epoch = 0
    for counter in range(0, max_steps):
        if adv_training:
            # When using adversarial training, need two dataloaders during training
            if counter % 2 == 0:
                sample_domain = 0
                try:
                    imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, index = next(straingenerator)
                except StopIteration:
                    # restart the generator if the previous generator is exhausted.
                    straingenerator = iter(strainloader)
                    imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, index = next(straingenerator)
                    source_epoch += 1
            else:
                sample_domain = 1
                try:
                    imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, index = next(ttraingenerator)
                except StopIteration:
                    # restart the generator if the previous generator is exhausted.
                    ttraingenerator = iter(ttrainloader)
                    imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, index = next(ttraingenerator)
                    target_epoch += 1
        else:
            # When NOT using adversarial training, need only one dataloaders during training
            try:
                imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, index = next(straingenerator)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                straingenerator = iter(strainloader)
                imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, index = next(straingenerator)
                source_epoch += 1
            
        if (sample_domain == 0 and teacher_student and train_student) or parser_name=='lidarinputdata':
            # normalize gt lidar
            lidar_sum = lidars.sum(dim=2, keepdim=True)
            lidars_teacher = lidars / torch.where(lidar_sum > 0.5, lidar_sum, torch.Tensor([1]))


        t0 = time()
        opt.zero_grad()
        # lidars: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        # depth: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        if teacher_student:
            depth, preds, x_mid, x_final, final_layer = model(imgs.to(device),
                                        rots.to(device),
                                        trans.to(device),
                                        intrins.to(device),
                                        post_rots.to(device),
                                        post_trans.to(device),
                                        lidars.to(device),
                                        )
            if train_student:
                _, _, _, _, final_layer_t = teacher_model(imgs.to(device),
                                            rots.to(device),
                                            trans.to(device),
                                            intrins.to(device),
                                            post_rots.to(device),
                                            post_trans.to(device),
                                            lidars_teacher.to(device),
                                            )
                
        else:
            depth, preds, x_mid, x_final = model(imgs.to(device),
                                        rots.to(device),
                                        trans.to(device),
                                        intrins.to(device),
                                        post_rots.to(device),
                                        post_trans.to(device),
                                        lidars.to(device),
                                        )
            
        binimgs = binimgs.to(device)
        lidars = lidars.to(device)

        if adv_training:
            loss_domain_m = loss_domain_mid(x_mid, sample_domain)
            loss_domain_f = loss_domain_final(x_final, sample_domain)
            if align_place == 'midfinal':   loss_domain = middomain_weight * loss_domain_m + loss_domain_f
            elif align_place == 'mid':      loss_domain = middomain_weight * loss_domain_m
            elif align_place == 'final':    loss_domain = loss_domain_f
            else:                           assert False, 'Error value for align_place.'
            
            loss = domainloss_lambda * loss_domain # if domain is '1', end here
            if sample_domain == 0:
                loss_f = loss_final(preds, binimgs)
                loss_t = loss_teacher(final_layer, final_layer_t) if (teacher_student and train_student) else torch.Tensor([0]) 
                loss = loss + teacher_lambda * loss_t if (teacher_student and train_student) else loss
                loss = loss + gt_lambda * loss_f if use_gt else loss
                loss = loss if (teacher_student and train_student) else domainloss_lambda * loss_domain + loss_f
            
                if use_depth:
                    loss_d, d_isnan = loss_depth(depth, lidars)
                    if d_isnan:
                        loss_nan_counter += 1
                        writer.add_scalar('train/loss_nan_counter', loss_nan_counter, counter)
                    else:
                        loss = loss + depth_lambda * loss_d   
                else:
                    loss_d = torch.Tensor([0])  
                        
        else:
            loss_f = loss_final(preds, binimgs)
            loss_t = loss_teacher(final_layer, final_layer_t) if (teacher_student and train_student) else torch.Tensor([0])  
            if teacher_student and train_student:
                loss = teacher_lambda * loss_t
                loss = loss + gt_lambda * loss_f if use_gt else loss
            else:
                loss = loss_f
            
            
            if use_depth:
                loss_d, d_isnan = loss_depth(depth, lidars)
                if d_isnan:
                    loss_nan_counter += 1
                    writer.add_scalar('train/loss_nan_counter', loss_nan_counter, counter)
                else:
                    loss = loss + depth_lambda * loss_d 
            else:
                loss_d = torch.Tensor([0])  
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        t1 = time()

        if counter > 0 and counter % 10 == 0:
            if adv_training:
                print(counter, loss_f.item(), loss_t.item(), loss_d.item(), loss_domain_m.item(), 
                        loss_domain_f.item(), loss.item())
                writer.add_scalar('train/loss_f', loss_f, counter)
                writer.add_scalar('train/loss_t', loss_t, counter)
                writer.add_scalar('train/loss_d', loss_d, counter)
                writer.add_scalar('train/loss_domain_mid', loss_domain_m, counter)
                writer.add_scalar('train/loss_domain_final', loss_domain_f, counter)
                writer.add_scalar('train/loss', loss, counter)
            else:                    
                print(counter, loss_f.item(), loss_t.item(), loss_d.item(), loss.item())
                writer.add_scalar('train/loss_f', loss_f, counter)
                writer.add_scalar('train/loss_t', loss_t, counter)
                writer.add_scalar('train/loss_d', loss_d, counter)
                writer.add_scalar('train/loss', loss, counter)

        if counter > 0 and counter % 50 == 0:
            _, _, iou = get_batch_iou(preds, binimgs)
            writer.add_scalar('train/iou', iou, counter)
            writer.add_scalar('train/source_epoch', source_epoch, counter)
            if adv_training:
                writer.add_scalar('train/target_epoch', target_epoch, counter)
            writer.add_scalar('train/step_time', t1 - t0, counter)

        if counter > 0 and counter % val_step == 0:
            val_info = get_val_info(model, tvalloader, parser_name, loss_final, device, teacher_student=teacher_student)
            print('VAL', val_info)
            writer.add_scalar('val/loss', val_info['loss'], counter)
            writer.add_scalar('val/iou', val_info['iou'], counter)

        if counter % val_step == 0 and counter != 0:
            model.eval()
            mname = os.path.join(logdir, "model{}.pt".format(counter))
            print('saving', mname)
            torch.save(model.state_dict(), mname)
            model.train()
