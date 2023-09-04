import os
import torch, numpy as np, copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches
import itertools

from .data import compile_data
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map, get_geom_points, 
                    project_depth_BEV, get_2d_boxes, get_batch_iou,
                    get_batch_iou_distance)
from .models import compile_model


def lidar_check(version,
                dataroot='/nuScenes/data',
                show_lidar=True,
                viz_train=False,
                nepochs=1,
                parser_name='imglidardata',

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=1,
                nworkers=10,

                domain_gap=False,
                source='boston',
                target='singapore'
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 5,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='vizdata', domain_gap=domain_gap, source=source, target=target)

    loader = trainloader if viz_train else valloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)

    rat = H / W
    val = 10.1
    fig = plt.figure(figsize=(val + val/3*2*rat*3, val/3*2*rat))
    gs = mpl.gridspec.GridSpec(2, 6, width_ratios=(1, 1, 1, 2*rat, 2*rat, 2*rat))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    for epoch in range(nepochs):
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, pts, binimgs) in enumerate(loader):

            img_pts = model.get_geometry(rots, trans, intrins, post_rots, post_trans)

            for si in range(imgs.shape[0]):
                plt.clf()
                final_ax = plt.subplot(gs[:, 5:6])
                for imgi, img in enumerate(imgs[si]):
                    ego_pts = ego_to_cam(pts[si], rots[si, imgi], trans[si, imgi], intrins[si, imgi])
                    mask = get_only_in_img_mask(ego_pts, H, W)
                    plot_pts = post_rots[si, imgi].matmul(ego_pts) + post_trans[si, imgi].unsqueeze(1)

                    ax = plt.subplot(gs[imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    plt.imshow(showimg)
                    if show_lidar:
                        plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=ego_pts[2, mask],
                                s=5, alpha=0.1, cmap='jet')
                                        
                    plt.axis('off')

                    plt.sca(final_ax)
                    plt.plot(img_pts[si, imgi, :, :, :, 0].view(-1), img_pts[si, imgi, :, :, :, 1].view(-1), '.', label=cams[imgi].replace('_', ' '))
                
                plt.legend(loc='upper right')
                final_ax.set_aspect('equal')
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))

                ax = plt.subplot(gs[:, 3:4])
                plt.scatter(pts[si, 0], pts[si, 1], c=pts[si, 2], vmin=-5, vmax=5, s=5)
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))
                ax.set_aspect('equal')

                ax = plt.subplot(gs[:, 4:5])
                plt.imshow(binimgs[si].squeeze(0).T, origin='lower', cmap='Greys', vmin=0, vmax=1)

                imname = f'lcheck{epoch:03}_{batchi:05}_{si:02}.jpg'
                print('saving', imname)
                plt.savefig(imname)


def cumsum_check(version,
                dataroot='/nuScenes/data',
                gpuid=1,

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=4,
                nworkers=10,
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
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': 6,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    loader = trainloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    model.eval()
    for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):

        model.use_quickcumsum = False
        model.zero_grad()
        out = model(imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
                )
        out.mean().backward()
        print('autograd:    ', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())

        model.use_quickcumsum = True
        model.zero_grad()
        out = model(imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
                )
        out.mean().backward()
        print('quick cumsum:', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())
        print()


def eval_model_iou(version,
                modelf,
                dataroot='/nuScenes/data',
                gpuid=1,

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=4,
                nworkers=10,
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
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': 5,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    loss_fn = SimpleLoss(1.0).cuda(gpuid)

    model.eval()
    val_info = get_val_info(model, valloader, loss_fn, device)
    print(val_info)


def viz_model_preds(modelf,
                    version='trainval',
                    dataroot='/nuScenes/data',
                    vis_folder='/visualization',
                    gpuid=1,
                    nsweeps=3,
                    viz_train=False,
                    viz_gt=False,
                    viz_depth=False,
                    adv_training=False,
                    strict_import=True,
                    limit=1000,
                    teacher_student=False,

                    H=900, W=1600,
                    H_lyft=1080, W_lyft=1920,
                    resize_lim=(0.22, 0.235),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    up_scale=2,
                    rand_flip=True,
                    color_jitter=True,
                    color_jitter_conf=[0.2, 0.2, 0.2, 0.1],
                    rand_resize=True,
                    ncams=5,
                    parser_name='segmentationdata',
                    class_name='vehicle',
                    sep_dpft=False,
                    new_model=False,
                    mask_2d=True,
                    fake_depth=False,
                    fake_type=None,

                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[4.0, 45.0, 1.0],

                    bsz=4,
                    nworkers=10,

                    domain_gap=True,
                    source='boston',
                    target='singapore',
                    ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
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
                    'cams': cams,
                    'Ncams': 5,
                }
    map_folder = dataroot
    downsample = 32 // (int)(data_aug_conf['up_scale'])
    strainloader, ttrainloader, tvalloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser_name, nsweeps=nsweeps, class_name=class_name, 
                                          domain_gap=domain_gap, source=source, target=target, adv_training=adv_training)
    
    loader = strainloader if viz_train else tvalloader
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, parser_name, adv_training, teacher_student, outC=1)
    print('loading', modelf)
    if strict_import:
        model.load_state_dict(torch.load(modelf, map_location=device))
    else:
        model.load_state_dict(torch.load(modelf, map_location=device), strict=False)
    model.to(device)

    if viz_gt:
        model_setting = modelf.split('/')[-2][5:]
        model_step = modelf.split('/')[-1][5:-3]
        model_dir = model_setting + '-' + model_step
        model_dir = model_dir + '-with-depth' if viz_depth else model_dir
        model_dir = model_dir + '-mask2d' if mask_2d else model_dir
        model_dir = model_dir + '-fakedepth-' + fake_type if fake_depth else model_dir

        model_dir = 'ground-truth-' + model_dir

        save_dir = os.path.join(vis_folder, model_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        model_setting = modelf.split('/')[-2][5:]
        model_step = modelf.split('/')[-1][5:-3]
        model_dir = model_setting + '-' + model_step
        model_dir = model_dir + '-with-depth' if viz_depth else model_dir
        model_dir = model_dir + '-mask2d' if mask_2d else model_dir
        model_dir = model_dir + '-fakedepth-' + fake_type if fake_depth else model_dir

        save_dir = os.path.join(vis_folder, model_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']


    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, index) in enumerate(loader):
            if batchi >= limit:
                break
            if parser_name=='lidarinputdata' and fake_depth:
                # Play with ground truth depth
                lidars[:, :, 25, :, :] += 5
                lidar_sum = lidars.sum(dim=2, keepdim=True)
                lidars = lidars / torch.where(lidar_sum > 0.5, lidar_sum, torch.Tensor([1]))

            depth, out, x_mid, x_final = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    lidars.to(device),
                    )
            out = out.sigmoid().cpu()
            depth = depth.cpu()

            rec = loader.dataset.ixes[index[0]]
            
            for si in range(imgs.shape[0]):
                plt.clf()
                # mask = np.zeros((1,6, 41, 128, 352)).astype(np.bool)
                mask = np.zeros(depth.shape).astype(np.bool)
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    lmask, lmask_full = get_2d_boxes(loader.dataset.nusc, rec, cams[imgi], post_rots[si][imgi], 
                                              post_trans[si][imgi], data_aug_conf['final_dim'], 
                                              downsample)
                    lmask, lmask_full = lmask.astype(np.bool), lmask_full.astype(np.bool)                
                    mask[si, imgi, :, lmask==True] = True
                    
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                        lmask_full[:, :] = lmask_full[:, ::-1]
                        # lidar_mask = lidar_mask.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.imshow(lmask_full, vmin=0, vmax=1, cmap='Blues', alpha=0.2)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')


                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='b', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                    mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
                ], loc=(0.01, 0.86))
                if viz_gt:
                    plt.imshow(binimgs[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')
                else:
                    plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')

                # plot static map (improves visualization)
                # plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                plt.xlim((out.shape[3], 0))
                plt.ylim((0, out.shape[3]))
                add_ego(bx, dx)

                if viz_depth:
                    geom_feats = get_geom_points(data_aug_conf, grid_conf, rots, trans, intrins, post_rots, post_trans)
                    if viz_gt:
                        lidars[torch.Tensor(mask) == False] = 0
                        lidar_sum = lidars.sum(dim=2, keepdim=True)
                        lidars = lidars / torch.where(lidar_sum > 0.5, lidar_sum, torch.Tensor([1]))
                        lidars[torch.nonzero(lidars, as_tuple=True)] += 0

                        dp_map = project_depth_BEV(lidars, geom_feats, grid_conf)
                        dp_map = dp_map.cpu().detach().numpy()
                        my_red_cmap = copy.copy(plt.cm.get_cmap('Reds'))
                        my_red_cmap.set_under(color="white", alpha="0")
                        plt.imshow(dp_map[si].squeeze(0), vmin=0, vmax=1, cmap=my_red_cmap, alpha=0.7)
                    else:
                        depth[torch.Tensor(mask) == False] = 0
                        depth[torch.nonzero(depth, as_tuple=True)] += 0
                        dp_map = project_depth_BEV(depth, geom_feats, grid_conf)
                        dp_map = dp_map.cpu().detach().numpy()
                        plt.imshow(dp_map[si].squeeze(0), vmin=0, vmax=1, cmap='Reds', alpha=0.55)
                    
                imname = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                
                save_filename = os.path.join(save_dir, imname)
                plt.savefig(save_filename)
                counter += 1



def viz_bad_samples(modelf,
                    version='trainval',
                    dataroot='/nuScenes/data',
                    vis_folder='/visualization',
                    gpuid=1,
                    nsweeps=3,
                    viz_train=False,
                    viz_gt=False,
                    viz_depth=False,
                    adv_training=False,
                    strict_import=True,
                    distance_iou=True,
                    no_vis=False,
                    teacher_student=False,
                    

                    H=900, W=1600,
                    H_lyft=1080, W_lyft=1920,
                    resize_lim=(0.22, 0.235),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    up_scale=2,
                    rand_flip=True,
                    color_jitter=True,
                    color_jitter_conf=[0.2, 0.2, 0.2, 0.1],
                    rand_resize=True,
                    ncams=5,
                    parser_name='segmentationdata',
                    class_name='vehicle',
                    sep_dpft=False,
                    new_model=False,
                    mask_2d=True,
                    fake_depth=False,
                    fake_type=None,
                    vis_good=False,
                    print_dist=True,

                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[4.0, 45.0, 1.0],

                    bsz=4,
                    nworkers=10,

                    domain_gap=True,
                    source='boston',
                    target='singapore',
                    ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
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
                    'cams': cams,
                    'Ncams': 5,
                }
    map_folder = dataroot
    downsample = 32 // (int)(data_aug_conf['up_scale'])
    strainloader, ttrainloader, tvalloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser_name, nsweeps=nsweeps, class_name=class_name, 
                                          domain_gap=domain_gap, source=source, target=target, adv_training=adv_training)
    
    loader = strainloader if viz_train else tvalloader

    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, parser_name, adv_training, teacher_student, outC=1)
    print('loading', modelf)
    if strict_import:
        model.load_state_dict(torch.load(modelf, map_location=device))
    else:
        model.load_state_dict(torch.load(modelf, map_location=device), strict=False)
    model.to(device)

    model_setting = modelf.split('/')[-2][5:]
    model_step = modelf.split('/')[-1][5:-3]
    model_dir = model_setting + '-' + model_step
    model_dir = model_dir + '-with-depth' if viz_depth else model_dir
    model_dir = model_dir + '-mask2d' if mask_2d else model_dir
    model_dir = model_dir + '-fakedepth-' + fake_type if fake_depth else model_dir
    model_dir = 'ground-truth-' + model_dir if viz_gt else model_dir
    model_dir = 'vis-good-' + model_dir if vis_good else 'vis-bad-' + model_dir

    save_dir = os.path.join(vis_folder, model_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']


    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()


    # Different start from here
    sample_iou, sample_union, sample_intersect = [], [], []
    total_intersect, total_union = 0.0, 0.0

    if distance_iou:
        easy_mask = torch.zeros(1, 1, 200, 200).bool().to(device)
        easy_mask[:, :, 70:130, 70:130] = True
        easy_intersect, easy_union = 0.0, 0.0

        
        mid_mask = torch.zeros(1, 1, 200, 200).bool().to(device)
        mid_mask[:, :, 40:160, 40:160] = True
        mid_mask[:, :, 70:130, 70:130] = False
        mid_intersect, mid_union = 0.0, 0.0

        
        hard_mask = torch.ones(1, 1, 200, 200).bool().to(device)
        hard_mask[:, :, 40:160, 40:160] = False
        hard_intersect, hard_union = 0.0, 0.0

    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, index = batch
            depth, preds, x_mid, x_final = model(allimgs.to(device), rots.to(device),
                                                trans.to(device), intrins.to(device), post_rots.to(device),
                                                post_trans.to(device), lidars.to(device),)
            
            binimgs = binimgs.to(device)

            # iou
            intersect, union, iou = get_batch_iou(preds, binimgs)
            
            total_intersect += intersect
            total_union += union
            
            sample_iou.append(iou)
            sample_union.append(union)
            sample_intersect.append(intersect)

            if distance_iou:
                distance_intersects, distance_unions = get_batch_iou_distance(preds, binimgs, easy_mask, mid_mask, hard_mask)
                
                easy_intersect += distance_intersects[0]
                mid_intersect += distance_intersects[1]
                hard_intersect += distance_intersects[2]
                
                easy_union += distance_unions[0]
                mid_union += distance_unions[1]
                hard_union += distance_unions[2]


    sample_iou = np.array(sample_iou)
    sample_union = np.array(sample_union)
    sample_intersect = np.array(sample_intersect)

    sample_ind = np.lexsort((sample_intersect, -1*sample_union, sample_iou))
    
    if print_dist:
        dist = np.zeros((7,))
        for ind in sample_ind:
            if sample_iou[ind] == 0:
                dist[0] += 1
            elif 0 < sample_iou[ind] <= 0.1:
                dist[1] += 1
            elif 0.1 < sample_iou[ind] <= 0.2:
                dist[2] += 1
            elif 0.2 < sample_iou[ind] <= 0.3:
                dist[3] += 1
            elif 0.3 < sample_iou[ind] <= 0.4:
                dist[4] += 1
            elif 0.4 < sample_iou[ind] <= 0.5:
                dist[5] += 1
            elif sample_iou[ind] > 0.5:
                dist[6] += 1
        for val in dist:
            print(val, val/np.sum(dist))
        
        print('Finished printing distribution.')
    
    print('Overall IoU is: %.3f' % (total_intersect / total_union))
    if distance_iou:
        print('Near (0m < 30m)  IoU is: %.3f' % (easy_intersect / easy_union))
        print('Mid (30m < 60m)  IoU is: %.3f' % (mid_intersect / mid_union))
        print('Far (60m < 100m) IoU is: %.3f' % (hard_intersect / hard_union))
    
    if no_vis:
        return 0
        
    if vis_good:
        sample_ind = sample_ind[::-1]
    
    # print(sample_ind)
    # =================  infer DataLoader by index  ==================
    # next(itertools.islice(dataloader, 5, None))

    counter = 0
    with torch.no_grad():
        for k in sample_ind:
            imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, index = \
                                                    next(itertools.islice(loader, int(k), None))
            depth, out, x_mid, x_final = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    lidars.to(device),
                    )
            out = out.sigmoid().cpu()
            depth = depth.cpu()

            rec = loader.dataset.ixes[index[0]]

            for si in range(imgs.shape[0]):
                plt.clf()
                # mask = np.zeros((1,6, 41, 128, 352)).astype(np.bool)
                mask = np.zeros(depth.shape).astype(np.bool)
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    lmask, lmask_full = get_2d_boxes(loader.dataset.nusc, rec, cams[imgi], post_rots[si][imgi], 
                                              post_trans[si][imgi], data_aug_conf['final_dim'], 
                                              downsample)
                    lmask, lmask_full = lmask.astype(np.bool), lmask_full.astype(np.bool)                
                    mask[si, imgi, :, lmask==True] = True
                    
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                        lmask_full[:, :] = lmask_full[:, ::-1]
                        # lidar_mask = lidar_mask.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.imshow(lmask_full, vmin=0, vmax=1, cmap='Blues', alpha=0.2)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')


                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='b', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                    mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
                ], loc=(0.01, 0.86))
                if viz_gt:
                    plt.imshow(binimgs[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')
                else:
                    plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')

                # plot static map (improves visualization)
                # rec = loader.dataset.ixes[counter]
                plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                plt.xlim((out.shape[3], 0))
                plt.ylim((0, out.shape[3]))
                add_ego(bx, dx)

                if viz_depth:
                    geom_feats = get_geom_points(data_aug_conf, grid_conf, rots, trans, intrins, post_rots, post_trans)
                    if viz_gt:
                        lidars[torch.Tensor(mask) == False] = 0
                        lidar_sum = lidars.sum(dim=2, keepdim=True)
                        lidars = lidars / torch.where(lidar_sum > 0.5, lidar_sum, torch.Tensor([1]))
                        lidars[torch.nonzero(lidars, as_tuple=True)] += 0

                        dp_map = project_depth_BEV(lidars, geom_feats, grid_conf)
                        dp_map = dp_map.cpu().detach().numpy()
                        my_red_cmap = copy.copy(plt.cm.get_cmap('Reds'))
                        my_red_cmap.set_under(color="white", alpha="0")
                        plt.imshow(dp_map[si].squeeze(0), vmin=0, vmax=1, cmap=my_red_cmap, alpha=0.7)
                    else:
                        depth[torch.Tensor(mask) == False] = 0
                        depth[torch.nonzero(depth, as_tuple=True)] += 0
                        dp_map = project_depth_BEV(depth, geom_feats, grid_conf)
                        dp_map = dp_map.cpu().detach().numpy()
                        plt.imshow(dp_map[si].squeeze(0), vmin=0, vmax=1, cmap='Reds', alpha=0.55)
                
                plt.annotate('IoU          = %8.3f' % (sample_iou[k]), (-0.4, 0.95), xycoords='axes fraction')
                plt.annotate('Intersection = %8.3f' % (sample_intersect[k]), (-0.4, 0.85), xycoords='axes fraction')
                plt.annotate('Union        = %8.3f' % (sample_union[k]), (-0.4, 0.75), xycoords='axes fraction')

                imname = f'eval{counter:04}_{k:04}.jpg'
                print('saving', imname)
                
                save_filename = os.path.join(save_dir, imname)
                plt.savefig(save_filename)
                counter += 1
