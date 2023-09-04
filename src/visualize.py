import os
import torch, numpy as np, copy, cv2
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



def viz_vehicle_road_lane(
                    version='trainval',
                    dataroot='/nuScenes/data',
                    vis_folder='/visualization',
                    gpuid=0,
                    nsweeps=3,
                    viz_gt=False,
                    viz_depth=False,
                    adv_training=False,
                    strict_import=True,
                    limit=2000,

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
                    parser_name='vizdata',
                    sep_dpft=False,
                    new_model=False,
                    vtype='ours',

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
    downsample = 32 // (int)(data_aug_conf['up_scale'])\
    
    _, _, loader_vehicle = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser_name, nsweeps=nsweeps, class_name='vehicle', 
                                          domain_gap=domain_gap, source=source, target=target, adv_training=adv_training)
    _, _, loader_road = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser_name, nsweeps=nsweeps, class_name='road', 
                                          domain_gap=domain_gap, source=source, target=target, adv_training=adv_training)
    _, _, loader_lane = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser_name, nsweeps=nsweeps, class_name='lane', 
                                          domain_gap=domain_gap, source=source, target=target, adv_training=adv_training)

    my_orange = copy.copy(plt.cm.get_cmap('Oranges'))
    my_orange.set_bad(alpha=0)

    my_green = copy.copy(plt.cm.get_cmap('Greens'))
    my_green.set_bad(alpha=0)
    
    my_blue = copy.copy(plt.cm.get_cmap('Blues'))
    my_blue.set_bad(alpha=0)

    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # grid_conf, data_aug_conf, parser_name, adv_training, teacher_student, outC
    model_vehicle = compile_model(grid_conf, data_aug_conf, parser_name, adv_training=True, teacher_student=False, outC=1)
    model_road = compile_model(grid_conf, data_aug_conf, parser_name, adv_training=True, teacher_student=False, outC=1)
    model_lane = compile_model(grid_conf, data_aug_conf, parser_name, adv_training=False, teacher_student=False, outC=1)
    
    # # our model
    vehicle_modelf = './output/runs-vehicle_modelf/model170000.pt'
    road_modelf = './output/runs-road_modelf/model170000.pt'
    lane_modelf = './output/runs-road_modelf/model170000.pt'
    

    print('loading', vehicle_modelf)
    if strict_import:
        model_vehicle.load_state_dict(torch.load(vehicle_modelf, map_location=device))
    else:
        model_vehicle.load_state_dict(torch.load(vehicle_modelf, map_location=device), strict=False)
    model_vehicle.to(device)

    print('loading', road_modelf)
    if strict_import:
        model_road.load_state_dict(torch.load(road_modelf, map_location=device))
    else:
        model_road.load_state_dict(torch.load(road_modelf, map_location=device), strict=False)
    model_road.to(device)

    print('loading', lane_modelf)
    if strict_import:
        model_lane.load_state_dict(torch.load(lane_modelf, map_location=device))
    else:
        model_lane.load_state_dict(torch.load(lane_modelf, map_location=device), strict=False)
    model_lane.to(device)


    if viz_gt:
        model_dir = 'three-layer-ground-truth'
        save_dir = os.path.join(vis_folder, model_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        model_dir = vtype
        save_dir = os.path.join(vis_folder, model_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader_vehicle.dataset.nusc.scene:
        log = loader_vehicle.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']


    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model_vehicle.eval()
    model_road.eval()
    model_lane.eval()
    
    max_steps = 10000
    
    loader_vehicle_generator = iter(loader_vehicle)
    loader_road_generator = iter(loader_road)
    loader_lane_generator = iter(loader_lane)
    
    with torch.no_grad():
        for counter in range(0, max_steps):
            try:
                imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_vehicle, index = next(loader_vehicle_generator)
                _   , _   , _    , _      , _        , _         , _     , binimgs_road   , _     = next(loader_road_generator)
                _   , _   , _    , _      , _        , _         , _     , binimgs_lane   , _     = next(loader_lane_generator)
            except StopIteration:
                # stop when dataset is traversed.
                break
            
            depth, out_vehicle, _, _ = model_vehicle(imgs.to(device), rots.to(device), trans.to(device),
                                                     intrins.to(device), post_rots.to(device), post_trans.to(device),
                                                     lidars.to(device))
            _, out_road, _, _ = model_road(imgs.to(device), rots.to(device), trans.to(device),
                                                     intrins.to(device), post_rots.to(device), post_trans.to(device),
                                                     lidars.to(device))
            _, out_lane, _, _ = model_lane(imgs.to(device), rots.to(device), trans.to(device),
                                                     intrins.to(device), post_rots.to(device), post_trans.to(device),
                                                     lidars.to(device))
            
            out_vehicle = out_vehicle.sigmoid().cpu()
            out_road = out_road.sigmoid().cpu()
            out_lane = out_lane.sigmoid().cpu()
            
            depth = depth.cpu()

            rec = loader_vehicle.dataset.ixes[index[0]]
            
            for si in range(imgs.shape[0]):
                plt.clf()
                # mask = np.zeros((1,6, 41, 128, 352)).astype(np.bool)
                mask = np.zeros(depth.shape).astype(np.bool)
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                        # lidar_mask = lidar_mask.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    # plt.imshow(lmask_full, vmin=0, vmax=1, cmap='Blues', alpha=0.2)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')


                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='b', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color='#3E3EE9', label='Output Vehicle Segmentation'),
                    mpatches.Patch(color='#FF9000', label='Output Road Segmentation'),
                    mpatches.Patch(color='#39BE28', label='Output Lane Marking Segmentation'),
                    mpatches.Patch(color='#00EFFF', label='Ego Vehicle'),
                    # mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
                ], loc=(0.01, 0.83))
                if viz_gt:
                    road_im = binimgs_road[si].squeeze(0)
                    lane_im = binimgs_lane[si].squeeze(0)
                    vehicle_im = binimgs_vehicle[si].squeeze(0)
                    
                    
                    road_im = road_im/2.0
                    lane_im = lane_im/1.7
                    vehicle_im = vehicle_im/1.2
                    road_im[road_im < 0.15] = np.nan
                    lane_im[lane_im < 0.1] = np.nan
                    vehicle_im[vehicle_im < 0.15] = np.nan
                    
                    plt.imshow(road_im, vmin=0, vmax=1, cmap=my_orange, alpha=0.8)
                    plt.imshow(lane_im, vmin=0, vmax=1, cmap=my_green, alpha=0.8)
                    plt.imshow(vehicle_im, vmin=0, vmax=1, cmap=my_blue, alpha=0.8)
                else:
                    road_im = out_road[si].squeeze(0)
                    lane_im = out_lane[si].squeeze(0)
                    vehicle_im = out_vehicle[si].squeeze(0)

                    
                    road_im = road_im/2.0
                    lane_im = lane_im/1.2
                    vehicle_im = vehicle_im/1.15
                    road_im[road_im < 0.195] = np.nan
                    lane_im[lane_im < 0.28] = np.nan
                    vehicle_im[vehicle_im < 0.25] = np.nan

                    plt.imshow(road_im, vmin=0, vmax=1, cmap=my_orange, alpha=0.8)
                    plt.imshow(lane_im, vmin=0, vmax=1, cmap=my_green, alpha=0.8)
                    plt.imshow(vehicle_im, vmin=0, vmax=1, cmap=my_blue, alpha=0.8)
                    

                # plot static map (improves visualization)
                # plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                plt.xlim((out_vehicle.shape[3], 0))
                plt.ylim((0, out_vehicle.shape[3]))
                add_ego(bx, dx)
                    
                imname = f'eval{counter:06}_{si:03}.jpg'
                print('saving', imname)
                
                save_filename = os.path.join(save_dir, imname)
                plt.savefig(save_filename)



def make_video(     version='trainval',
                    dataroot='/nuScenes/data',
                    vis_folder='/visualization',
                    gpuid=0,
                    nsweeps=3,
                    viz_gt=False,
                    viz_depth=False,
                    adv_training=False,
                    strict_import=True,
                    limit=2000,

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
                    parser_name='vizdata',
                    sep_dpft=False,
                    new_model=False,
                    vtype='ours',
                    start_frame=0,
                    end_frame=100,
                    

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
    
    images_dir = os.path.join(vis_folder, vtype)
    
    
    frame = cv2.imread(os.path.join(images_dir, 'eval000000_000.jpg'))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_name = os.path.join('../visualization', vtype+'_%04d.mp4' % start_frame)
    print(video_name)
    video = cv2.VideoWriter(video_name, fourcc, 24, (width,height))

    for number in range(start_frame, end_frame+1):
        imname = f'eval{number:06}_000.jpg'
        video.write(cv2.imread(os.path.join(images_dir, imname)))

    print('save here.')
    cv2.destroyAllWindows()
    video.release()
