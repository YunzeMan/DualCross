import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from lyft_dataset_sdk.lyftdataset import LyftDataset
from glob import glob

from .utils.splits import create_splits_scenes
from .tools import (get_lidar_data, get_lidar_data_to_img, img_transform, 
                    normalize_img, gen_dx_bx, get_local_map, get_nusc_maps)


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf, nsweeps, class_name, domain_gap, 
                domain, domain_type, is_lyft):
        self.nusc = nusc
        self.is_lyft = is_lyft
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.domain_gap = domain_gap
        self.domain = domain
        self.downsample = 32 // (int)(self.data_aug_conf['up_scale'])
        self.nsweeps = nsweeps
        self.class_name=class_name
        if self.is_lyft is False:
            self.nusc_maps = get_nusc_maps(self.nusc.dataroot)
        self.domain_type = domain_type
        

        self.scene2map = {}
        for rec in self.nusc.scene:
            log = self.nusc.get('log', rec['log_token'])
            self.scene2map[rec['name']] = log['location']


        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        if self.is_lyft is False:
            self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        # filter by scene split
        if self.domain_gap:
            split = {
                'boston': {'strain': 'boston', 'ttrain': 'boston_train', 'tval': 'boston_val'},
                'singapore': {'strain': 'singapore', 'ttrain': 'singapore_train', 'tval': 'singapore_val'},
                'singapore_day': {'strain': 'singapore_day', 'ttrain': 'singapore_day_train', 'tval': 'singapore_day_val'},
                'singapore_day_train': {'strain': 'singapore_day_train', 'ttrain': None, 'tval': None},
                'day': {'strain': 'day', 'ttrain': None, 'tval': None},
                'night': {'strain': None, 'ttrain': 'night_train', 'tval': 'night_val'},
                'dry': {'strain': 'dry', 'ttrain': None, 'tval': None},
                'rain': {'strain': None, 'ttrain': 'rain_train', 'tval': 'rain_val'},
                'nuscenes': {'strain': 'train', 'ttrain': 'train', 'tval': 'val'},
                'lyft': {'strain': 'lyft_train', 'ttrain': 'lyft_train', 'tval': 'lyft_val'},
            }[self.domain][self.domain_type]
        else:
            split = {
                'v1.0-trainval': {True: 'train', False: 'val'},
                'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
            }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):
        if self.is_lyft:
            H, W = self.data_aug_conf['H_lyft'], self.data_aug_conf['W_lyft']
        else:
            H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            if self.data_aug_conf['rand_resize']: 
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
                resize_dims = (int(W*resize), int(H*resize))
                newW, newH = resize_dims
                crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                resize = max(fH/H, fW/W)
                resize_dims = (int(W*resize), int(H*resize))
                newW, newH = resize_dims
                crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
            if self.data_aug_conf['color_jitter']: 
                colorjt = True
            else:
                colorjt = False    
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            colorjt = False
        return resize, resize_dims, crop, flip, rotate, colorjt

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        # ims = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            if self.is_lyft is False:
                imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            else:
                imgname = os.path.join(self.nusc.data_path, 'train_'+samp['filename'])
                
            img = Image.open(imgname)
            # ims.append(normalize_img(img))
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate, colorjt = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     colorjt=colorjt,
                                                     colorjt_conf=self.data_aug_conf['color_jitter_conf']
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_lidar_data_to_img(self, rec, post_rots, post_trans, cams, downsample, nsweeps):
        points_imgs = get_lidar_data_to_img(self.nusc, rec, data_aug_conf=self.data_aug_conf, 
                                            grid_conf=self.grid_conf, post_rots=post_rots, post_trans=post_trans, 
                                            cams=cams, downsample=downsample, nsweeps=nsweeps, min_distance=2.2,
                                            is_lyft=self.is_lyft)
        return torch.Tensor(points_imgs)  # [N_cam, D, H//downsample, W//downsample]

    def get_binimg(self, rec):
        img = np.zeros((self.nx[0], self.nx[1]))
        if self.class_name == 'vehicle':
            egopose = self.nusc.get('ego_pose',
                                    self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
            trans = -np.array(egopose['translation'])
            rot = Quaternion(egopose['rotation']).inverse
            for tok in rec['anns']:
                inst = self.nusc.get('sample_annotation', tok)
                if self.is_lyft: # this is Lyft dataset
                    if not inst['category_name'] in ['car', 'bus', 'truck', 'other_vehicle', 'bicycle']:
                        continue
                else: # this is nuScenes dataset
                    if not inst['category_name'].split('.')[0] == 'vehicle':
                        continue
                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)
            return torch.Tensor(img).unsqueeze(0)

        elif self.class_name == 'road':
            egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
            map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

            rot = Quaternion(egopose['rotation']).rotation_matrix
            rot = np.arctan2(rot[1, 0], rot[0, 0])
            center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

            poly_names = ['road_segment', 'lane']
            line_names = []
            lmap = get_local_map(self.nusc_maps[map_name], center,
                                50.0, poly_names, line_names)
            for name in poly_names:
                for la in lmap[name]:
                    pts = np.round((la - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    cv2.fillPoly(img, [pts], 1.0)
            return torch.Tensor(img).unsqueeze(0)

        elif self.class_name == 'lane':
            egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
            map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

            rot = Quaternion(egopose['rotation']).rotation_matrix
            rot = np.arctan2(rot[1, 0], rot[0, 0])
            center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

            poly_names = []
            line_names = ['road_divider', 'lane_divider']
            lmap = get_local_map(self.nusc_maps[map_name], center,
                                50.0, poly_names, line_names)
            for name in poly_names:
                for la in lmap[name]:
                    pts = np.round((la - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    cv2.fillPoly(img, [pts], 1.0)
            for name in line_names:
                for la in lmap[name]:
                    pts = np.round((la - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    # valid_pts = np.logical_and((pts[:, 0] < 200), np.logical_and((pts[:, 0] >= 0), 
                    #             np.logical_and((pts[:, 1] >= 0), (pts[:, 1] < 200))))
                    # img[pts[valid_pts, 0], pts[valid_pts, 1]] = 1.0
                    # cv2.fillPoly(img, [pts], 1.0)
                    cv2.polylines(img, [pts], isClosed=False, color=1.0, thickness=2)

            return torch.Tensor(img).unsqueeze(0)
        else:
            assert False, 'class_name invalid. Should be one of [vehicle, road, lane].'


    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data_to_img(rec, post_rots=post_rots, post_trans=post_trans,
                                                cams=cams, downsample=self.downsample, nsweeps=self.nsweeps)
        binimg = self.get_binimg(rec)
                
        # imgs: [cams, 3+D, fH, fW]
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg, index


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
                
        return imgs, rots, trans, intrins, post_rots, post_trans, 0, binimg, index


class ImgLiDARData(NuscData):
    def __init__(self, *args, **kwargs):
        super(ImgLiDARData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data_to_img(rec, post_rots=post_rots, post_trans=post_trans,
                                                cams=cams, downsample=self.downsample, nsweeps=self.nsweeps)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg, index


class LiDARInputData(NuscData):
    def __init__(self, *args, **kwargs):
        super(LiDARInputData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data_to_img(rec, post_rots=post_rots, post_trans=post_trans,
                                                cams=cams, downsample=self.downsample, nsweeps=self.nsweeps)
        binimg = self.get_binimg(rec)
                
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg, index


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz, nworkers, parser_name, 
                nsweeps, class_name, domain_gap, source, target, adv_training):
    # nusc = NuScenes(version='v1.0-{}'.format(version),
    #                 dataroot=os.path.join(dataroot, version),
    #                 verbose=False)
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot,
                    verbose=False)
    if source == 'lyft' or target == 'lyft':
        lyft5 = LyftDataset(data_path='/shared/rsaas/yunzem2/Lyft/', 
                        json_path='/shared/rsaas/yunzem2/Lyft/train_data/', 
                        verbose=False)
    
    if source is not 'lyft':
        source_dataset = nusc
        source_is_lyft = False
    else: 
        source_dataset = lyft5
        source_is_lyft = True
        
    if target is not 'lyft':
        target_dataset = nusc
        target_is_lyft = False
    else:
        target_dataset = lyft5
        target_is_lyft = True
    
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
        'imglidardata': ImgLiDARData,
        'lidarinputdata': LiDARInputData,
    }[parser_name]
    straindata = parser(source_dataset, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf, nsweeps=nsweeps, class_name=class_name, 
                         domain_gap=domain_gap, domain=source, domain_type='strain', is_lyft=source_is_lyft)
    
    tvaldata = parser(target_dataset, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf, nsweeps=nsweeps, class_name=class_name, 
                       domain_gap=domain_gap, domain=target, domain_type='tval', is_lyft=target_is_lyft)

    strainloader = torch.utils.data.DataLoader(straindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init
                                              )
    # strainloader = torch.utils.data.DataLoader(straindata, batch_size=bsz,
    #                                           shuffle=True,
    #                                           num_workers=nworkers,
    #                                           drop_last=True,
    #                                           worker_init_fn=worker_rnd_init,
    #                                           collate_fn=lambda x: x
    #                                           )
    
    tvalloader = torch.utils.data.DataLoader(tvaldata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)
    # tvalloader = torch.utils.data.DataLoader(tvaldata, batch_size=bsz,
    #                                         shuffle=False,
    #                                         num_workers=nworkers,
    #                                         collate_fn=lambda x: x)

    if adv_training:
        ttraindata = SegmentationData(target_dataset, is_train=True, data_aug_conf=data_aug_conf,
                            grid_conf=grid_conf, nsweeps=nsweeps, class_name=class_name, 
                            domain_gap=domain_gap, domain=target, domain_type='ttrain', is_lyft=target_is_lyft)
        ttrainloader = torch.utils.data.DataLoader(ttraindata, batch_size=bsz,
                                                shuffle=True,
                                                num_workers=nworkers,
                                                drop_last=True,
                                                worker_init_fn=worker_rnd_init
                                                )
        # ttrainloader = torch.utils.data.DataLoader(ttraindata, batch_size=bsz,
        #                                         shuffle=True,
        #                                         num_workers=nworkers,
        #                                         drop_last=True,
        #                                         worker_init_fn=worker_rnd_init,
        #                                         collate_fn=lambda x: x
        #                                         )

        return strainloader, ttrainloader, tvalloader
    else:
        return strainloader, None, tvalloader
