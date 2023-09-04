import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from pyquaternion import Quaternion
from PIL import Image
import cv2
from functools import reduce
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.geometry import MultiPoint, box


def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)


        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points


###################################################
def project_to_image(points: np.ndarray, view: np.ndarray, normalize: bool, keep_depth: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        if keep_depth:
            points = np.concatenate((points[0:2, :] / points[2:3, :], points[2:3, :]), axis=0)
        else:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


#######################################################
def get_lidar_data_to_img(nusc, sample_rec, data_aug_conf, grid_conf, post_rots, post_trans, cams, downsample, nsweeps, min_distance, is_lyft):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((4, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        if is_lyft is False:
            current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        else:
            current_pc = LidarPointCloud.from_file(os.path.join(nusc.data_path, 'train_'+current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)
        new_points = current_pc
        
        # print(new_points.points, 'Here')
        # exit
        # # Add time vector which can be used as a temporal feature.
        # time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        # times = time_lag * np.ones((1, current_pc.nbr_points()))

        # new_points = np.concatenate((current_pc.points, times), 0)

        points = np.concatenate((points, new_points.points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    # points [4, N]

    ogfH, ogfW = data_aug_conf['final_dim']
    fH, fW = ogfH // downsample, ogfW // downsample
    depth_bins = np.array(grid_conf['dbound'])
    D = np.rint((depth_bins[1]-depth_bins[0])/depth_bins[2]).astype(int) # [4, 5, ..., 44]
    
    # print((cams, D, fH, fW))
    # Get camera poses and timestamp, project point clouds into camera/image planes
    points_imgs = np.zeros((len(cams), D, fH, fW))
    for count, cam in enumerate(cams): 
        # points_lidar = LidarPointCloud(np.copy(points))
        # points_lidar = LidarPointCloud(points)
        cam_rec = nusc.get('sample_data', sample_rec['data'][cam])
        # imgname = os.path.join(nusc.dataroot, cam_rec['filename'])
        # img = Image.open(imgname)
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        cam_calib_rec = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
        cam_pose_rec = nusc.get('ego_pose', cam_rec['ego_pose_token'])

        # Step 1: From ego car to global
        car_to_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                            inverse=False)

        # Step 2: From global to ego
        global_to_cam_ego = transform_matrix(cam_pose_rec['translation'],
                                            Quaternion(cam_pose_rec['rotation']), inverse=True)

        # Step 3: From ego to camera
        cam_ego_to_cam = transform_matrix(cam_calib_rec['translation'],
                                            Quaternion(cam_calib_rec['rotation']), inverse=True)

        # Fuse three transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [cam_ego_to_cam, global_to_cam_ego, car_to_global])
        # points_lidar.transform(trans_matrix)
        points_lidar = trans_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]


        # Step 4: project to image plane
        points_lidar = project_to_image(points_lidar, np.array(cam_calib_rec['camera_intrinsic']), 
                                        normalize=True, keep_depth=True)

        # Do augmentation if needed
        # post_trans: [Ncam, 3], post_rots: [Ncam, 3, 3]
        # points_lidar: [3, N]

        # mask_ori = np.ones(points_lidar.shape[1], dtype=bool)
        # mask_ori = np.logical_and(mask_ori, points_lidar[0, :] > 0)
        # mask_ori = np.logical_and(mask_ori, (points_lidar[0, :]) < oriW)
        # mask_ori = np.logical_and(mask_ori, points_lidar[1, :] > 0)
        # mask_ori = np.logical_and(mask_ori, (points_lidar[1, :]) < oriH)

        # Step 5: project to augmented image plane
        points_lidar = np.dot(post_rots.numpy()[count], points_lidar)
        points_lidar = points_lidar + post_trans.numpy()[count].reshape(3, 1)

        # Change to the points_imgs: [N_cam, D, H//downsample, W//downsample] representation
        mask = np.ones(points_lidar.shape[1], dtype=bool)
        mask = np.logical_and(mask, points_lidar[2, :] > (depth_bins[0] - depth_bins[2]/2.0))
        mask = np.logical_and(mask, points_lidar[2, :] < (depth_bins[1] - depth_bins[2]/2.0))
        mask = np.logical_and(mask, points_lidar[0, :] > 0)
        mask = np.logical_and(mask, (points_lidar[0, :] // downsample) < fW)
        mask = np.logical_and(mask, points_lidar[1, :] > 0)
        mask = np.logical_and(mask, (points_lidar[1, :] // downsample) < fH)
        # mask = np.logical_and(mask, mask_ori)
        points_lidar = points_lidar[:, mask]

        # fill in the volume
        # points_imgs[count, np.rint(points_lidar[2, :]).astype(int)-np.rint(depth_bins[0]).astype(int), 
        #             points_lidar[1, :].astype(int) // downsample, points_lidar[0, :].astype(int) // downsample] += 1
        points_imgs[count, np.rint((points_lidar[2, :] - depth_bins[0])/depth_bins[2]).astype(int), 
                    points_lidar[1, :].astype(int) // downsample, points_lidar[0, :].astype(int) // downsample] += 1

    return points_imgs


def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    points = intrins.matmul(points)
    points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate, colorjt, colorjt_conf):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)
    if colorjt:
        img = torchvision.transforms.ColorJitter(brightness=colorjt_conf[0], contrast=colorjt_conf[1], 
                                                 saturation=colorjt_conf[2], hue=colorjt_conf[3])(img)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))


normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats 

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


class L2Loss(torch.nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def forward(self, ystudent, yteacher):
        loss = self.loss_fn(ystudent, yteacher)
        return loss


class DepthLoss(torch.nn.Module):
    def __init__(self, depth_weight=None, gt_type='probability'):
        super(DepthLoss, self).__init__()
        self.gt_type = gt_type
        self.depth_weight = depth_weight

    def cross_entropy(self, pred, soft_targets):
        if self.depth_weight is None:
            return torch.mean(torch.sum(-soft_targets * torch.log(pred), 1))
        else:
            return torch.mean(torch.matmul(-soft_targets * torch.log(pred), self.depth_weight))

    def filter_empty(self, depth, lidars):
        # lidars, depth: [B, N, final_dim[0]//downsample, final_dim[1]//downsample, D]
        _, _, _, _, D = lidars.shape
        lidars = lidars.reshape(-1, D)
        depth = depth.reshape(-1, D)

        # filter pixels without points
        mask = torch.logical_and(torch.sum(lidars, 1) > 0, torch.min(depth, 1)[0] > 0)
        lidars = lidars[mask, :]
        depth = depth[mask, :]

        return depth, lidars

  
    def forward(self, depth, lidars):
        # lidars, depth: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        depth = depth.permute(0, 1, 3, 4, 2)
        lidars = lidars.permute(0, 1, 3, 4, 2)
        
        depth, lidars = self.filter_empty(depth, lidars)
        if self.gt_type == 'probability':
            lidars = lidars / lidars.sum(dim=1).view(-1, 1)
        else: # Change to other ways here
            lidars = lidars / lidars.sum(dim=1).view(-1, 1)

        loss = self.cross_entropy(depth, lidars)
                
        return loss, torch.isnan(loss).item()

class DomainLoss(torch.nn.Module):
    def __init__(self):
        super(DomainLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_label):
        y_truth_tensor = torch.FloatTensor(y_pred.size())
        y_truth_tensor.fill_(y_label)
        y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
        return self.loss_fn(y_pred, y_truth_tensor)



def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0


def get_batch_iou_distance(preds, binimgs, easy_mask, mid_mask, hard_mask):
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        
        easy_intersect = (pred & tgt & easy_mask).sum().float().item()
        easy_union = ((pred | tgt) & easy_mask).sum().float().item()
        
        mid_intersect = (pred & tgt & mid_mask).sum().float().item()
        mid_union = ((pred | tgt) & mid_mask).sum().float().item()
        
        hard_intersect = (pred & tgt & hard_mask).sum().float().item()
        hard_union = ((pred | tgt) & hard_mask).sum().float().item()
        
    return (easy_intersect, mid_intersect, hard_intersect), (easy_union, mid_union, hard_union)


def get_val_info(model, valloader, parser_name, loss_fn, device, use_tqdm=False, teacher_student=True):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, rec = batch
            if teacher_student:
                depth, preds, x_mid, x_final, final_layer = model(allimgs.to(device), rots.to(device),
                                                    trans.to(device), intrins.to(device), post_rots.to(device),
                                                    post_trans.to(device), lidars.to(device),)
            else:
                depth, preds, x_mid, x_final = model(allimgs.to(device), rots.to(device),
                                                    trans.to(device), intrins.to(device), post_rots.to(device),
                                                    post_trans.to(device), lidars.to(device),)
                
            
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

    model.train()
    return {
            'loss': total_loss / len(valloader.dataset),
            'iou': total_intersect / total_union,
            }


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#00EFFF')


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys


def get_geom_points(data_aug_conf, grid_conf, rots, trans, intrins, post_rots, post_trans):
    """Determine the (x,y,z) locations (in the ego frame)
    of the points in the point cloud.
    Returns B x N x D x H/downsample x W/downsample x 3
    """
    downsample = 32 // (int)(data_aug_conf['up_scale'])

    # Get frustum
    ogfH, ogfW = data_aug_conf['final_dim']
    fH, fW = ogfH // downsample, ogfW // downsample
    ds = torch.arange(*grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
    D, _, _ = ds.shape
    xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
    ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

    # D x H x W x 3
    frustum = torch.stack((xs, ys, ds), -1)
    # frustum = nn.Parameter(frustum, requires_grad=False)


    B, N, _ = trans.shape

    # undo post-transformation
    # B x N x D x H x W x 3
    points = frustum - post_trans.view(B, N, 1, 1, 1, 3)
    points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

    # cam_to_ego
    points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                        points[:, :, :, :, :, 2:3]
                        ), 5)
    combine = rots.matmul(torch.inverse(intrins))
    points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    points += trans.view(B, N, 1, 1, 1, 3)
    
    return points




def project_depth_BEV(x, geom_feats, grid_conf):
    """Determine the (x,y,z) locations (in the ego frame)
    of the points in the point cloud.
    Returns B x N x D x H/downsample x W/downsample x 3
    """
    use_quickcumsum = True
    dx, bx, nx = gen_dx_bx(grid_conf['xbound'],
                           grid_conf['ybound'],
                           grid_conf['zbound'],)


    # x: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample]
    B, N, D, H, W = x.shape
    Nprime = B*N*D*H*W

    # flatten x
    x = x.reshape(Nprime, 1)

    # flatten indices
    geom_feats = ((geom_feats - (bx - dx/2.)) / dx).long()
    geom_feats = geom_feats.view(Nprime, 3)
    batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                            device=x.device, dtype=torch.long) for ix in range(B)])
    geom_feats = torch.cat((geom_feats, batch_ix), 1)

    # filter out points that are outside box
    kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0])\
        & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1])\
        & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
    x = x[kept]
    geom_feats = geom_feats[kept]

    # get tensors from the same voxel next to each other
    ranks = geom_feats[:, 0] * (nx[1] * nx[2] * B)\
        + geom_feats[:, 1] * (nx[2] * B)\
        + geom_feats[:, 2] * B\
        + geom_feats[:, 3]
    sorts = ranks.argsort()
    x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

    # cumsum trick
    if not use_quickcumsum:
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)
    else:
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

    # griddify (B x C x Z x X x Y)
    final = torch.zeros((B, 1, nx[2], nx[0], nx[1]), device=x.device)
    final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

    # collapse Z
    final = torch.cat(final.unbind(dim=2), 1)

    return final


def post_process_coords(corner_coords, imsize):
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def get_2d_boxes(nusc, rec, cam, post_rots, post_trans, imsize, downsample):
    img = np.zeros((imsize[0]//downsample, imsize[1]//downsample))
    img_full = np.zeros((imsize[0], imsize[1]))

    sd_rec = nusc.get('sample_data', rec['data'][cam])

    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError('The 2D re-projections are available only for keyframes.')

    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    for tok in rec['anns']:
        inst = nusc.get('sample_annotation', tok)
        # add category for lyft
        if not inst['category_name'].split('.')[0] == 'vehicle':
            continue
        # box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
        box = nusc.get_box(inst['token'])
        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corners_2d = project_to_image(corners_3d,camera_intrinsic, normalize=True, keep_depth=True)
        # corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        corners_2d = np.dot(post_rots.numpy(), corners_2d)
        corners_2d = corners_2d + post_trans.numpy().reshape(3, 1)

        corners_2d = corners_2d.T
        # corners_2d[:, [1, 0]] = corners_2d[:, [0, 1]]

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corners_2d[:, :2].tolist(), imsize=(imsize[1], imsize[0]))

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords
            pts = np.array([[min_x, min_y],
                            [min_x, max_y],
                            [max_x, max_y],
                            [max_x, min_y]]).astype(int) // downsample
            pts_full = np.array([[min_x, min_y],
                            [min_x, max_y],
                            [max_x, max_y],
                            [max_x, min_y]]).astype(int)

            cv2.fillPoly(img, [pts], 1.0)
            cv2.fillPoly(img_full, [pts_full], 1.0)

    return img, img_full
