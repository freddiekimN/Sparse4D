import torch
import cv2
import numpy as np
from mmcv import Config, DictAction
from mmdet.models import build_detector
from mmdet.datasets import replace_ImageToTensor, build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
import sys
import os
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import matplotlib.pyplot as plt
from PIL import Image
import mmcv

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor


frame_index = 0

def _img_transform(img, aug_configs):
    H, W = img.shape[:2]
    resize = aug_configs.get("resize", 1)
    resize_dims = (int(W * resize), int(H * resize))
    crop = aug_configs.get("crop", [0, 0, *resize_dims])
    flip = aug_configs.get("flip", False)
    rotate = aug_configs.get("rotate", 0)

    origin_dtype = img.dtype
    if origin_dtype != np.uint8:
        min_value = img.min()
        max_vaule = img.max()
        scale = 255 / (max_vaule - min_value)
        img = (img - min_value) * scale
        img = np.uint8(img)
    img = Image.fromarray(img)
    img = img.resize(resize_dims).crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)
    img = np.array(img).astype(np.float32)
    if origin_dtype != np.uint8:
        img = img.astype(np.float32)
        img = img / scale + min_value

    transform_matrix = np.eye(3)
    transform_matrix[:2, :2] *= resize
    transform_matrix[:2, 2] -= np.array(crop[:2])
    if flip:
        flip_matrix = np.array(
            [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
        )
        transform_matrix = flip_matrix @ transform_matrix
    rotate = rotate / 180 * np.pi
    rot_matrix = np.array(
        [
            [np.cos(rotate), np.sin(rotate), 0],
            [-np.sin(rotate), np.cos(rotate), 0],
            [0, 0, 1],
        ]
    )
    rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
    transform_matrix = rot_matrix @ transform_matrix
    extend_matrix = np.eye(4)
    extend_matrix[:3, :3] = transform_matrix
    return img, extend_matrix

def plot_rect3d_on_img_matplotlib(
    img, num_rects, rect_corners, color=(0, 1, 0), linewidth=1
):
    """Plot the boundary lines of 3D rectangulars on 2D images using matplotlib.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[float], optional): The color to draw bboxes (normalized RGB).
            Default: (0, 1, 0).
        linewidth (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )

    h, w = img.shape[:2]
    plt.imshow(img)
    
    for i in range(num_rects):
        corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)
        for start, end in line_indices:
            if (
                (corners[start, 1] >= h or corners[start, 1] < 0)
                or (corners[start, 0] >= w or corners[start, 0] < 0)
            ) and (
                (corners[end, 1] >= h or corners[end, 1] < 0)
                or (corners[end, 0] >= w or corners[end, 0] < 0)
            ):
                continue
            
            plt.plot(
                [corners[start, 0], corners[end, 0]],
                [corners[start, 1], corners[end, 1]],
                linewidth=linewidth,
            )
    
    plt.axis('off')  # Hide the axes for a cleaner look
    plt.show()

def plot_rect3d_on_img(
    img, num_rects, rect_corners, color=(0, 255, 0), thickness=1
):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    h, w = img.shape[:2]
    for i in range(num_rects):
        corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)
        for start, end in line_indices:
            if (
                (corners[start, 1] >= h or corners[start, 1] < 0)
                or (corners[start, 0] >= w or corners[start, 0] < 0)
            ) and (
                (corners[end, 1] >= h or corners[end, 1] < 0)
                or (corners[end, 0] >= w or corners[end, 0] < 0)
            ):
                continue
            if isinstance(color[0], int):
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color[i],
                    thickness,
                    cv2.LINE_AA,
                )

    return img.astype(np.uint8)

def draw_lidar_bbox3d_on_img(
    bboxes3d, raw_img, lidar2img_rt, img_metas=None, color=(0, 255, 0), thickness=1
):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    # corners_3d = bboxes3d.corners
    corners_3d = box3d_to_corners(bboxes3d)
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1
    )
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)

def box3d_to_corners(box3d):
    YAW = 6  # decoded
    X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
    CNS, YNS = 0, 1  # centerness and yawness indices in qulity
    YAW = 6  # decoded
    
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    rot_cos = np.cos(box3d[:, YAW])
    rot_sin = np.sin(box3d[:, YAW])
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    return corners

def plot_bev_orthogonal(
    bboxes_3d, bev_size, bev_range=115, color=(255, 0, 0), thickness=3
):
    if isinstance(bev_size, (list, tuple)):
        bev_h, bev_w = bev_size
    else:
        bev_h, bev_w = bev_size, bev_size
    bev = np.zeros([bev_h, bev_w, 3])

    marking_color = (127, 127, 127)
    bev_resolution = bev_range / bev_h

    # 중앙 좌표
    center_x, center_y = bev_w // 2, bev_h // 2
    
    step_plot = int(5 / bev_resolution)
    start_pixel = int(2 / bev_resolution)
    max_pixel = int(bev_h/step_plot)*step_plot
    # 직교 좌표계 그리드 그리기
    grid_interval=10
    
    for i in range(0, center_y, int(grid_interval / bev_resolution)):
    
        # 수평선 그리기
        cv2.line(
            bev,
            (0, center_y - i),
            (bev_w, center_y - i),
            marking_color,
            thickness=1,
        )
        
        # 수평선 그리기
        cv2.line(
            bev,
            (0, center_y + i),
            (bev_w, center_y + i),
            marking_color,
            thickness=1,
        )
        
        if i != 0:  # 중앙은 생략    
            
            distance_label = f"{np.round(i* bev_resolution)}m"
            cv2.putText(
                bev,
                distance_label,
                (center_y + i, bev_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # 글자 크기 절반으로 줄임
                (255, 255, 255),
                2,
            )            
        
            distance_label = f"-{np.round(i* bev_resolution)}m"
            cv2.putText(
                bev,
                distance_label,
                (center_y - i, bev_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # 글자 크기 절반으로 줄임
                (255, 255, 255),
                2,
            )            
        else:
            distance_label = f"{np.round(i* bev_resolution)}m"
            cv2.putText(
                bev,
                distance_label,
                (center_y + i, bev_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # 글자 크기 절반으로 줄임
                (255, 255, 255),
                2,
            )            
        
    for i in range(0, center_x, int(grid_interval / bev_resolution)):
    
        # 수직선 그리기
        cv2.line(
            bev,
            (center_x+i, 0),
            (center_x+i, bev_h),
            marking_color,
            thickness=1,
        )
        
        # 수직선 그리기
        cv2.line(
            bev,
            (center_x-i, 0),
            (center_x-i, bev_h),
            marking_color,
            thickness=1,
        )        
        if i != 0:  # 중앙은 생략    
            
            distance_label = f"-{np.round(i* bev_resolution)}m"
            cv2.putText(
                bev,
                distance_label,
                (0, center_x+i + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # 글자 크기 절반으로 줄임
                (255, 255, 255),
                2,
            )            
        
            distance_label = f"{np.round(i* bev_resolution)}m"
            cv2.putText(
                bev,
                distance_label,
                (0, center_x-i + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # 글자 크기 절반으로 줄임
                (255, 255, 255),
                2,
            )                  
        else:
            distance_label = f"{np.round(i* bev_resolution)}m"
            cv2.putText(
                bev,
                distance_label,
                (0, center_x+i + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # 글자 크기 절반으로 줄임
                (255, 255, 255),
                2,
            )            

    if len(bboxes_3d) != 0:
        bev_corners = box3d_to_corners(bboxes_3d)[:, [0, 3, 4, 7]][
            ..., [0, 1]
        ]
        xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
        ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
        for obj_idx, (x, y) in enumerate(zip(xs, ys)):
            for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                if isinstance(color[0], (list, tuple)):
                    tmp = color[obj_idx]
                else:
                    tmp = color
                cv2.line(
                    bev,
                    (int(x[p1]), int(y[p1])),
                    (int(x[p2]), int(y[p2])),
                    tmp,
                    thickness=thickness,
                )
    return bev.astype(np.uint8)

def get_projection_matrix(nusc, sample, camera_sensors, T_global_lidar, imgs, aug_config):
    projection_matrices = []

    for idx, cam in enumerate(camera_sensors):
        cam_token = sample['data'][cam]
        
        # Get camera data, calibration, and ego pose
        cam_data = nusc.get('sample_data', cam_token)
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])

        # Calculate camera transformation matrices
        cam_translation = np.array(cam_calib['translation'])
        cam_rotation = Quaternion(cam_calib['rotation']).rotation_matrix
        ego_translation_cam = np.array(ego_pose_cam['translation'])
        ego_rotation_cam = Quaternion(ego_pose_cam['rotation']).rotation_matrix

        T_global_ego_cam = np.eye(4)
        T_global_ego_cam[:3, :3] = ego_rotation_cam
        T_global_ego_cam[:3, 3] = ego_translation_cam

        T_ego_cam = np.eye(4)
        T_ego_cam[:3, :3] = cam_rotation
        T_ego_cam[:3, 3] = cam_translation    

        # Calculate the transformation matrix from lidar to camera
        T_cam_lidar_rt2 = np.linalg.inv(T_ego_cam) @ np.linalg.inv(T_global_ego_cam) @ T_global_lidar

        # Camera intrinsic matrix
        intrinsic = copy.deepcopy(cam_calib["camera_intrinsic"])
        viewpad = np.eye(4)
        viewpad[:3, :3] = intrinsic

        # Calculate the final projection matrix
        lidar2img_rt2 = viewpad @ T_cam_lidar_rt2

        # Apply image transformation (augmentation)
        img, mat = _img_transform(imgs[idx], aug_config)

        # Convert to torch tensor
        lidar2img_rt2_tensor = torch.tensor(mat @ lidar2img_rt2)

        projection_matrices.append(lidar2img_rt2_tensor)

    return torch.stack(projection_matrices)

# 이미지 리사이징 함수 정의
def resize_image(image, size):
    return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

def get_image_wh():
    image_wh_value = torch.tensor([[[704., 256.],
                                        [704., 256.],
                                        [704., 256.],
                                        [704., 256.],
                                        [704., 256.],
                                        [704., 256.]]], device='cuda:0')
    return image_wh_value
    


ID_COLOR_MAP = [
    (59, 59, 238),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
    (0, 127, 255),
    (71, 130, 255),
    (127, 127, 0),
]
# BGR to RGB conversion
ID_COLOR_MAP_RGB = [(r, g, b) for (b, g, r) in ID_COLOR_MAP]

# 현재 스크립트 파일의 디렉터리 경로
current_dir = os.path.dirname(__file__)

# 최상위 프로젝트 디렉터리 경로
# project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 최상위 프로젝트 디렉터리를 Python 경로에 추가
sys.path.insert(0, current_dir)

from projects.mmdet3d_plugin.datasets.builder import build_dataloader

config = 'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'
cfg = Config.fromfile(config)
checkpoint = 'ckpt/sparse4dv3_r50.pth'

# cfg.data.test.test_mode = True

dataset = build_dataset(cfg.data.test)

samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
distributed = False
data_loader = build_dataloader_origin(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False,
)

cfg.model.train_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))


fp16_cfg = cfg.get("fp16", None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")

model.CLASSES = checkpoint["meta"]["CLASSES"]

model = MMDataParallel(model, device_ids=[0])

model.eval()

data_iter = data_loader.__iter__()
from mmcv.parallel import scatter
plt.figure(figsize=(20, 10))

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import copy
import scipy
from scipy.spatial.transform import Rotation as R
nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)
scene = nusc.scene[2]  # Adjust the scene index as needed
current_sample_token = scene['first_sample_token']

import torchvision.transforms as transforms
# 이미지 리사이징 변환 정의
resize_transform = transforms.Compose([
    transforms.Resize((256, 704)),  # 이미지 리사이즈
    transforms.ToTensor()  # PIL 이미지를 텐서로 변환
])

aug_config = {'resize': 0.44, 'resize_dims': (704, 396), 'crop': (0, 140, 704, 396), 'flip': False, 'rotate': 0, 'rotate_3d': 0}

plt.ion()


def get_global_inv(ego_pose_lidar, lidar_calib):
    
    #lidar_translation = np.array(lidar_calib['translation'])
    #lidar_rotation = Quaternion(lidar_calib['rotation']).rotation_matrix
    lidar_translation = np.array([0, 0.0, 1.84019])
    
    # 쿼터니언을 Rotation 객체로 변환
    lidar_rot = [lidar_calib['rotation'][1],lidar_calib['rotation'][2],lidar_calib['rotation'][3], lidar_calib['rotation'][0]]
    rotation = R.from_quat(lidar_rot) 
    euler_angles = rotation.as_euler('xyz', degrees=False) 
    rotation = R.from_euler('xyz', [0, 0, euler_angles[2]])
    x, y, z,w = rotation.as_quat()
    lidar_rotation = Quaternion([w, x, y, z]).rotation_matrix
    
    # Ego pose 정보에서 변환 행렬 계산
    ego_translation_lidar = np.array(ego_pose_lidar['translation'])
    #ego_rotation_lidar = Quaternion(ego_pose_lidar['rotation']).rotation_matrix
    
    # 쿼터니언을 Rotation 객체로 변환
    ego_rot = [ego_pose_lidar['rotation'][1],ego_pose_lidar['rotation'][2],ego_pose_lidar['rotation'][3], ego_pose_lidar['rotation'][0]]    
    rotation = R.from_quat(ego_rot) 
    euler_angles = rotation.as_euler('xyz', degrees=False) 
    rotation = R.from_euler('xyz', [0, 0, euler_angles[2]])
    x, y, z,w = rotation.as_quat()
    ego_rotation_lidar = Quaternion([w, x, y, z]).rotation_matrix
    
    T_global_ego_lidar = np.eye(4)
    T_global_ego_lidar[:3, :3] = ego_rotation_lidar
    T_global_ego_lidar[:3, 3] = ego_translation_lidar


    T_ego_lidar = np.eye(4)
    T_ego_lidar[:3, :3] = lidar_rotation
    T_ego_lidar[:3, 3] = lidar_translation

    # global frame_index
    # frame_index = frame_index + 1
    # t_x, t_y, t_z = lidar_calib['translation']
    # w, x, y, z = lidar_calib['rotation']
    # ysqr = y * y
    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + ysqr)
    # roll_x = np.degrees(np.arctan2(t0, t1))
    
    # t2 = +2.0 * (w * y - z * x)
    # t2 = np.clip(t2, -1.0, 1.0)
    # pitch_y = np.degrees(np.arcsin(t2))
    
    # t3 = +2.0 * (w * z + x * y)
    # t4 = +1.0 - 2.0 * (ysqr + z * z)
    # yaw_z = np.degrees(np.arctan2(t3, t4))
    # plt.figure(1)
    # plt.subplot(3,1,1)
    # plt.plot(frame_index,roll_x,marker='o')
    # plt.title('Roll')
    # plt.ylabel('Roll (degrees)')  # y축에 도 단위 포함    
    # plt.grid(True)  # 그리드 추가        
        
    # plt.subplot(3,1,2)
    # plt.plot(frame_index,pitch_y,marker='o')
    # plt.title('Pitch')
    # plt.ylabel('Pitch (degrees)')  # y축에 도 단위 포함    
    # plt.grid(True)  # 그리드 추가        
        
    # plt.subplot(3,1,3)
    # plt.plot(frame_index,yaw_z,marker='o')
    # plt.title('Yaw')
    # plt.ylabel('Yaw (degrees)')  # y축에 도 단위 포함
    # plt.grid(True)  # 그리드 추가        
        
    # plt.figure(2)
    # plt.subplot(3,1,1)
    # plt.plot(frame_index,t_x,marker='o')
    # plt.title('x')
    # plt.ylabel('x (m)')  # y축에 도 단위 포함    
    # plt.grid(True)  # 그리드 추가        
        
    # plt.subplot(3,1,2)
    # plt.plot(frame_index,t_y,marker='o')
    # plt.title('y')
    # plt.ylabel('y (m)')  # y축에 도 단위 포함
    # plt.grid(True)  # 그리드 추가        
    
    # plt.subplot(3,1,3)
    # plt.plot(frame_index,t_z,marker='o')
    # plt.title('z')
    # plt.ylabel('z (m)')  # y축에 도 단위 포함
    # plt.grid(True)  # 그리드 추가        
    
    # plt.draw()
    # plt.pause(1)

    # 글로벌 좌표계에서 카메라 좌표계로 변환하는 행렬 계산
    T_global_lidar = T_global_ego_lidar @ T_ego_lidar
    T_global_lidar_inv = np.linalg.inv(T_global_lidar)
    return T_global_lidar,T_global_lidar_inv

while current_sample_token:

    # data2 = next(data_iter)
    # gpu_id = 0
    # data2 = scatter(data2, [gpu_id])[0]

    data = {
        'projection_mat': torch.zeros((1, 6, 4, 4)), # 1x6x4x4 크기의 텐서
        'img_metas': list()
    }
   
    sample = nusc.get('sample', current_sample_token)

    camera_sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT','CAM_BACK_RIGHT']

    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

    ego_pose_lidar = nusc.get("ego_pose", lidar_data["ego_pose_token"])

    lidar_calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

    # 카메라 이미지 추출
    radar_tokens = {
        'front': sample['data']['CAM_FRONT'],
        'front_right': sample['data']['CAM_FRONT_RIGHT'],
        'front_left': sample['data']['CAM_FRONT_LEFT'],
        'back': sample['data']['CAM_BACK'],        
        'back_left': sample['data']['CAM_BACK_LEFT'],
        'back_right': sample['data']['CAM_BACK_RIGHT'],
    }    
    # raw_imgs = data["img"][0].permute(0, 2, 3, 1).cpu().numpy()
    img_norm_mean = np.array(cfg.img_norm_cfg["mean"])
    img_norm_std = np.array(cfg.img_norm_cfg["std"])

    # image = raw_imgs * img_norm_std + img_norm_mean

    mean = [123.675, 116.28 , 103.53 ]
    std = [58.395, 57.12 , 57.375]
    to_rgb = True
    color_type = 'unchanged'    
    imgs = []
    for key, img_token in radar_tokens.items():
        img_sensor = nusc.get('sample_data', img_token)
        img = mmcv.imread(os.path.join(os.path.join(nusc.dataroot, img_sensor['filename'])), color_type)
        img = img.astype(np.float32)
        imgs.append(img)
        
        
    N = len(imgs)
    new_imgs = []
    for i in range(N):
        img, mat = _img_transform(
            np.uint8(imgs[i]), aug_config,
        )
        new_imgs.append(np.array(img).astype(np.float32))
        
    imgs_nor = []
    for img in new_imgs:
        img_nor = mmcv.imnormalize(img, np.array(mean), np.array(std), to_rgb)
        imgs_nor.append(img_nor)        

    # # 리스트를 (1, 6, 3, 256, 704) 크기의 텐서로 변환
    # # numpy 배열을 PyTorch 텐서로 변환하고 차원 순서를 (C, H, W)로 변경
    resized_imgs_tensor = [torch.tensor(img).permute(2, 0, 1).to('cuda:0') for img in imgs_nor]
    imgs_tensor = torch.stack(resized_imgs_tensor).unsqueeze(0)  # (6, 3, 256, 704) -> (1, 6, 3, 256, 704)
    
    data['img'] = imgs_tensor.to(torch.float32)
    

    # 카메라 캘리브레이션 정보에서 변환 행렬 계산
    T_global_lidar, T_global_lidar_inv = get_global_inv(ego_pose_lidar, lidar_calib)
    
    data['img_metas'] = [data['img_metas']]

    data['img_metas'][0] = dict()
    data['img_metas'][0]['T_global'] = T_global_lidar
    data['img_metas'][0]['T_global_inv'] = T_global_lidar_inv
    
    raw_imgs = data["img"][0].permute(0, 2, 3, 1).cpu().numpy()
    imgs = []
    img_norm_mean = np.array(cfg.img_norm_cfg["mean"])
    img_norm_std = np.array(cfg.img_norm_cfg["std"])
    image = raw_imgs * img_norm_std + img_norm_mean
    for i in range(6):
        img = image[i]
        if i > 2:
            img = cv2.flip(img, 1)
            
        resized_img = cv2.resize(img,(1600, 900))
        ubyte_img = resized_img.astype(np.uint8)
        imgs.append(ubyte_img)    
    
    # 이미지 비교하는 구문
    # raw_imgs2 = data2["img"][0].permute(0, 2, 3, 1).cpu().numpy()
    # imgs = []
    # img_norm_mean = np.array(cfg.img_norm_cfg["mean"])
    # img_norm_std = np.array(cfg.img_norm_cfg["std"])

    # image2 = raw_imgs2 * img_norm_std + img_norm_mean    
    
    # for i in range(6):
    #     plt.figure(1)
    #     plt.imshow(image[i].astype(np.uint8))
    #     plt.draw()  # 그래프를 그립니다.
        
    #     plt.figure(2)
    #     plt.imshow(image2[i].astype(np.uint8))
    #     plt.draw()  # 그래프를 그립니다.
    #     plt.pause(1)  # 1초 대기    

    float_value = sample['timestamp']/1000000.0
    data['img_metas'][0]['timestamp'] = float_value
    data['timestamp'] = torch.tensor(float_value, device='cuda:0',dtype=torch.float64).unsqueeze(0)
    
    data['image_wh'] = get_image_wh()
    
    data['projection_mat'][0] = get_projection_matrix(nusc, sample, camera_sensors, T_global_lidar, imgs, aug_config)
    
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
        
    raw_imgs = data["img"][0].permute(0, 2, 3, 1).cpu().numpy()
    imgs = []
    img_norm_mean = np.array(cfg.img_norm_cfg["mean"])
    img_norm_std = np.array(cfg.img_norm_cfg["std"])

    image = raw_imgs * img_norm_std + img_norm_mean
    for i in range(6):

        img = image[i]
        
        if i > 2:
            img = cv2.flip(img, 1)
            
        resized_img = cv2.resize(img,(1600, 900))
        ubyte_img = resized_img.astype(np.uint8)
        # rgb_img = cv2.cvtColor(ubyte_img, cv2.COLOR_BGR2RGB)
        imgs.append(ubyte_img)        
        
    images = np.concatenate(
        [
            np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
            np.concatenate([imgs[4], imgs[3], imgs[5]], axis=1),
        ],
        axis=0,
    )  

    result = result[0]["img_bbox"]
    vis_score_threshold = 0.3    
    pred_bboxes_3d = result["boxes_3d"][
        result["scores_3d"] > vis_score_threshold
    ]

    color = []
    for id in result["labels_3d"].cpu().numpy().tolist():
        color.append(ID_COLOR_MAP_RGB[id])

    bev = plot_bev_orthogonal(
        pred_bboxes_3d,
        bev_size=900 * 2,
        color=color,
    )
    
    images = np.concatenate([images, bev], axis=1)
    
    plt.clf()  # 현재 화면을 지우고 갱신할 수 있도록 합니다.
    plt.imshow(images)
    plt.draw()  # 그래프를 그립니다.
    plt.pause(1)  # 1초 대기
    cv2.destroyAllWindows()
    current_sample = nusc.get('sample', current_sample_token)    
    current_sample_token = current_sample['next']  # Move to the next sample


debug = 1


