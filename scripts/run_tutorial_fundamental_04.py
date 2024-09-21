import os
import sys
import cv2
from mmcv.parallel import MMDataParallel
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
from nuscenes.nuscenes import NuScenes
import torch
import mmcv
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import copy
import matplotlib.pyplot as plt

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
# 함수 정의
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

def get_global_inv(ego_x_m,ego_y_m,yaw_degree):
    
    # lidar translation 계산
    lidar_translation = np.array([0, 0.0, 1.84019])
    
    # lidar rotation 계산
    euler_angles_1 = np.radians(-89.8835)
    rotation = R.from_euler('xyz', [0, 0, euler_angles_1])
    x, y, z,w = rotation.as_quat()
    lidar_rotation = Quaternion([w, x, y, z]).rotation_matrix
    
    # Ego pose 정보에서 변환 행렬 계산
    ego_translation_lidar = np.array([ego_x_m,ego_y_m,0])
    
    
    # 쿼터니언을 Rotation 객체로 변환
    euler_angles_2 = np.radians(yaw_degree)
    rotation = R.from_euler('xyz', [0, 0, euler_angles_2])
    x, y, z,w = rotation.as_quat()
    ego_rotation_lidar = Quaternion([w, x, y, z]).rotation_matrix
    
    # print(np.degrees(euler_angles_1),np.degrees(euler_angles_2))
    
    T_global_ego_lidar = np.eye(4)
    T_global_ego_lidar[:3, :3] = ego_rotation_lidar
    T_global_ego_lidar[:3, 3] = ego_translation_lidar


    T_ego_lidar = np.eye(4)
    T_ego_lidar[:3, :3] = lidar_rotation
    T_ego_lidar[:3, 3] = lidar_translation

    # 글로벌 좌표계에서 카메라 좌표계로 변환하는 행렬 계산
    T_global_lidar = T_global_ego_lidar @ T_ego_lidar
    T_global_lidar_inv = np.linalg.inv(T_global_lidar)
    return T_global_lidar,T_global_lidar_inv, euler_angles_2, ego_translation_lidar

def get_image_wh():
    image_wh_value = torch.tensor([[[704., 256.],
                                        [704., 256.],
                                        [704., 256.],
                                        [704., 256.],
                                        [704., 256.],
                                        [704., 256.]]], device='cuda:0')
    return image_wh_value

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
    bboxes_3d,bboxes_gt, bev_size, bev_range=115, color=(255, 0, 0), thickness=3
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
                
    if len(bboxes_gt) != 0:
        for boxes in bboxes_gt:
            bev_corners = boxes[0].corners()[:, [1,0,6,7]][[0,1],...].T
            xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
            ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
            for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                tmp = [255, 255, 255]
                cv2.line(
                    bev,
                    (int(xs[p1]), int(ys[p1])),
                    (int(xs[p2]), int(ys[p2])),
                    tmp,
                    thickness=thickness,
                )
                
    return bev.astype(np.uint8)

def get_3d_lines(corners,offset_x,offset_y,resize_div):
    ret = []
    for st, ed in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                   [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        if corners[st, -1] > 0 and corners[ed, -1] > 0:
            ret.append([[corners[st, 0]/resize_div+offset_x, corners[ed, 0]/resize_div+offset_x],
                        [corners[st, 1]/resize_div+offset_y, corners[ed, 1]/resize_div+offset_y]])
    return ret

# 현재 스크립트 파일의 디렉터리 경로
current_dir = os.path.dirname(__file__)

project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 최상위 프로젝트 디렉터리를 Python 경로에 추가
sys.path.insert(0, project_root)

# 모델 생성
config = 'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'
cfg = Config.fromfile(config)
checkpoint = 'ckpt/sparse4dv3_r50.pth'
cfg.data.test.test_mode = True
cfg.model.train_cfg = None

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))

fp16_cfg = cfg.get("fp16", None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")

model.CLASSES = checkpoint["meta"]["CLASSES"]

model = MMDataParallel(model, device_ids=[0])

model.eval()

##
model_img = fasterrcnn_resnet50_fpn(pretrained=True)
model_img.eval()

# 이미지 전처리
transform_img = transforms.Compose([transforms.ToTensor()])


# Nuscene 설정
nusc = NuScenes(version='v1.0-mini', dataroot='/media/joo/6b227b2d-8477-45e1-af10-7dde9f85afe4/data/sets/nuscenes', verbose=True)
a = input('시나리오선택하시오(0~9): ')
scene = nusc.scene[int(a)]  # Adjust the scene index as needed

from nuscenes.can_bus.can_bus_api import NuScenesCanBus
nusc_can = NuScenesCanBus(dataroot='/media/joo/6b227b2d-8477-45e1-af10-7dde9f85afe4/data/sets/nuscenes')
scene_name = scene['name']
# wheel_speed = nusc_can.get_messages(scene_name, 'zoe_veh_info')

veh_speed = nusc_can.get_messages(scene_name, 'vehicle_monitor')

current_sample_token = scene['first_sample_token']
frame_count = 0

from scripts.KalmanFilter import LinearKalmanFilter
dt = 0.5  # delta time (100ms)
process_noise_std = 0.1  # 프로세스 노이즈의 표준편차
measurement_noise_std = [1.0, 0.5]  # 측정 노이즈 표준 편차 (v, yaw rate)
initial_state = [0,0,0,0,0,0]
kf = LinearKalmanFilter(dt, initial_state, process_noise_std, measurement_noise_std)

# 이미지 입력 변수 추가 자료
mean = [123.675, 116.28 , 103.53 ] # np.array(cfg.img_norm_cfg["mean"])
std = [58.395, 57.12 , 57.375] # np.array(cfg.img_norm_cfg["std"])
aug_config = {'resize': 0.44, 'resize_dims': (704, 396), 'crop': (0, 140, 704, 396), 'flip': False, 'rotate': 0, 'rotate_3d': 0}

# frame index 초기화
frame_index = 0

camera_sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT','CAM_BACK_RIGHT']

ori_cam_rotation={}
ori_cam_rotation['CAM_FRONT'] = [-90, 0 , -90]
ori_cam_rotation['CAM_FRONT_RIGHT'] = [-90, 0 , -146]  
ori_cam_rotation['CAM_FRONT_LEFT'] = [-90, 0 , -34]
ori_cam_rotation['CAM_BACK'] = [-90, 0 , 90]  
ori_cam_rotation['CAM_BACK_LEFT'] = [-90, 0 , 18]  
ori_cam_rotation['CAM_BACK_RIGHT'] = [-90, 0 , 159]    
ori_cam_translation={}
ori_cam_translation['CAM_FRONT'] = [1.70079119, 0.01594563, 1.51095764]
ori_cam_translation['CAM_FRONT_RIGHT'] = [1.55084775, -0.4934048, 1.49574801]  
ori_cam_translation['CAM_FRONT_LEFT'] = [1.52387798, 0.49463134, 1.50932822]
ori_cam_translation['CAM_BACK'] = [0.02832603, 0.00345137, 1.57910346]  
ori_cam_translation['CAM_BACK_LEFT'] = [1.035691, 0.48479503, 1.59097015]  
ori_cam_translation['CAM_BACK_RIGHT'] = [1.0148781, -0.48056822, 1.56239545]  

ori_cam_intrinsic = {}

# BGR to RGB conversion
ID_COLOR_MAP = [
    (59, 59, 238), # vibrant blue
    (0, 255, 0),   # green
    (0, 0, 255),   # blue
    (255, 255, 0), # Yellow
    (0, 255, 255), # Cyan
    (255, 0, 255), # Magenta
    (255, 255, 255),# White
    (0, 127, 255),  # medium sky blue
    (71, 130, 255), # medium cornflower blue
    (127, 127, 0),  # Olive or dark yellow-green
]
ID_COLOR_MAP_RGB = [(r, g, b) for (b, g, r) in ID_COLOR_MAP]

score_threshold=0.5
coco_classes = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
target_classes = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]

# 이미지에 박스 그리기
resize_div = 2
resize_dim = ((int)(1600/resize_div), (int)(900/resize_div))  # 원하는 크기로 설정

while current_sample_token:
    
    sample = nusc.get('sample', current_sample_token)
    
    lidar = sample['data']['LIDAR_TOP']
    bboxes_gt = []
    for  index  in range(len(sample['anns'])):
        my_annotation_token = sample['anns'][index]
        visibility_token = nusc.get('sample_annotation', my_annotation_token)
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[my_annotation_token])
        bboxes_gt.append(boxes)

    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    
    ego_pose_lidar = nusc.get("ego_pose", lidar_data["ego_pose_token"])

    data = {
        'projection_mat': torch.zeros((1, 6, 4, 4)), # 1x6x4x4 크기의 텐서
        'img_metas': list()
    }

    # 1. 이미지 입력 변수 만들기
    radar_tokens = {
        'front': sample['data']['CAM_FRONT'],
        'front_right': sample['data']['CAM_FRONT_RIGHT'],
        'front_left': sample['data']['CAM_FRONT_LEFT'],
        'back': sample['data']['CAM_BACK'],        
        'back_left': sample['data']['CAM_BACK_LEFT'],
        'back_right': sample['data']['CAM_BACK_RIGHT'],
    }    
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
    # # numpy 배열을 PyTorch 텐서로 변환하고 차원 순서를(H, W, C) --> (C, H, W)로 변경
    resized_imgs_tensor = [torch.tensor(img).permute(2, 0, 1).to('cuda:0') for img in imgs_nor]
    imgs_tensor = torch.stack(resized_imgs_tensor).unsqueeze(0)  # (6, 3, 256, 704) -> (1, 6, 3, 256, 704)
    
    data['img'] = imgs_tensor.to(torch.float32)

    # 2. 시간 정보 입력
    data['img_metas'] = list()
    data['img_metas'].append(None)
    float_value = sample['timestamp']/1000000.0    
    data['img_metas'][0] = dict()
    data['img_metas'][0]['timestamp'] = float_value
    data['timestamp'] = torch.tensor(float_value, device='cuda:0',dtype=torch.float64).unsqueeze(0)

    # 3. meta 정보 입력
    # 차량 정보를 이용해서 구함.
    while(1):  
        if sample['timestamp'] < veh_speed[frame_index]['utime']:
            break
        else:
            frame_index = frame_index + 1
    
    if frame_index == 0:
        ego_rot = [ego_pose_lidar['rotation'][1],ego_pose_lidar['rotation'][2],ego_pose_lidar['rotation'][3], ego_pose_lidar['rotation'][0]]    
        rotation = R.from_quat(ego_rot) 
        euler_angles_2 = rotation.as_euler('xyz', degrees=True) 
        ego_transition = np.array(ego_pose_lidar['translation'])
        # 초기위치만 라이다의 정보를 이용해서 setting한다.
        initial_state = [ego_transition[0], ego_transition[1],0,0,euler_angles_2[2],0]  # 초기 위치와 방향 [x, y, theta]
        kf.set_init(initial_state)
        start_logic_timestamp = sample['timestamp']
        pre_gt_x, pre_gt_y, start_logic_z = ego_transition
        pre_timestamp = sample['timestamp']
    else:
        # 그 다음부터는 차량 정보를 이용한다.
        omega = veh_speed[frame_index]['yaw_rate']#/180*np.pi
        vs = veh_speed[frame_index]['vehicle_speed']/3.6
        u = np.array([vs, omega]) # speed와 yaw rate
        dif_time = sample['timestamp'] - pre_logic_timestamp
        dt = (dif_time*1e-6)
        kf.predict(dt,u)
        kf.update([vs, omega])

    kf_x = kf.get_state()      
    ego_x_m = kf_x[0]
    ego_y_m = kf_x[1]
    ego_yaw_degree = kf_x[4]
    
    T_global_lidar, T_global_lidar_inv, ego_yaw, ego_transition = get_global_inv(ego_x_m,ego_y_m,ego_yaw_degree)

    # 디버깅을 위한 plot code
    if False:
        plt.figure(4)
        plt.subplot(3,1,1)
        ego_transition = np.array(ego_pose_lidar['translation'])
        gt_x_m = ego_transition[0]
        gt_y_m = ego_transition[1]
        plt.plot(gt_x_m,gt_y_m,marker='o')
        plt.plot(kf_x[0],kf_x[1],marker='x')
        plt.grid(True)  # 그리드 추가  

        ego_rot = [ego_pose_lidar['rotation'][1],ego_pose_lidar['rotation'][2],ego_pose_lidar['rotation'][3], ego_pose_lidar['rotation'][0]]    
        rotation = R.from_quat(ego_rot) 
        euler_angles_2 = rotation.as_euler('xyz', degrees=True) 
        
        ego_yaw_degree_lidar = euler_angles_2[2]    
        
        plt.subplot(3,1,2)
        plt.plot((sample['timestamp'] - start_logic_timestamp)/10000,ego_yaw_degree_lidar,marker='o')
        plt.plot((sample['timestamp'] - start_logic_timestamp)/10000,kf_x[4],marker='x')
        plt.title('ego Yaw ')
        plt.ylabel('Yaw (degrees)')  # y축에 도 단위 포함    
        plt.grid(True)  # 그리드 추가    

        plt.subplot(3,1,3)
        dif_x = gt_x_m - pre_gt_x
        dif_y = gt_y_m - pre_gt_y
        diff_time = sample['timestamp'] - pre_timestamp
        if diff_time == 0 or np.isnan(diff_time):
            gt_vel = kf_x[2]
        else:
            gt_vel = np.sqrt(dif_x**2 + dif_y**2)/(diff_time/1e+6)
        pre_timestamp = sample['timestamp']
        pre_gt_x = gt_x_m
        pre_gt_y = gt_y_m   
        plt.plot((sample['timestamp'] - start_logic_timestamp)/10000,gt_vel,marker='o')
        plt.plot((sample['timestamp'] - start_logic_timestamp)/10000,kf_x[2],marker='x')
        plt.title('velocity')
        plt.ylabel('velocity (km/h)')  # y축에 도 단위 포함    
        plt.grid(True)  # 그리드 추가    

    data['img_metas'][0]['T_global'] = T_global_lidar
    data['img_metas'][0]['T_global_inv'] = T_global_lidar_inv
    data['img_metas'][0]['timestamp'] = float_value
    
    data['image_wh'] = get_image_wh()
    
    
    # projection_matrices 계산하기
    projection_matrices = []
    # 0 CAM_FRONT
    # 1 CAM_FRONT_RIGHT
    # 2 CAM_FRONT_LEFT
    # 3 CAM_BACK
    # 4 CAM_BACK_LEFT
    # 5 CAM_BACK_RIGHT
    for idx, cam in enumerate(camera_sensors):
        cam_token = sample['data'][cam]
        
        # Get camera data, calibration, and ego pose
        cam_data = nusc.get('sample_data', cam_token)
        # # print(idx,cam,cam_data['channel'])
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        # ego_pose_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])

        cam_translation = np.array(ori_cam_translation[cam])
        euler_angles_2 = np.radians(ori_cam_rotation[cam])
        rotation = R.from_euler('xyz', euler_angles_2)
        x, y, z,w = rotation.as_quat()
        cam_rotation = Quaternion([w, x, y, z]).rotation_matrix

        # Ego pose 정보에서 변환 행렬 계산
        ego_translation_cam = np.array([ego_x_m,ego_y_m,0])
        euler_angles_2 = np.radians(ego_yaw_degree)
        rotation = R.from_euler('xyz', [0, 0, euler_angles_2])
        x, y, z,w = rotation.as_quat()
        ego_rotation_cam = Quaternion([w, x, y, z]).rotation_matrix


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
        ori_cam_intrinsic[cam] = intrinsic

        # Calculate the final projection matrix
        lidar2img_rt2 = viewpad @ T_cam_lidar_rt2

        # Apply image transformation (augmentation)
        img, mat = _img_transform(imgs[idx], aug_config)

        # Convert to torch tensor
        lidar2img_rt2_tensor = torch.tensor(mat @ lidar2img_rt2)

        projection_matrices.append(lidar2img_rt2_tensor)

    # return torch.stack(projection_matrices)
    data['projection_mat'][0] = torch.stack(projection_matrices)    
    
    pre_logic_timestamp = sample['timestamp']

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)


    # 6개 카메라 이미지 합치기
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


    # 여기까지가 출력 결과 임.
    result = result[0]["img_bbox"]
    vis_score_threshold = 0.3    
    pred_bboxes_3d = result["boxes_3d"]#[result["scores_3d"] > vis_score_threshold]
    
    # BEV에서 사용할 Color값 매칭하기
    color = []
    for id in result["labels_3d"].cpu().numpy().tolist():
        color.append(ID_COLOR_MAP_RGB[id])    
    
    # 추가 DNN 실행하기
    img_pil = Image.fromarray(images)
    image_tensor = transform_img(img_pil).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model_img(image_tensor)
    
    Debug_Enable = False
    # 디버그용
    if Debug_Enable:
        plt.figure(4)
        plt.clf()
        plt.subplot(2,1,1)
        plt.imshow(images)
        fov_boxes = []
        for idx, (box, label, score) in enumerate(zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores'])):
            if score > score_threshold and label in target_classes:
                x1, y1, x2, y2 = box
                fov_boxes.append(box)
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=2))
                if label == 3:
                    plt.text(x1, y1, f'{len(fov_boxes)-1},', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
                else:
                    plt.text(x1, y1, f'{len(fov_boxes)-1}:{coco_classes[label]},', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
                
            # 라벨 숫자 그리기
                # plt.text(x1, y1, f'{coco_classes[label]}', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
                
        
        plt.draw()  # 그래프를 그립니다.
        plt.pause(1)  # 1초 대기

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
    corners_3d = box3d_to_corners(pred_bboxes_3d)
    ratio_x = 1600/704
    ratio_y = 900/256   
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1
    )    
    imgfov_pts_2d_CAM = []
    for k, key in enumerate(camera_sensors):
        lidar2img_rt = copy.deepcopy(data["projection_mat"][0][k]).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        pts_2d = pts_4d @ lidar2img_rt.T

        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)        
        imgfov_pts_2d_CAM.append(imgfov_pts_2d)
    

    if Debug_Enable:
        # projection matrix계산하기        
        for k, key in enumerate(camera_sensors):
            if 'CAM_FRONT' == key:
                offset_x = 1600
                offset_y = 0
            elif 'CAM_FRONT_RIGHT' == key:
                offset_x = 3200
                offset_y = 0
            elif 'CAM_FRONT_LEFT' == key:
                offset_x = 0
                offset_y = 0
            elif 'CAM_BACK' == key:
                offset_x = 3200
                offset_y = 900
            elif 'CAM_BACK_LEFT' == key:
                offset_x = 1600
                offset_y = 900
            elif 'CAM_BACK_RIGHT' == key:                              
                offset_x = 4800
                offset_y = 900

            if 'BACK' in key:
                sign = -1
            else:
                sign = 1        
            # color=(255, 0, 0)
            img = image[k].copy()
            
            thickness = 2                
            h, w = 256,704
            fov_corners = []
            for i in range(num_bbox):
                corners = np.clip(imgfov_pts_2d_CAM[k][i], -1e4, 1e5).astype(np.int32)
                if result["scores_3d"][i] > vis_score_threshold:
                    fov_corners.append(corners)
                    for start, end in line_indices:
                        if (
                            (corners[start, 1] >= h or corners[start, 1] < 0)
                            or (corners[start, 0] >= w or corners[start, 0] < 0)
                        ) and (
                            (corners[end, 1] >= h or corners[end, 1] < 0)
                            or (corners[end, 0] >= w or corners[end, 0] < 0)
                        ):
                            continue
                        
                        tmp = color[i]
                        if sign > 0:
                            cv2.line(
                                images,
                                (int(corners[start, 0]*ratio_x+offset_x), int(corners[start, 1]*ratio_y+offset_y)),
                                (int(corners[end, 0]*ratio_x+offset_x), int(corners[end, 1]*ratio_y+offset_y)),
                                tmp,
                                thickness,
                                cv2.LINE_AA,
                            )       
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            font_color = (255, 255, 255)  # 텍스트 색상 (흰색)
                            line_type = 2                     
                            mid_point = (int(corners[start, 0]*ratio_x+offset_x), int(corners[start, 1]*ratio_y+offset_y))
                            text = str(i) #str(start)
                            cv2.putText(images, text, mid_point, font, font_scale, font_color, line_type)        
                        else:
                            cv2.line(
                                images,
                                (offset_x - int(corners[start, 0]*ratio_x), int(corners[start, 1]*ratio_y+offset_y)),
                                (offset_x - int(corners[end, 0]*ratio_x), int(corners[end, 1]*ratio_y+offset_y)),
                                tmp,
                                thickness,
                                cv2.LINE_AA,
                            )       
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            font_color = (255, 255, 255)  # 텍스트 색상 (흰색)
                            line_type = 2                     
                            mid_point = (offset_x - int(corners[start, 0]*ratio_x), int(corners[start, 1]*ratio_y+offset_y))
                            text = str(i) #str(start)
                            cv2.putText(images, text, mid_point, font, font_scale, font_color, line_type)                
        plt.subplot(2,1,2)
        plt.imshow(images)
        plt.draw()  # 그래프를 그립니다.
        plt.pause(1)  # 1초 대기

    
    # 기존에 있는 박스에 결과값 매칭하기
    fastrnn_boxes = []
    for idx, (box, label, score) in enumerate(zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores'])):
        if score > score_threshold and label in target_classes:
            x1, y1, x2, y2 = box
                
            box1 = [
                (x1, y1),  # 좌상단
                (x2, y1),  # 우상단
                (x1, y2),  # 좌하단
                (x2, y2)   # 우하단
            ]            
            fastrnn_boxes.append([box1,score,label,False])
    
    # 결과값이용해서 박스 그리기
    h, w = 256,704
    sparse4d_boxes = []
    for k, key in enumerate(camera_sensors):    
        if 'CAM_FRONT' == key:
            offset_x = 1600
            offset_y = 0
        elif 'CAM_FRONT_RIGHT' == key:
            offset_x = 3200
            offset_y = 0
        elif 'CAM_FRONT_LEFT' == key:
            offset_x = 0
            offset_y = 0
        elif 'CAM_BACK' == key:
            offset_x = 3200
            offset_y = 900
        elif 'CAM_BACK_LEFT' == key:
            offset_x = 1600
            offset_y = 900
        elif 'CAM_BACK_RIGHT' == key:                              
            offset_x = 4800
            offset_y = 900

        if 'BACK' in key:
            sign = -1
        else:
            sign = 1        
            
        for i in range(num_bbox):
            corners = np.clip(imgfov_pts_2d_CAM[k][i], -1e4, 1e5).astype(np.int32)
            if result["scores_3d"][i] > 0.1:
                if sign > 0:
                    box2 = [
                            (int(corners[2][0]*ratio_x+offset_x),int(corners[2][1]*ratio_y+offset_y)), # 좌상단
                            (int(corners[5][0]*ratio_x+offset_x),int(corners[5][1]*ratio_y+offset_y)), # 우상단
                            (int(corners[3][0]*ratio_x+offset_x),int(corners[3][1]*ratio_y+offset_y)), # 좌하단
                            (int(corners[4][0]*ratio_x+offset_x),int(corners[4][1]*ratio_y+offset_y)), # 우하단
                        ]                    
                else:
                    box2 = [
                            (offset_x - int(corners[2][0]*ratio_x),int(corners[2][1]*ratio_y+offset_y)), # 좌상단
                            (offset_x - int(corners[5][0]*ratio_x),int(corners[5][1]*ratio_y+offset_y)), # 우상단
                            (offset_x - int(corners[3][0]*ratio_x),int(corners[3][1]*ratio_y+offset_y)), # 좌하단
                            (offset_x - int(corners[4][0]*ratio_x),int(corners[4][1]*ratio_y+offset_y)), # 우하단
                        ]                      
                sparse4d_boxes.append([box2,score,i,False])                 
    
    
    for idx1, fastrnn_box in enumerate(fastrnn_boxes):
        box1,score,label,checked = fastrnn_box
        if checked == False:
            for idx2, sparse4d_box in enumerate(sparse4d_boxes):
                box2,score2,label2,checked2 = sparse4d_box
                if checked2 == False:
                    
                    # box1 = sparse4d_boxes[8][0]
                    # box2 = fastrnn_boxes[4][0]                   
                    
                    # 각 사각형의 좌상단 및 우하단 좌표 추출
                    (x1_min, y1_min), (x1_max, y1_max) = box1[0], box1[3]
                    (x2_min, y2_min), (x2_max, y2_max) = box2[0], box2[3]

                    # 교차 영역의 좌상단과 우하단 좌표 계산
                    inter_x_min = max(x1_min, x2_min)  # 교차 영역의 좌상단 X 좌표
                    inter_y_min = max(y1_min, y2_min)  # 교차 영역의 좌상단 Y 좌표
                    inter_x_max = min(x1_max, x2_max)  # 교차 영역의 우하단 X 좌표
                    inter_y_max = min(y1_max, y2_max)  # 교차 영역의 우하단 Y 좌표

                    # 교차 영역의 면적 계산
                    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

                    # 각 사각형의 면적 계산
                    area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
                    area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

                    # 합집합 영역의 면적 계산
                    union_area = area_box1 + area_box2 - inter_area

                    # IoU 계산
                    if union_area > 0:
                        iou = inter_area / union_area
                    else:
                        iou = 0

                    # 겹침 비율이 특정 임계값 이상일 때
                    threshold = 0.3  # 50% 겹침을 기준으로
                    if iou > threshold:
                        fastrnn_boxes[idx1][3] = True
                        sparse4d_boxes[idx2][3] = True
                        i = sparse4d_boxes[idx2][2]
                        result["scores_3d"][i] = torch.tensor(0.5)
                        break
                        #pred_bboxes_3d = torch.cat((pred_bboxes_3d, pred_bboxes_outofspec_3d[buf_index].unsqueeze(0)), dim=0)
                        #pred_bboxes_3d = result["boxes_3d"]#[result["scores_3d"] > vis_score_threshold]
                    
                  
    pred_bboxes_3d_wCorr = result["boxes_3d"][
        result["scores_3d"] > vis_score_threshold
    ]

    bev = plot_bev_orthogonal(
        pred_bboxes_3d_wCorr,
        bboxes_gt,
        bev_size=900 * 2,
        color=color,
    )    
    
    images = np.concatenate([images, bev], axis=1)
    
    plt.figure(1)
    plt.clf()  # 현재 화면을 지우고 갱신할 수 있도록 합니다.
    plt.imshow(images)
    plt.draw()  # 그래프를 그립니다.
    plt.title
    
    
    plt.pause(1)  # 1초 대기
    current_sample = nusc.get('sample', current_sample_token)    
    current_sample_token = current_sample['next']  # Move to the next sample

debug = 1    