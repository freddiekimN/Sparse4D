
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
try:
    while True:
        data = next(data_iter)
        gpu_id = 0
        data = scatter(data, [gpu_id])[0]

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
            
        # image = data['img'].data[0]   
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
        # cv2.imshow(f'Image', images.astype(np.uint8))
        # cv2.waitKey(0)
        # 
        
        plt.clf()  # 현재 화면을 지우고 갱신할 수 있도록 합니다.
        plt.imshow(images)
        plt.draw()  # 그래프를 그립니다.
        plt.pause(1)  # 1초 대기
        # cv2.destroyAllWindows()

except StopIteration:
    # 이터레이터의 끝에 도달한 경우
    print("Iterator has been exhausted.")

