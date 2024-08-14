import warnings
warnings.filterwarnings('ignore')
import os
# cwd = os.getcwd()
# if cwd.endswith("tutorial"):
#     os.chdir("../")

from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import torch
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv.parallel import scatter
from mmcv.cnn.utils.flops_counter import add_flops_counting_methods

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets.utils import draw_lidar_bbox3d


gpu_id = 0
config = 'sparse4dv3_temporal_r50_1x8_bs6_256x704'
checkpoint = 'ckpt/sparse4dv3_r50.pth'
cfg = Config.fromfile(f"projects/configs/{config}.py")
# cfg.model["use_deformable_func"] = False
# cfg.model["head"]["deformable_model"]["use_deformable_func"] = False
img_norm_mean = np.array(cfg.img_norm_cfg["mean"])
img_norm_std = np.array(cfg.img_norm_cfg["std"])


dataset = build_dataset(cfg.data.val)
dataloader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=0,
    dist=False,
    shuffle=False,
)
data_iter = dataloader.__iter__()
data = next(data_iter)
data = scatter(data, [gpu_id])[0]

## build model

model = build_detector(cfg.model)
model = model.cuda(gpu_id)
_ = model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)
model = model.eval()
assert model.use_deformable_func, "Please compile deformable aggregation first !!!"


raw_imgs = data["img"][0].permute(0, 2, 3, 1).cpu().numpy()
raw_imgs = raw_imgs * img_norm_std + img_norm_mean

plt.figure(figsize=(20, 10))
anchor = model.head.decoder.decode_box(model.head.instance_bank.anchor)
plt.imshow(
    draw_lidar_bbox3d(anchor[::5], raw_imgs, data["projection_mat"][0])
)


feature_maps = model.extract_feat(data["img"], metas=data)
if model.use_deformable_func:
    print("Feature map format for deformable aggregation: (flatten feature maps, feature size, start inidces).")
    print("Deformable aggregation can accept feature maps from multiple views with different resolutions as input !!!")
    
    print(f"N_camera = {feature_maps[1].shape[0]}")
    print(f"N_scale= {feature_maps[1].shape[1]}\n")
    print(
        "Multi-camera multi-scale feature maps, "
        "flatten to shape [batch_size, N_feature, channel]: "
        f"{feature_maps[0].shape}\n"
    )
    print("Deformable aggregation")
    print("Size of all feature maps, [N_camera, N_scale, 2]: ")
    pprint(feature_maps[1].tolist())
    
    print("\nStart indices of all feature maps, [N_camera, N_scale]:")
    pprint(feature_maps[2].tolist())

model.head.instance_bank.reset()
model_outs = model.head(feature_maps, data)
print("Output Keys:", model_outs.keys())
print("Number of transformer layers: len(model_outs['classification']).")
print(
    f"Shape of classification is [batch_size, N_instance, N_class] = "
    f"{model_outs['classification'][-1].shape}"
)
print(
    f"Shape of prediction is [batch_size, N_instance, box_dim] = "
    f"{model_outs['prediction'][-1].shape}."
)
if "quality" in model_outs:
    print(
        f"Shape of quality is [batch_size, N_instance, (centerness, yawness)] = "
        f"{model_outs['quality'][-1].shape}."
    )
if "instance_id" in model_outs:
    print(
        f"Shape of instance_id is [batch_size, N_instance] = "
        f"{model_outs['instance_id'].shape}."
    )

plt.figure(figsize=(20, 10))
pred_bbox3d = model.head.decoder.decode_box(model_outs["prediction"][-1][0])
confidence = model_outs["classification"][-1].max(dim=-1)[0][0].sigmoid()
mask = confidence > 0.35
num_det = mask.sum()
img = draw_lidar_bbox3d(
    torch.cat([pred_bbox3d[mask], anchor[mask]]),
    raw_imgs, data["projection_mat"][0],
    color=[(0, 255, 0)] * num_det + [(255, 0, 0)] * num_det
)
plt.imshow(img)  # The green boxes denotes model detections, and red ones are the corresponding anchors.
plt.draw()  # 그래프를 그립니다.
plt.pause(1)  # 1초 대기
end = 1