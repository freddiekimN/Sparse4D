import numpy as np
from sklearn.cluster import KMeans
import mmcv
import sys
import os

# 현재 스크립트 파일의 디렉터리 경로
current_dir = os.path.dirname(__file__)
# 최상위 프로젝트 디렉터리 경로
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 최상위 프로젝트 디렉터리를 Python 경로에 추가
sys.path.insert(0, project_root)

from projects.mmdet3d_plugin.core.box3d import *


def get_kmeans_anchor(
    ann_file,
    num_anchor=900,
    detection_range=55,
    output_file_name="nuscenes_kmeans900.npy",
    verbose=False,
):
    data = mmcv.load(ann_file, file_format="pkl")
    gt_boxes = np.concatenate([x["gt_boxes"] for x in data["infos"]], axis=0)
    distance = np.linalg.norm(gt_boxes[:, :3], axis=-1, ord=2)
    mask = distance <= detection_range
    gt_boxes = gt_boxes[mask]
    clf = KMeans(n_clusters=num_anchor, verbose=verbose)
    print("===========Starting kmeans, please wait.===========")
    clf.fit(gt_boxes[:, [X, Y, Z]])
    anchor = np.zeros((num_anchor, 11))
    anchor[:, [X, Y, Z]] = clf.cluster_centers_
    anchor[:, [W, L, H]] = np.log(gt_boxes[:, [W, L, H]].mean(axis=0))
    anchor[:, COS_YAW] = 1
    np.save(output_file_name, anchor)
    print(f"===========Done! Save results to {output_file_name}.===========")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="anchor kmeans")
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--num_anchor", type=int, default=900)
    parser.add_argument("--detection_range", type=float, default=55)
    parser.add_argument(
        "--output_file_name", type=str, default="_nuscenes_kmeans900.npy"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    get_kmeans_anchor(
        args.ann_file,
        args.num_anchor,
        args.detection_range,
        args.output_file_name,
        args.verbose,
    )
