import argparse
import os
import shutil
import pickle as pkl
import json
import cv2
import numpy as np
import PIL
import glob
# import mmcv

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='', help='path of input image')
parser.add_argument('--ann_path', type=str, default='', help='path of annotation file')
parser.add_argument('--output_path', type=str, default='', help='output dir')

opt = parser.parse_args()


def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def draw_bboxes(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
    # draw pillars in minor darker color
    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)

    return img


if __name__ == '__main__':

    img_path = opt.img_path
    ann_path = opt.ann_path
    output_path = opt.output_path
    class_name = 'bottle'
    img_paths = sorted(glob.glob('./test_set/{}/**/**/images/*.jpg'.format(class_name)))
    ann_paths = sorted(glob.glob('./test_set/pkl_annotations/{}/*.pkl'.format(class_name)))

    for ann_path in ann_paths:
        anns = pkl.load(open(ann_path, 'rb'))
        for ann in anns['annotations']:
            cls_n, seq_idx, obj_idx, frame_idx = ann['name'].split('/')
            base_path = 'test_set/{}/{}/{}'.format(cls_n, seq_idx, obj_idx)
            img_path = os.path.join(base_path, 'images/{}.jpg'.format(int(frame_idx)))
            if int(frame_idx) == 0:
                shutil.rmtree(os.path.join(base_path, 'vis_pose/'))
            os.makedirs(os.path.join(base_path, 'vis_pose/'), exist_ok=True)
            output_path = os.path.join(base_path, 'vis_pose/{:04d}.jpg'.format(int(frame_idx)))
            meta_path = os.path.join(base_path, 'metadata')

            # load image and meta file
            img = cv2.imread(img_path)
            meta = json.load(open(meta_path, 'rb'))

            # load 6d pose annotations
            scale = ann['size']
            rot = ann['rotation']
            trans = ann['translation']
            RTs = np.eye(4)
            RTs[:3, :3] = rot
            RTs[:3, 3] = trans
            K = np.array(meta['K']).reshape(3, 3).T
            noc_cube = get_3d_bbox(scale, 0)
            bbox_3d = transform_coordinates_3d(noc_cube, RTs)
            projected_bbox = calculate_2d_projections(bbox_3d, K)
            img = draw_bboxes(img, projected_bbox, (0, 255, 0))

            cv2.imwrite(output_path, img)
        # mmcv.video.frames2video(os.path.join(base_path, 'vis_pose'),
        #                         os.path.join(base_path, 'vis.mp4'), filename_tmpl='{:04d}.jpg', fps=10, end=10)




























