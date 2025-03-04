import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.CLGrasp import CLGraspNet
from lib.align import estimateSimilarityTransform
from lib.utils import get_bbox

def predict_pose(image_path, model, mean_shapes, n_cat=6, nv_prior=1024, num_structure_points=256,
                 img_size=192, n_pts=1024, cam_params=None, device='cuda'):
    # Ensure CUDA is available and select the device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load the model
    estimator = CLGraspNet(n_cat, nv_prior, num_structure_points=num_structure_points)
    estimator.to(device)
    estimator = nn.DataParallel(estimator)
    estimator.load_state_dict(torch.load(model))
    estimator.eval()

    # Camera parameters
    if cam_params is None:
        cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5  # Default values
    else:
        cam_fx, cam_fy, cam_cx, cam_cy = cam_params

    # Image processing
    raw_rgb = cv2.imread(image_path)[:, :, :3]
    raw_rgb = raw_rgb[:, :, ::-1]  # Convert BGR to RGB

    # Generate masks and bounding boxes (dummy implementation for example)
    # Replace with actual detection results
    num_insts = 1
    mrcnn_result = {
        'class_ids': [1],  # Dummy class ID
        'rois': [[0, 0, raw_rgb.shape[1], raw_rgb.shape[0]]],  # Dummy bounding box covering the whole image
        'masks': np.ones((raw_rgb.shape[0], raw_rgb.shape[1], 1), dtype=np.uint8),  # Dummy mask
        'scores': [1.0]  # Dummy score
    }

    # Prepare frame data
    xmap = np.array([[i for i in range(raw_rgb.shape[1])] for j in range(raw_rgb.shape[0])])
    ymap = np.array([[j for i in range(raw_rgb.shape[1])] for j in range(raw_rgb.shape[0])])
    norm_color = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    f_rgb, f_choose, f_catId, f_prior = [], [], [], []
    valid_inst = []

    for i in range(num_insts):
        cat_id = mrcnn_result['class_ids'][i] - 1
        prior = mean_shapes[cat_id]
        rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][i])
        mask = mrcnn_result['masks'][:, :, i]

        # Use a simple bounding box for demonstration purposes
        mask = np.ones_like(raw_rgb[:, :, 0])
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) < 32:
            continue

        valid_inst.append(i)

        if len(choose) > n_pts:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:n_pts] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, n_pts - len(choose)), 'wrap')

        rgb = raw_rgb[rmin:rmax, cmin:cmax, :]
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        rgb = norm_color(rgb)
        crop_w = rmax - rmin
        ratio = img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * img_size + np.floor(col_idx * ratio)).astype(np.int64)

        f_rgb.append(rgb)
        f_choose.append(choose)
        f_catId.append(cat_id)
        f_prior.append(prior)

    # if len(valid_inst):
    #     f_rgb = torch.stack(f_rgb, dim=0).to(device)
    #     f_choose = torch.cuda.LongTensor(f_choose).to(device)
    #     f_catId = torch.cuda.LongTensor(f_catId).to(device)
    #     f_prior = torch.cuda.FloatTensor(f_prior).to(device)

        # Inference
        with torch.no_grad():
            structure_points, assign_mat, deltas = estimator(None, f_rgb, f_choose, f_catId, f_prior)

        inst_shape = f_prior + deltas
        assign_mat = F.softmax(assign_mat, dim=2)
        f_coords = torch.bmm(assign_mat, inst_shape)  # bs x n_pts x 3

        f_coords = f_coords.cpu().numpy()
        f_choose = f_choose.cpu().numpy()
        f_insts = inst_shape.cpu().numpy()

        results = []
        for i in range(len(valid_inst)):
            inst_idx = valid_inst[i]
            choose = f_choose[i]
            _, choose = np.unique(choose, return_index=True)
            nocs_coords = f_coords[i, choose, :]
            f_size = 2 * np.amax(np.abs(f_insts[i]), axis=0)
            _, _, _, pred_sRT = estimateSimilarityTransform(nocs_coords, np.zeros_like(nocs_coords))  # No depth points
            if pred_sRT is None:
                pred_sRT = np.identity(4, dtype=float)
            results.append({
                'pred_class_id': cat_id,
                'pred_bboxes': mrcnn_result['rois'][i],
                'pred_scores': mrcnn_result['scores'][i],
                'pred_RTs': pred_sRT,
                'pred_scales': f_size
            })

        return results

# Example usage
if __name__ == "__main__":
    image_path = 'data/Real_test/test/scene_1/0000_color.png'
    model_path = 'realbestmodel_1.pth'
    mean_shapes = np.load('assets1/mean_points_emb.npy')
    cam_params = (577.5, 577.5, 319.5, 239.5)  # Example parameters
    results = predict_pose(image_path, model_path, mean_shapes, cam_params=cam_params)
    print(results)
