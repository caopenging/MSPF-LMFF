import numpy as np
from tools.visual_points import visual_points

mean_shapes = np.load('assets1/mean_points_emb.npy')

point = mean_shapes[2]

visual_points(points4=point)