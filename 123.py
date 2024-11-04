import os
import numpy as np
path = '/data/laiyan/datasets/ScanNet/sparse_depth_multi/scans_test/'
scenes = os.listdir(path)
scenes.sort()
for enu, scene in enumerate(scenes):
    scene_path = os.path.join(path, scene,'sensor_data')
    for data_path in os.listdir(scene_path):
        real_path = os.path.join(scene_path, data_path)
        data = np.load(real_path)['arr_0']
        if data.sum()==0 or np.isnan(data).sum()>0:
            print(enu, real_path)
            print(123)
