import os
import numpy as np
from torch.utils.data.dataset import T_co

from tqdm import tqdm
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, Source_root='trainval_fullarea', Target_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0,
                 sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        Source_areas = sorted(os.listdir(Source_root))
        Source_areas = [area for area in Source_areas if '.npy' in area]  # 所有的文件的输入格式应该为npy
        Target_areas = sorted(os.listdir(Target_root))
        Target_areas = [area for area in Target_areas if '.npy' in area]  # 所有的文件的输入格式应该为npy

        self.Source_points, self.Source_labels = [], []
        self.Source_coord_min, self.Source_coord_max = [], []
        self.Target_points = []  # Target 数据没有 labels
        self.Target_coord_min, self.Target_coord_max = [], []

        Source_num_point_all = []
        Target_num_point_all = []
        Source_labelweights = np.zeros(13)

        for Source_area_name in tqdm(Source_areas, total=len(Source_areas)):
            area_path = os.path.join(Source_root, Source_area_name)
            area_data = np.load(area_path)  # xyzrgbl, N*7
            points, labels = area_data[:, 0:6], area_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))
            Source_labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.Source_points.append(points), self.Source_labels.append(labels)
            self.Source_coord_min.append(coord_min), self.Source_coord_max.append(coord_max)
            Source_num_point_all.append(labels.size)
        for Target_area_name in tqdm(Target_areas, total=len(Target_areas)):
            area_path = os.path.join(Target_root, Target_area_name)
            area_data = np.load(area_path)  # xyzrgb, N*6
            points = area_data  # xyzrgb, N*6;
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.Target_points.append(points),
            self.Target_coord_min.append(coord_min), self.Target_coord_max.append(coord_max)
            Target_num_point_all.append(points.size[0])

        Source_labelweights = Source_labelweights.astype(np.float32)
        Source_labelweights = Source_labelweights / np.sum(Source_labelweights)
        self.Source_labelweights = np.power(np.amax(Source_labelweights) / Source_labelweights, 1 / 3.0)

        # source
        Source_sample_prob = Source_num_point_all / np.sum(Source_num_point_all)
        Source_num_iter = int(np.sum(Source_num_point_all) * sample_rate / num_point)  # 创建迭代器
        Source_area_idxs = []
        for index in range(len(Source_areas)):
            Source_area_idxs.extend([index] * int(round(Source_sample_prob[index] * Source_num_iter)))
        self.Source_area_idx = np.array(Source_area_idxs)


    def __getitem__(self, index) -> T_co:
        pass
