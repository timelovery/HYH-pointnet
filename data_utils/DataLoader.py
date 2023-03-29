import os
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, Source_root='trainval_fullarea', Target_root='trainval_fullarea', num_point=4096, test_area=5,
                 block_size=1.0,
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
        # target
        Target_sample_prob = Target_num_point_all / np.sum(Target_num_point_all)
        Target_sum_iter = int(np.sum(Target_num_point_all) * sample_rate / num_point)
        Target_area_idxs = []
        for index in range(len(Target_areas)):
            Target_area_idxs.extend([index] * int(round(Target_sample_prob[index] * Target_sum_iter)))

        # 实现数据量对齐
        if len(Source_area_idxs) > len(Target_area_idxs):
            for i in range(len(Target_area_idxs), len(Source_area_idxs)):
                Target_area_idxs.append(random.choice(Target_area_idxs))
        else:
            for i in range(len(Source_area_idxs), len(Target_area_idxs)):
                Source_area_idxs.append(random.choice(Source_area_idxs))
        self.Source_area_idxs = np.array(Source_area_idxs)
        self.Target_area_idxs = np.array(Target_area_idxs)
        print("Totally {} samples in {} set.".format(len(self.Source_area_idxs), "Source_area"))
        print("Totally {} samples in {} set.".format(len(self.Target_area_idxs), "Target_area"))

    def __getitem__(self, idx):
        Source_area_idx = self.Source_area_idxs[idx]
        Target_area_idx = self.Target_area_idxs[idx]
        Source_points = self.Source_points[Source_area_idx]
        Source_labels = self.Source_labels[Source_area_idx]
        Target_points = self.Target_points[Target_area_idx]
        Source_N_points = Source_points.shape[0]
        Target_N_points = Target_points.shape[0]

        Source_selected_point_idx, Source_center = self.point_idxs(Source_points, Source_N_points)  # 选择源场景数据
        Target_selected_point_idx, Target_center = self.point_idxs(Target_points, Target_N_points)  # 选择目标场景数据
        Source_current_points = self.normalize(Source_points, Source_selected_point_idx,
                                               Source_area_idx, Source_center, self.Source_coord_max)
        Target_current_points = self.normalize(Target_points, Target_selected_point_idx,
                                               Target_area_idx, Target_center, self.Target_coord_max)

        Source_current_labels = Source_labels[Source_selected_point_idx]
        if self.transform is not None:
            Source_current_points, Source_current_labels = self.transform(Source_current_points, Source_current_labels)
        return Source_current_points, Source_current_labels, Target_current_points

    def point_idxs(self, points, N_points):
        while True:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                        points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
        return selected_point_idxs, center

    def normalize(self, points, selected_point_idxs, room_idx, center, coord_max):
        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        return current_points
