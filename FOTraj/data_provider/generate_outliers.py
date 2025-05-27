import ast
import os
from datetime import datetime, timedelta
import random

import numpy as np

np.random.seed(42)


def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            traj = ast.literal_eval(line.strip())
            data.append(traj)
    return data


def inject_outliers(data, ratio, detour_level, switch_level, time_level, loop_level, point_prob, otype):
    traj_num = len(data)
    selected_idx = np.random.randint(0, traj_num, size=int(traj_num * ratio))
    outliers = []

    if otype == 'detour':
        for idx in selected_idx:
            traj = data[idx]
            detour_traj = detour_batch([traj], detour_level, point_prob)
            outliers.append(detour_traj[0])
    elif otype == 'switch':
        for idx in selected_idx:
            traj = data[idx]
            switched_traj = switch_batch([traj], level=3, prob=switch_level, vary=False)
            outliers.append(switched_traj[0])
    elif otype == 'time':
        for idx in selected_idx:
            traj = data[idx]
            time_traj = time_batch([traj], time_level)
            outliers.append(time_traj[0])
    elif otype == 'loop':
        for idx in selected_idx:
            traj = data[idx]
            loop_traj = loop_batch([traj], loop_level)
            outliers.append(loop_traj[0])
    elif otype == 'all':
        selected_idx = np.random.randint(0, traj_num, size=int(4 * traj_num * ratio))
        for idx in selected_idx:
            traj = data[idx]
            if np.random.random() < 0.25:
                detour_traj = detour_batch([traj], detour_level, point_prob)
                outliers.append(detour_traj[0])
            elif 0.25 <= np.random.random() < 0.5:
                switched_traj = switch_batch([traj], switch_level, point_prob, vary=False)
                outliers.append(switched_traj[0])
            elif 0.5 <= np.random.random() < 0.75:
                time_traj = time_batch([traj], time_level)
                outliers.append(time_traj[0])
            else:
                loop_traj = loop_batch([traj], loop_level)
                outliers.append(loop_traj[0])
    else:
        raise ValueError("otype must be 'detour' or 'switch' or 'time' or 'all'.")

    for i, idx in enumerate(selected_idx):
        data[idx] = outliers[i]

    return data, selected_idx


def detour_point(point, level, offset=None):
    if dataset == 'chengdu':
        map_size = (111, 112)
    x, y = int(point // map_size[1]), int(point % map_size[1])
    if offset is None:
        offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
        x_offset, y_offset = offset[np.random.randint(0, len(offset))]
    else:
        x_offset, y_offset = offset
    if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:
        x += x_offset * level
        y += y_offset * level
    return int(x * map_size[1] + y)


def detour_batch(batch_x, level, prob):
    noisy_batch_x = []
    for traj in batch_x:
        noisy_batch_x.append([traj[0]] + [(detour_point(p[0], level), p[1])
                                          if not p == 0 and np.random.random() < prob else p
                                          for p in traj[1:-1]] + [traj[-1]])
    return noisy_batch_x


def switch_batch(batch_x, level, prob, vary=False):
    if dataset == 'chengdu':
        map_size = (111, 112)
    noisy_batch_x = []
    if vary:
        level += np.random.randint(-2, 3)
        if np.random.random() > 0.5:
            prob += 0.2 * np.random.random()
        else:
            prob -= 0.2 * np.random.random()

    for traj in batch_x:
        num_points_to_process = min(len(traj), 64)
        anomaly_len = int((num_points_to_process - 2) * prob)
        anomaly_st_loc = np.random.randint(1, num_points_to_process - anomaly_len - 2)
        anomaly_ed_loc = anomaly_st_loc + anomaly_len
        offset = [
            (traj[anomaly_st_loc][0] // map_size[1]) - (traj[anomaly_ed_loc][0] // map_size[1]),
            (traj[anomaly_st_loc][0] % map_size[1]) - (traj[anomaly_ed_loc][0] % map_size[1])
        ]

        if offset[0] == 0:
            div0 = 1
        else:
            div0 = abs(offset[0])

        if offset[1] == 0:
            div1 = 1
        else:
            div1 = abs(offset[1])

        if np.random.random() < 0.5:
            offset = [-offset[0] / div0, offset[1] / div1]
        else:
            offset = [offset[0] / div0, -offset[1] / div1]

        noisy_batch_x.append(
            traj[:anomaly_st_loc] +
            [(detour_point(p[0], level, offset), p[1]) for p in traj[anomaly_st_loc:anomaly_ed_loc]] +
            traj[anomaly_ed_loc:]
        )
    return noisy_batch_x


def time_batch(batch_x, level, prob=0.1):
    noisy_batch_x = []
    for traj in batch_x:
        num_points_to_process = min(len(traj), 64)
        anomaly_start_idx_list = sorted(np.random.choice(
            num_points_to_process,
            size=int(prob * num_points_to_process),
            replace=False
        ))
        noisy_traj = []
        for i, (point_id, timestamp) in enumerate(traj):
            num_greater = sum(i > idx for idx in anomaly_start_idx_list)
            if num_greater > 0:
                dt = datetime(*timestamp) + timedelta(minutes=int(level * num_greater))
                new_timestamp = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
                noisy_traj.append((point_id, new_timestamp))
            else:
                noisy_traj.append((point_id, timestamp))
        noisy_batch_x.append(noisy_traj)
    return noisy_batch_x


def loop_batch(batch_x, level, prob=0.1):
    if dataset == 'chengdu':
        map_size = (111, 112)
    noisy_batch_x = []
    for traj in batch_x:
        num_points_to_process = min(len(traj), 64)
        anomaly_st_loc_list = sorted(np.random.choice(
            range(1, num_points_to_process - 1),
            size=int(prob * num_points_to_process),
            replace=False
        ))
        noisy_traj = []
        cnt = 0
        for i, (point_id, timestamp) in enumerate(traj):
            if cnt < len(anomaly_st_loc_list) and i == anomaly_st_loc_list[cnt]:
                anomaly_ed_loc = i + 1
                off = [
                    (traj[i][0] // map_size[1]) - (traj[anomaly_ed_loc][0] // map_size[1]),
                    (traj[i][0] % map_size[1]) - (traj[anomaly_ed_loc][0] % map_size[1])
                ]
                off = [np.clip(off[0], -1, 1), np.clip(off[1], -1, 1)]
                offlist = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
                while True:
                    offset = random.choice(offlist)
                    if offset != off:
                        break
                noisy_traj.append((detour_point(point_id, level, offset), timestamp))
                cnt += 1
            else:
                noisy_traj.append((point_id, timestamp))
        noisy_batch_x.append(noisy_traj)
    return noisy_batch_x


if __name__ == "__main__":
    dataset = 'chengdu'
    otype = ['detour', 'switch', 'time', 'loop']
    subset = ['train', 'test']
    detour_level = 3
    point_prob = 0.3
    switch_level = 0.3
    time_level = 15
    loop_level = 1
    for outlier in otype:
        if outlier == 'detour':
            folder = f'detour_{detour_level}_prob_{point_prob}'
        elif outlier == 'switch':
            folder = f'switch_{switch_level}'
        elif outlier == 'time':
            folder = f'time_{time_level}'
        elif outlier == 'loop':
            folder = f'loop_{loop_level}'
        elif outlier == 'all':
            folder = f'all'
            subset = ['train']
        elif outlier == 'noise':
            pass
        else:
            raise ValueError("outlier must be 'detour' or 'switch' or 'time' or 'loop' or 'all'.")

        if outlier != 'noise':
            for sub in subset:
                input_file = f'../../datasets/{dataset}/processed_{dataset}_{sub}.csv'
                output_folder = f'../../datasets/{dataset}/{outlier}/{folder}'
                output_file = f'{output_folder}/{dataset}_{sub}_{outlier}.csv'
                output_idx_file = f'{output_folder}/{dataset}_{sub}_{outlier}_idx.csv'
                os.makedirs(output_folder, exist_ok=True)

                data = load_dataset(input_file)
                outlier_data, outlier_indices = inject_outliers(data, ratio=0.05, detour_level=detour_level,
                                                                switch_level=switch_level, time_level=time_level,
                                                                loop_level=loop_level, point_prob=point_prob, otype=outlier)

                with open(output_file, 'w') as f:
                    for traj in outlier_data:
                        f.write(f"{traj}\n")
                print(f"Saved {len(outlier_data)} trajectories to {output_file}")

                with open(output_idx_file, 'w') as f:
                    for idx in outlier_indices:
                        f.write(f"{idx}\n")
                print(f"Saved {len(outlier_indices)} outlier indices to {output_idx_file}")
        else:
            pass

