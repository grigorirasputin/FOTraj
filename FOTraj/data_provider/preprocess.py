import ast
import os
import random
import time

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, Manager

np.random.seed(42)

def height2lat(height):
    return height / 110.574


def width2lon(width):
    return width / 111.320 / 0.99974


def in_boundary(lon, lat, b):
    return b['min_lon'] < lon < b['max_lon'] and b['min_lat'] < lat < b['max_lat']


def cutting_trajs(traj, longest, shortest):
    cutted_trajs = []
    while len(traj) > longest:
        random_length = np.random.randint(shortest, longest)
        cutted_traj = traj[:random_length]
        cutted_trajs.append(cutted_traj)
        traj = traj[random_length:]
    return cutted_trajs


def chengdu_process(data_dir, data_name, noise=False):
    lat_size, lon_size = height2lat(grid_height), width2lon(grid_width)

    lat_grid_num = int((chengdu_boundary['max_lat'] - chengdu_boundary['min_lat']) / lat_size) + 1
    lon_grid_num = int((chengdu_boundary['max_lon'] - chengdu_boundary['min_lon']) / lon_size) + 1

    trajectories = pd.read_csv("{}/{}.txt".format(data_dir, data_name), header=None)
    trajectories.columns = ['id', 'lat', 'lon', 'state', 'timestamp']
    trajectories = trajectories[trajectories['state'] == 1]
    trajectories = trajectories.sort_values(by=['id', 'timestamp'])
    trajectories['timestamp'] = pd.to_datetime(trajectories['timestamp'], format='%Y/%m/%d %H:%M:%S')
    trajectories['timestamp'] = trajectories['timestamp'].apply(lambda x: x.timestamp())
    trajectories['timestamp_gap'] = trajectories.groupby('id')['timestamp'].diff()

    trajs = []
    traj_seq = []
    valid = True
    noise_std = 0.0005
    shortest, longest = 30, 120

    for i in tqdm(range(1, len(trajectories))):
        point = trajectories.iloc[i]
        pre_point = trajectories.iloc[i - 1]

        if point.id == pre_point.id and point['timestamp_gap'] <= 180:
            lat, lon = point.lat, point.lon
            if noise and len(traj_seq) >= 2:
                if np.random.rand() < 0.2:
                    noisy_lat = lat + np.random.normal(0, noise_std)
                    noisy_lon = lon + np.random.normal(0, noise_std)
                    lat, lon = noisy_lat, noisy_lon

            if in_boundary(lon, lat, chengdu_boundary):
                lat_grid = int((lat - chengdu_boundary['min_lat']) / lat_size)
                lon_grid = int((lon - chengdu_boundary['min_lon']) / lon_size)
                grid_value = int(lat_grid * lon_grid_num + lon_grid)
                traj_seq.append([grid_value, point.timestamp])
            else:
                valid = False
        else:
            if valid:
                if shortest <= len(traj_seq) <= longest:
                    trajs.append(traj_seq)
                elif len(traj_seq) > longest:
                    trajs += cutting_trajs(traj_seq, longest, shortest)
            traj_seq = []
            valid = True

    print("Valid trajectory num:", len(trajs))
    if noise:
        data_name = f"{data_name}_noise"
    output_file = f"{data_dir}/processed_{data_name}.csv"
    with open(output_file, 'w') as processed_file:
        for traj in tqdm(trajs):
            processed_file.write("[")
            for grid_id, timestamp in traj[:-1]:
                processed_file.write("(%s, %s), " % (str(int(grid_id)), str(int(timestamp))))
            last_grid, last_timestamp = traj[-1]
            processed_file.write("(%s, %s)]\n" % (str(int(last_grid)), str(int(last_timestamp))))

    print(f"Processed trajectories saved to {output_file}")


def chengdu_merge(noise=False):
    data_dir = '../../datasets/chengdu'
    file_list = ['20140803_train', '20140804_train', '20140805_train', '20140806_train', '20140808_train', '20140809_train', '20140810_train', '20140811_train', '20140812_train', '20140814_train']
    if noise:
        file_list = [f"{file_name}_noise" for file_name in file_list]
        output_file = 'processed_chengdu_noise.csv'
    else:
        output_file = 'processed_chengdu.csv'
    with open(f"{data_dir}/{output_file}", 'w') as merged_file:
        for file_name in file_list:
            with open(f"{data_dir}/processed_{file_name}.csv", 'r') as file:
                for line in file:
                    merged_file.write(line)


def process_sd(data_dir, data_name, noise=False):
    min_sd_traj_num = 25
    test_traj_num = 5
    if noise:
        file_path = f"{data_dir}/processed_{data_name}_noise.csv"
    else:
        file_path = f"{data_dir}/processed_{data_name}.csv"
    with open(file_path, 'r') as file:
        lines = file.readlines()

    sd_cnt = defaultdict(list)
    for eachline in tqdm(lines):
        traj = eval(eachline.strip())
        if not traj:
            continue
        s, d = traj[0][0], traj[-1][0]
        converted_traj = [(grid, datetime.fromtimestamp(timestamp).timetuple()[:6]) for grid, timestamp in traj]
        sd_cnt[(s, d)].append(converted_traj)
    if noise:
        data_name = f"{data_name}_noise"
    train_file_path = f"{data_dir}/processed_{data_name}_train.csv"
    test_file_path = f"{data_dir}/processed_{data_name}_test.csv"
    with open(train_file_path, 'w') as train_file, open(test_file_path, 'w') as test_file:
        for sd_pair, trajs in sd_cnt.items():
            if len(trajs) >= min_sd_traj_num:
                train_trajs = trajs[:-test_traj_num]
                test_trajs = trajs[-test_traj_num:]

                for traj in train_trajs:
                    train_file.write(f"{traj}\n")

                for traj in test_trajs:
                    test_file.write(f"{traj}\n")


def count_sd_trajs(file_path):
    sd_count = defaultdict(int)
    with open(file_path, 'r') as file:
        for _, line in tqdm(enumerate(file)):
            traj = ast.literal_eval(line.strip())
            if traj:
                s, d = traj[0][0], traj[-1][0]
                sd_count[(s, d)] += 1
    print(len(sd_count))
    cnt = 0
    for sd, count in sd_count.items():
        cnt += count
    print(cnt)


if __name__ == '__main__':
    chengdu_data_dir = '../../datasets/chengdu'
    chengdu_data_name = "chengdu"
    grid_height, grid_width = 0.1, 0.1
    chengdu_boundary = {'min_lat': 30.50, 'max_lat': 30.80, 'min_lon': 103.9, 'max_lon': 104.2}
    file_list = ['20140803_train', '20140804_train', '20140805_train', '20140806_train', '20140808_train',
                 '20140809_train', '20140810_train', '20140811_train', '20140812_train', '20140814_train']
    for i in range(10):
        chengdu_process(chengdu_data_dir, file_list[i], noise=False)
    chengdu_merge()
    process_sd(chengdu_data_dir, chengdu_data_name, noise=False)
    count_sd_trajs(f"{chengdu_data_dir}/processed_{chengdu_data_name}_test.csv")
