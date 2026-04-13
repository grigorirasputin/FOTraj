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


import ast
import time
from collections import defaultdict
from tqdm import tqdm

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

# --- OPTIMIZATION: Wrapper for Multiprocessing ---
def process_file_parallel(args):
    data_dir, file_name, noise = args
    chengdu_process(data_dir, file_name, noise=noise)

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
    

    # ... (Keep the pandas dataframe setup above this) ...
    trajectories['timestamp_gap'] = trajectories.groupby('id')['timestamp'].diff()

    # --- OPTIMIZATION: Extract to raw NumPy arrays for lightning-fast loop access ---
    ids = trajectories['id'].values
    lats = trajectories['lat'].values
    lons = trajectories['lon'].values
    timestamps = trajectories['timestamp'].values
    gaps = trajectories['timestamp_gap'].values

    trajs = []
    traj_seq = []
    valid = True
    noise_std = 0.0005
    shortest, longest = 30, 120

    # Iterate using standard indices on the NumPy arrays
    for i in tqdm(range(1, len(trajectories)), desc=f"Processing {data_name}"):
        curr_id = ids[i]
        pre_id = ids[i - 1]
        gap = gaps[i]

        if curr_id == pre_id and gap <= 180:
            lat = lats[i]
            lon = lons[i]
            
            if noise and len(traj_seq) >= 2:
                if np.random.rand() < 0.2:
                    # Apply noise
                    lat += np.random.normal(0, noise_std)
                    lon += np.random.normal(0, noise_std)

            if in_boundary(lon, lat, chengdu_boundary):
                lat_grid = int((lat - chengdu_boundary['min_lat']) / lat_size)
                lon_grid = int((lon - chengdu_boundary['min_lon']) / lon_size)
                grid_value = int(lat_grid * lon_grid_num + lon_grid)
                traj_seq.append([grid_value, timestamps[i]])
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

    print(f"Valid trajectory num for {data_name}:", len(trajs))
    # ... (Keep the file writing logic below this exactly the same) ...
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
    data_dir = r'datasets/chengdu'
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


def process_sd_ultra_low_ram(data_dir, data_name, noise=False):
    min_sd_traj_num = 25
    test_traj_num = 5
    
    if noise:
        file_path = f"{data_dir}/processed_{data_name}_noise.csv"
        data_name = f"{data_name}_noise"
    else:
        file_path = f"{data_dir}/processed_{data_name}.csv"

    print(f"Loading merged file: {file_path}")
    
    # ---------------------------------------------------------
    # PASS 1: Indexing (Ultra-low RAM, stores only line integers)
    # ---------------------------------------------------------
    sd_lines = defaultdict(list)
    
    with open(file_path, 'r') as file:
        for line_idx, line in enumerate(tqdm(file, desc="Pass 1: Indexing SD Pairs")):
            line = line.strip()
            if not line: continue
            
            # Fast parsing just to find Source and Destination
            traj = ast.literal_eval(line)
            if traj:
                s, d = traj[0][0], traj[-1][0]
                sd_lines[(s, d)].append(line_idx)
                
    # Figure out exactly which lines belong in train vs test
    train_targets = set()
    test_targets = set()
    
    for (s, d), lines in sd_lines.items():
        if len(lines) >= min_sd_traj_num:
            train_targets.update(lines[:-test_traj_num]) # All but last 5
            test_targets.update(lines[-test_traj_num:])  # Last 5
            
    # Free up the dictionary from RAM completely
    del sd_lines 
    print(f"\nFound {len(train_targets) + len(test_targets)} valid trajectories across {len(test_targets)//5} SD pairs.")

    # ---------------------------------------------------------
    # PASS 2: Writing (Streams directly to disk, RAM stays empty)
    # ---------------------------------------------------------
    train_file_path = f"{data_dir}/processed_{data_name}_train.csv"
    test_file_path = f"{data_dir}/processed_{data_name}_test.csv"
    
    with open(file_path, 'r') as file, \
         open(train_file_path, 'w') as train_file, \
         open(test_file_path, 'w') as test_file:
             
        for line_idx, line in enumerate(tqdm(file, desc="Pass 2: Writing Splits to Disk")):
            if line_idx in train_targets:
                traj = ast.literal_eval(line.strip())
                converted = [(grid, time.localtime(ts)[:6]) for grid, ts in traj]
                train_file.write(f"{converted}\n")
                
            elif line_idx in test_targets:
                traj = ast.literal_eval(line.strip())
                converted = [(grid, time.localtime(ts)[:6]) for grid, ts in traj]
                test_file.write(f"{converted}\n")

def count_sd_trajs(file_path):
    sd_count = defaultdict(int)
    print(f"\nCounting final SD Trajectories in {file_path}...")
    
    with open(file_path, 'r') as file:
        for line in tqdm(file, desc="Verifying Test Set"):
            line = line.strip()
            if not line: continue
            
            traj = ast.literal_eval(line)
            if traj:
                s, d = traj[0][0], traj[-1][0]
                sd_count[(s, d)] += 1
                
    cnt = sum(sd_count.values())
    print(f"Total Unique SD Pairs in Test Set: {len(sd_count)}")
    print(f"Total Trajectories in Test Set: {cnt}")


# --- MOVE THESE TO GLOBAL SCOPE FOR WINDOWS MULTIPROCESSING ---
grid_height, grid_width = 0.3, 0.3 # Was noted as 0.1, 0.1 in the code, but according to the paper it should be 0.3, 0.3    
chengdu_boundary = {'min_lat': 30.50, 'max_lat': 30.80, 'min_lon': 103.9, 'max_lon': 104.2}

if __name__ == '__main__':
    chengdu_data_dir = r'datasets/chengdu'
    chengdu_data_name = "chengdu"
   
  
    file_list = ['20140803_train', '20140804_train', '20140805_train', '20140806_train', '20140808_train',
                 '20140809_train', '20140810_train', '20140811_train', '20140812_train', '20140814_train']
    
    # --- OPTIMIZATION: Execute in Parallel ---
    print("Starting parallel preprocessing...")
    tasks = [(chengdu_data_dir, f, False) for f in file_list]
    
    # Run 4 files at a time to manage RAM
    with Pool(processes=4) as pool:
        pool.map(process_file_parallel, tasks)
  
    #print("All files processed. Merging...")
    chengdu_merge()
    print("Merging complete. Processing Source-Destination (SD) splits...")
    process_sd_ultra_low_ram(chengdu_data_dir, chengdu_data_name, noise=False)
    print("Counting SD trajectories...")
    count_sd_trajs(f"{chengdu_data_dir}/processed_{chengdu_data_name}_test.csv")
    print("Preprocessing fully complete!")