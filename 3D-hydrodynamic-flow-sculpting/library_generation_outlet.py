#!/usr/bin/env python
# -*-coding:utf-8 -*-

from pathlib import Path
import os
import glob
import gzip
import shutil
import pandas as pd
import numpy as np
import traceback 
from multiprocessing import Pool
from scipy.interpolate import LinearNDInterpolator


"""
backward mapping: end plane -> start plane
parameters for the outlet operator:
shape
"""

PARAM_KEYS = ['shape']
SRC_DIR = 'streamline_files_outlet'

# 250*250 grid to match with the 62500 streamline distribution in COMSOL
ny = 250 
nz = 250 


def parse_fname(fname):
    """
    get the parameter information from the file name
    e.g. shapetriangle.txt
    """
    fname_stem = Path(fname).stem

    if 'shape' not in fname_stem:
        raise ValueError(f"Filename does not contain shape: {fname_stem}")

    shape_name = fname_stem.removeprefix('shape')

    if not shape_name:
        raise ValueError(f"No text after shape: {fname_stem}")

    return shape_name, [shape_name]


def convert_streamline_to_index_map(fname, ny=ny, nz=nz):
    """
    streamlines are distributed uniformly from the end plane in COMSOL model
    """
    print(f'Processing {fname}')
    data = pd.read_csv(fname, sep='\t', header=7)

    # x is the flow direction in COMSOL model
    x_start = data['% x start'].to_numpy()
    y_start = data['y start'].to_numpy()
    z_start = data['z start'].to_numpy()
    y_end = data['y end'].to_numpy()
    z_end = data['z end'].to_numpy()

    # backward displacement
    dy_back = - (y_end - y_start)
    dz_back = - (z_end - z_start)

    # keep only streamlines that start from start plane
    valid_streamline =  x_start < 0.05
    y_end_valid = y_end[valid_streamline]
    z_end_valid = z_end[valid_streamline]
    dy_back_valid = dy_back[valid_streamline]
    dz_back_valid = dz_back[valid_streamline]

    # use end plane data to build the grid
    y_min, y_max = y_end.min(), y_end.max()
    z_min, z_max = z_end.min(), z_end.max()

    y_grid = np.linspace(y_min, y_max, ny)
    z_grid = np.linspace(z_min, z_max, nz)
    [zq, yq] = np.meshgrid(z_grid, y_grid, indexing='ij')

    # interpolation
    points_valid_at_end = np.c_[y_end_valid, z_end_valid]
    interp_dy_lin = LinearNDInterpolator(points_valid_at_end, dy_back_valid)
    interp_dz_lin = LinearNDInterpolator(points_valid_at_end, dz_back_valid)
    dy_interp = interp_dy_lin(yq, zq)
    dz_interp = interp_dz_lin(yq, zq)

    y_source = yq + dy_interp
    z_source = zq + dz_interp

    y_start_min, y_start_max = y_start.min(), y_start.max()
    z_start_min, z_start_max = z_start.min(), z_start.max()

    y_s_idx = np.round(
        (y_source - y_start_min) / (y_start_max - y_start_min) * (ny - 1)
    ).astype(int)

    z_s_idx = np.round(
        (z_source - z_start_min) / (z_start_max - z_start_min) * (nz - 1)
    ).astype(int)

    y_s_idx = np.clip(y_s_idx, 0, ny - 1)
    z_s_idx = np.clip(z_s_idx, 0, nz - 1)

    index_map = (z_s_idx * ny + y_s_idx).reshape(ny * nz)

    # build valid mask (outlet channel shape) from end plane data
    end_occupied = np.zeros((nz, ny), dtype=bool)

    end_y_idx = np.round((y_end_valid - y_min) / (y_max - y_min) * (ny - 1)).astype(int) # Boundary clamping
    end_z_idx = np.round((z_end_valid - z_min) / (z_max - z_min) * (nz - 1)).astype(int)

    end_y_idx = np.clip(end_y_idx, 0, ny - 1)
    end_z_idx = np.clip(end_z_idx, 0, nz - 1)

    end_occupied[end_z_idx, end_y_idx] = True

    valid_mask = end_occupied.reshape(ny * nz)

    return index_map, valid_mask 


def single_file_worker(fname):
    try:
        print(f'Processing {fname}...')
        fname_stem, value_from_name = parse_fname(fname)
        index_map, valid_mask = convert_streamline_to_index_map(fname)

        return fname_stem, {
            "map": index_map,
            "valid_mask": valid_mask,
        }, value_from_name
        
    except Exception as e:
        print(f"\n❌ ERROR processing {fname}: {e}")
        traceback.print_exc()
        return None
    

def all_files_parallel():
    num_cpus = os.cpu_count()

    all_files = glob.glob(f'{SRC_DIR}/*.txt')
    all_files.sort()
    print(f'Found {len(all_files)} text files. Using {num_cpus} CPUs.')

    advection_lib = {}
    value_dict = {k: [] for k in PARAM_KEYS}
    tree = {}

    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(single_file_worker, [(fname,) for fname in all_files])
        
    for res in results:
        if res is not None:  
            fname, index_map, value_from_name = res
            advection_lib[fname] = index_map
            for i, key in enumerate(PARAM_KEYS):
                value_dict[key].append(value_from_name[i])
                
            curr_level = tree
            for val in value_from_name:
                if val not in curr_level:
                    curr_level[val] = {}
                curr_level = curr_level[val]

    final_value_dict = {k: sorted(list(set(v))) for k, v in value_dict.items()}
    
    final_value_dict['tree'] = tree

    return advection_lib, final_value_dict


def build_and_save_library():
    advection_lib, final_value_dict = all_files_parallel()
    
    if not advection_lib:
        print("Library is empty. Something went wrong or no files found.")
        return

    for key in PARAM_KEYS:
        values = final_value_dict[key]
        print(f"The range of parameter {key}: {min(values)} to {max(values)}")

    save_dir = './generated_library_outlet'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'library_outlet')

    advection_lib_fname = f'{save_path}.npy'
    np.save(advection_lib_fname, advection_lib)
    print(f'\n{advection_lib_fname} saved.')

    advection_dict = f'{save_path}_dict.npy'
    np.save(advection_dict, final_value_dict)
    print(f'{advection_dict} saved.')

    lib_gz_fname = f'{save_path}.npy.gz'
    with open(advection_lib_fname, 'rb') as f_in:
        with gzip.open(lib_gz_fname, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f'{lib_gz_fname} saved.')

if __name__ == '__main__':
    build_and_save_library()