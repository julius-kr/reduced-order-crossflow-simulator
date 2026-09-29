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
parameters for the pillar operator:
channel width, cw
channel height, ch
pillar height, ph
y-coordinate of point A, ya
y-coordinate of point B, yb
x-coordinate of point C, xc
y-coordinate of point C, yc
"""

PARAM_KEYS = ['cw', 'ch', 'ph', 'ya', 'yb', 'xc', 'yc']
SRC_DIR = 'streamline_files_pillar'

# 250*250 grid to match with the 62500 streamline distribution in COMSOL
ny = 250 
nz = 250 


def parse_fname(fname):
    """
    get the parameter information from the file name
    e.g. cw1.5000_ch1.5000_ph0.3750_ya0.0000_yb0.3750_xc3.4375_yc0.0000.txt
    """
    fname_stem = Path(fname).stem
    split_name = fname_stem.split('_')

    values = []
    for i, key in enumerate(PARAM_KEYS):
        val_str = split_name[i].replace(key, '')
        values.append(float(val_str))

    return fname_stem, np.array(values)


def convert_streamline_to_advection_map(fname, ny=ny, nz=nz):
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
    dy = - (y_end - y_start)
    dz = - (z_end - z_start)

    # filter streamlines that dont start from start plane
    failed_streamline = x_start > 0.05 
    dy[failed_streamline] = 0 # set these failed streamlines as 0
    dz[failed_streamline] = 0

    # use end plane data to build the grid
    y_min, y_max = y_end.min(), y_end.max()
    z_min, z_max = z_end.min(), z_end.max()

    y_grid = np.linspace(y_min, y_max, ny)
    z_grid = np.linspace(z_min, z_max, nz)
    [zq, yq] = np.meshgrid(z_grid, y_grid, indexing='ij')

    # interpolation
    displacements = np.c_[dy, dz]
    interp_func = LinearNDInterpolator((y_end, z_end), displacements)
    interpolation = interp_func((yq, zq))
    dy_interp = np.nan_to_num(interpolation[..., 0], nan=0.0)
    dz_interp = np.nan_to_num(interpolation[..., 1], nan=0.0)

    # normalize displacement and quantize it to grid resolution
    y_span = y_max - y_min
    z_span = z_max - z_min
    
    dy_norm = dy_interp / y_span if y_span > 0 else dy_interp
    dz_norm = dz_interp / z_span if z_span > 0 else dz_interp

    dy_interp_flat = dy_norm.reshape((ny*nz,))
    dz_interp_flat = dz_norm.reshape((ny*nz,))

    advection_map = np.concatenate((dy_interp_flat, dz_interp_flat), axis=0)

    return advection_map


def convert_advection_map_to_index_map(advection_map, ny=ny, nz=nz):

    origins = range(ny*nz)
    sources = np.zeros(ny*nz, dtype=np.int32)

    for cell in origins:
        y_e = int(cell % ny)
        z_e = int(cell / ny)

        dy = round(advection_map[cell] * ny)
        dz = round(advection_map[cell + ny * nz] * nz)

        y_s = int(y_e + dy)
        z_s = int(z_e + dz)

        # Boundary clamping
        if y_s < 0: y_s = 0
        if z_s < 0: z_s = 0
        if y_s > ny - 1: y_s = ny - 1
        if z_s > nz - 1: z_s = nz - 1

        sources[cell] = int((z_s * ny) + y_s)

    return sources


def single_file_worker(fname):
    try:
        print(f'Processing {fname}...')
        fname_stem, value_from_name = parse_fname(fname)
        advection_map_dx_dz = convert_streamline_to_advection_map(fname)
        index_map = convert_advection_map_to_index_map(advection_map_dx_dz)
        
        return fname_stem, index_map, value_from_name
        
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

    print(f'\nLibrary created with {len(advection_lib)} maps')

    for key in PARAM_KEYS:
        values = final_value_dict[key]
        print(f"The range of parameter {key}: {min(values)} to {max(values)}")

    
    save_dir = './generated_library_pillar'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'library_pillar')

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