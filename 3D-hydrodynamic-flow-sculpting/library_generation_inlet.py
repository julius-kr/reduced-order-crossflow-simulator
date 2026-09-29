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
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


"""
forward mapping: start plane -> end plane
parameters for the inlet operator:
channel height, ch
operator width, ow
inlet number, inlet
inlet position, ip
previous inlet number, pin
"""

PARAM_KEYS = ['ch', 'ow', 'inlet', 'ip', 'pin'] 
SRC_DIR = 'streamline_files_inlet'

# 250*250 grid to match with the 62500 streamline distribution in COMSOL
ny = 250 
nz = 250 


def parse_fname(fname):
    """
    get the parameter information from the main file name
    e.g. ch1.5000_ow0.3750_inlet1.0000_ip0.0000_pin1.0000.txt
    corresponding emptytargets file
    e.g. emptytargets_ch1.5000_ow0.3750_inlet1.0000_ip0.0000_pin1.0000.txt
    """
    raw_stem = Path(fname).stem
    fname_stem = raw_stem
    is_emptytargets = False

    if raw_stem.startswith('emptytargets_'):
        fname_stem = raw_stem[len('emptytargets_'):]
        is_emptytargets = True

    split_name = fname_stem.split('_')

    if len(split_name) != len(PARAM_KEYS):
        raise ValueError(
            f'Filename format error: {raw_stem}. '
            f'Expected {len(PARAM_KEYS)} fields after optional prefix: {PARAM_KEYS}'
        )

    values = []
    for i, key in enumerate(PARAM_KEYS):
        token = split_name[i]
        if not token.startswith(key):
            raise ValueError(
                f'Filename token mismatch: {token}, expected prefix {key}'
            )
        val_str = token.replace(key, '', 1)
        values.append(float(val_str))

    base_key = fname_stem
    return base_key, np.array(values), is_emptytargets


def convert_main_streamline_to_advection_map(fname, ny=ny, nz=nz):
    """
    process the main streamlines in the main file
    main streamlines are distributed uniformly from the start plane in COMSOL model
    """
    print(f'Processing {fname}')
    data = pd.read_csv(fname, sep='\t', header=7)

    # x is the flow direction in COMSOL model
    x_end = data['x end'].to_numpy()
    y_start = data['y start'].to_numpy()
    z_start = data['z start'].to_numpy()
    y_end = data['y end'].to_numpy()
    z_end = data['z end'].to_numpy()

    # forward displacement
    dy = (y_end - y_start)
    dz = (z_end - z_start)

    # keep only streamlines that reach close to the outlet
    valid_streamline = x_end > 0.95 * x_end.max() 
    y_start_valid = y_start[valid_streamline]
    z_start_valid = z_start[valid_streamline]
    dy_valid = dy[valid_streamline]
    dz_valid = dz[valid_streamline]

    # use start plane data to build the grid
    y_min, y_max = y_start.min(), y_start.max()
    z_min, z_max = z_start.min(), z_start.max()

    y_grid = np.linspace(y_min, y_max, ny)
    z_grid = np.linspace(z_min, z_max, nz)
    [zq, yq] = np.meshgrid(z_grid, y_grid, indexing='ij')

    # interpolation
    points_valid_at_start = np.c_[y_start_valid, z_start_valid]
    interp_dy_lin = LinearNDInterpolator(points_valid_at_start, dy_valid)
    interp_dz_lin = LinearNDInterpolator(points_valid_at_start, dz_valid)
    dy_interp = interp_dy_lin(yq, zq)
    dz_interp = interp_dz_lin(yq, zq)

    # fix the empty cells created by forward mapping
    empty_cells = np.isnan(dy_interp) | np.isnan(dz_interp)

    interp_dy_near = NearestNDInterpolator(points_valid_at_start, dy_valid) 
    interp_dz_near = NearestNDInterpolator(points_valid_at_start, dz_valid)

    if np.any(empty_cells):
        dy_interp[empty_cells] = interp_dy_near(
            yq[empty_cells], zq[empty_cells]
        )
        dz_interp[empty_cells] = interp_dz_near(
            yq[empty_cells], zq[empty_cells]
        )

    # turn the displacement into cell numbers in index map
    dy_step = (y_max - y_min) / (ny - 1)
    dz_step = (z_max - z_min) / (nz - 1)

    dy_idx = np.round(dy_interp / dy_step)
    dz_idx = np.round(dz_interp / dz_step)

    dy_interp_flat = dy_idx.reshape(ny * nz)
    dz_interp_flat = dz_idx.reshape(ny * nz)

    advection_map = np.concatenate((dy_interp_flat, dz_interp_flat), axis=0)

    return advection_map 


def define_empty_target_region(fname, value_from_name, ny=ny, nz=nz):
    """
    process the newly introduced streamlines in the emptytargets file
    to define an empty target region
    streamlines are distributed uniformly from the end plane in COMSOL model
    """
    print(f'Processing emptytargets file: {fname}')
    data = pd.read_csv(fname, sep='\t', header=7)

    y_start = data['y start'].to_numpy()
    z_start = data['z start'].to_numpy()

    y_end = data['y end'].to_numpy()
    z_end = data['z end'].to_numpy()

    ch, ow, inlet, ip, pin = value_from_name
    inlet = int(round(float(inlet)))
    ip = float(ip)

    # keep streamlines that come from the new inlet
    keep = z_start > 1.8 

    # build a mask at the end plane with the kept streamlines
    y_end_mask = y_end[keep] 
    z_end_mask = z_end[keep]

    y_start_for_emptytargets_division = y_start[keep]

    if len(y_end_mask) == 0:
        return {
            'empty_streamline_count': 0,
            'empty_point_indices': np.array([], dtype=np.int32),
            'empty_point_target_label': np.array([], dtype=np.int32),
            'empty_targets': np.zeros(ny * nz, dtype=bool),
            'empty_target_plane_masks': {},
            'empty_target_plane_counts': {},
            'empty_streamline_meta': {
                'rule': 'z_start > 1.8, classify by y_start around ip + k*0.75, project to end plane',
                'inlet': inlet,
                'ip': ip,
            }
        }

    y_min, y_max = y_end.min(), y_end.max()
    z_min, z_max = z_end.min(), z_end.max()

    y_span = y_max - y_min
    z_span = z_max - z_min

    if y_span <= 0:
        y_span = 1.0
    if z_span <= 0:
        z_span = 1.0

    iy = np.round((y_end_mask - y_min) / y_span * (ny - 1)).astype(np.int32)
    iz = np.round((z_end_mask - z_min) / z_span * (nz - 1)).astype(np.int32)

    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)

    point_indices = iz * ny + iy

    # divide the region for streamlines from different new inlets
    target_label = np.zeros(len(y_end_mask), dtype=np.int32)
    target_masks = {}
    target_counts = {}

    for k in range(inlet):
        center = ip + 0.75 * k # 0.75 is the diameter of the inlet
        left = center - 0.375
        right = center + 0.375
        label_id = k + 1

        in_region = (y_start_for_emptytargets_division >= left) & (y_start_for_emptytargets_division < right)
        target_label[in_region] = label_id

        region_mask = np.zeros(ny * nz, dtype=bool)
        if np.any(in_region):
            region_mask[point_indices[in_region]] = True

        target_masks[f'emptytarget{label_id}'] = region_mask
        target_counts[f'emptytarget{label_id}'] = int(np.sum(in_region))

    point_mask = np.zeros(ny * nz, dtype=bool)
    point_mask[point_indices] = True

    return {
        'empty_streamline_count': int(len(y_end_mask)),
        'empty_point_indices': point_indices,
        'empty_point_target_label': target_label,
        'empty_targets': point_mask,
        'empty_target_plane_masks': target_masks,
        'empty_target_plane_counts': target_counts,
        'empty_streamline_meta': {
            'rule': 'z_start > 1.8, classify by y_start around ip + k*0.75, project to end plane',
            'inlet': inlet,
            'ip': ip,
            'y_min_end': float(y_min),
            'y_max_end': float(y_max),
            'z_min_end': float(z_min),
            'z_max_end': float(z_max),
            'y_min_start': float(y_start.min()),
            'y_max_start': float(y_start.max()),
            'z_min_start': float(z_start.min()),
            'z_max_start': float(z_start.max()),
        }
    }


def convert_advection_map_to_index_map(advection_map, ny=ny, nz=nz): 

    destinations = np.zeros(ny*nz, dtype=np.int32)

    for cell in range(ny * nz):
        y_s = int(cell % ny)
        z_s = int(cell / ny)

        dy = int(advection_map[cell])
        dz = int(advection_map[cell + ny*nz])

        y_e = y_s + dy 
        z_e = z_s + dz

        y_e = max(0, min(ny - 1, y_e))
        z_e = max(0, min(nz - 1, z_e))

        destinations[cell] = z_e * ny + y_e

    return destinations


def single_file_worker(fname):
    try:
        base_key, value_from_name, is_emptytargets = parse_fname(fname)

        # process emptytargets file
        if is_emptytargets:
            aux_data = define_empty_target_region(
                fname,
                value_from_name=value_from_name,
                ny=ny,
                nz=nz
            )

            return {
                'base_key': base_key,
                'value_from_name': value_from_name,
                'is_emptytargets': True,
                'payload': aux_data,
            }

        # process main file
        advection_map = convert_main_streamline_to_advection_map(fname)
        index_map = convert_advection_map_to_index_map(advection_map)

        return {
            'base_key': base_key,
            'value_from_name': value_from_name,
            'is_emptytargets': False,
            'payload': {
                'map': index_map,
            }
        }    

    except Exception as e:
        traceback.print_exc()
        return None


def all_files_parallel():
    advection_lib = {}
    value_dict = {k: [] for k in PARAM_KEYS}
    tree = {}
    num_cpus = os.cpu_count()

    all_files = glob.glob(f'{SRC_DIR}/*.txt')
    all_files.sort()

    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(single_file_worker, [(fname,) for fname in all_files])
        
    for res in results:
        if res is not None:
            base_key = res['base_key']
            value_from_name = res['value_from_name']
            payload = res['payload']

            if base_key not in advection_lib:
                advection_lib[base_key] = {}

            advection_lib[base_key].update(payload)

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
        print("Something went wrong or no files found.")
        return
    
    for key in PARAM_KEYS:
        values = final_value_dict[key]
        print(f"The range of parameter {key}: {min(values)} to {max(values)}")

    save_dir = './generated_library_inlet'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'library_inlet')

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
    