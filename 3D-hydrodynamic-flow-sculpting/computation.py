#!/usr/bin/env python

import numpy as np
from PIL import Image
from scipy import ndimage


class flowsculpting:
    def __init__(self):
        # advection map dimensions
        self.nx = 250 
        self.nz = 250

        self.map_lib = None

        self.color_table_for_image()

        default_img = 255*np.ones((self.nz, self.nx), dtype=np.uint8)
        self.default_img = Image.fromarray(default_img)

        self.outlet_color = self.default_img
        self.inlet_color = self.default_img
        self.outlet_valid_mask_color = self.default_img


    def color_table_for_image(self):

        COLORS = np.array([
        [255/255, 208/255, 203/255],   # light pink
        [240/255, 162/255, 165/255],   # dark pink
        [225/255, 173/255, 37/255],    # yellow
        [130/255, 191/255, 229/255],   # light blue
        [180/255, 160/255, 210/255],   # light purple
        ])

        self.all_colors = COLORS


    def load_libraries(self, library_fname_path):
        self.map_lib = np.load(library_fname_path, allow_pickle=True).item()


    def get_library_info(self, key):
        """
        index_map, from 3 libraries
        empty_target_masks, from inlet library
        empty_point_indices, from inlet library
        empty_point_target_label, from inlet library
        outlet_shape, from outlet libraby
        """
        entry = self.map_lib[key]

        if isinstance(entry, dict):

            index_map = entry['map']

            empty_target_masks = entry.get('empty_target_plane_masks', None)
            if empty_target_masks is None:
                empty_target_masks = entry.get('empty_target_masks', None)

            empty_point_indices = entry.get('empty_point_indices', None)
            empty_point_target_label = entry.get('empty_point_target_label', None)

            outlet_shape = entry.get('valid_mask', None)

        else:
            index_map = entry
            empty_target_masks = None
            empty_point_indices = None
            empty_point_target_label = None
            outlet_shape = None

        return (
            index_map,
            empty_target_masks,
            empty_point_indices,
            empty_point_target_label,
            outlet_shape,
        )


    def create_initial_flow_profile(self, dividers):
        # get the stream number from GUI, up to 5 horizontally stacked streams

        num_channels = len(dividers)

        inlet = np.zeros((num_channels, self.nx*self.nz), dtype=np.float32)

        div_start = 0

        for n in range(num_channels):

            if n == num_channels - 1:
                div_end = self.nx
            else:
                div_end = int(dividers[n] * self.nx)

            inlet_c = np.zeros(self.nx, dtype=np.uint8)
            inlet_c[div_start:div_end] = 1

            div_start = div_end

            inlet[n, :] = np.tile(inlet_c, self.nz)

        self.inlet = inlet
        self.inlet_color = self.generate_color_image(inlet)

        return inlet
    

    def compute_and_generate_new_flow_profile(self, op_sequence, treat_empty_as_new_source=True):

        outlet = self.inlet.copy().astype(np.float32)

        for op_type, key in op_sequence:
            map_info = self.get_library_info(key)

            if op_type == 'pillar':
                outlet = self.pillar_operator(outlet, map_info)

            elif op_type == 'outlet':
                outlet = self.outlet_operator(outlet, map_info)

            elif op_type == 'inlet':
                outlet = self.inlet_operator(
                    outlet,
                    map_info,
                    treat_empty_as_new_source=treat_empty_as_new_source
                )

        num_channels = outlet.shape[0]
        self.outlet = outlet.reshape((num_channels, self.nz, self.nx))
        self.outlet_color = self.generate_color_image(self.outlet)

        return self.outlet_color
    

    def pillar_operator(self, outlet, map_info):
        index_map = map_info[0]

        outlet = outlet[:, index_map]

        return outlet
    

    def outlet_operator(self, outlet, map_info):
        index_map = map_info[0]
        valid_mask = map_info[4]

        outlet = outlet[:, index_map]

        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
            outlet[:, ~valid_mask] = 0.0
            self.outlet_valid_mask_color = self.show_the_outlet_shape_in_outlet_operator(valid_mask)

        return outlet
    
    
    def inlet_operator(self, outlet, map_info, treat_empty_as_new_source=True):
        (
            index_map,
            empty_target_masks,
            empty_point_indices,
            empty_point_target_label,
            valid_mask,
        ) = map_info

        old_outlet = outlet
        old_num_channels = old_outlet.shape[0]

        new_outlet = np.zeros(
            (old_num_channels, old_outlet.shape[1]),
            dtype=np.float32
        )

        for ch in range(old_outlet.shape[0]):
            np.maximum.at(new_outlet[ch], index_map, old_outlet[ch])

        if treat_empty_as_new_source:
            new_outlet = self.set_empty_targets_as_new_channels_in_inlet_operator(
                new_outlet,
                old_outlet,
                empty_target_masks,
                empty_point_indices,
                empty_point_target_label
            )

            new_outlet = self.fill_the_empty_cells_in_inlet_operator(new_outlet)

        return new_outlet
    

    def set_empty_targets_as_new_channels_in_inlet_operator(
        self,
        new_outlet,
        old_outlet,
        empty_target_masks,
        empty_point_indices,
        empty_point_target_label
    ):
        appended_channels = []

        if (
            empty_target_masks is None
            or not isinstance(empty_target_masks, dict)
            or len(empty_target_masks) == 0
        ):
            return new_outlet

        target_names = sorted(
            empty_target_masks.keys(),
            key=lambda x: int(x.replace('emptytarget', ''))
            if x.replace('emptytarget', '').isdigit() else 999
        )

        if (
            empty_point_indices is not None
            and empty_point_target_label is not None
            and len(empty_point_indices) > 0
        ):
            for target_name in target_names:
                try:
                    label_id = int(target_name.replace("emptytarget", ""))
                except Exception:
                    continue

                target_mask = np.zeros(old_outlet.shape[1], dtype=bool)
                idx_sel = np.asarray(empty_point_target_label) == label_id

                if np.any(idx_sel):
                    target_mask[np.asarray(empty_point_indices)[idx_sel]] = True

                target_mask = target_mask & np.all(new_outlet == 0, axis=0)

                if np.any(target_mask):
                    empty_channel = np.zeros(
                        (1, old_outlet.shape[1]),
                        dtype=np.float32
                    )
                    empty_channel[0, target_mask] = 1.0
                    appended_channels.append(empty_channel)

        else:
            for target_name in target_names:
                target_mask = np.asarray(
                    empty_target_masks[target_name],
                    dtype=bool
                ).reshape(-1)

                target_mask = target_mask & np.all(new_outlet == 0, axis=0)

                if np.any(target_mask):
                    empty_channel = np.zeros(
                        (1, old_outlet.shape[1]),
                        dtype=np.float32
                    )
                    empty_channel[0, target_mask] = 1.0
                    appended_channels.append(empty_channel)

        if len(appended_channels) > 0:
            new_outlet = np.vstack([new_outlet] + appended_channels)

        return new_outlet
    
    
    def fill_the_empty_cells_in_inlet_operator(self, new_outlet):

        filled_outlet = new_outlet.copy()

        uncovered_mask = np.all(filled_outlet == 0, axis=0)
        if not np.any(uncovered_mask):
            return filled_outlet

        occupied_mask = np.any(filled_outlet > 0, axis=0)
        if not np.any(occupied_mask):
            return filled_outlet

        dominant_channel = np.argmax(filled_outlet, axis=0).astype(np.int32)

        occupied_2d = occupied_mask.reshape((self.nz, self.nx))
        dominant_2d = dominant_channel.reshape((self.nz, self.nx))

        _, nearest_idx = ndimage.distance_transform_edt(
            ~occupied_2d,
            return_indices=True
        )

        nearest_z = nearest_idx[0]
        nearest_y = nearest_idx[1]

        nearest_channel_2d = dominant_2d[nearest_z, nearest_y]
        nearest_channel_flat = nearest_channel_2d.reshape(-1)

        for ch in range(filled_outlet.shape[0]):
            assign_mask = uncovered_mask & (nearest_channel_flat == ch)
            if np.any(assign_mask):
                filled_outlet[ch, assign_mask] = 1.0

        return filled_outlet
    

    def show_the_outlet_shape_in_outlet_operator(self, valid_mask):
        
        if valid_mask is None:
            img = np.full((self.nz, self.nx, 3), [160, 160, 160], dtype=np.uint8)
            img = np.flipud(img)
            return Image.fromarray(img)
        
        mask = np.asarray(valid_mask, dtype=bool).reshape((self.nz, self.nx))

        img = np.full((self.nz, self.nx, 3), [211, 211, 211], dtype=np.uint8)

        img[mask] = [160, 160, 160]

        img = np.flipud(img)
        return Image.fromarray(img)
    
    
    def generate_color_image(self, flow_img):
    
        img = np.full((self.nz, self.nx, 3), [211, 211, 211], dtype=np.uint8)

        num_channels = flow_img.shape[0]

        for n in range(num_channels):
            mask_2d = flow_img[n, :].reshape((self.nz, self.nx)).astype(bool)
            color_idx = n % len(self.all_colors)
            img[mask_2d] = (self.all_colors[color_idx] * 255).astype(np.uint8)

        img = np.flipud(img)
        return Image.fromarray(img)
    

    def reset(self):
        self.outlet_color = self.default_img
        self.inlet_color = self.default_img
