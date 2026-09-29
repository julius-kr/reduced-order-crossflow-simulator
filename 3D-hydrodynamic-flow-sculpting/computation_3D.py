import numpy as np
from PIL import Image


class particleengineering:
    def __init__(self):
        self.nx = 70
        self.ny = 140
        self.nz = 70

    def get_transformed_photomask_mask(
        self,
        photomask_mask,
        canvas_size,
        scale,
        offset_x,
        offset_y,
        rotation
    ):
        canvas_w, canvas_h = canvas_size

        if photomask_mask is None:
            return None

        x_left = canvas_w * 0.25
        x_right = canvas_w * 0.75
        channel_width = x_right - x_left

        margin_y = 20
        channel_height = canvas_h - 2 * margin_y

        mask_h, mask_w = photomask_mask.shape

        # Crop photomask to the actual exposed region before scaling
        ys, xs = np.where(photomask_mask)

        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            photomask_mask = photomask_mask[y_min:y_max + 1, x_min:x_max + 1]

        mask_h, mask_w = photomask_mask.shape

        # scale=1 means exposed mask width equals channel width
        fit_scale = channel_width / mask_w

        img_w = max(1, int(mask_w * fit_scale * scale))
        img_h = max(1, int(mask_h * fit_scale * scale))

        mask_img = Image.fromarray(
            (photomask_mask * 255).astype(np.uint8)
        ).convert("L")

        mask_img = mask_img.resize(
            (img_w, img_h),
            Image.NEAREST
        )

        if rotation != 0:
            mask_img = mask_img.rotate(
                rotation,
                resample=Image.BICUBIC,
                expand=True
            )

        mask_arr = np.array(mask_img) > 0

        img_w, img_h = mask_img.size

        full_mask = np.zeros((canvas_h, canvas_w), dtype=bool)

        channel_center = (x_left + x_right) / 2

        x0 = int(channel_center - img_w / 2 + offset_x)
        y0 = int((canvas_h - img_h) / 2 + offset_y)

        x1 = x0 + img_w
        y1 = y0 + img_h

        src_x0 = max(0, -x0)
        src_y0 = max(0, -y0)
        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)

        dst_x1 = min(canvas_w, x1)
        dst_y1 = min(canvas_h, y1)

        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            full_mask[dst_y0:dst_y1, dst_x0:dst_x1] = mask_arr[
                src_y0:src_y1,
                src_x0:src_x1
            ]

        return full_mask

    def get_transformed_flow_mask(self, flow_mask, canvas_size):
        canvas_w, canvas_h = canvas_size

        if flow_mask is None:
            return None

        mask_img = Image.fromarray(
            (flow_mask * 255).astype(np.uint8)
        ).convert("L")

        mask_img = mask_img.resize(
            (canvas_w, canvas_h),
            Image.NEAREST
        )

        mask_arr = np.array(mask_img) > 0

        full_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        full_mask[:, :] = mask_arr

        return full_mask

    def compute_particle_volume(
        self,
        flow_mask,
        photomask_mask,
        canvas_size,
        scale,
        offset_x,
        offset_y,
        rotation
    ):
        canvas_w, canvas_h = canvas_size

        nx = self.nx
        ny = self.ny
        nz = self.nz

        flow_top = self.get_transformed_flow_mask(
            flow_mask,
            canvas_size
        )

        if flow_top is None:
            return None

        flow_img = Image.fromarray(
            (flow_top * 255).astype(np.uint8)
        )

        flow_img = flow_img.resize((nx, nz), Image.NEAREST)

        flow_2d = np.array(flow_img) > 0
        flow_xz = np.fliplr(flow_2d.T)

        flow_volume = np.repeat(
            flow_xz[:, :, None],
            ny,
            axis=2
        )

        photomask_top = self.get_transformed_photomask_mask(
            photomask_mask=photomask_mask,
            canvas_size=canvas_size,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
            rotation=rotation
        )

        if photomask_top is None:
            return None

        x_left = int(canvas_w * 0.25)
        x_right = int(canvas_w * 0.75)

        photomask_channel = photomask_top[:, x_left:x_right]

        photo_img = Image.fromarray(
            (photomask_channel * 255).astype(np.uint8)
        )

        photo_img = photo_img.resize((nx, ny), Image.NEAREST)

        photo_2d = np.array(photo_img) > 0
        photo_xy = photo_2d.T

        photo_volume = np.repeat(
            photo_xy[:, None, :],
            nz,
            axis=1
        )

        particle_volume = flow_volume & photo_volume

        return particle_volume