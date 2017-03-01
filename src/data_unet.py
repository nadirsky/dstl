from random import Random

import h5py
import numpy as np

from keras.preprocessing import image

channels = 20

LARGE_VEHICLES = 9
SMALL_VEHICLES = 10
MISC = 2

hf = h5py.File('/mnt/ml-team/satellites/files/dstl-train-20channels_ver2.h5')
hf_masks = h5py.File('/mnt/ml-team/satellites/files/dstl-train-masks_ver2.h5')


def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = rg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

means = np.array([433.63702393, 470.16989136, 336.4604187, 505.84042358,
                  295.93945312, 336.45907593, 470.146698, 476.11135864,
                  433.62802124, 520.07617188, 692.80700684, 521.65710449,
                  4398.10644531, 4629.50585938, 4317.39990234, 3875.63598633,
                  3026.12573242, 2718.59057617, 2659.33374023, 2568.97021484], dtype=np.float16)
std = np.array([218.58804321, 172.68545532, 109.00450134, 166.98269653,
                40.36304474, 109.01099396, 172.72686768, 175.36598206,
                218.5912323, 158.89251709, 232.90725708, 142.55833435,
                1896.10668945, 2496.32739258, 2206.86889648, 2129.56494141,
                1881.29187012, 1666.98059082, 1668.1427002, 1703.44152832], dtype=np.float16)

channels_cache = {}
for key in hf.keys():
    channels_cache[key] = hf[key].value

masks_cache = {}
masks_cache_small = {}
masks_cache_large = {}
for key in hf_masks.keys():
    path_small = '{}/{}'.format(key, str(SMALL_VEHICLES))
    mask_small = hf_masks[path_small].value
    masks_cache_small[key] = mask_small

    path_large = '{}/{}'.format(key, str(LARGE_VEHICLES))
    mask_large = hf_masks[path_large].value
    masks_cache_large[key] = mask_large

    mask = mask_large + mask_small

    masks_cache[key] = mask


def train_generator(batch_size, crop_width, crop_height):
    """Returns batches of (X,Y)"""
    return unet_generator(batch_size, crop_width, crop_height, 'train')


def val_generator(batch_size, crop_width, crop_height):
    """Returns batches of (X,Y)"""
    return unet_generator(batch_size, crop_width, crop_height, 'validation')


def unet_generator(batch_size, crop_width, crop_height,
                   train_or_validation='train'):
    seed = 0

    random = Random(seed)
    while True:
        channels_crops = []
        mask_crops = []
        for ix in xrange(batch_size):
            should_have_cars = ix % 2 == 0

            (channels_single_crop, mask_single_crop) = random_crops(crop_width, crop_height, train_or_validation,
                                                                    should_have_cars, random)

            # augementation
            degrees= random.randint(0, 360)

            if np.random.random() < 0.5:
                mask_single_crop = image.flip_axis(mask_single_crop, 0)
                channels_single_crop = image.flip_axis(channels_single_crop, 0)

            if np.random.random() < 0.5:
                mask_single_crop = image.flip_axis(mask_single_crop, 1)
                channels_single_crop = image.flip_axis(channels_single_crop, 1)

            channels_single_crop = random_rotation(channels_single_crop, degrees, 0, 1, 2, fill_mode='mirror',
                                                         cval=0)

            mask_single_crop = mask_single_crop.reshape((crop_width, crop_height, 1)) # rotate needs channel dim
            mask_single_crop = random_rotation(mask_single_crop, degrees, 0, 1, 2, fill_mode='mirror',
                                                         cval=0)
            mask_single_crop = mask_single_crop.reshape((crop_width, crop_height))

            mask_single_crop = mask_single_crop.astype(np.uint8)

            channels_single_crop = channels_single_crop.astype(np.float16)
            channels_single_crop -= means
            channels_single_crop /= std

            channels_crops.append(channels_single_crop)
            mask_crops.append(mask_single_crop)

        channels_crops = np.stack(channels_crops)
        mask_crops = np.stack(mask_crops)

        # channels_crops = np.transpose(np.stack(channels_crops), (0, 3, 1, 2))
        # mask_crops = np.transpose(np.stack(mask_crops), (0, 3, 1, 2))

        yield (channels_crops, mask_crops)


def random_crops(crop_width, crop_height, train_or_validation, should_have_cars, random):
    while True: # must be iterative because of recursion depth limits
        image_id = random.choice(hf.keys())
        image = channels_cache[image_id]
        mask = masks_cache[image_id]

        image_width = image.shape[0]
        image_height = image.shape[1]

        crop_cords = random_crop_cords(image_width, image_height, crop_width, crop_height, train_or_validation, random)
        (x_start, x_end, y_start, y_end) = crop_cords

        image_crop = image[x_start:x_end, y_start:y_end, :channels]
        mask_crop = mask[x_start:x_end, y_start:y_end]

        car_pixels = np.count_nonzero(mask_crop[10:-10, 10:-10])
        has_cars = car_pixels > 10

        if should_have_cars == has_cars:
            return image_crop, mask_crop
        # else retry in loop


def random_crop_cords(image_width, image_height, crop_width, crop_height, train_or_validation, random):
    x = random.randint(0, image_width - crop_width)
    y = random.randint(0, image_height - crop_height)

    #########
    # 1 # 2 # Quarters 1,2 and 3 are train set
    #########
    # 3 # 4 # Quarter 4 is validation
    #########
    if x < int(image_width * 3 / 4) - crop_width:
        if y < int(image_height * 3 / 4) - crop_height:
            quarter = 1
        else:
            quarter = 3
    else:
        if y < int(image_height * 3 / 4) - crop_height:
            quarter = 2
        else:
            quarter = 4

    if quarter <= 3:
        crop_for_train_or_validation = 'train'
    else:
        crop_for_train_or_validation = 'validation'

    if train_or_validation == crop_for_train_or_validation:
        return x, x + crop_width, y, y + crop_height
    else:
        # try again
        return random_crop_cords(image_width, image_height, crop_width, crop_height, train_or_validation, random)
