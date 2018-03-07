import numpy as np


def float_to_uint8(input_, output_bounding_box):
    return np.round(input_ * 255.).astype('uint8')


def clip_float_to_uint8(input_, output_bounding_box, float_range=(0., 1.), safe_scale=True):
    """Convert float values in the range float_range to uint8. Crop values to (0, 255).

    Args:
        input_ (np.array): Input data as produced by the network.
        output_bounding_box (slice): Bounding box of the current block in the full dataset.
        float_range (tuple, list): Range of values of data.
        safe_scale (bool): If True, values are scaled such that all values within float_range fall within (0, 255).
            and are not cropped.  If False, values at the lower end of float_range may be scaled to < 0 and then
            cropped to 0.

    Returns:
        np.array: Postprocessed output, dtype is uint8
    """
    print(list(output_bounding_box[k].start for k in range(len(output_bounding_box))))
    if safe_scale:
        mult = np.floor(255./(float_range[1]-float_range[0]))
    else:
        mult = np.ceil(255./(float_range[1]-float_range[0]))
    add = 255 - mult*float_range[1]
    return np.clip((input_*mult+add).round(), 0, 255).astype('uint8')
