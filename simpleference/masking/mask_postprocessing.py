import numpy as np

from scipy.ndimage.morphology import binary_dilation


def postprocess_ilastik_predictions(prediction,
                                    dilation_iterations=1,
                                    threshold=.5):
    """
    Postprocess ilastik predictions to get a prediction mask.

    Arguments:
        prediction [np.ndarray]   - Array with the channel predictions
        dilation_iterations [int] - Number of iterations used for the dilation (default: 1)
        threshold [floatt]        - Threshold for the mask channel (default: 0.5)
    """
    import vigra

    print("Making mask for prediction with shape:", prediction.shape)

    # ilastik channel is in the last axis
    mask = prediction > threshold
    dilated = binary_dilation(mask, iterations=dilation_iterations)

    # filter connected components
    ccs = vigra.analysis.labelVolumeWithBackground(dilated.astype('uint8'))
    components, sizes = np.unique(ccs, return_counts=True)
    biggest_component = components[np.argmax(sizes)]

    if biggest_component == 0:
        components = components[1:]
        sizes = sizes[1:]
        biggest_component = components[np.argmax(sizes)]

    return (ccs == biggest_component).astype('uint8')
