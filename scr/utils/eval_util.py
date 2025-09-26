from pathlib import Path
from warnings import warn
import sklearn.preprocessing as pre
import scipy
import numpy as np
import pandas as pd
import sed_eval
import dcase_util

def calculate_sed_metric(ref_list, pred_list):
    reference_event_list = dcase_util.containers.MetaDataContainer(ref_list)
    estimated_event_list = dcase_util.containers.MetaDataContainer(pred_list)

    segment_based_metrics_1s = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        time_resolution=1.0
    )
    segment_based_metrics_002s = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        time_resolution=0.02
    )
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        t_collar=0.250
    )

    for filename in reference_event_list.unique_files:
        reference_event_list_for_current_file = reference_event_list.filter(
            filename=filename
        )

        estimated_event_list_for_current_file = estimated_event_list.filter(
            filename=filename
        )

        segment_based_metrics_1s.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

        segment_based_metrics_002s.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

        event_based_metrics.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    # Get only certain metrics
    #print("segment_based 1s Result:", segment_based_metrics_1s.results_overall_metrics(),)
    #print("segment_based 0.02s Result:", segment_based_metrics_002s.results_overall_metrics())
    #print("event_based Result:", event_based_metrics.results_overall_metrics())

    output = {
        "segment_based_1s": segment_based_metrics_1s.results_overall_metrics()['f_measure']['f_measure'],
        "segment_based_0.02s": segment_based_metrics_002s.results_overall_metrics()['f_measure']['f_measure'],
        "event_based": event_based_metrics.results_overall_metrics()['f_measure']['f_measure'],
    }
    return output

def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))

def binarize(x, threshold=0.5):
    if x.ndim == 3:
        return np.array(
            [pre.binarize(sub, threshold=threshold) for sub in x])
    else:
        return pre.binarize(x, threshold=threshold)

def median_filter(x, window_size, threshold=0.5):
    x = binarize(x, threshold=threshold)
    if x.ndim == 3: # (batch_size, time_steps, num_classes)
        size = (1, window_size, 1)
    elif x.ndim == 2 and x.shape[0] == 1: # (batch_size, time_steps)
        size = (1, window_size)
    elif x.ndim == 2 and x.shape[0] > 1: # (time_steps, num_classes)
        size = (window_size, 1)
    return scipy.ndimage.median_filter(x, size=size)
