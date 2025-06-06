import numpy as np
from scipy import stats
from tqdm.notebook import tqdm

from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, polygonize

## THIS IS ADAPTED FROM https://github.com/ulissigroup/uncertainty_benchmarking
## Tran et al. 2023, "Uncertainty Benchmarking for Materials Property Prediction"
## https://doi.org/10.1088/2632-2153/ab7e1a

def calculate_density(percentile, residuals, stdevs):
    '''
    Calculate the fraction of the residuals that fall within the lower
    `percentile` of their respective Gaussian distributions, which are
    defined by their respective uncertainty estimates.
    '''
    

    norm = stats.norm(loc=0, scale=1)
    # Find the normalized bounds of this percentile
    upper_bound = norm.ppf(percentile)

    # Normalize the residuals so they all should fall on the normal bell curve
    normalized_residuals = residuals.reshape(-1) / stdevs.reshape(-1)

    # Count how many residuals fall inside here
    num_within_quantile = 0
    for resid in normalized_residuals:
        if resid <= upper_bound:
            num_within_quantile += 1

    # Return the fraction of residuals that fall within the bounds
    density = num_within_quantile / len(residuals)
    return density

def calculate_miscalibration_area(predicted_pi, observed_pi):
    """
    Calculate the miscalibration area between predicted and observed probabilities.

    Args:
        predicted_pi (array-like): Predicted probabilities (e.g., expected frequency).
        observed_pi (array-like): Observed probabilities (e.g., observed frequency).

    Returns:
        float: The miscalibration area.
    """
    # Create polygon points
    polygon_points = []
    for point in zip(predicted_pi, observed_pi):
        polygon_points.append(point)
    for point in zip(reversed(predicted_pi), reversed(predicted_pi)):
        polygon_points.append(point)
    polygon_points.append((predicted_pi[0], observed_pi[0]))

    # Create polygon and calculate area
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy  # Original data
    ls = LineString(np.c_[x, y])  # Closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    return miscalibration_area

def calculate_calibration_error(predicted_pi, observed_pi):
    return ((predicted_pi - observed_pi)**2).sum()

def calculate_calibration(residuals, stddevs):

    predicted_pi = np.linspace(0, 1, 100)

    return [calculate_density(quantile, residuals, stddevs)
               for quantile in tqdm(predicted_pi, desc='Calibration')]


def err_vs_num_samples(n_trials, id_pred_set, ood_pred_set, original_targets_id, original_targets_ood):
    num_trials = []
    list_of_cal_err_id = []
    list_of_cal_area_id = []
    list_of_cal_err_ood = []
    list_of_cal_area_ood = []

    for i in np.arange(10, n_trials+10, 10):
        residuals_id = np.array(original_targets_id) - np.mean(np.array(id_pred_set)[:i], axis=0)
        stddev_id = np.std(np.array(id_pred_set)[:i], axis=0)

        residuals_ood = np.array(original_targets_ood) - np.mean(np.array(ood_pred_set)[:i], axis=0)
        stddev_ood = np.std(np.array(ood_pred_set)[:i], axis=0)

        predicted_pi = np.linspace(0, 1, 100)
        obsv_pi_id = calculate_calibration(residuals_id, stddev_id)
        obsv_pi_ood = calculate_calibration(residuals_ood, stddev_ood)

        cal_err_id = ((predicted_pi - obsv_pi_id)**2).sum()
        cal_area_id = calculate_miscalibration_area(predicted_pi, obsv_pi_id)

        cal_err_ood = ((predicted_pi - obsv_pi_ood)**2).sum()
        cal_area_ood = calculate_miscalibration_area(predicted_pi, obsv_pi_ood)

        num_trials.append(i)
        list_of_cal_err_id.append(cal_err_id)
        list_of_cal_area_id.append(cal_area_id)
        list_of_cal_err_ood.append(cal_err_ood)
        list_of_cal_area_ood.append(cal_area_ood)
    return num_trials, list_of_cal_err_id, list_of_cal_area_id, list_of_cal_err_ood, list_of_cal_area_ood              