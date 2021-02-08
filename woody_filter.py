"""Python implementation of Woody adaptive filter for variable latency ERP detection. The module contains function
for processing of raw epoch data passed as NumPy array and a wrapper function for python-MNE package. """

from typing import Tuple
import numpy.typing as npt

import numpy as np
import logging
from mne import Epochs, EpochsArray


def print_help():
    print("Python implementation of Woody adaptive filter. To use it, import either `woody_filter` or "
          "`woody_filter_raw` into your python script.")


def woody_filter(epochs: Epochs, eps: float = 0.001, max_iterations: int = 100,
                 template: npt.ArrayLike = None) -> Tuple[EpochsArray, np.ndarray, np.ndarray]:
    """
    Woody filter for python-MNE module. Aligns Epochs to maximize correlation.
    :param epochs: mne.Epochs instance containing epochs of signal
    :param eps: float number specifying convergence condition
    :param max_iterations: integer specifying maximum of iteration steps
    :param template: Numpy array representing template signal to correlate with each epoch. Must have the same
    dimensions as an epoch.
    :return: Tuple with EpochsArray instance, array of latencies and averaged correlation coefficients.
    """
    result, latencies, avg_correlations = woody_filter_raw(epochs.get_data(), eps, max_iterations, template)
    return EpochsArray(result, epochs.info), latencies, avg_correlations


def woody_filter_raw(data: npt.ArrayLike, eps: float = 0.001, max_iterations: int = 100,
                     template: npt.ArrayLike = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligns input signal to maximize correlation.
    :param data: Numpy array
    :param eps: float number specifying convergence condition
    :param max_iterations: integer specifying maximum of iteration steps
    :param template: Numpy array representing template signal to correlate with each epoch. Must have the same
    dimensions as an epoch.
    :return: Tuple with shifted epochs in Numpy array, array of latencies and averaged correlation coefficients.
    """
    data = np.asarray(data)
    [n_epochs, n_channels, n_times] = np.shape(data)
    result = np.zeros((n_epochs, n_channels, n_times))
    latencies = np.zeros((n_epochs, n_channels))
    avg_correlations = np.zeros(n_channels)
    for i in range(n_channels):
        result[:, i, :], latencies[:, i], avg_correlations[i] = __process_single_channel(data[:, i, :], eps,
                                                                                         max_iterations, template)
    return result, latencies, avg_correlations


def __process_single_channel(epochs: npt.ArrayLike, eps: float, max_iterations: int,
                             template: npt.ArrayLike = None) -> Tuple[np.ndarray, np.ndarray, float]:
    avg = np.mean(epochs, axis=0)
    n_epochs, n_times = np.shape(epochs)
    if template is None:  # If template is not specified use time-locked average
        logging.info('Template was not specified, using mean average.')
        template = avg
    else:
        template = np.asarray(template)
        if template.shape[0] != n_times:
            raise ValueError("Template must have the same dimensions as an epoch.")

    current_template = template
    convergence = False
    avg_corr = __get_averaged_correlation_coefficients(current_template, n_epochs, epochs)
    n_iterations = 0
    result = None
    latencies = np.zeros(n_epochs)
    correction = n_times - 1

    while (not convergence) and n_iterations < max_iterations:
        n_iterations = n_iterations + 1
        signal_bin = np.zeros((n_epochs, n_times))
        for i in range(n_epochs):
            signal = epochs[i, :]
            correlation = np.correlate(current_template, signal, 'full')
            index = np.argmax(correlation)
            shift = index - correction
            # align signal
            signal_bin[i, :] = np.roll(signal, shift)
            latencies[i] = latencies[i] + shift
        avg = np.mean(signal_bin, axis=0)
        old_avg_corr = avg_corr
        avg_corr = __get_averaged_correlation_coefficients(avg, n_epochs, signal_bin)
        if abs(avg_corr - old_avg_corr) < eps:
            convergence = True
            result = signal_bin
        else:
            current_template = avg
    logging.info('Epochs processed in {} iterations with avg correlation coefficient value {}'.format(n_iterations,
                                                                                                      avg_corr))
    return result, latencies, avg_corr


def __get_averaged_correlation_coefficients(template: npt.ArrayLike, n_epochs: int, signal_bin: npt.ArrayLike) -> float:
    matrix = np.vstack((template, signal_bin))
    corr_matrix = np.corrcoef(matrix)

    return np.sum(corr_matrix[0, 1:]) / n_epochs


if __name__ == "__main__":
    print_help()
