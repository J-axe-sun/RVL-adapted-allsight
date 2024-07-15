import sys
import numpy as np
import torch
import json


def save_df_as_json(df_data, save_path, file_name):
    """
    Save a pandas DataFrame as a JSON file.

    Args:
        df_data (pandas.DataFrame): The DataFrame to be saved.
        save_path (str): The path where the JSON file will be saved.
        file_name (str): The name of the JSON file.

    Returns:
        None
    """
    to_dict = {}
    for index, row in list(df_data.iterrows()):
        to_dict[index] = dict(row)
    with open(r'{}{}.json'.format(save_path, file_name), 'w') as json_file:
        json.dump(to_dict, json_file, indent=3)


def find_strings_with_substring(string_list, substring):
    """
    Find strings in a list containing a specific substring.

    Args:
        string_list (list): List of strings to search through.
        substring (str): Substring to search for in the strings.

    Returns:
        list: List of strings containing the specified substring (excluding those containing 'finetune').
    """
    result = []
    for string in string_list:
        if substring in string and 'finetune' not in string:
            result.append(string)
    return result


class ColorPrint:
    """
    A class to print colored messages to the console.

    Attributes:
        None
    """

    @staticmethod
    def print_fail(message, end='\n'):
        sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_pass(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_info(message, end='\n'):
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_bold(message, end='\n'):
        sys.stdout.write('\x1b[1;37m' + message.strip() + '\x1b[0m' + end)


def pandas_col_to_numpy(df_col):
    """
    Convert a pandas DataFrame column containing string representations of arrays to a numpy array.

    Args:
        df_col (pandas.Series): The DataFrame column containing string representations of arrays.

    Returns:
        numpy.ndarray: Numpy array containing the converted arrays.
    """
    df_col = df_col.apply(
        lambda x: np.fromstring(x.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "), sep=", "))
    df_col = np.stack(df_col)
    return df_col


def pandas_string_to_numpy(arr_str):
    """
    Convert a string representation of an array to a numpy array.

    Args:
        arr_str (str): String representation of an array.

    Returns:
        numpy.ndarray: Numpy array containing the converted array.
    """
    arr_npy = np.fromstring(arr_str.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "), sep=", ")
    return arr_npy


def medfilter(x, W=20):
    """
    Apply a median filter to a 1D numpy array.

    Args:
        x (numpy.ndarray): 1D numpy array to filter.
        W (int, optional): Window size of the median filter (default is 20).

    Returns:
        numpy.ndarray: Median filtered array.
    """
    w = int(W / 2)
    x_new = np.copy(x)
    for i in range(0, x.shape[0]):
        if i < w:
            x_new[i] = np.mean(x[:i + w])
        elif i > x.shape[0] - w:
            x_new[i] = np.mean(x[i - w:])
        else:
            x_new[i] = np.mean(x[i - w:i + w])
    return x_new


def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Inverse normalization of a torch tensor.

    Args:
        tensor (torch.Tensor): Tensor to be normalized.
        mean (tuple, optional): Mean values used for normalization (default is (0.485, 0.456, 0.406)).
        std (tuple, optional): Standard deviation values used for normalization (default is (0.229, 0.224, 0.225)).

    Returns:
        torch.Tensor: Inverse normalized tensor.
    """
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def normalize(data, mean, std, eps=1e-8):
    """
    Normalize data using mean and standard deviation.

    Args:
        data (numpy.ndarray): Data to be normalized.
        mean (numpy.ndarray): Mean values for normalization.
        std (numpy.ndarray): Standard deviation values for normalization.
        eps (float, optional): Epsilon value to avoid division by zero (default is 1e-8).

    Returns:
        numpy.ndarray: Normalized data.
    """
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    """
    Unnormalize data using mean and standard deviation.

    Args:
        data (numpy.ndarray): Data to be unnormalized.
        mean (numpy.ndarray): Mean values used for normalization.
        std (numpy.ndarray): Standard deviation values used for normalization.

    Returns:
        numpy.ndarray: Unnormalized data.
    """
    return data * std + mean


def normalize_max_min(data, dmax, dmin, eps=1e-8):
    """
    Normalize data using maximum and minimum values.

    Args:
        data (numpy.ndarray): Data to be normalized.
        dmax (numpy.ndarray): Maximum values for normalization.
        dmin (numpy.ndarray): Minimum values for normalization.
        eps (float, optional): Epsilon value to avoid division by zero (default is 1e-8).

    Returns:
        numpy.ndarray: Normalized data.
    """
    return (data - dmin) / (dmax - dmin + eps)


def unnormalize_max_min(data, dmax, dmin):
    """
    Unnormalize data using maximum and minimum values.

    Args:
        data (numpy.ndarray): Data to be unnormalized.
        dmax (numpy.ndarray): Maximum values used for normalization.
        dmin (numpy.ndarray): Minimum values used for normalization.

    Returns:
        numpy.ndarray: Unnormalized data.
    """
    dmax = np.array(dmax)
    dmin = np.array(dmin)
    return data * (dmax - dmin) + dmin


def rolling_window(a, window):
    """
    Create a rolling window view of a numpy array.

    Args:
        a (numpy.ndarray): Input numpy array.
        window (int): Size of the rolling window.

    Returns:
        numpy.ndarray: Rolling window view of the input array.
    """
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window - 1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad, mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def mean_and_plot(x, y, ax, ylabel, window=100):
    """
    Calculate the mean and standard deviation of data and plot it.

    Args:
        x (numpy.ndarray): X-axis data.
        y (numpy.ndarray): Y-axis data.
        ax (matplotlib.axes.Axes): Axes object for plotting.
        ylabel (str): Label for the Y-axis.
        window (int, optional): Window size for calculating mean and standard deviation (default is 100).

    Returns:
        None
    """
    mean = np.mean(rolling_window(y, window), axis=-1)
    std = np.std(rolling_window(y, window * 2), axis=-1)

    ax.plot(x, y, 'ko', markersize=1, alpha=0.3)
    ax.plot(x, mean, 'bo', markersize=1, alpha=0.5)
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, edgecolor='none')
    ax.set_xlabel('Force [N]')
    ax.set_ylabel(ylabel)
    ax.set_xlim([min(x), max(x)])


