"""utility functions for general machine learning purposes in Pytorch"""
import torch

# general param
device_cache = None


def get_device():
    """Return the used device (either cuda or cpu) by pytorch

    Returns
    -------
    torch.device
        Can be device(cuda) or device(cpu)
    """
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        #device_cache = torch.device("cpu")
    return device_cache