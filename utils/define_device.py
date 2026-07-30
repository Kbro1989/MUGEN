import torch

def define_device(device=None):
    """Resolve the compute device.

    If ``device`` is given (e.g. 'cuda:1', '0', 'cpu', 'mps') it is honored
    explicitly; a bare integer/digit string is treated as a CUDA index.
    Otherwise fall back to auto-detection.
    """
    if device is not None:
        device = str(device)
        if device.isdigit():
            device = f'cuda:{device}'
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')