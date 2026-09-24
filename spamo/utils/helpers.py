import torch
import importlib
import random
import os
import glob


def derangement(lst):
    assert len(lst) > 1, "List must have at least two elements."
    
    while True:
        shuffled = lst[:]
        random.shuffle(shuffled)
        if all(original != shuffled[i] for i, original in enumerate(lst)):
            return shuffled


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def instantiate_from_config(config):
    """Instantiates an object based on a configuration."""
    if 'target' not in config:
        raise KeyError('Expected key "target" to instantiate.')
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """Get an object from a string reference."""
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def create_mask(seq_lengths: list, device="cpu"):
    """Creates a mask tensor based on sequence lengths."""
    max_len = max(seq_lengths)
    mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device)[:, None]
    return mask.to(torch.bool)


def get_img_list(ds_name, vid_root, path):
    # Handling for different datasets
    if ds_name == 'Phoenix14T':
        img_path = os.path.join(vid_root, 'features', 'fullFrame-256x256px', path)
    elif ds_name == 'CSL-Daily':
        img_path = os.path.join(vid_root, 'CSL-Daily_256x256px', path)
    else:
        raise ValueError(f"Dataset {ds_name} is not supported.")
    return sorted(glob.glob(img_path))


# Credit by https://stackoverflow.com/questions/77782599/how-can-i-extract-all-the-frames-from-a-particular-time-interval-in-a-video
def read_video(fname, start_time=None, end_time=None):
    """Extracts frames from a video, optionally bounded by start and end times."""
    try:
        container = av.open(fname)
        duration = container.duration * (1 / av.time_base)
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = duration
        if start_time >= end_time:
            print("Start time must be less than end time")
            return []
        if end_time > duration:
            print("End time exceeds video duration")
            return []
        stream = container.streams.video[0]
        container.seek(int(start_time / stream.time_base), stream=stream)
        frames = []
        for frame in container.decode(stream):
            if frame.time > end_time:
                break
            elif frame.time < start_time:
                continue
            else:
                frames.append(frame.to_image())
        return frames
    except Exception as e:
        print(e)
        return []


def sliding_window_for_list(data_list, window_size, overlap_size):
    """Apply a sliding window to a list."""
    step_size = window_size - overlap_size
    windows = [data_list[i:i + window_size] for i in range(0, len(data_list), step_size) if i + window_size <= len(data_list)]
    return windows