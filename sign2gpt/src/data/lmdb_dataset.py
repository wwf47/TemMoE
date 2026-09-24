"""LMDB Video Dataset for Sign Language."""

import lmdb
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import io
import pickle
from typing import Optional, List, Dict, Union


class LMDBVideoReader:
    def __init__(self, lmdb_path: str):
        self.lmdb_path = lmdb_path
        with lmdb.open(
            path=lmdb_path,
            readonly=True,
            readahead=False,
            lock=False,
            meminit=False,
        ).begin(write=False) as txn:
            self.details = pickle.loads(txn.get(key=b"details"))

    def get_num_frames(self) -> int:
        return self.details["num_frames"]

    def get_frames(self, indices: List[int]) -> List[np.ndarray]:
        with lmdb.open(
            path=self.lmdb_path,
            readonly=True,
            readahead=False,
            lock=False,
            meminit=False,
        ).begin(write=False) as txn:
            frames = [
                np.array(Image.open(io.BytesIO(txn.get(key=f"{idx}".encode("ascii")))))
                for idx in indices
            ]
        return frames


def sample_frame_indices(
    num_frames: int,
    max_frames: int = 256,
    stride: int = 2,
    random_shift: int = 4,
    training: bool = True,
    min_frames: int = 2,  # Minimum frames to return (prevents BatchNorm issues)
) -> np.ndarray:
    if training:
        start_frame = np.random.randint(0, max(1, random_shift))
        end_frame = np.random.randint(max(1, num_frames - random_shift), num_frames + 1)
    else:
        start_frame = 0
        end_frame = num_frames

    selection_frames = np.arange(start_frame, end_frame, stride)
    if len(selection_frames) == 0:
        selection_frames = np.arange(0, num_frames)

    if len(selection_frames) > max_frames:
        selection_frames = np.random.choice(selection_frames, size=max_frames, replace=False)
        selection_frames = np.sort(selection_frames)

    # Ensure minimum number of frames (fixes edge case for normalization layers)
    if len(selection_frames) < min_frames and num_frames >= min_frames:
        selection_frames = np.arange(0, min(num_frames, min_frames))

    return selection_frames


class ISignDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str,
        lmdb_root: str,
        split: str,
        transform: Optional[object] = None,
        max_frames: int = 256,
        stride: int = 2,
        random_shift: int = 4,
        pseudo_gloss_dict: Optional[Union[Dict, str]] = None,
        min_translation_length: int = 0,  # Filter out samples with translation < this many chars
        max_samples: Optional[int] = None,  # Limit number of samples for fast training
    ):
        self.metadata_csv = metadata_csv
        self.lmdb_root = Path(lmdb_root)
        self.split = split
        self.transform = transform
        self.max_frames = max_frames
        self.stride = stride
        self.random_shift = random_shift
        self.min_translation_length = min_translation_length
        self.max_samples = max_samples

        # Load pseudo-gloss dict if path provided
        if isinstance(pseudo_gloss_dict, str):
            with open(pseudo_gloss_dict, "rb") as f:
                pseudo_gloss_dict = pickle.load(f)
        self.dict_sentence = pseudo_gloss_dict.get("dict_sentence") if pseudo_gloss_dict else None
        self.dict_lem_to_id = pseudo_gloss_dict.get("dict_lem_to_id") if pseudo_gloss_dict else None
        self.dict_lem_counter = pseudo_gloss_dict.get("dict_lem_counter") if pseudo_gloss_dict else None

        df = pd.read_csv(metadata_csv)
        videos = df[df['split'] == split].reset_index(drop=True)
        
        # Filter by minimum translation length
        if min_translation_length > 0:
            original_count = len(videos)
            videos = videos[videos['text'].str.len() >= min_translation_length].reset_index(drop=True)
            filtered_count = original_count - len(videos)
            print(f"[{split}] Filtered {filtered_count}/{original_count} samples with translation < {min_translation_length} chars")
        
        # Limit number of samples for fast training
        if max_samples is not None and max_samples > 0 and len(videos) > max_samples:
            original_count = len(videos)
            # Shuffle and take first max_samples (for reproducibility, use fixed seed based on split)
            videos = videos.sample(n=max_samples, random_state=42 if split == 'train' else 43).reset_index(drop=True)
            print(f"[{split}] Limited to {max_samples}/{original_count} samples for fast training")
        
        self.videos = videos
        self._lmdb_readers = {}

    def __len__(self) -> int:
        return len(self.videos)

    def _get_lmdb_reader(self, video_id: str) -> LMDBVideoReader:
        if video_id not in self._lmdb_readers:
            lmdb_path = self.lmdb_root / video_id
            self._lmdb_readers[video_id] = LMDBVideoReader(str(lmdb_path))
        return self._lmdb_readers[video_id]

    def __getitem__(self, idx: int) -> Dict:
        video_info = self.videos.iloc[idx]
        video_id = video_info['uid']
        sentence = video_info['text']

        reader = self._get_lmdb_reader(video_id)
        num_frames = reader.get_num_frames()

        is_training = (self.split == 'train')
        frame_indices = sample_frame_indices(
            num_frames=num_frames,
            max_frames=self.max_frames,
            stride=self.stride,
            random_shift=self.random_shift,
            training=is_training,
        )

        frames = reader.get_frames(frame_indices)

        if self.transform:
            frames = self.transform(frames, is_valid=not is_training)
        else:
            frames = torch.tensor(np.stack(frames)).float()

        output = {
            "index": torch.tensor(idx).long(),
            "frames": frames,
            "sentence": sentence,
            "file_name": video_id,
        }

        if self.dict_sentence is not None:
            pseudo_gloss_ids = []
            if sentence in self.dict_sentence:
                lems = self.dict_sentence[sentence]
                pseudo_gloss_ids = [
                    self.dict_lem_to_id[lem]
                    for lem in lems
                    if self.dict_lem_counter[lem] / len(self.dict_sentence) < 0.4
                ]
            if pseudo_gloss_ids:
                output["pseudo_gloss_ids"] = torch.tensor(pseudo_gloss_ids).long()

        return output

    def collate_fn(self, batch: List[Dict]) -> Dict:
        key_set = {k for k in batch[0].keys()}
        val_list = lambda k: [d.get(k) for d in batch if d.get(k) is not None]
        return {k: val_list(k) for k in key_set}
