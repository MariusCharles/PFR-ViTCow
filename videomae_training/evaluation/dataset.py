"""
This project reuses and adapts the `VideoClsDataset` class originally introduced
in the official VideoMAE implementation (see kinetics.py).

Tong, Z., Song, Y., Wang, J., & Wang, L. (2022).
VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.
Advances in Neural Information Processing Systems (NeurIPS 2022).
arXiv preprint arXiv:2203.12602.

"""

import os
import numpy as np
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union

def encode_labels(
    labels: List[Union[int, str]]
) -> Tuple[List[int], Optional[Dict[str, int]]]:
    """
    Encode labels to integers if needed.

    Args:
        labels: List of int (already encoded) or str labels.

    Returns:
        encoded_labels: List[int]
        label_to_idx: Mapping dict if encoding was applied, else None.
    """
    arr = np.array(labels)

    if np.issubdtype(arr.dtype, np.integer):
        return list(labels), None

    unique = sorted(set(labels))
    mapping = {l: i for i, l in enumerate(unique)}
    return [mapping[l] for l in labels], mapping

class TestingVideoClsDataset(Dataset):
    """
    Video dataset for inference / embedding extraction with VideoMAE.

    This dataset:
    - Loads full videos
    - Applies temporal slicing according to clip_len and test_num_segment
    - Applies spatial multi-crop according to crop_size and test_num_crop
    - Resizes final output to output_size x output_size
    - Returns normalized tensor compatible with pretrained VideoMAE

    Output tensor shape:
        (C, T, H, W)
    """

    def __init__(
        self,
        anno_path:str,
        clip_len:int=8,
        frame_sample_rate:int=1,
        crop_size:int=224,
        test_num_segment:int=1,
        test_num_crop:int=1,
        output_size:int=224,
        sep:str=' '
    ):
        """
        Args:
            anno_path (str): Path to annotation csv file (video_path label).
            clip_len (int): Number of frames per clip.
            frame_sample_rate (int): Temporal stride between frames.
            crop_size (int): Spatial crop size before final resize.
            test_num_segment (int): Number of temporal segments.
            test_num_crop (int): Number of spatial crops.
            output_size (int): Final spatial size fed to VideoMAE.
        """
        self.anno_path = anno_path
        self.sep = sep
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.test_num_segment = max(1, test_num_segment)
        self.test_num_crop = max(1, test_num_crop)
        self.output_size = output_size

        if VideoReader is None:
            raise ImportError("decord required")

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=self.sep)
        self.dataset_samples = list(cleaned.values[:, 0])
        raw_labels = list(cleaned.values[:, 1])
        self.label_array, self.label_mapping = encode_labels(raw_labels)

        # Final resize to match VideoMAE input resolution
        self.data_resize = video_transforms.Resize(
            size=(self.output_size, self.output_size),
            interpolation='bilinear'
        )

        self.data_transform = video_transforms.Compose([
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.test_seg = []
        self.test_dataset = []
        self.test_label_array = []

        for ck in range(self.test_num_segment):
            for cp in range(self.test_num_crop):
                for idx in range(len(self.label_array)):
                    self.test_label_array.append(self.label_array[idx])
                    self.test_dataset.append(self.dataset_samples[idx])
                    self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        """
        Returns:
            tensor (C, T, H, W),
            label,
            video_path,
            temporal_segment_index,
            spatial_crop_index
        """
        sample = self.test_dataset[index]
        chunk_nb, split_nb = self.test_seg[index]
        buffer = self.loadvideo_decord(sample)

        while len(buffer) == 0:
            warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                str(self.test_dataset[index]), chunk_nb, split_nb))
            index = np.random.randint(self.__len__())
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)

        if isinstance(buffer, list):
            buffer = np.stack(buffer, 0)

        # Temporal slicing parameters
        if self.test_num_segment == 1:
            temporal_start = max((buffer.shape[0] - self.clip_len) // 2, 0)
        else:
            temporal_step = max((buffer.shape[0] - self.clip_len) / (self.test_num_segment - 1), 0)
            temporal_start = int(chunk_nb * temporal_step)
        
        # Spatial slicing parameters
        if self.test_num_crop == 1: #One center crop depending on crop_size
            spatial_start = max((max(buffer.shape[1], buffer.shape[2]) - self.crop_size) // 2, 0)
        else:
            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.crop_size) \
                                    / (self.test_num_crop - 1)
            spatial_start = int(split_nb * spatial_step)
        
        # Spatio-temporal slicing depending on which side is the shortest
        if buffer.shape[1] >= buffer.shape[2]:
            buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                   spatial_start:spatial_start + self.crop_size, :, :]
        else:
            buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                   :, spatial_start:spatial_start + self.crop_size, :]
        
        buffer = self.data_resize(buffer) #output_size x output_size
        buffer = self.data_transform(buffer)
        return (buffer, 
                self.test_label_array[index], 
                sample,
                chunk_nb, 
                split_nb)


    def loadvideo_decord(self, sample):
        """
        Load video using Decord and sample frames according to frame_sample_rate.
        Returns numpy array of shape (T, H, W, C).
        """
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
        while len(all_index) < self.clip_len:
            all_index.append(all_index[-1])
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        """Total number of (segment, crop, video) combinations."""
        return len(self.test_dataset)

