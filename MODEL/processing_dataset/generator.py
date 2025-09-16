import os
import math
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from MODEL.constants import SPLIT, PREPROCESS
from MODEL.processing_dataset.reader import IAMReader, xmlpath2npypath
from MODEL.decoding.decoding_converter import mlf2label, mlf2txt


class IAMDataset(Dataset):

    def __init__(
        self,
        split=SPLIT.ALL,
        npz: bool = False,
        preprocess: Optional[int] = None,
        npz_dir: Optional[str] = None,
        preload: bool = True,
        pad_to: Optional[Tuple[int, int]] = None,
        inout_ratio: int = 4,
        pred: bool = False,
    ):
        super().__init__()
        reader = IAMReader(split)
        self.samples = reader.get_samples()
        self.n = len(self.samples)
        self.npz = bool(npz)
        self.preprocess = preprocess
        self.npz_dir = npz_dir if npz_dir is not None else "npz-" + str(preprocess)
        self.preload = bool(preload)
        self.pad_to = pad_to
        self.inout_ratio = int(inout_ratio)
        self.pred = bool(pred)
        self._xs: Optional[List[np.ndarray]] = []
        self._ys: Optional[List] = []

        if self.preload:
            if not self.npz:
                print("Preprocessing dataset on the fly (this may take time)...")
                preprocess_scheme = None
                if self.preprocess is not None:
                    preprocess_scheme = getattr(
                        PREPROCESS, "SCHEME" + str(self.preprocess)
                    )
                for s in tqdm(self.samples, desc="Preparing samples"):
                    x = (
                        s.generate_features(preprocess_scheme)
                        if preprocess_scheme is not None
                        else s.generate_features()
                    )
                    x = np.asarray(x)
                    y = s.ground_truth
                    self._xs.append(x)
                    self._ys.append(y)
                    npz_path = xmlpath2npypath(s.xml_path, npz_dir=self.npz_dir)
                    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
                    np.savez_compressed(npz_path, x=x, y=y)
            else:
                print("Loading preprocessed npz files...")
                for s in tqdm(self.samples, desc="Loading npz"):
                    npz_path = xmlpath2npypath(s.xml_path, self.npz_dir)
                    data = np.load(npz_path)
                    self._xs.append(np.asarray(data["x"]))
                    self._ys.append(data["y"])
        else:
            self._xs = None
            self._ys = None

    def __len__(self):
        return self.n

    def _load_one(self, idx: int):
        """Return (x: np.ndarray (T,feat), labels: np.ndarray (L,), sample_obj) loaded or computed for sample idx."""
        sample = self.samples[idx]
        if self._xs is not None:
            x = self._xs[idx]
            y_mlf = self._ys[idx]
        else:
            if self.npz:
                npz_path = xmlpath2npypath(sample.xml_path, self.npz_dir)
                data = np.load(npz_path)
                x = np.asarray(data["x"])
                y_mlf = data["y"]
            else:
                preprocess_scheme = None
                if self.preprocess is not None:
                    preprocess_scheme = getattr(
                        PREPROCESS, "SCHEME" + str(self.preprocess)
                    )
                x = (
                    sample.generate_features(preprocess_scheme)
                    if preprocess_scheme is not None
                    else sample.generate_features()
                )
                y_mlf = sample.ground_truth

        labels = mlf2label(y_mlf, multiple=False)
        labels = np.asarray(labels, dtype=np.int64)
        return np.asarray(x, dtype=np.float32), labels, sample

    def __getitem__(self, idx: int):
        x, labels, sample = self._load_one(idx)
        if self.pred:
            return x, sample
        x_len = x.shape[0]
        y_len = labels.shape[0]
        return x, labels, x_len, y_len, sample

    def sample_at_idx(self, idx: int, pad: int = 10):
        x, labels, sample = self._load_one(idx)
        padded = pad_2d(x, pad_to=x.shape[0] + pad, pad_value=0.0)
        return np.asarray([padded]), mlf2txt(sample.ground_truth), sample.pointset

    def random_sample(self, pad: int = 10):
        return self.sample_at_idx(np.random.randint(0, self.n), pad=pad)


def make_iam_collate_fn(
    inout_ratio: int = 4,
    pad_value_x: float = 0.0,
    pad_value_y: int = -1,
    fixed_pad: Optional[Tuple[int, int]] = None,
):
    def collate_fn(batch):
        if len(batch) == 0:
            return {}

        first = batch[0]
        pred_mode = len(first) == 2

        if pred_mode:
            xs, samples = zip(*batch)
            lengths = [int(x.shape[0]) for x in xs]
            max_T = max(lengths) if fixed_pad is None else fixed_pad[0]
            feat = xs[0].shape[1]
            batch_size = len(xs)
            inputs = np.full((batch_size, max_T, feat), pad_value_x, dtype=np.float32)
            for i, x in enumerate(xs):
                inputs[i, : x.shape[0], :] = x
            return {
                "inputs": torch.from_numpy(inputs),  # (B, T, feat)
                "input_lengths": torch.LongTensor(lengths),  # (B,)
                "samples": list(samples),
            }

        xs, labels_list, x_lens, y_lens, samples = zip(*batch)
        batch_size = len(xs)
        x_lens = [int(v) for v in x_lens]
        y_lens = [int(v) for v in y_lens]

        max_T = max(x_lens) if fixed_pad is None else fixed_pad[0]
        if fixed_pad is not None:
            y_pad = fixed_pad[1]
        else:
            y_pad = int(math.ceil(max_T / float(inout_ratio)))

        feat = xs[0].shape[1]

        inputs = np.full((batch_size, max_T, feat), pad_value_x, dtype=np.float32)
        for i, x in enumerate(xs):
            inputs[i, : x.shape[0], :] = x

        if len(labels_list) > 0:
            targets = np.concatenate(
                [np.asarray(lbl, dtype=np.int64) for lbl in labels_list]
            ).astype(np.int64)
        else:
            targets = np.array([], dtype=np.int64)

        labels_padded = np.full((batch_size, y_pad), pad_value_y, dtype=np.int64)
        for i, lbl in enumerate(labels_list):
            L = min(len(lbl), y_pad)
            if L > 0:
                labels_padded[i, :L] = np.asarray(lbl, dtype=np.int64)

        input_lengths = torch.LongTensor(x_lens)
        ypred_lengths = torch.LongTensor(
            [((l + inout_ratio - 1) // inout_ratio) for l in x_lens]
        )
        target_lengths = torch.LongTensor([len(lbl) for lbl in labels_list])
        targets_tensor = (
            torch.LongTensor(targets) if targets.size > 0 else torch.LongTensor([])
        )

        batch_dict = {
            "inputs": torch.from_numpy(inputs),  # (B, T, feat)
            "input_lengths": input_lengths,  # (B,)
            "ypred_lengths": ypred_lengths,  # (B,)
            "targets": targets_tensor,  # (sum_targets,)
            "target_lengths": target_lengths,  # (B,)
            "labels_padded": torch.from_numpy(labels_padded),  # (B, y_pad)
            "samples": list(samples),
        }
        return batch_dict

    return collate_fn


def pad_2d(x: np.ndarray, pad_to: int, pad_value: float = 0.0):
    result = np.ones((pad_to, x.shape[1]), dtype=np.float32) * pad_value
    result[: x.shape[0], :] = x
    return result


def pad_1d(x: np.ndarray, pad_to: int, pad_value: int = -1):
    result = np.ones((pad_to,), dtype=np.int64) * pad_value
    result[: x.shape[0]] = x
    return result
