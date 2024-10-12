import os
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from collections.abc import Callable
from pathlib import Path
from scipy.signal._short_time_fft import ShortTimeFFT
from scipy.signal import resample
from scipy.signal.windows import hamming


def convert_bytes_to_time(filename: str, n_channels: int, fs: float) -> int:
    return os.stat(filename).st_size / n_channels / fs / 8

class EEGED(Dataset):
    r"""Dataset of EEG recordings, which are split based on the subject, substance and time."""
    def __init__(self, folder_in: str, negative_mode: str = 'hard',
                 max_positive_delta_t: float = 15.0, fs: float = 250.0,
                 sample_duration: float = 5.0,
                 signal_transformation_function: str = 'identity') -> None:
        r"""
        Args:
            folder_in (str): Data folder.
            negative_mode (str, optional): Negative mining strategy. Defaults to `'hard'`.
            max_positive_delta_t (float, optional): Maximal distance (in seconds) of the positive sample from the anchor. Defaults to `15.0`.
            fs (float, optional): Sampling rate of the signal in Hz. Defaults to `250`.
            sample_duration (float, optional): Duration of the sample in seconds. Defaults to `5`.
        """
        super().__init__()
        
        possible_negative_modes = ['easy', 'hard', 'all']
        assert negative_mode in possible_negative_modes, f"Mode must be one of the {possible_negative_modes}"
        
        self.max_positive_delta_t = max_positive_delta_t
        self.fs = fs
        self.sample_duration = int(sample_duration * fs)
        self.time_reserve = self.sample_duration + self.max_positive_delta_t + 1
        
        self.folder_in = folder_in
        self.filenames = os.listdir(folder_in)
        self.files = [os.path.join(folder_in, f) for f in self.filenames]
        
        # Filter out short signals
        thr = 100
        lengths = [convert_bytes_to_time(f, 18, fs) for f in self.files]
        possible_to_use = [l > thr for l in lengths]
        self.filenames = [f for f, p in zip(self.filenames, possible_to_use) if p]
        self.files = [f for f, p in zip(self.files, possible_to_use) if p]
        
        # Mining values
        self.ids   = [f.split("_")[0] for f in self.filenames]
        self.drugs = [f.split("_")[1] for f in self.filenames]
        self.times = [f.split("_")[2] for f in self.filenames]
        
        self.possible_drugs = np.unique(self.drugs).tolist()
        
        # Domain adaptation variables
        self.anchor_id = None
        self.negative_id = None
        self.anchor_class = None
        self.negative_class = None
        
        match negative_mode:
            case 'easy':
                self.mining_function = self._negative_strategy_easy
            case 'hard':
                self.mining_function = self._negative_strategy_hard
            case 'all':
                self.mining_function = self._negative_strategy_all
            case _:
                raise NotImplementedError
        
        match signal_transformation_function:
            case 'identity':
                self.transformation_function = self._transform_identity
            case 'spectrogram':
                hop_time = 0.25 # Calculated every 100 miliseconds
                self.fs_new = 50
                self.stft = ShortTimeFFT(hamming(int(self.fs_new * sample_duration)), int(self.fs_new * hop_time), self.fs_new, scale_to='psd')
                self.transformation_function = self._transform_spectrogram
            case _:
                raise NotImplementedError


    def _transform_identity(self, signal: ndarray) -> ndarray:
        return signal

    def _transform_spectrogram(self, signal: ndarray) -> ndarray:
        try:
            signal = resample(signal, int(len(signal) * (self.fs_new / self.fs_new)))
            S = np.abs(np.clip(self.stft.spectrogram(signal), 0, 15)).astype(float)
            S = S[:, :150, :]
            S = S.reshape((-1, S.shape[-1]))
            return S
        except ValueError:
            return np.empty((0, 0))
    
 
    def _load(self, index: int = -1, path: str = "") -> ndarray:
        def assign_id(subject_id):
            si = int(subject_id) - 1
            if self.anchor_id == None:
                self.anchor_id = si
                self.anchor_class = self.possible_drugs.index(self.drugs[index])
            else:
                self.negative_id = si
                drug = path.split("_")[-2]
                self.negative_class = self.possible_drugs.index(drug)

        if index != -1:
            subject_id = self.ids[index]
            assign_id(subject_id)
            return np.load(self.files[index]).T
        elif path != "":
            subject_id = Path(path).stem.split("_")[0]
            assign_id(subject_id)
            return np.load(path).T
        else:
            raise RuntimeError

    def _find_anchor(self, anchor_id: int) -> tuple[ndarray, ndarray, int]:
        """Sample the anchor.

        Args:
            anchor_id (int): Position of the anchor in files.

        Returns:
            tuple[ndarray, ndarray, int]: anchor, anchor signal, anchor start time.
        """
        signal: ndarray = self._load(anchor_id)
        signal_length = signal.shape[1]
        t_anchor = np.random.randint(self.max_positive_delta_t + 1, signal_length - (2 * self.sample_duration + self.max_positive_delta_t + 1))
        anchor = signal[:, t_anchor:(t_anchor + self.sample_duration)]
        return anchor, signal, t_anchor
        
    def _negative_strategy_hard(self, anchor_subject: int, anchor_drug: str, anchor_time: int,
                                anchor_position: int, anchor_signal: ndarray) -> ndarray:
        filter_fcn = lambda f: f.split("_")[0] == anchor_subject and f.split("_")[1] == anchor_drug and f.split("_")[2] != anchor_time
        possible_files = list(filter(filter_fcn, self.filenames))
        negative_file = possible_files[np.random.randint(0, len(possible_files))]
        
        signal_negative = self._load(path=os.path.join(self.folder_in, negative_file))
        
        signal_length = signal_negative.shape[1]
        t_negative = np.random.randint(0, signal_length - self.sample_duration - 1)
        
        negative = signal_negative[:, t_negative:(t_negative + self.sample_duration)]
        return negative
    
    def _negative_strategy_easy(self, anchor_subject: int, anchor_drug: str, anchor_time: int,
                                anchor_position: int, anchor_signal: ndarray) -> ndarray:
        filter_fcn = lambda f: f.split("_")[0] != anchor_subject and f.split("_")[1] != anchor_drug
        possible_files = list(filter(filter_fcn, self.filenames))
        negative_id = np.random.randint(0, len(possible_files))
        negative_file = possible_files[negative_id]
        
        signal_negative = self._load(path=os.path.join(self.folder_in, negative_file))
        
        signal_length = signal_negative.shape[1]
        t_negative = np.random.randint(0, signal_length - self.sample_duration - 1)
        
        negative = signal_negative[:, t_negative:(t_negative + self.sample_duration)]
        return negative
    
    def _negative_strategy_all(self, anchor_subject: int, anchor_drug: str, anchor_time: int,
                                anchor_position: int, anchor_signal: ndarray) -> ndarray:
        raise NotImplementedError
    

    def _find_negative(self, anchor_id: int, time_position: int = None, anchor_signal: ndarray = None) -> ndarray:
        r"""Find a sample considered negative under the current `negative_mode`.

        Args:
            anchor_id (int): Position of the anchor in files.
            time_position (int): Time index needed for `negative_mode = hard`. 
            anchor_signal (ndarray, optional): Signal from which the anchor sample is sampled. (Usage not implemented)
        Returns:
            ndarray: Negative sample.
        """
        anchor_subject = self.ids[anchor_id]
        anchor_drug = self.drugs[anchor_id]
        anchor_time = self.times[anchor_id]
        
        return self.mining_function(anchor_subject, anchor_drug, anchor_time,
                                    time_position, anchor_signal)
        
    def _find_positive(self, signal: ndarray, t_anchor: int, anchor_id: int = -1) -> ndarray:
        """Find a sample considered positive under the current `positive_mode`. # TODO Implement more options

        Args:
            signal (ndarray): Signal from which the anchor sample is sampled.
            t_anchor (int): Time stamp of the anchor
            anchor_id (int, optional): Position of the anchor file. Defaults to -1. (Future implementation)

        Returns:
            ndarray: Positive sample.
        """
        if np.random.uniform(0, 1) >= 0.5:
            time_delta = self.sample_duration + np.random.randint(0, self.max_positive_delta_t)
        else:
            time_delta = np.random.randint(-self.sample_duration - self.max_positive_delta_t, -self.sample_duration - 1)
        t_positive = t_anchor + time_delta
 
        positive = signal[:, t_positive:(t_positive + self.sample_duration)]
        return positive   
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, int, int, int, int, int, int]:
        """Load and sample signals to get triplets.

        Args:
            index (int): Position of the anchor.

        Returns:
            tuple[Tensor, Tensor, Tensor]: [anchor, positive, negative] samples.
        """
        anchor, anchor_signal, anchor_start = self._find_anchor(index)
        positive = self._find_positive(anchor_signal, anchor_start, index)
        negative = self._find_negative(index, anchor_start, anchor_signal)
        
        # Apply additional transofrmations (if not `identity`)
        anchor = self.transformation_function(anchor)
        positive = self.transformation_function(positive)
        negative = self.transformation_function(negative)
        
        # Prepare for torch
        anchor = torch.from_numpy(anchor).float()
        positive = torch.from_numpy(positive).float()
        negative = torch.from_numpy(negative).float()
        
        # Domain specific variables
        subject_anchor_id = self.anchor_id
        subject_positive_id = subject_anchor_id
        subject_negative_id = self.negative_id
        
        class_anchor = self.anchor_class
        class_positive = self.anchor_class
        class_negative = self.negative_class
        
        
        # Reset for next triplet
        self.anchor_id = None
        self.negative_id = None
        self.anchor_class = None
        self.negative_class = None
        
        return anchor, positive, negative, subject_anchor_id, subject_positive_id, subject_negative_id, class_anchor, class_positive, class_negative


def collate(batch: tuple[tuple[Tensor, Tensor, Tensor, int, int, int, int, int, int]]):
    n_batch = len(batch)
    usable = n_batch * [True]
    remove_needed = False
    A = [b[0] for b in batch]
    P = [b[1] for b in batch]
    N = [b[2] for b in batch]
    
    sA = [b[3] for b in batch]
    sP = [b[4] for b in batch]
    sN = [b[5] for b in batch]
    
    cA = [b[6] for b in batch]
    cP = [b[7] for b in batch]
    cN = [b[8] for b in batch]
    
    for i in range(n_batch):
        lA = A[i].shape[1]
        lP = P[i].shape[1]
        lN = N[i].shape[1]
        if lA != lP != lN:
            usable[i] = False
            remove_needed = True
        if lA == 0:
            usable[i] == False
            remove_needed = True
    
    if remove_needed:
        A = [a for a, u in zip(A, usable) if u]
        P = [p for p, u in zip(P, usable) if u]
        N = [n for n, u in zip(N, usable) if u]
        sA = [a for a, u in zip(sA, usable) if u]
        sP = [p for p, u in zip(sP, usable) if u]
        sN = [n for n, u in zip(sN, usable) if u]
        cA = [a for a, u in zip(cA, usable) if u]
        cP = [p for p, u in zip(cP, usable) if u]
        cN = [n for n, u in zip(cN, usable) if u]

    return torch.stack(A), torch.stack(P), torch.stack(N), torch.tensor(sA), torch.tensor(sP), torch.tensor(sN), torch.tensor(cA), torch.tensor(cP), torch.tensor(cN)

def get_dataloader_EEGED(folder_in: str = "data", negative_mode: str = "hard", max_positive_delta: float = 15.0, sample_duration: float = 15.0, fs: float = 250,
                         transformation_function: str = 'identity',
                         batch_size: int = 64, shuffle: bool = True,
                         num_workers: int = 4, collate_fn: Callable = collate) -> DataLoader:

    return DataLoader(EEGED(folder_in, negative_mode, max_positive_delta, fs, sample_duration, transformation_function),
                      batch_size, shuffle,
                      num_workers=num_workers, collate_fn=collate_fn)


if __name__ == "__main__":
    loader = get_dataloader_EEGED(negative_mode='easy', transformation_function='spectrogram', batch_size=12, num_workers=0)
    
    for A, P, N, sA, sP, sN, cA, cP, cN in loader:
        print(A.shape, P.shape, N.shape, sA.shape, sP.shape, sN.shape, cA.shape, cP.shape, cN.shape)