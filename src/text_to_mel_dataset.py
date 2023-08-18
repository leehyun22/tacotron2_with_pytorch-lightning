import torch
import torchaudio
from torchaudio.transforms import Resample

from torch.utils.data import Dataset
from text_utils import load_filepaths_and_text, text_to_sequence

from typing import List
from omegaconf import DictConfig
import random

from audio_processing import mel_spectrogram


class TextToMelDataset(Dataset):
    """
    1) load audio, text pairs
    2) normalize text and convert embedded vector
    3) compute mel-spec from audio
    """
    def __init__(self, audio_text_path: str, hparams: DictConfig):
        self.audiopaths_text = load_filepaths_and_text(audio_text_path)
        self.data_path = hparams.data_path
        self.is_resample = hparams.is_resample
        self.hparams = hparams
        self.sampling_rate = hparams.sampling_rate

        if self.is_resample:
            self.resample = Resample(
                orig_freq=hparams.ori_sampling_rate,
                new_freq=hparams.sampling_rate
            )
        random.shuffle(self.audiopaths_text)

    def get_mel_text_pair(self, audiopath_text: List):
        audiopath, text = audiopath_text[0], audiopath_text[1]
        audiopath = self.data_path + '/' + audiopath
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_text(self, text: str):
        text_norm = torch.IntTensor(text_to_sequence(text))
        return text_norm

    def get_mel(self, audiopath: str):
        audio, sampling_rate = torchaudio.load(audiopath, normalize=True)
        # change to mono channel
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)
        if self.is_resample:
            audio = self.resample(audio)
        else:
            if sampling_rate != self.sampling_rate:
                raise ValueError(f"{sampling_rate} {self.sampling_rate} SR doesn't match target SR")

        audio = audio * 0.95
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = mel_spectrogram(
            y=audio,
            n_fft=self.hparams.melkwargs.win_length,
            num_mels=self.hparams.melkwargs.n_mels,
            sampling_rate=self.hparams.melkwargs.sampling_rate,
            hop_size=self.hparams.melkwargs.hop_length,
            win_size=self.hparams.melkwargs.win_length,
            fmin=self.hparams.melkwargs.f_min,
            fmax=self.hparams.melkwargs.f_max,
            center=False
        )
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_text[index])

    def __len__(self):
        return len(self.audiopaths_text)
