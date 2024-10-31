import os
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from tqdm import tqdm

class MUSDBHQ18(Dataset):
    def __init__(
        self, 
        data_loc = "/home/venkatakrishnan/Desktop/datasets/MUSDBHQ-18/", 
        train = True,
        chunk_dur = 4
    ):
        super(MUSDBHQ18, self).__init__()
        if train:
            data_loc += "train/"
        else:
            data_loc += "test/"

        self.data = []
        
        for path, _, files in tqdm(os.walk(data_loc), position=0):
            print(path)
            components = [None, None]   # Vocals, Mixture
            sr = 44100
            for file in files:
                if file=="vocals.wav":
                    components[0], sr = torchaudio.load(path+"/"+file, normalize=True, channels_first=True)
                    components[0] = Resample(orig_freq=sr, new_freq=16000)(components[0])
                elif file=="mixture.wav":
                    components[1], sr = torchaudio.load(path+"/"+file, normalize=True, channels_first=True)
                    components[1] = Resample(orig_freq=sr, new_freq=16000)(components[1])
                # elif file=="bass.wav":
                #     components[2], sr = torchaudio.load(path+"/"+file, normalize=True, channels_first=True)
                # elif file=="drums.wav":
                #     components[3], sr = torchaudio.load(path+"/"+file, normalize=True, channels_first=True)
                # elif file=="other.wav":
                #     components[4], sr = torchaudio.load(path+"/"+file, normalize=True, channels_first=True)
            
            flag = True
            for component in components:
                if component is None:
                    flag = False

            chunk_samples = sr*chunk_dur

            if flag:    # If all components have been initialized
                new_components = components
                for i, component in enumerate(components):
                    duration = component.shape[-1]
                    if duration % chunk_samples > 0:
                        extra_length_reqd = chunk_samples - (duration % chunk_samples)
                        new_components[i] = self._right_pad(
                            waveform=component, 
                            length_req=duration+extra_length_reqd
                        )
                self.add_to_data(new_components, chunk_samples, chunk_samples//2)
            
    def _right_pad(
        self, 
        waveform: torch.Tensor, 
        length_req: int
    ):
        """
            Arguments
            ---------
            waveform: Time domain waveform of audio - Shape: (num_channels, num_samples)
            length_req: Length required

        """
        length_signal = waveform.shape[-1]
        if(length_signal < length_req):
            num_missing_samples = length_req - length_signal
            last_dimension_padding = (0, num_missing_samples)
            waveform = torch.nn.functional.pad(waveform, last_dimension_padding)
        else:
            waveform = waveform[:length_req]
        assert waveform.shape[-1]==length_req
        return waveform
    
    def add_to_data(self, new_components, chunk_samples, hop_size):
        new_duration = new_components[0].shape[-1]
        pointer = 0
        while pointer<new_duration:
            self.data.append(
                (new_components[0][:, pointer : pointer + chunk_samples], 
                 new_components[1][:, pointer : pointer + chunk_samples])
            )
            pointer+=hop_size
                
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
if __name__=="__main__":
    dataset = MUSDBHQ18(train=True, chunk_dur=2)
    torch.save(dataset, "data/train.pt")

    dataset = MUSDBHQ18(train=False, chunk_dur=2)
    torch.save(dataset, "data/test.pt")
    print(len(dataset))
    print(dataset[100])


