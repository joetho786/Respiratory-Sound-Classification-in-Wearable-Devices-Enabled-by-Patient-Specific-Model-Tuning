from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm
import torch
import numpy as np
import torchvision
import torchaudio
from utils import *

class ImageLoader(Dataset):
  def __init__(self,dataset_file,dataset_dir,train_flag=True,transform=None):
    self.input_transform = transform
    self.dataset_file = dataset_file
    self.dataset_dir = dataset_dir
    # self.dataset = pd.read_csv(os.path.join(dataset_dir,dataset_file),sep="\t",header=None)
    self.dataset = dataset_file
    self.audio_data = []
    # self.train_flag = False
    # Spectrogram parameters
    self.train_flag = train_flag
    self.sample_rate = 4000
    self.desired_length = 8
    self.n_mels = 128 #128
    self.nfft = 1024 #2048
    self.win_length = int(60/1000*self.sample_rate)
    self.hop = self.win_length//2
    self.f_max = 2000
    self.device_to_files = []  # mapping the filename to device
    # self.patient_to_device = {}
    self.patient_to_samples ={}
    self.patient_to_idx = {}

    # files = os.listdir(dataset_dir)
    failed_files=[]
    print("LOADING AUDIO FILES")
    for i,f in enumerate(tqdm(self.dataset[0])):
      # idx_0: patient_id, idx_1: recording_index, idx_2:Chest location, idx_3:A cquistation mode, idx_4: device
      tokens = f.strip().split("_")
      try:
        _,annotation_data = get_annotation_data(f,dataset_dir)
        sample_data = get_samples(f,dataset_dir,annotation_data,sample_rate=self.sample_rate)
        if tokens[0] not in self.patient_to_samples.keys():
          self.patient_to_samples[tokens[0]] = sample_data
        else:
          self.patient_to_samples[tokens[0]].extend(sample_data)
        
        if tokens[0] not in self.patient_to_idx.keys():
          self.patient_to_idx[tokens[0]] = [i]
        else:
          self.patient_to_idx[tokens[0]].append(i)
        
        
        self.audio_data.extend(sample_data)
        
      except Exception as e:
        print(e)
        failed_files.append(self.dataset.iloc[i,0])
        continue

  
  def augment_audio(self,audio):
    effects = [["lowpass", "-1", "300"],
           ["speed", "0.8"],
           ["rate", f"{self.sample_rate}"],
           ["reverb", "-w"],
           ["channels", "1"],
           ]
    if self.train_flag:
      audio,sr = torchaudio.sox_effects.apply_effects_tensor(audio, self.sample_rate, effects)
    # print(audio.shape)
    return audio
    

  def __getitem__(self, index):
    if torch.is_tensor(index):
      index = index.tolist()
    audio = self.audio_data[index][0]
    # print("Augmenting data")
    audio,_ = torchaudio.sox_effects.apply_effects_tensor(audio,self.sample_rate,effects=[["channels","1"]])
    if np.random.random() > 0.5:
      audio = self.augment_audio(audio)

    # pad the audio to desired length using 
    # print("Before padding: ",audio.shape)
    if audio.shape[1] < self.desired_length*self.sample_rate:
      audio = torch.nn.functional.pad(audio,(0,self.desired_length*self.sample_rate-audio.shape[1]))
    else:
      audio = audio[:,:self.desired_length*self.sample_rate]
    audio_mel_image = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.nfft, win_length=self.win_length, hop_length=self.hop, n_mels=self.n_mels, f_max=self.f_max)(audio)
    
    # blank Region Clipping
    audio_mel_image_raw = audio_mel_image.squeeze(0).numpy()
    for row in range(audio_mel_image_raw.shape[0]):
        black_percent = len(np.where(audio_mel_image_raw[row,:]==-100)[0])/len(audio_mel_image_raw[row,:])
        if black_percent > 0.80:
            break
    audio_mel_image_raw = audio_mel_image_raw[:row+1,:]

    for column in range(audio_mel_image_raw.shape[1]):
        black_percent = len(np.where(audio_mel_image_raw[:,column]==-100)[0])/len(audio_mel_image_raw[:,column])
        if black_percent > 0.90:
            break
    
    audio_mel_image_raw = audio_mel_image_raw[:,:column+1]
    audio_mel_image = torch.from_numpy(audio_mel_image_raw).unsqueeze(0)

    
    label = self.audio_data[index][1]
    audio_mel_image = torchvision.transforms.Resize((256,256))(audio_mel_image)
  
    if self.input_transform is not None:
      audio_mel_image = self.input_transform(audio_mel_image)
    
    label = torch.from_numpy(np.array(label)).float()
    return audio_mel_image,label

  def __len__(self):
    # print("Length of dataset: ",len(self.audio_data))
    return len(self.audio_data)