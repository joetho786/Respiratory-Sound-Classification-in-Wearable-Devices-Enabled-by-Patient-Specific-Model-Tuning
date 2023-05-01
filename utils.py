import os
import cv2
import pandas as pd
import os
from tqdm.notebook import tqdm as tqdm

import torchaudio

def get_annotation_data(file_name,data_dir): # Function returns the recording_data and the annotation_data
  '''
  Parameters
  ----------
  file_name: the file name of which data is to be retrived
  data_dir: Directory where file is present
  '''
  annotation_data = pd.read_csv(os.path.join(data_dir,file_name+".txt"),sep="\t")
  annotation_data.columns = ["start","end","crackle","wheeze"]
  file_data = file_name.split("_")
  recording_data = pd.DataFrame([file_data],columns = ["pid","recording_index","chest_location","acquisition_mode","equipment"])
  return recording_data,annotation_data 


def save_images(image, train_flag):
    save_dir = 'dump_image'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    if train_flag:
        save_dir = os.path.join(save_dir, 'train')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, image[1]+'_'+str(image[2])+'_'+str(image[3])+'_'+str(image[4])+'.jpg'), cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR))
    else:
        save_dir = os.path.join(save_dir, 'test')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, image[1]+'_'+str(image[2])+'_'+str(image[3])+'_'+str(image[4])+'.jpg'), cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR))

def get_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0:
        return 0
    elif crackle == 1 and wheeze == 0:
        return 1
    elif crackle == 0 and wheeze == 1:
        return 2
    else:
        return 3

def get_samples(file_name,dataset_dir,annotation_data,sample_rate):
  samples = []
  audio_file,original_sr = torchaudio.load(os.path.join(dataset_dir,file_name+".wav"))
  audio_file = torchaudio.transforms.Resample(original_sr, sample_rate)(audio_file)
#   print("Audio Chunk shape: ",audio_file.shape)
#   print("Annotation Data: ",annotation_data)
  for i in range(len(annotation_data.index)):
        row = annotation_data.loc[i]
        # print("Row: ",row)
        start = row['start']
        # print("Start: ",start)
        end = row['end']
        crackles = row['crackle']
        wheezes = row['wheeze']
        max_ind = audio_file.shape[1]
        # print("Max Ind",max_ind) 
        # split signal
        start_ind = min(int(start * sample_rate), max_ind)
        end_ind = min(int(end * sample_rate), max_ind)
        # print("Start Ind",start_ind,"End Ind",end_ind)
        audio_chunk = audio_file[:,start_ind:end_ind]
        # print("Audio Chunk",audio_chunk.shape)
        samples.append((audio_chunk, get_label(crackles, wheezes), start,end))
  return samples

def get_train_test_names(train_test_file_names):
    '''
    Get List of file names belonging in train and test datasets
    ---------
    train_test_file_names:  txt file containing names of all samples in train
                            and test as given by ICBHI
    return: train_names,test_names
    '''
    train_test = pd.read_csv(train_test_file_names,sep="\t",header=None)
    train_names = train_test[train_test[1]=="train"]
    test_names = train_test[train_test[1]=="test"]
    return train_names,test_names



