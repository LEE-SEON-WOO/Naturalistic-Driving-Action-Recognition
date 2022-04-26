import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split
def pil_loader(path):
    """
    open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    :param path: image path
    :return: image data
    """
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
            # return img.convert('L')

def accimage_loader(path):
    """
    compared with PIL, accimage loader eliminates useless function within class, so that it is faster than PIL
    :param path: image path
    :return: image data
    """
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    """
    choose accimage as image loader if it is available, PIL otherwise
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def get_video(video_path, frame_indices):
    """
    generate a video clip which is a list of selected frames
    :param video_path: path of video folder which contains video frames
    :param frame_indices: list of selected indices of frames. e.g. if index is 1, then selected frame's name is "img_1.png"
    :return: a list of selected frames which are PIL.Image or accimage form
    """
    image_reader = get_default_image_loader()
    video = []
    for image_index in frame_indices:
        #image_name = 'img_' + str(image_index) + '.png' # 김종구, 우리 포맷에 맞게 image_name 수정, 이게 원본
        image_name = f'{image_index:05d}.jpg'
        image_path = os.path.join(video_path, image_name)
        img = image_reader(image_path)
        video.append(img)
    return video

def get_clips(video_path, video_begin, video_end, label, view, sample_duration):
    """
    be used when validation set is generated. be used to divide a video interval into video clips
    :param video_path: validation data path
    :param video_begin: begin index of frames
    :param video_end: end index of frames
    :param label: 1(normal) / 0(anormal)
    :param view: Dashboard / Right / Rear
    :param sample_duration: how many frames should one sample contain
    :return: a list which contains  validation video clips
    """
    clips = []
    sample = {
        'video': video_path,
        'label': label,
        'subset': 'validation',
        'view': view,
    }
    interval_len = (video_end - video_begin + 1)
    num = int(interval_len / sample_duration)
    for i in range(num):
        sample_ = sample.copy()
        sample_['frame_indices'] = list(range(video_begin, video_begin + sample_duration))
        clips.append(sample_)
        video_begin += sample_duration
    if interval_len % sample_duration != 0:
        sample_ = sample.copy()
        sample_['frame_indices'] = list(range(video_begin, video_end+1)) + [video_end] * (sample_duration - (video_end - video_begin + 1))
        clips.append(sample_)
    return clips


def listdir(path):
    """
    show every files or folders under the path folder
    """
    for f in os.listdir(path):
            yield f
            

# def class_parsing(csv_path):
#     df = pd.read_csv(csv_path, header=None)
#     df.fillna(axis=0, method='ffill', inplace=True)
#     df.rename(columns={0:'userID', 1:'case', 2:'start', 3:'end', 4:'is_ano', 5:'cls_num'}, inplace=True)
#     df[['userID', 'case']] = df[['userID', 'case']].astype(int)
#     return df
    
    
def make_dataset(root_path, subset, view, sample_duration,random_state=42, type=None):
    """
    :param root_path: root path of the dataset"
    :param subset: train / validation
    :param view: Dashboard / Rear / Right
    :param sample_duration: how many frames should one sample contain
    :param type: during training process: type = normal / anormal ; during validation or test process: type = None
    :return: list of data samples, each sample is in form {'video':video_path, 'label': 0/1, 'subset': 'train'/'validation', 'view': 'front_depth' / 'front_IR' / 'top_depth' / 'top_IR', 'action': 'normal' / other anormal actions}
    """
    video_df = pd.read_csv(os.path.join(root_path,'LABEL.csv'), header=None)
    video_df.rename(columns={0:'userID', 1:'case', 2:'start', 3:'end', 4:'is_ano', 5:'cls_num'}, inplace=True)
    video_df.fillna(axis=0, method='ffill', inplace=True)
    video_df[['userID', 'case', 'start', 'end']] = video_df[['userID', 'case', 'start', 'end']].astype(int)
    
    
    X_train, X_test, y_train, y_test = train_test_split(video_df[['userID', 'case', 'start', 'end', 'is_ano']], video_df[['cls_num']], 
                                                        random_state=random_state, stratify=video_df[['cls_num']])
    dataset=[]

    if subset == 'train' and type == 'normal':
        item = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        for idx in item.index:
            if item.at[idx, 'cls_num'] == 0:
                sample= {
                    'video' : os.path.join(root_path,str(item.at[idx, 'userID']),str(item.at[idx, 'case']),view),
                    'label' : 1,
                    'subset' : 'train',
                    'view' : view,
                    'action' : 'normal',
                }
                start = int(item.at[idx, 'start'])
                end = int(item.at[idx, 'end'])
                
                for i in range(start, end,sample_duration):
                    sample_=sample.copy()
                    sample_['frame_indices'] = list(range(i, min(end, i + sample_duration)))
                    if len(sample_['frame_indices']) < sample_duration:
                        for j in range(sample_duration-len(sample_['frame_indices'])):
                            sample_['frame_indices'].append(sample_['frame_indices'][-1])
                    dataset.append(sample_)
    elif subset == 'train' and type == 'anormal':
        item = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        for idx in item.index:
            if item.at[idx, 'cls_num'] != 0:
                
                sample= {
                        'video' : os.path.join(root_path,str(item.at[idx, 'userID']),str(item.at[idx, 'case']),view),
                        'label' : 0,
                        'subset' : 'train',
                        'view' : view,
                        'action' : item.at[idx, 'cls_num']
                }
                start = int(item.at[idx, 'start'])
                end = int(item.at[idx, 'end'])
                for i in range(start,end,sample_duration):
                    sample_=sample.copy()
                    sample_['frame_indices'] = list(range(i, min(end, i + sample_duration)))
                    if len(sample_['frame_indices']) < sample_duration:
                        for j in range(sample_duration-len(sample_['frame_indices'])):
                            sample_['frame_indices'].append(sample_['frame_indices'][-1])
                    dataset.append(sample_)
    elif subset == 'validation' and type == None:
        #load valiation data as well as thier labels
        
        for idx, row in pd.concat([X_test, y_test], axis=1).reset_index(drop=True).iterrows():
            if idx==0:
                continue
            if row[0] != '':
                which_val_path = os.path.join(root_path, str(row[0]))
            if row[1] != '':
                video_path = os.path.join(which_val_path, str(row[1]), view) 
            video_begin = int(row[2])
            video_end = int(row[3])
            if row[4] == 'N':
                label = 1
            elif row[4] == 'A':
                label = 0
            clips = get_clips(video_path, video_begin, video_end, label, view, sample_duration)
            dataset = dataset + clips
    else:
        print('!!!DATA LOADING FAILURE!!!CANT FIND CORRESPONDING DATA!!!PLEASE CHECK INPUT!!!')
    return dataset
from typing import List
from torch.utils.data import Dataset
class DAC(data.Dataset):                   
    """
    generate normal training/ anormal training/ validation dataset according to requirement
    """
    def __init__(self,
                root_path:str,
                subset:str,
                view:str,
                sample_duration:int=16,
                type:str=None,
                random_state:int=42,
                get_loader:List=get_video,
                spatial_transform=None,
                temporal_transform=None)->None:
        self.data = make_dataset(root_path, subset, view, sample_duration, random_state, type) #view 빼기!!
        self.sample_duration = sample_duration
        self.subset = subset
        self.loader = get_loader
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, index):
        if self.subset == 'train':
            video_path = self.data[index]['video']
            frame_indices = self.data[index]['frame_indices']
            #print(frame_indices)
            if self.temporal_transform:
                frame_indices = self.temporal_transform(frame_indices)
            #print(frame_indices)
            clip = self.loader(video_path, frame_indices)
            try:
                self.spatial_transform.randomize_parameters()
            except:
                pass
            clip = [self.spatial_transform(img) for img in clip]
            
            clip = torch.stack(clip, dim=0).permute(1, 0, 2, 3)     #data with shape (channels, timesteps, height, width)
            
            return clip, index
        elif self.subset == 'validation':
            video_path = self.data[index]['video']
            ground_truth = self.data[index]['label']
            frame_indices = self.data[index]['frame_indices']

            clip = self.loader(video_path, frame_indices)
            try:
                self.spatial_transform.randomize_parameters()
            except:
                pass
            clip = [self.spatial_transform(img) for img in clip]
            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

            return clip, ground_truth

        else:
            print('!!!DATA LOADING FAILURE!!!CANT FIND CORRESPONDING DATA!!!PLEASE CHECK INPUT!!!')
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    df = pd.read_csv('../A1/newFrame/LABEL.csv', header=None)
    from main import parse_args
    
    args = parse_args()

    #print(df.fillna(axis=0, method='ffill'))
    
    print("=================================Loading Normal-Driving Training Data!=================================")
    training_normal_data = DAC(root_path='../A1/newFrame',
                                subset='validation',
                                view='Rear',
                                type=None
                                )
    print(len(training_normal_data))