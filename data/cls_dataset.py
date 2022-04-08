import torch
import torch.utils.data as data
from PIL import Image
import os
import glob
from glob import iglob
import csv
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
            return img.convert('L')

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

def get_clips(video_path, video_begin, video_end, label, view, sample_duration=16):
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

def get_test_clips(video_path, view, sample_duration=16):
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
        'view': view,
    }
    video_begin = 0
    dirListing = os.listdir(os.path(video_path,view))
    interval_len = (len(dirListing))
    num = int(interval_len / sample_duration)
    for i in range(num):
        sample_ = sample.copy()
        sample_['frame_indices'] = list(range(video_begin, video_begin + sample_duration))
        clips.append(sample_)
        video_begin += sample_duration
    if interval_len % sample_duration != 0:
        sample_ = sample.copy()
        sample_['frame_indices'] = list(range(video_begin, interval_len)) + [interval_len - 1] * (sample_duration - (interval_len - video_begin))
        clips.append(sample_)
    return clips

def listdir(path):
    """
    show every files or folders under the path folder
    """
    for f in os.listdir(path):
        yield f

def make_dataset_classification(root_path, subset, view, sample_duration, random_state, type=None):
    """
    Only be used at test time
    :param root_path: root path, e.g. "/usr/home/aicity/datasets/DAD/DAD/"
    :param subset: validation
    :param view: Dashboard | Rear | Right
    :param sample_duration: how many frames should one sample contain
    :param type: during training process: type = None
    :return: list of data samples, each sample is in form {'video':video_path, 
                                                            'label': 0/1, 
                                                            'subset': 'train'/'validation', 
                                                            'view': 'Dashboard' / 'Rear' / 'Right', 
                                                            'action': 'normal' / other anormal actions }
    """
    video_df = pd.read_csv(os.path.join(root_path,'LABEL.csv'), header=None)
    video_df.rename(columns={0:'userID', 1:'case', 2:'start', 3:'end', 4:'is_ano', 5:'cls_num'}, inplace=True)
    video_df.fillna(axis=0, method='ffill', inplace=True)
    video_df[['userID', 'case', 'start', 'end']] = video_df[['userID', 'case', 'start', 'end']].astype(int)
    
    dataset = []
    X_train = video_df[['userID', 'case', 'start', 'end', 'is_ano']]
    y_train = video_df[['cls_num']]
    _, X_test, _, y_test = train_test_split(video_df[['userID', 'case', 'start', 'end', 'is_ano']], video_df[['cls_num']], 
                                                        random_state=random_state, stratify=video_df[['cls_num']], test_size=0.2)
    if subset=='train':
        for idx, row in pd.concat([X_train, y_train], axis=1).reset_index(drop=True).iterrows():
            if idx==0:
                continue
            if row[0] != '':
                which_val_path = os.path.join(root_path, str(row[0]))
            if row[1] != '':
                video_path = os.path.join(which_val_path, str(row[1]), view)
                
            video_begin = int(row[2])
            video_end = int(row[3])
            
            label = int(row[5])
            
            clips = get_clips(video_path=video_path, video_begin=video_begin, video_end=video_end, 
                            label=label, view=view, sample_duration=sample_duration)
            
            dataset = dataset + clips
    elif subset == 'validation' and type == None:
        for idx, row in pd.concat([X_test, y_test], axis=1).reset_index(drop=True).iterrows():
            if idx==0:
                continue
            if row[0] != '':
                which_val_path = os.path.join(root_path, str(row[0]))
            if row[1] != '':
                video_path = os.path.join(which_val_path, str(row[1]), view)
                
            video_begin = int(row[2])
            video_end = int(row[3])
            
            label = int(row[5])
            
            clips = get_clips(video_path=video_path, video_begin=video_begin, video_end=video_end, 
                              label=label, view=view, sample_duration=sample_duration)
            
            dataset = dataset + clips
    else:
        print('!!!DATA LOADING FAILURE!!!CANT FIND CORRESPONDING DATA!!!PLEASE CHECK INPUT!!!')
    
    
    return dataset

def make_test_dataset(root_path, view, sample_duration):
    """
    Only be used at test time
    :param root_path: root path, e.g. "/usr/home/aicity/datasets/DAD/DAD/"
    :param subset: validation
    :param view: Dashboard | Rear | Right
    :param sample_duration: how many frames should one sample contain
    :param type: during training process: type = None
    :return: list of data samples, each sample is in form {'video':video_path, 
                                                            'label': 0/1, 
                                                            'subset': 'train'/'validation', 
                                                            'view': 'Dashboard' / 'Rear' / 'Right', 
                                                            'action': 'normal' / other anormal actions }
    """
    dataset = []
    video_list = [lists for lists in glob.iglob(os.path.join(root_path,'./*'), recursive=True)]
    for video_path in video_list:
        clips = get_test_clips(video_path=video_path, view=view, sample_duration=sample_duration)
        dataset = dataset + clips
    return dataset

class CLS(data.Dataset):
    """
    This dataset is only used at test time to genrate consecutive video samples.
    """
    def __init__(self,
                root_path,
                subset,
                view,
                sample_duration=16,
                type=None,
                random_state=42,
                get_loader=get_video,
                spatial_transform=None,
                temporal_transform=None,
                ):
        if subset == 'train' or subset=='validation' :
            self.data = make_dataset_classification(root_path=root_path, subset=subset, view=view, sample_duration=sample_duration, 
                                                                random_state=random_state, 
                                                                type=type)
        elif subset == 'test' :
            self.data = make_test_dataset(root_path=root_path, view=view, sample_duration=sample_duration)
        self.sample_duration = sample_duration
        self.subset = subset
        self.loader = get_loader
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, index):
        if self.subset == 'train':
            video_path = self.data[index]['video']
            ground_truth = self.data[index]['label']
            frame_indices = self.data[index]['frame_indices']
            #Data Augmentation During traing phase
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
            clip = self.loader(video_path, frame_indices)
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            return clip, ground_truth
        
        elif self.subset == 'validation':
            video_path = self.data[index]['video']
            ground_truth = self.data[index]['label']
            frame_indices = self.data[index]['frame_indices']
            
            clip = self.loader(video_path, frame_indices)
            
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            return clip, ground_truth
        
        elif self.subset == 'test':
            video_path = self.data[index]['video']
            frame_indices = self.data[index]['frame_indices']
            clip = self.loader(video_path, frame_indices)
            clip = [img for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            return clip
        else:
            print('!!!DATA LOADING FAILURE!!!THIS DATASET IS ONLY USED IN TESTING MODE!!!PLEASE CHECK INPUT!!!')
            
    def __len__(self):
        return len(self.data)