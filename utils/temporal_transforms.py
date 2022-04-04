import random
import math
import numpy as np

class Compose(object):

    def __init__(self, transforms): #Random Sample
        self.transforms = transforms

    def __call__(self, frame_indices):
        for i, t in enumerate(self.transforms):
            if isinstance(frame_indices[0], list):
                next_transforms = Compose(self.transforms[i:])
                dst_frame_indices = [
                    next_transforms(clip_frame_indices)
                    for clip_frame_indices in frame_indices
                ]

                return dst_frame_indices
            else:
                frame_indices = t(frame_indices)
        return frame_indices

class LoopPadding(object):

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample
        out = frame_indices

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample

        out = frame_indices[:clip_duration]

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return selected_frames


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (clip_duration // 2))
        end_index = min(begin_index + clip_duration, vid_duration)

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return selected_frames


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample

        rand_end = max(0, vid_duration - clip_duration - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + clip_duration, vid_duration)

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return selected_frames


class TemporalSelectCrop(object):
    def __init__(self, size, downsample, number_clips=6, clip_interval=0):
        self.size = size
        self.downsample = downsample
        self.clip_duration = self.size * self.downsample
        self.number_clips = number_clips
        self.clip_interval = clip_interval

    def __call__(self, frame_indices):
        vid_duration = len(frame_indices)

        outs = []
        if (self.clip_duration + self.clip_interval) * self.number_clips - self.clip_interval <= vid_duration:
            center_index = len(frame_indices) // 2
            begin_index = max(0, center_index -
                              (self.clip_duration // 2 +
                               self.number_clips // 2 * (self.clip_duration + self.clip_interval)))
            for i in range(self.number_clips):
                end_index = min(begin_index + self.clip_duration, vid_duration)
                out = frame_indices[begin_index:end_index]
                outs.append(out)
                begin_index = end_index + self.clip_interval
        else:
            for i in range(self.number_clips):
                rand_end = max(0, vid_duration - self.clip_duration - 1)
                begin_index = random.randint(0, rand_end)
                end_index = min(begin_index + self.clip_duration, vid_duration)
                out = frame_indices[begin_index:end_index]
                outs.append(out)

        total_frames = []
        for out in outs:
            for index in out:
                if len(out) >= self.clip_duration:
                    break
                out.append(index)
            frames = [out[i] for i in range(0, self.clip_duration, self.downsample)]
            total_frames.append(frames)

        return total_frames


class TemporalBeginEndCrop(object):
    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample
        self.clip_duration = self.size * self.downsample

    def __call__(self, frame_indices):
        vid_duration = len(frame_indices)

        outs = []
        begin = frame_indices[:self.clip_duration]
        end = frame_indices[-self.clip_duration:]
        outs.append(begin)
        outs.append(end)

        total_frames = []
        for out in outs:
            for index in out:
                if len(out) >= self.clip_duration:
                    break
                out.append(index)
            frames = [out[i] for i in range(0, self.clip_duration, self.downsample)]
            total_frames.append(frames)

        return total_frames


class TemporalRandomMultipleCrop(object):
    def __init__(self, size, downsample, number_clips=4, clip_interval=-1):
        self.size = size
        self.downsample = downsample
        self.clip_duration = self.size * self.downsample
        self.number_clips = number_clips
        self.clip_interval = clip_interval

    def __call__(self, frame_indices):
        vid_duration = len(frame_indices)

        outs = []
        if self.clip_interval < 0:
            # randomly choose clips
            for i in range(self.number_clips):
                rand_end = max(0, vid_duration - self.clip_duration - 1)
                begin_index = random.randint(0, rand_end)
                end_index = min(begin_index + self.clip_duration, vid_duration)
                out = frame_indices[begin_index : end_index]
                outs.append(out)
        else:
            rand_begin = vid_duration - self.clip_duration * (self.number_clips-1) - self.clip_interval * (self.number_clips-1)
            if rand_begin < 0:
                begin_index = random.randint(0, rand_begin)
            else:
                begin_index = 0
            for i in range(self.number_clips):
                end_index = min(begin_index + self.clip_duration, vid_duration)
                out = frame_indices[begin_index : end_index]
                outs.append(out)
                begin_index = min(begin_index)

        total_frames = []
        for out in outs:
            for index in out:
                if len(out) >= self.clip_duration:
                    break
                out.append(index)
            frames = [out[i] for i in range(0, self.clip_duration, self.downsample)]
            total_frames.append(frames)

        return total_frames

class TemporalSequentialCrop(object):
    def __init__(self, duration=32, downsample=2):
        self.duration = duration
        self.downsample = downsample
        if self.duration % self.downsample != 0:
            print('Error! Sample duration should be be an integral multiple of downsample!')
            assert 0
    def __call__(self, frame_indices):
        help = []
        step = self.downsample
        for i in range(0, self.duration, step):
            help.append(frame_indices[i])
        return help

class TemporalEvenCrop(object):

    def __init__(self, size, n_samples=1):#random sample
        self.size = size
        self.n_samples = n_samples
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        n_frames = len(frame_indices)
        stride = max(
            1, math.ceil((n_frames - 1 - self.size) / (self.n_samples - 1)))

        out = []
        for begin_index in frame_indices[::stride]:
            if len(out) >= self.n_samples:
                break
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)

        return out
class SlidingWindow(object):

    def __init__(self, size, stride=0):#random sample
        self.size = size
        if stride == 0:
            self.stride = self.size
        else:
            self.stride = stride
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        out = []
        for begin_index in frame_indices[::self.stride]:
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)

        return out

class TemporalSubsampling(object):

    def __init__(self, stride):#random sample
        self.stride = stride

    def __call__(self, frame_indices):
        return frame_indices[::self.stride]


class TemporalSubsampling(object):

    def __init__(self, stride):#random sample
        self.stride = stride

    def __call__(self, frame_indices):
        return frame_indices[::self.stride]
    




