import os
import numpy as np
import imageio
import time
from scipy.ndimage import gaussian_filter

os.environ['IMAGEIO_FFMPEG_LOG_LEVEL'] = 'fatal'

class ImageStack:
    def __init__(self, filename: str, channel=None):
        self.filename = filename
        self.channel  = channel

        # read all frames into a list of float32 arrays
        reader = imageio.get_reader(filename, 'ffmpeg')
        meta   = reader.get_meta_data()
        self.fps = meta.get('fps', None)

        frames = []
        start = time.perf_counter()
        print("Reading video frames...")
        for frame in reader:
            frames.append(frame.astype(np.float32))
        reader.close()
        print(f"[completed in {time.perf_counter() - start:.2f}s]")

        # stack into shape (n_frames, H, W, C)
        self.frames = np.stack(frames, axis=0)
        self.frame_count = self.frames.shape[0]
        _, h, w, c = self.frames.shape

        # collapse channel dimension if signal is greyscale
        if c == 3 and self.channel is None:
            # compare channels of first frame to check for "fake" RGB
            f0 = self.frames[0]
            if (np.allclose(f0[...,0], f0[...,1]) and
                np.allclose(f0[...,1], f0[...,2])):
                # collapse last axis
                self.frames = self.frames[..., 0]
                c = 1

        if c == 1:
            # single channel
            self.shape = (h, w)
        else:
            # true multi‐channel
            self.shape = (h, w, c)

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        """Return frame(s) by index or slice."""
        # support negative indices
        if isinstance(idx, int):
            if idx < 0:
                idx = self.frame_count + idx
            if not (0 <= idx < self.frame_count):
                raise IndexError("Frame index out of range")
            frame = self.frames[idx]
        elif isinstance(idx, slice):
            frame = self.frames[idx]
        else:
            raise TypeError("Index must be int or slice")

        # extract channel or convert to grayscale
        if self.frames.ndim == 4:
            if self.channel is not None:
                return frame[..., self.channel]
            else:
                # mean across RGB channels
                return frame.mean(axis=-1)
        else:
            # already single‐channel
            return frame

    def pre_load_stack(self, renormalise=False, gauss_sigma=10.):
        """
        Returns a numpy array of shape (n_frames, H, W),
        optionally with background subtraction + gaussian filter.
        """
        frames = self.frames

        # applies gaussian filter to all frames 
        # helps sharpen peaks corresponding to particle positions
        # filtering logic can be changed - see https://docs.scipy.org/doc/scipy/reference/signal.html
        if renormalise:
            start = time.perf_counter()
            print("\nApplying background noise filter...")
            background = gaussian_filter(self.frames.mean(axis=0), sigma=gauss_sigma)
            frames = frames - background
            frames[frames < 0] = 0
            print(f"[completed in {time.perf_counter() - start:.2f}s]")

        return frames.astype(np.float32)

