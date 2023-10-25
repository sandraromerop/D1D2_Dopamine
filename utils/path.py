import os


class PathConfig:
    def __init__(self, g_dir=None):
        self.save_dir = os.path.join('analysis','lhb_data')
        if g_dir is None:
            self.g_dir = os.path.join('data','osfstorage','raw_data') # modify 'data' for any path if necessary
