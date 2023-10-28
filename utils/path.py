import os


class PathConfig:
    def __init__(self, g_dir=None,g2_dir=None):
        self.save_dir = os.path.join('analysis','lhb_data')
        if g_dir is None:
            self.g_dir = os.path.join('data','osfstorage') # modify 'data' for any path if necessary
        if g2_dir is None:
            self.g2_dir = os.path.join('data','googledrive') # modify 'data' for any path if necessary

