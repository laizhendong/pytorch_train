from data.default_dataset import DualClassificationDataset,ClassificationDataset

class ROBBIN_BREAK_TOPBORDER(DualClassificationDataset):
    def __init__(self,**kwargs):
        super(ROBBIN_BREAK_TOPBORDER, self).__init__(merge_mode="concat",
                                                     **kwargs)
        return
    def input_channels(self):
        return 4

class ROBBIN_BREAK_TOPBORDER_BASELINE(ClassificationDataset):
    def __init__(self,**kwargs):
        super(ROBBIN_BREAK_TOPBORDER_BASELINE, self).__init__(**kwargs)
        return
    def input_channels(self):
        return 3
