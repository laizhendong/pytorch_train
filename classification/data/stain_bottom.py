from data.default_dataset import DualClassificationDataset,ClassificationDataset


class STAIN_BOTTOM(ClassificationDataset):
    def __init__(self,**kwargs):
        super(STAIN_BOTTOM, self).__init__(**kwargs)
        return
    def input_channels(self):
        return 3
