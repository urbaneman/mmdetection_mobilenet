from .xml_style import XMLDataset
from .registry import DATASETS

@DATASETS.register_module
class HelmetDataset(XMLDataset):
    CLASSES = ('person', 'blue', 'white', 'yellow', 'red', 'none', 'light_jacket', 'red_life_jacket')

    def __init__(self, **kwargs):
        super(HelmetDataset, self).__init__(**kwargs)
        if 'VOC2012' in self.img_prefix:
            self.year = 2012

