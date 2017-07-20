import os
from PIL import Image
import numpy as np
from keras.utils.np_utils import to_categorical

class ImageDataGenerator(object):

    def __init__(self, directory, img_size, batch_size=128, 
                get_label_func=None, augment_func=None, ref_file=None,
                class_mode='train'):
        
        if not os.path.isdir(directory):
            raise Exception('Directory not found.')
        if (ref_file is not None) and not os.path.isfile(ref_file):
            raise Exception('Reference file not found.')

        if class_mode not in ['train', 'test']:
            raise Exception(('class_mode "{}" not found - should be one of '
                '"train" or "test"').format(class_mode)) 
        if (class_mode == 'train') and get_label_func is None:
            raise Exception('No get_label_func provided for class_mode '
                '"train"')
        if (img_size is not tuple) and len(img_size) != 2:
            raise Exception('img_size should be a tuple of length 2')


        self.directory = directory
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment_func = augment_func
        self.class_mode = class_mode

        self.label_mapping, self.labels = get_label_func(ref_file)

        self.generator = self.create_generator()

    def __next__(self):
        return self.next()

    def next(self):
        return self.generator.next()

    def create_generator(self):
        images = []
        labels = []
        while True:
            for _,_,files in os.walk(self.directory):
                for fn in files:
                    basename = os.path.splitext(os.path.basename(fn))[0]
                    label_vec = self.onehot_labels(self.label_mapping[basename])
                    img = Image.open(self.directory+fn)
                    img.thumbnail(self.img_size) 

                    if self.augment_func is not None:
                        img = self.augment_func(img)

                    img_array = np.asarray(img.convert("RGB"),dtype=np.float32)
                    img_array = img_array/255
                    images.append(img_array)
                    labels.append(label_vec)
                    if len(images)==self.batch_size:
                        yield (np.stack(images), np.stack(labels))
                        images = []
                        labels = []
                yield (np.stack(images), np.stack(labels))
                images = []
                labels = []

    def onehot_labels(self, strings):
        vector = np.zeros(len(self.labels))
        if type(strings) is list:
            for s in strings:
                vector = self._onehot_label(vector, s)
        else:
            vector = self._onehot_label(vector, strings)
        return vector

    def _onehot_label(self, vector, string):
        try:
            idx = self.labels.index(string)
            vector += to_categorical([idx], len(self.labels)).squeeze()
            return vector
        except ValueError:
            raise Exception('Unrecognised label '+string)
