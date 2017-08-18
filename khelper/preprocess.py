import os
from PIL import Image
import numpy as np
from keras.utils.np_utils import to_categorical

class ImageDataGenerator(object):
    """Rewrite of keras.preprocessing.image.ImageDataGenerator. Allows the user
    to give a reference file to specify the class label(s), and apply custom
    data augmentation (see examples)
    
    Initialise then use as an iterator. Each call returns a tuple, with first
    element an np.array representing a batch of image data, and if applicable
    second tuple element an np.array representing a batch of labels."""

    def __init__(self, directory, img_size, batch_size=128, 
                get_label_func=None, augment_func=None, ref_file=None,
                class_mode='train'):
        """
            directory -- string directory path
            img_size -- int[] 2-tuple of image size to return 
            batch_size -- int (default 128) 
            get_label_func -- callable function to get label, takes ref_file as
                an argument (see examples) (default None)
            augment_func -- callable function to augment image, takes and
                returns Image.image (see examples) (default None)
            ref_file -- string path to a reference file to give to the
                get_label_func (default None)
            class_mode -- string 'train' or 'test' whether to call label
                function to find labels (or skip for test data)
        """

        
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

        if self.class_mode == 'train':
            self.label_mapping, self.labels = get_label_func(ref_file)

        self.generator = self.create_generator()

    def __next__(self):
        return self.next()

    def next(self):
        return self.generator.next()

    def create_generator(self):
        """Returns a generator of (np.array, np.array) tuple of stacked image
        data and stacked labels, or (np.array) stacked image data if
        class_mode=test so there are no labels."""
        images = []
        labels = []
        while True:
            for _,_,files in os.walk(self.directory):
                for fn in files:
                    img = Image.open(self.directory+fn)
                    img.thumbnail(self.img_size) 

                    if self.augment_func is not None:
                        img = self.augment_func(img)

                    img_array = np.asarray(img.convert("RGB"),dtype=np.float32)
                    img_array = img_array/255

                    basename = os.path.splitext(os.path.basename(fn))[0]
                    if self.class_mode == 'train':
                        file_labels = self.label_mapping[basename]
                        label_vec = self.onehot_labels(file_labels)
                        labels.append(label_vec)
                    images.append(img_array)

                    if len(images)==self.batch_size:
                        if self.class_mode == 'train':
                            yield (np.stack(images), np.stack(labels))
                        else: 
                            yield np.stack(images)
                        images = []
                        labels = []

                if self.class_mode == 'train':
                    yield (np.stack(images), np.stack(labels))
                else: 
                    yield np.stack(images)
                images = []
                labels = []

    def onehot_labels(self, strings):
        """Turns string(s) into a (sum of) onehot vector(s) to represent labels.
            
            strings -- string[] or string an array of string label names or a
                single string label name
        """
        vector = np.zeros(len(self.labels))
        if type(strings) is list:
            for s in strings:
                vector = self._onehot_label(vector, s)
        else:
            vector = self._onehot_label(vector, strings)
        return vector

    def _onehot_label(self, vector, string):
        """Adds a onehot vector representing a label to an existing vector."""
        try:
            idx = self.labels.index(string)
            vector += to_categorical([idx], len(self.labels)).squeeze()
            return vector
        except ValueError:
            raise Exception('Unrecognised label '+string)
