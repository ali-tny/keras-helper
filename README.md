# keras-helper: helpers for keras
In progress, obviously.
#### Contents:
- *PersistentHistory*: a class to replace the Keras History callback. Remembers 
across training sessions, and has methods to load, save and plot.
- *ImageDataGenerator*: a class to replace (in some cases) the Keras 
ImageDataGenerator. Allows the user to provide functions to get labels for each 
file (for eg, from a reference csv) and allows the user to define their own 
image augmentation.
#### Todo:
1. Document the two 
2. Add examples for *ImageDataGenerator* callback functions
3. Add some tests
4. Write some submission making helpers (if a general one is possible)
