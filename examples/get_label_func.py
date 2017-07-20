import pandas as pd

def get_label_func(ref_file):
    """Example get_label_func for ImageDataGenerator. Reads a CSV (found in this
    directory, 'example_ref_file.csv' and converts it into a list of unique label
    names, and a mapping of file basename (without extension) to an array of 
    labels. Note this function can also return a single string label name for 
    each file."""

    df = pd.read_csv(ref_file)
    labels = pd.concat([pd.Series(row.image_name, row.tags.split(' ')) 
           for _,row in df.iterrows()]).reset_index()['index'].unique()
    labels.sort()
    label_mapping = {row.image_name:row.tags.split(' ') 
					for _,row in df.iterrows()}
    return label_mapping, list(labels)
