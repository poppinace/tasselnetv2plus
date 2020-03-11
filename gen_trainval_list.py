import os
import glob
import random

root = './data/wheat_ears_counting_dataset'
image_folder = 'images'
label_folder = 'labels'
train = 'train'
val = 'val'

train_path = os.path.join(root, train)
with open('train.txt', 'w') as f:
    for image_path in glob.glob(os.path.join(train_path, image_folder, '*.JPG')):
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.JPG', '.xml')
        f.write(im_path+'\t'+gt_path+'\n')

val_path = os.path.join(root, val)
with open('val.txt', 'w') as f:
    for image_path in glob.glob(os.path.join(val_path, image_folder, '*.JPG')):
        im_path = image_path.replace(root, '')
        gt_path = im_path.replace(image_folder, label_folder).replace('.JPG', '.xml')
        f.write(im_path+'\t'+gt_path+'\n')