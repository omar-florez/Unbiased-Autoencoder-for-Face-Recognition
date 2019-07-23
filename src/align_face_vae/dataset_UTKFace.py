import numpy as np

#dataset
import pandas as pd
import matplotlib.image as mpimg
import os
from sklearn.utils import shuffle
import random

from src.vggface import VGGface

#image
from scipy.ndimage import imread

#face align
from src.align_face_vae import face_alignment
import ipdb

from keras.utils import to_categorical

use_cuda = False
device = 'cpu' if not use_cuda else 'gpu'

data_folder     = '/Users/ost437/Documents/OneDrive/workspace/WorkspaceLiClipse/ethnicGAN/'
saved_folder    = './saved/input/'
model_path      = './model/vggface.pt-adj-255.pkl'
image_path      = data_folder + './data/crop_part1'
metadata_path   = data_folder + 'data/list_attr_celeba.txt'
#size of each image for the CelebA dataset
image_size      = (256, 256)
num_workers     = 1
batch_size      = 64
output_size     = 256

class CelebaUTKDataset():
    def __init__(self, image_path, metadata_path, mode=''):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.image_format = 'jpg'
        self.rotate = False

        #df = pd.read_csv(metadata_path, sep='\s+', skiprows=1)
        #df.columns:
        #['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        # 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        # 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        # 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        # 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        # 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        # 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        # 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        # 'Wearing_Necklace', 'Wearing_Necktie', 'Young'])

        exist_files = os.path.exists(os.path.join(saved_folder, 'latino_images.npy'))
        exist_files &= os.path.exists(os.path.join(saved_folder, 'latina_images.npy'))
        exist_files = False

        if not exist_files:
            print('Build dataset files')
            output = self.build_dataset(image_path=image_path,
                                        metadata_path=metadata_path,
                                        image_size=image_size)
            self.male_images = output[0]
            self.female_images = output[1]
            self.male_images_proto = output[2]
            self.female_images_proto = output[3]
        else:
            print('Load dataset files')
            self.male_images = np.load(os.path.join(saved_folder, 'male_images.npy'))
            self.female_images = np.load(os.path.join(saved_folder, 'female_images.npy'))
            self.male_images_proto = np.load(os.path.join(saved_folder, 'male_images_proto.npy'))
            self.female_images_proto = np.load(os.path.join(saved_folder, 'female_images_proto.npy'))

    def classification_generator(self, batch_size=32, is_validation=False):
        while True:
            yield self.get_batch(batch_size=32, is_validation=is_validation)

    def reconstruction_generator(self, batch_size=32, is_validation=False):
        while True:
            x, y = self.get_batch(batch_size=32, is_validation=is_validation)
            yield (x, None)

    def get_batch(self, batch_size=32, is_validation=False):
        if not is_validation:
            index_male = range(0, int(len(self.male_images)*0.8))
            index_female = range(0, int(len(self.female_images) * 0.8))
        else:
            index_male = range(int(len(self.male_images) * 0.8), len(self.male_images))
            index_female = range(int(len(self.female_images) * 0.8), len(self.female_images))

        index_male = random.sample(index_male, batch_size//2)
        index_female = random.sample(index_female, batch_size//2)

        x_batch = np.concatenate((self.male_images[index_male], self.female_images[index_female]))
        # class 0: male, class 1: female
        y_batch = np.concatenate((np.ones(batch_size//2)*0, np.ones(batch_size//2)*1))
        y_batch = to_categorical(y_batch, num_classes=2)
        return  shuffle(x_batch, y_batch)

    # def load_and_build_lite_dataset(self, number_per_class=1000):
    #     aa = self.get_batch(batch_size=number_per_class, is_validation=False)


    #Labels:
    # The labels of each face image is embedded in the file name, formated like `[age]_[gender]_[race]_[date&time].jpg`
    #       `[age]` is an integer from 0 to 116, indicating the age
    #       `[gender]` is either 0 (male) or 1 (female)
    #       `[race]` is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
    #       `[date&time]` is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
    def build_dataset(self, image_path, metadata_path, image_size, save_vectors=True):
        exist_files = os.path.exists(os.path.join(saved_folder, 'latino_images.npy'))
        exist_files &= os.path.exists(os.path.join(saved_folder, 'latina_images.npy'))
        image_files = os.listdir(image_path)

        #get latinx images
        idx=0
        valid_male_filenames = []
        valid_female_filenames = []
        valid_filenames = []
        for root, _, files in os.walk(image_path):
            print('Reading files from %s', root)
            for file_name in files:
                if not file_name.endswith(self.image_format):
                    continue
                full_file_name = os.path.join(root, file_name)
                img = imread(full_file_name, flatten=False)
                file_name = full_file_name.split('/')[-1]
                age, gender, race, date_time = file_name.split('_')
                age, gender, race = int(age), int(gender), int(race)


                if race == 4 and gender == 0:
                    valid_male_filenames.append(file_name)
                if race == 4 and gender == 1:
                    valid_female_filenames.append(file_name)
                if race == 4:
                    valid_filenames.append(file_name)

        # male_images = [mpimg.imread(os.path.join(image_path, fn)) for fn in valid_male_filenames]
        # female_images = [mpimg.imread(os.path.join(image_path, fn)) for fn in valid_female_filenames]
        # Align faces:
        male_images = face_alignment.face_aligner([os.path.join(image_path, fn) for fn in valid_male_filenames], output_size=output_size)
        female_images = face_alignment.face_aligner([os.path.join(image_path, fn) for fn in valid_female_filenames], output_size=output_size)
        both_images = face_alignment.face_aligner([os.path.join(image_path, fn) for fn in valid_filenames], output_size=output_size)

        # set image pixels to 0-1
        male_images = np.array(male_images)/255.0
        female_images = np.array(female_images)/255.0
        both_images = np.array(both_images)/255.0

        #create prototype imagesL
        male_images_proto = np.mean(male_images, axis=0)
        female_images_proto = np.mean(female_images, axis=0)
        both_images_proto = np.mean(both_images, axis=0)

        if save_vectors:
            #save prototype images
            np.save(os.path.join(saved_folder, 'latino_images.npy'), male_images)
            np.save(os.path.join(saved_folder, 'latina_images.npy'), female_images)
            np.save(os.path.join(saved_folder, 'latinx_images.npy'), both_images)

            np.save(os.path.join(saved_folder,'latino_images_proto.npy'), male_images_proto)
            np.save(os.path.join(saved_folder,'latina_images_proto.npy'), female_images_proto)
            np.save(os.path.join(saved_folder,'latinx_images_proto.npy'), both_images_proto)
            mpimg.imsave(os.path.join(saved_folder, 'latino_images_proto.png'), male_images_proto)
            mpimg.imsave(os.path.join(saved_folder, 'latina_images_proto.png'), female_images_proto)
            mpimg.imsave(os.path.join(saved_folder, 'latinx_images_proto.png'), both_images_proto)
        return male_images, female_images, male_images_proto, female_images_proto

    def blend_image_prototype(self, cur_image, image_proto_same, image_proto_op, factor=0.0):
        scaled_image = cur_image/255.
        mpimg.imsave('./current_image.png', scaled_image)

        output_image_same = (1.0-factor)*scaled_image + factor*image_proto_same
        #output_image_same /= 255.
        mpimg.imsave('./blended_image_same.png', output_image_same)

        output_image_op = (1.0-factor)*scaled_image + factor*image_proto_op
        #output_image_op /= 255.
        mpimg.imsave('./blended_image_op.png', output_image_op)
        return scaled_image, output_image_same, output_image_op

if __name__ == "__main__":
    dataset = CelebaUTKDataset(image_path, metadata_path)
    #dataset.load_and_build_lite_dataset()










