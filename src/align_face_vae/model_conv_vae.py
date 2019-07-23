'''
    Automatic bias removal for Face Recognition
    Author: Omar U. Florez
    November, 2018
''' 

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import shutil
import imageio

import keras.utils as utils
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda, Conv2DTranspose, BatchNormalization
from keras.models import Model, model_from_json
from keras import metrics
import keras.backend as K

from keras.callbacks import Callback

from src.align_face_vae import face_alignment
from src.align_face_vae.dataset_celeba import CelebaDatasetSimple
import ipdb
import warnings
warnings.filterwarnings('ignore')
import argparse
from keras.activations import softmax, relu, sigmoid
from keras.utils import plot_model

# saved_folder    = './saved'
# image_path      = data_folder + 'data/images-dpmcrop-test/'
# metadata_path   = data_folder + 'data/list_attr_celeba.txt'
#image_size      = (224, 224)
# num_workers     = 1
# batch_size      = 64
# img_rows        = 256
# img_cols        = 256
# img_chans       = 3
data_folder     = '/Users/ost437/Documents/OneDrive/workspace/datasets/celebrity/'

parser = argparse.ArgumentParser()
parser.add_argument('--saved_folder',   default='./saved')
parser.add_argument('--image_path',     default=data_folder + 'data/images-dpmcrop-test/')
parser.add_argument('--metadata_path',  default=data_folder + 'data/list_attr_celeba.txt')
parser.add_argument('--img_rows',       default=256, type=int)
parser.add_argument('--img_cols',       default=256, type=int)
parser.add_argument('--img_chans',      default=3, type=int)
parser.add_argument('--batch_size',     default=64, type=int)
parser.add_argument('--load_model',     default=False, action='store_true')
args = parser.parse_args()

# load dataset
## male_images = np.load(os.path.join(args.saved_folder, 'input', 'male_images.npy'))
# female_images = np.load(os.path.join(saved_folder, 'input', 'female_images.npy'))
male_proto = np.load(os.path.join(args.saved_folder, 'input', 'male_images_proto.npy'))
female_proto = np.load(os.path.join(args.saved_folder, 'input', 'female_images_proto.npy'))

def load_external_model():
    #10000/10000 - 4031s 403ms/step - loss: 0.3303 - acc: 0.8574 - val_loss: 0.2131 - val_acc: 0.9062
    json_file = open(os.path.join(args.saved_folder, "model", "model_gender.json"), "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model_gender = model_from_json(loaded_model_json)
    model_gender.load_weights(os.path.join(args.saved_folder, "model", "model_gender.h5"))
    return model_gender


def vae_unbias_images(vae, gender, images, file_name=None):
    indices = np.random.choice(len(images), 10)
    predicted_images = vae.predict(images[indices])
    actual_images = images[indices]

    plt.figure(figsize=(30, 8))
    for i, index in enumerate(indices):
        plt.subplot(2, len(indices), i+1), plt.xticks(()), plt.yticks(())
        predicted_image = predicted_images[i]
        prediction = gender.predict(np.array([predicted_image]))
        plt.imshow(predicted_image)
        #plt.title('Predicted class: %s Distribution: %s' % (i2label(np.argmax(prediction)), str(prediction)))
        plt.title('Distribution:\n%s' % (str(prediction)))

        plt.subplot(2, len(indices), i + 1 + len(indices)), plt.xticks(()), plt.yticks(())
        actual_image = actual_images[i]
        prediction = gender.predict(np.array([actual_image]))
        plt.imshow(actual_image)
        # plt.title('Predicted class: %s Distribution: %s' % (i2label(np.argmax(prediction)), str(prediction)))
        plt.title('Distribution:\n%s' % (str(prediction)))
    plt.savefig(os.path.join(args.saved_folder, 'output', file_name))

dataset = CelebaDatasetSimple(args.image_path, args.metadata_path)

def i2label(class_id):
    return "male" if class_id == 0 else "female"

def l2_norm(x):
    return x/K.sqrt(K.sum(x**2)+0.000001)

def var_sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#-----------------------------------------------------------------------------------------------------------------------
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
# Callbacks:
from keras.callbacks import Callback, CSVLogger, ModelCheckpoint, EarlyStopping
class ComputeMetrics(Callback):
    def on_epoch_end(self, epoch, logs):
        logs['val_metric'] = epoch ** 2  # replace it with your metrics
        if (epoch + 1) % 10 == 0:
            logs['test_metric'] = epoch ** 3  # same
        else:
            logs['test_metric'] = np.nan

class Histories(Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        #ipdb.set_trace()
        # self.losses.append(logs.get('loss'))
        # y_pred = self.model.predict(self.model.validation_data[0])
        # #self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        test_images = dataset.get_batch(batch_size=256)[0]
        file_name = 'unbias_gender_recognition_%s-%s.png'%(str(epoch), self.params['steps'])
        gender_model = load_external_model()
        vae_unbias_images(self.model, gender_model, test_images, file_name=file_name)
        del gender_model
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# Convolutional Variational Auto Encoder
def ConvolutionalVAE(img_rows, img_cols, img_chans, mse=True, external_model=None):
    filters             = 16
    kernel_size         = 3
    latent_dim          = 100
    original_img_size   = (img_rows, img_cols, img_chans)

    # ------------------------------------------------------------------------------------------------------------------
    # Encoder:
    inputs = Input(shape=original_img_size, name='encoder_input')
    x = inputs
    for i in range(2):
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)

    encoder_shape = K.int_shape(x)
    x = Flatten()(x)
    # z_mean = z_log_var = z = (None, latent_dim)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    ## use reparameterization trick to push the sampling out as input
    z = Lambda(var_sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    #encoder = Model(inputs=inputs, outputs = [z_mean, z_log_var, z], name='encoder')
    encoder = Model(inputs=inputs, outputs=z, name='encoder')
    encoder.summary()

    # ------------------------------------------------------------------------------------------------------------------
    # Decoder:
    latent_inputs = Input(shape=(latent_dim, ), name='z_sampling')
    x = Dense(encoder_shape[1]*encoder_shape[2]*encoder_shape[3], activation='relu')(latent_inputs)
    x = Reshape((encoder_shape[1], encoder_shape[2], encoder_shape[3]))(x)
    for i in range(2):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2
    outputs = Conv2DTranspose(filters=img_chans,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    decoder = Model(inputs=latent_inputs, outputs=outputs, name='decoder')
    decoder.summary()

    pred_outputs = decoder(encoder(inputs))

    reconstruction_loss = metrics.mse(K.flatten(pred_outputs), K.flatten(inputs)) * img_rows * img_cols
    # KL(p(z|x), p(z  = -1/2*(e^d*u^2)/d))
    kl_loss             = K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) * -0.5
    #vae_loss            = K.mean(sigmoid(reconstruction_loss) + sigmoid(kl_loss))
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    external_loss = 0
    if external_model:
        external_model.summary()

        # entropy_loss: our goal is to maxmimize entropy for gender prediction,
        # so we negate its original definition
        # p_x = external_model(pred_outputs)
        # entropy_gender_loss = -K.mean(-p_x * K.log(p_x))
        # vae_loss += entropy_gender_loss

        # penalize similarity between gender predictions for original and reconstructed images
        external_loss = K.mean(metrics.mse(external_model(pred_outputs), (1. - external_model(inputs))))

        #reranking loss as in Adversarial Transformation Networks
        #external_l = external_loss(external_model)

        vae_loss = vae_loss + 100.*external_loss

    vae = Model(inputs=inputs, outputs=pred_outputs, name='vae')
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder, decoder

def save_models(vae, encoder, decoder, context=''):
    vae_json = vae.to_json()
    with open(os.path.join(args.saved_folder, "model", "vae%s.json"%context), "w") as json_file:
        json_file.write(vae_json)
    vae.save_weights(os.path.join(args.saved_folder, "model", "vae%s.h5"%context))

    encoder_json = encoder.to_json()
    with open(os.path.join(args.saved_folder, "model", "encoder%s.json"%context), "w") as json_file:
        json_file.write(encoder_json)
        encoder.save_weights(os.path.join(args.saved_folder, "model", "encoder%s.h5"%context))

    decoder_json = decoder.to_json()
    with open(os.path.join(args.saved_folder, "model", "decoder%s.json"%context), "w") as json_file:
        json_file.write(decoder_json)
    decoder.save_weights(os.path.join(args.saved_folder, "model", "decoder%s.h5"%context))
    print("Saved models to disk")


def load_models(context=''):
    json_file = open(os.path.join(args.saved_folder, "model", "vae%s.json"%context), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    vae = model_from_json(loaded_model_json)
    vae.load_weights(os.path.join(args.saved_folder, "model", "vae%s.h5"%context))

    json_file = open(os.path.join(args.saved_folder, "model", "encoder%s.json"%context), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    encoder = model_from_json(loaded_model_json)
    encoder.load_weights(os.path.join(args.saved_folder, "model", "encoder%s.h5"%context))

    json_file = open(os.path.join(args.saved_folder, "model", "decoder%s.json"%context), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    decoder = model_from_json(loaded_model_json)
    decoder.load_weights(os.path.join(args.saved_folder, "model", "decoder%s.h5"%context))

    #10000/10000 - 4031s 403ms/step - loss: 0.3303 - acc: 0.8574 - val_loss: 0.2131 - val_acc: 0.9062
    json_file = open(os.path.join(args.saved_folder, "model", "model_gender.json"), "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model_gender = model_from_json(loaded_model_json)
    model_gender.load_weights(os.path.join(args.saved_folder, "model", "model_gender.h5"))

    return vae, encoder, decoder, model_gender



def unbias_images_original(encoder, decoder, gender_model, images, file_name=None):
    def softmax(x):
        return np.exp(x) / (0.00001 + np.sum(np.exp(x)))

    indices = np.arange(len(images))[:10]
    #predicted_images = decoder.predict(encoder.predict(images[indices])[2])
    predicted_images = decoder.predict(encoder.predict(images[indices]))
    actual_images = images[indices]

    plt.figure(figsize=(30, 8))
    for i, index in enumerate(indices):
        plt.subplot(2, len(indices), i+1), plt.xticks(()), plt.yticks(())
        predicted_image = predicted_images[i]
        prediction = softmax(gender_model.predict(np.array([predicted_image])))
        plt.imshow(predicted_image)
        #plt.title('Predicted class: %s Distribution: %s' % (i2label(np.argmax(prediction)), str(prediction)))
        plt.title('Classification:\nFemale: %.2f%%\nMale: %.2f%%' % (100. * prediction[0][1], 100. * prediction[0][0]))

        plt.subplot(2, len(indices), i + 1 + len(indices)), plt.xticks(()), plt.yticks(())
        actual_image = actual_images[i]
        prediction = softmax(gender_model.predict(np.array([actual_image])))
        plt.imshow(actual_image)
        # plt.title('Predicted class: %s Distribution: %s' % (i2label(np.argmax(prediction)), str(prediction)))
        plt.title('Classification:\nFemale: %.2f%%\nMale: %.2f%%' % (100. * prediction[0][1], 100. * prediction[0][0]))
    plt.savefig(os.path.join(args.saved_folder, 'output', file_name))
    return

def vae_morph(images_):
    faces_len = np.min([20,images_.shape[0]])
    pb = display.ProgressBar(faces_len)
    pb.display()
    os.remove('movie.gif')
    with imageio.get_writer('movie.gif', mode='I') as writer:
        for f in range(faces_len - 1):
            A = encoder.predict(images_[[f,f+1]]/256.)
            for i,a in enumerate(np.linspace(0,1,7)):
                blend_latent = A[0][np.newaxis,0] * (1. - a) + A[0][np.newaxis,1] * a
                blended = np.squeeze(generator.predict(blend_latent))
                writer.append_data((blended * 255.).astype(np.uint8))
            pb.progress = f+1
    display.clear_output(wait=True)
    return HTML('<img src="movie.gif?%s">'%(random.randint(0,1000)))

def morph_random_faces(encoder, decoder, male_images, file_name=None):
    # generate one embeddeding per observation: len(idx)=2
    # A = 3x(2, latent_dim)
    idx = np.random.randint(male_images.shape[0], size=(2,))
    A = encoder.predict(male_images[idx])

    # morph between faces
    plt.figure(figsize=(30, 8))
    plt.subplot(1, 7, 1), plt.xticks(()), plt.yticks(()), plt.imshow(male_images[idx[0]])
    plt.subplot(1, 7, 7), plt.xticks(()), plt.yticks(()), plt.imshow(male_images[idx[1]])
    ipdb.set_trace()
    # blend the embedding vectors with different proportions
    for i, alpha in enumerate(np.linspace(0, 1, 5)):
        embedding_a = A[0][np.newaxis, 0]
        embedding_b = A[0][np.newaxis, 1]
        blend = embedding_a * (alpha) + embedding_b * (1. - alpha)
        # reconstruction = (1, 256, 256, 3)
        reconstruction = decoder.predict(blend)
        plt.subplot(1, 7, i + 2), plt.xticks(()), plt.yticks(())
        plt.imshow(np.squeeze(reconstruction))
    file_name = file_name if file_name else 'morph_random_faces.png'
    plt.savefig(os.path.join(args.saved_folder, 'output', file_name))

def morph_two_faces(encoder, decoder, face_a, face_b, file_name=None):
    faces = np.array([face_a, face_b])
    A = encoder.predict(faces)

    # morph between faces
    plt.figure(figsize=(30, 8))
    plt.subplot(1, 7, 1), plt.xticks(()), plt.yticks(()), plt.imshow(face_a)
    plt.subplot(1, 7, 7), plt.xticks(()), plt.yticks(()), plt.imshow(face_b)
    # blend the embedding vectors with different proportions
    for i, alpha in enumerate(np.linspace(0, 1, 5)):
        embedding_a = A[0][np.newaxis, 0]
        embedding_b = A[0][np.newaxis, 1]
        blend = embedding_a * (1.-alpha) + embedding_b * (alpha)
        # reconstruction = (1, 256, 256, 3)
        reconstruction = decoder.predict(blend)
        plt.subplot(1, 7, i + 2), plt.xticks(()), plt.yticks(())
        plt.imshow(np.squeeze(reconstruction))
    file_name = file_name if file_name else 'morph_two_faces.png'
    plt.savefig(os.path.join(args.saved_folder, 'output', file_name))

def run_autoencoder():
    # check for male images
    #for i in range(min(28, male_images.shape[0])):
    #    plt.subplot(4, 7, i+1), plt.xticks(()), plt.yticks(())
    #    plt.imshow(male_images[i])
    #plt.savefig(os.path.join(args.saved_folder, 'input', 'male_images.png'))

    ckpt_dir = './'
    checkpoint_path = os.path.join(ckpt_dir, "checkpoint-{epoch:02d}-{val_loss:.6f}.hdf5")
    #checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode="min")
    stahp = EarlyStopping(min_delta=0.0001, patience=5)
    histories = Histories()

    K.clear_session()
    if args.load_model:
        print("Load json and create model")
        vae, encoder, decoder, gender_model = load_models(context='_reranking')
    else:
        gender_model = load_external_model()
        # freeze gender_model layers
        for l in gender_model.layers: l.trainable = False
        vae, encoder, decoder = ConvolutionalVAE(args.img_rows, args.img_cols,
                                                 args.img_chans, external_model=gender_model)

        ##vae.fit(male_images, epochs=20, batch_size=32, validation_split=0.1)
        history = vae.fit_generator(dataset.reconstruction_generator(is_validation=False),
                                    steps_per_epoch=40,
                                    epochs=100,
                                    validation_data=dataset.reconstruction_generator(is_validation=True),
                                    validation_steps=1,
                                    callbacks=[histories])
        ipdb.set_trace()
        save_models(vae, encoder, decoder, context='_reranking')
        #plot_model(vae, to_file="./model.svg", show_layer_names=True, show_shapes=True, rankdir="TB")

    #test autoencoder with different faces
    test_images = dataset.get_batch(batch_size=256)[0]
    ipdb.set_trace()
    morph_random_faces(encoder, decoder, test_images)
    unbias_images_original(encoder, decoder, gender_model, test_images, file_name='unbias_gender_recognition.png')
    vae_unbias_images(vae, gender_model, test_images, file_name='unbias_gender_recognition.png')
    #morph_two_faces(encoder, decoder, male_images[0], female_proto, file_name='morph_male_femaleproto.png')
    #morph_two_faces(encoder, decoder, male_images[0], male_proto, file_name='morph_male_maleproto.png')

    indices = np.random.choice(len(test_images), 10)
    run_gender_detection(gender_model, test_images[indices], file_name='gender_prediction.png')
    return

def run_gender_detection(model, images, file_name='gender_prediction.png'):
    plt.figure(figsize=(30, 8))
    for index in range(len(images)):
        plt.subplot(1, len(images), index+1), plt.xticks(()), plt.yticks(())
        prediction = model.predict(np.array([images[index]]))
        plt.imshow(images[index])
        #plt.title('Predicted class: %s\nDistribution: ' %(i2label(np.argmax(prediction)), str(prediction)))
        plt.title('Distribution:\n%s' %str(prediction))
    plt.savefig(os.path.join(args.saved_folder, 'output', file_name))


#images contains two faces used as start/end states of the transition
def gif_vae_morph(images):
    faces_len = np.min(20, len(images))


# http://www.morethantechnical.com/2018/05/20/aligning-faces-with-py-opencv-dlib-combo/
# Run:
#       python src/align_face_vae/model_conv_vae.py --load_model
if __name__ == '__main__':
    run_autoencoder()
    #run_gender_detection(male_images, file_name='gender_prediction_males.png')