import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, rc
import random
import warnings
import cv2
from IPython import display
from IPython.display import HTML
import os
import shutil
import time
import urllib
import zipfile
import tarfile
import keras.utils as utils
import progressbar
import imageio

import dlib
from sklearn.ensemble import IsolationForest
import ipdb

def face_aligner(face_image_paths,
                 images_directory_output=None,
                 output_size=256,
                 limit_num_faces_=None,
                 limit_num_files_=None):

    def write_faces_to_disk(directory, faces):
        print("writing faces to disk...")
        if os.path.exists(directory):
            shutil.rmtree(directory)
        print('creating output directory: %s' % (directory))
        os.mkdir(directory)
        for i in range(faces.shape[0]):
            cv2.imwrite(''.join([directory, "%03d.jpg" % i]), faces[i, :, :, ::-1])
        print("wrote %d faces" % (faces.shape[0]))

    if images_directory_output and images_directory_output[-1] != '/':
        images_directory_output += '/'

    faces = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        '/Users/ost437/Documents/OneDrive/workspace/WorkspaceLiClipse/semi-adversarial-networks/shape_predictor_68_face_landmarks.dat')

    max_val = len(face_image_paths) if limit_num_files_ is None else limit_num_files_
    pb = display.ProgressBar(max_val)
    pb.display()

    face_counter = 0
    for img_idx, img_file_path in enumerate(face_image_paths):
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(img_file_path)

        if image is None:
            continue

        image = image[:, :, ::-1]  # BGR to RGB
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)

        if len(rects) > 0:
            # loop over the face detections
            for (i, rect) in enumerate(rects):
                ######### Align with facial features detector #########

                shape = predictor(gray, rect)  # get facial features
                shape = np.array([(shape.part(j).x, shape.part(j).y) for j in range(shape.num_parts)])

                # center and scale face around mid point between eyes
                center_eyes = shape[27].astype(np.int)
                eyes_d = np.linalg.norm(shape[36] - shape[45])
                face_size_x = int(eyes_d * 2.)
                if face_size_x < 50: continue

                # rotate to normalized angle
                d = (shape[45] - shape[36]) / eyes_d  # normalized eyes-differnce vector (direction)
                a = np.rad2deg(np.arctan2(d[1], d[0]))  # angle
                ###scale_factor = float(output_size) / float(face_size_x * 2.)  # scale to fit in output_size
                scale_factor = float(output_size) / float(face_size_x)  # scale to fit in output_size
                # rotation (around center_eyes) + scale transform
                M = np.append(cv2.getRotationMatrix2D((center_eyes[0], center_eyes[1]), a, scale_factor),
                              [[0, 0, 1]], axis=0)
                # apply shift from center_eyes to middle of output_size
                M1 = np.array([[1., 0., -center_eyes[0] + output_size / 2.],
                               [0., 1., -center_eyes[1] + output_size / 2.],
                               [0, 0, 1.]])
                # concatenate transforms (rotation-scale + translation)
                M = M1.dot(M)[:2]
                # warp
                try:
                    face = cv2.warpAffine(image, M, (output_size, output_size), borderMode=cv2.BORDER_REPLICATE)
                except:
                    continue
                face_counter += 1
                face = cv2.resize(face, (output_size, output_size))
                faces.append(face)
        pb.progress = img_idx + 1

        if limit_num_faces_ is not None and faces.shape[0] > limit_num_faces_:
            break
        if limit_num_files_ is not None and img_idx >= limit_num_files_:
            break

    faces = np.asarray(faces)

    if images_directory_output:
        write_faces_to_disk(images_directory_output, faces)
    return faces

def face_data_normalizer(images_directory_input,
                         images_directory_output,
                         output_size=256,
                         align_faces_=True,
                         limit_num_faces_=None,
                         limit_num_files_=None,
                         remove_outliers_=False):
    def write_faces_to_disk(directory, faces):
        print("writing faces to disk...")
        if os.path.exists(directory):
            shutil.rmtree(directory)
        print('creating output directory: %s' % (directory))
        os.mkdir(directory)
        for i in range(faces.shape[0]):
            cv2.imwrite(''.join([directory, "%03d.jpg" % i]), faces[i, :, :, ::-1])
        print("wrote %d faces" % (faces.shape[0]))

    if images_directory_input[-1] != '/':
        images_directory_input += '/'
    if images_directory_output[-1] != '/':
        images_directory_output += '/'

    faces = []

    if os.path.exists(images_directory_output):
        print('data already preprocessed? loading preprocessed files...')
        for img_idx, img_file in enumerate(os.listdir(images_directory_output)):
            # load the input image, resize it, and convert it to grayscale
            image = cv2.imread(''.join([images_directory_output, img_file]))
            if image is None: continue
            image = image[:, :, ::-1]  # BGR to RGB
            faces.append(np.expand_dims(image, 0))
        faces = np.asarray(faces)
        print('loaded %d preprocessed images' % (faces.shape[0]))
        if remove_outliers_:
            faces, num_outliers = remove_outliers(faces)
        write_faces_to_disk(images_directory_output, faces)
        return faces

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        '/Users/ost437/Documents/OneDrive/workspace/WorkspaceLiClipse/semi-adversarial-networks/shape_predictor_68_face_landmarks.dat')

    max_val = len(os.listdir(images_directory_input)) if limit_num_files_ is None else limit_num_files_
    pb = display.ProgressBar(max_val)
    pb.display()

    face_counter = 0
    for img_idx, img_file in enumerate(os.listdir(images_directory_input)):
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(''.join([images_directory_input, img_file]))

        if image is None:
            continue

        image = image[:, :, ::-1]  # BGR to RGB
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)

        if len(rects) > 0:
            # loop over the face detections
            for (i, rect) in enumerate(rects):
                if align_faces_:
                    ######### Align with facial features detector #########

                    shape = predictor(gray, rect)  # get facial features
                    shape = np.array([(shape.part(j).x, shape.part(j).y) for j in range(shape.num_parts)])

                    # center and scale face around mid point between eyes
                    center_eyes = shape[27].astype(np.int)
                    eyes_d = np.linalg.norm(shape[36] - shape[45])
                    face_size_x = int(eyes_d * 2.)
                    if face_size_x < 50: continue

                    # rotate to normalized angle
                    d = (shape[45] - shape[36]) / eyes_d  # normalized eyes-differnce vector (direction)
                    a = np.rad2deg(np.arctan2(d[1], d[0]))  # angle
                    scale_factor = float(output_size) / float(face_size_x * 2.)  # scale to fit in output_size
                    # rotation (around center_eyes) + scale transform
                    M = np.append(cv2.getRotationMatrix2D((center_eyes[0], center_eyes[1]), a, scale_factor),
                                  [[0, 0, 1]], axis=0)
                    # apply shift from center_eyes to middle of output_size
                    M1 = np.array([[1., 0., -center_eyes[0] + output_size / 2.],
                                   [0., 1., -center_eyes[1] + output_size / 2.],
                                   [0, 0, 1.]])
                    # concatenate transforms (rotation-scale + translation)
                    M = M1.dot(M)[:2]
                    # warp
                    try:
                        face = cv2.warpAffine(image, M, (output_size, output_size), borderMode=cv2.BORDER_REPLICATE)
                    except:
                        continue
                    face_counter += 1
                    face = cv2.resize(face, (output_size, output_size))
                    faces.append(face)
                else:
                    ######### "No align" with just the detector #########
                    if rect.width() < 50: continue

                    # find scale factor
                    scale_factor = float(output_size) / float(rect.width() * 2.)  # scale to fit in output_size

                    # scale around the center of the face (shift a bit for the approximate y-position of the eyes)
                    M = np.append(cv2.getRotationMatrix2D((rect.center().x, rect.center().y - rect.height() / 6.), 0,
                                                          scale_factor), [[0, 0, 1]], axis=0)
                    # apply shift from center_eyes to middle of output_size
                    M1 = np.array([[1., 0., -rect.center().x + output_size / 2.],
                                   [0., 1., -rect.center().y + output_size / 2. + rect.height() / 6.],
                                   [0, 0, 1.]])
                    # concatenate transforms (rotation-scale + translation)
                    M = M1.dot(M)[:2]
                    try:
                        face = cv2.warpAffine(image, M, (output_size, output_size), borderMode=cv2.BORDER_REPLICATE)
                    except:
                        continue
                    face_counter += 1

                    faces.append(face)

        pb.progress = img_idx + 1

        if limit_num_faces_ is not None and faces.shape[0] > limit_num_faces_:
            break
        if limit_num_files_ is not None and img_idx >= limit_num_files_:
            break

    faces = np.asarray(faces)

    write_faces_to_disk(images_directory_output, faces)

    return faces