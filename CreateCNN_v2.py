import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import optimizers
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import callbacks
# import time
import os
from random import shuffle
from tqdm import tqdm

# БАЗОВАЯ ДИРЕКТОРИЯ ДЛЯ КОМПЬЮТЕРА
# BASE_DIR = '/home/user/Рабочий стол/VostokCNN/Image_for_CNN_4000_image/'
# БАЗОВАЯ ДИРЕКТОРИЯ ДЛЯ НОУТБУКА
BASE_DIR = 'C:/Users/Geomags/Desktop/VostokCNN/Image_for_CNN_4000_image/'

TRAIN_DIR = BASE_DIR + 'train'
VALIDATION_DIR = BASE_DIR + 'validation'
TEST_DIR = BASE_DIR + 'test'

TRAIN_DATA_FILE = 'train_data.npy'
VALIDATION_DATA_FILE = 'validation_data.npy'

IMAGE_SIZE = [206, 398, 1]
NB_EPOCH = 50

MODEL_NAME = 'Vostok_model_v{}.json'
WEIGHTS_NAME = 'Vostok_weights_v{}.h5'

def label_img(img):
    '''conversion to one-hot array [fail, pass]'''
    word_label = img.split(' ')[0][-4:]
    if word_label == 'pass':
        return 1
    elif word_label == 'fail':
        return 0

def create_data(train_fail, validation_fail,
                    train_pass, validation_pass, rate):
    '''
    create dataset as .npy file
    :param train_fail: loading images from folder "train/fail"
    :param validation_fail: loading images from folder "validation/fail"
    :param train_pass: loading images from folder "train/pass"
    :param validation_pass: loading images from folder "validation/pass"
    :param rate: training_data = data // rate
    :return: list image arrays with labels for training
    '''
    training_fail_data = []
    training_pass_data = []

    if train_fail == True:
        for img in tqdm(os.listdir(TRAIN_DIR + '/fail'), ncols=100, desc='fail folder'):
            label = label_img(img)
            path = os.path.join(TRAIN_DIR + '/fail', img)
            img = image.load_img(path, grayscale=True)
            x = image.img_to_array(img)
            x /= 255
            training_fail_data.append([x, label])

    if validation_fail ==True:
        for img in tqdm(os.listdir(VALIDATION_DIR + '/fail'), ncols=100, desc='fail folder'):
            label = label_img(img)
            path = os.path.join(VALIDATION_DIR + '/fail', img)
            img = image.load_img(path, grayscale=True)
            x = image.img_to_array(img)
            x /= 255
            training_fail_data.append([x, label])

    if train_pass == True:
        for img in tqdm(os.listdir(TRAIN_DIR + '/pass'), ncols=100, desc='pass folder'):
            label = label_img(img)
            path = os.path.join(TRAIN_DIR + '/pass', img)
            img = image.load_img(path, grayscale=True)
            x = image.img_to_array(img)
            x /= 255
            training_pass_data.append([x, label])

    if validation_pass == True:
        for img in tqdm(os.listdir(VALIDATION_DIR + '/pass'), ncols=100, desc='pass folder'):
            label = label_img(img)
            path = os.path.join(VALIDATION_DIR + '/pass', img)
            img = image.load_img(path, grayscale=True)
            x = image.img_to_array(img)
            x /= 255
            training_pass_data.append([x, label])

    first_number_fail = int(len(training_fail_data) * rate)
    second_number_fail = len(training_fail_data) - first_number_fail

    first_number_pass = int(len(training_pass_data) * rate)
    second_number_pass = len(training_pass_data) - first_number_pass

    training_data = []
    training_data[:first_number_fail] = training_fail_data[:first_number_fail]
    training_data[first_number_fail:(first_number_fail + first_number_pass)] = training_pass_data[:first_number_pass]

    validation_data = []
    validation_data[:second_number_fail] = training_fail_data[:second_number_fail]
    validation_data[second_number_fail:(second_number_fail + second_number_pass)] = training_pass_data[:second_number_pass]

    shuffle(training_data)
    shuffle(validation_data)
    np.save(TRAIN_DATA_FILE, training_data)
    np.save(VALIDATION_DATA_FILE, validation_data)
    return training_data, validation_data

def create_model(summary):
    model = Sequential()
    model.add(Conv2D(32, (7, 7),
                     # количество каналов изображения (ПРОВЕРИТЬ!)
                     input_shape=IMAGE_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    if summary == True: print(model.summary())

    return model

def save_model_and_weights(model, save_model, version_model, save_weights, version_weights):
    if save_model == True:
        json_file = open(MODEL_NAME.format(version_model), 'w')
        json_file.write(model.to_json())
        json_file.close()

    if save_weights == True:
        model.save(WEIGHTS_NAME.format(version_weights))

# train_data, validation_data = create_data(train_fail=True, validation_fail=True,
#                                     train_pass=True, validation_pass=True, rate=0.8)
# if you have already created the dataset
train_data = np.load(TRAIN_DATA_FILE)
validation_data = np.load(VALIDATION_DATA_FILE)




# model = Sequential()
# model.add(Conv2D(32, (7, 7),
#                  # количество каналов изображения (ПРОВЕРИТЬ!)
#                  input_shape=IMAGE_SIZE, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# print(model.summary())
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['acc'])
#
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    rotation_range=40,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True)
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
#
#
# train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
#                                                     target_size=IMAGE_SIZE,
#                                                     batch_size=20,
#                                                     shuffle=True,
#                                                     class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(VALIDATION_DIR,
#                                                     target_size=IMAGE_SIZE,
#                                                     batch_size=20,
#                                                     shuffle=True,
#                                                     class_mode='binary')
#
# test_generator = test_datagen.flow_from_directory(TEST_DIR,
#                                                     target_size=IMAGE_SIZE,
#                                                     batch_size=20,
#                                                     shuffle=True,
#                                                     class_mode='binary')
#
# # Обратные вызовы
# # callbacks_list = [
# #     callbacks.EarlyStopping(
# #         monitor='val_acc',
# #         patience=5,
# #     ),
# #     callbacks.ModelCheckpoint(
# #         filepath='VostokCNN_4_206x398_4000_image.h5',
# #         monitor='val_loss',
# #         save_best_only=True,
# #     ),
# #     callbacks.ReduceLROnPlateau(
# #         monitor='val_loss',
# #         factor=0.1,
# #         patience=10,
# #     ),
# #     callbacks.TensorBoard(
# #         log_dir='my_log_dir',
# #         histogram_freq=1,
# #     )
# # ]
#
#
# history = model.fit_generator(train_generator,
#                               steps_per_epoch=160,  # =(все изображения из директории)/batch_size
#                               epochs=50,  # 3 -> 50
#                               validation_data=validation_generator,
#                               # callbacks=callbacks_list,
#                               # class_weight={0: 7.25, 1: 1},
#                               validation_steps=40)  # =(все изображения из директории)/batch_size
#
#
# json_file = open('VostokCNN_v4.json', 'w')
# json_file.write(model.to_json())
# json_file.close()
# model.save('VostokCNN_4_206x398_4000_image.h5')
#
# # model.load_weights('VostokCNN_4_206x398_4000_image.h5')
# # model.load_weights('VostokCNN_3_206x398_all_image_class_weight.h5')
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.legend()
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.legend()
# plt.show()
