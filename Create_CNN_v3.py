import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import callbacks

# БАЗОВАЯ ДИРЕКТОРИЯ ДЛЯ КОМПЬЮТЕРА
BASE_DIR = '/home/user/Рабочий стол/Dataset_Train_and_Test/'
# БАЗОВАЯ ДИРЕКТОРИЯ ДЛЯ НОУТБУКА
# BASE_DIR = 'C:/Users/Geomags/Desktop/VostokCNN/Image_for_CNN_4000_image/'

TRAIN_DIR = BASE_DIR + 'train'
VALIDATION_DIR = BASE_DIR + 'validation'
TEST_DIR = BASE_DIR + 'test'

IMAGE_SIZE = [206, 398, 1]
TARGET_SIZE = [206, 398]
NB_EPOCH = 30
BATCH_SIZE = 50

MODEL_NAME = 'Vostok_model_v{}.json'
WEIGHTS_NAME = 'Vostok_weights_v{}.h5'


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

def create_generator():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=20, #40
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       # zoom_range=0.2,
                                       horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(directory=TRAIN_DIR,
                                                        target_size=TARGET_SIZE,
                                                        color_mode='grayscale',
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                  target_size=TARGET_SIZE,
                                                                  color_mode='grayscale',
                                                                  batch_size=BATCH_SIZE,
                                                                  shuffle=True,
                                                                  class_mode='binary')

    return train_generator, validation_generator

def create_callbacks(early_stopping, model_checkpoint, reduce_lr_on_plateau, tensor_board):
    callbacks_list = []

    if early_stopping == True:
        callbacks_list.append(callbacks.EarlyStopping(monitor='val_acc', patience=5))

    if model_checkpoint == True:
        callbacks_list.append(callbacks.ModelCheckpoint(filepath='weight_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                        monitor='val_loss', save_best_only=True))

    if reduce_lr_on_plateau == True:
        callbacks_list.append(callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10))

    if tensor_board == True:
        callbacks_list.append(callbacks.TensorBoard(log_dir='log_dir', histogram_freq=1))

    return callbacks_list

def fit_model(model, train_generator, validation_generator, callbacks_list, nb_train_step, nb_val_step):
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=nb_train_step,
                                  epochs=NB_EPOCH,
                                  validation_data=validation_generator,
                                  callbacks=callbacks_list,
                                  validation_steps=nb_val_step)
    return history

def save_model_and_weights(model, save_model, version_model, save_weights, version_weights):
    if save_model == True:
        json_file = open(MODEL_NAME.format(version_model), 'w')
        json_file.write(model.to_json())
        json_file.close()

    if save_weights == True:
        model.save(WEIGHTS_NAME.format(version_weights))

def show_results(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.legend()
    plt.show()


model = create_model(summary=True)

train_generator, validation_generator = create_generator()

callbacks_list = create_callbacks(early_stopping=True, model_checkpoint=True,
                                  reduce_lr_on_plateau=True, tensor_board=False)

history = fit_model(model=model, train_generator=train_generator,
                    validation_generator=validation_generator, callbacks_list=callbacks_list,
                    nb_train_step=200, nb_val_step=40)

save_model_and_weights(model=model, save_model=True, version_model=1,
                       save_weights=True, version_weights=1)

show_results(history=history)

