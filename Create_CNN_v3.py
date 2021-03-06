import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import callbacks
import csv, os

# БАЗОВАЯ ДИРЕКТОРИЯ ХРАНИЛИЩА ДАННЫХ
BASE_DIR = 'E:/Only gap/Dataset/'

TRAIN_DIR = BASE_DIR + 'train'
VALIDATION_DIR = BASE_DIR + 'validation'
TEST_DIR = BASE_DIR + 'test'

IMAGE_SIZE = [38, 390, 1]  # 206, 398, 1
TARGET_SIZE = [38, 390]  # 206, 398
NB_EPOCH = 40
BATCH_SIZE = 20  # 25
LR = 1e-4
MODEL_VERSION = 16
WEIGHTS_VERSION = 1
LOSS_FUNCTION = 'binary_crossentropy'
NB_TRAIN_STEP = 146  # 850
NB_VAL_STEP = 31  # 150

MODEL_NAME = 'Vostok_model_v{}'
CONFIG_NAME = 'Vostok_configuration_v{}.csv'

CONFIG_DIR = 'C:/Users/geoma/Documents/GitHub/Vostok/configuration/'


def create_model(summary):
    '''
    Формирование архитектуры сверточной нейронной сети.
    Создание сети.

    :param summary:
    True - выводить описание сети
    Felse - не выводить

    :return:
    model - объект сети

    Архитектура сети оптимизирована для изображений размера [38, 390, 1]
    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3),  # (7, 7) вместо (3, 3)
                     # количество каналов изображения (ПРОВЕРИТЬ!)
                     input_shape=IMAGE_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))  # (5, 5) вместо (3, 3)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))  # (5, 5) вместо (3, 3)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=LOSS_FUNCTION,
                  optimizer=optimizers.RMSprop(lr=LR),
                  metrics=['acc'])

    if summary == True: print(model.summary())

    return model


def create_generator():
    '''
    Создание генераторов изображений:
    train_generator
    validation_generator
    Data Augmentation (расширение данных) - выключено
    :return:
    train_generator, validation_generator
    '''
    train_datagen = ImageDataGenerator(rescale=1. / 255)
                                       # rotation_range=20,
                                       # width_shift_range=0.2,
                                       # height_shift_range=0.2,
                                       # shear_range=0.2,
                                       # # zoom_range=0.2,
                                       # horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(directory=TRAIN_DIR,
                                                        target_size=TARGET_SIZE,
                                                        color_mode='grayscale',
                                                        batch_size=BATCH_SIZE,
                                                        # classes=None, # автоматически подгружаются классы из
                                                        # названий подпапок
                                                        shuffle=True,
                                                        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                  target_size=TARGET_SIZE,
                                                                  color_mode='grayscale',
                                                                  batch_size=BATCH_SIZE,
                                                                  # classes=None,  # автоматически подгружаются классы из
                                                                  # названий подпапок
                                                                  shuffle=True,
                                                                  class_mode='binary')
    print(train_generator.class_indices)
    print(validation_generator.class_indices)

    return train_generator, validation_generator


def create_callbacks(early_stopping, model_checkpoint, reduce_lr_on_plateau, tensor_board):
    '''
    Создание списка callbacks

    :param early_stopping: остановка обучения, если параметр 'monitor' не меняется в течении 'patience' эпох
    :param model_checkpoint:  сохранение весов сети с лучшим показателем параметра 'monitor'
    :param reduce_lr_on_plateau: уменьшение learning rate в процессе обучения
    :param tensor_board:
    :return:
    '''
    callbacks_list = []

    # if early_stopping == True:
    #     callbacks_list.append(callbacks.EarlyStopping(monitor='val_acc', patience=7))

    if model_checkpoint == True:
        callbacks_list.append(callbacks.ModelCheckpoint(filepath='weight_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                        monitor='val_loss', save_best_only=True))

    if reduce_lr_on_plateau == True:
        callbacks_list.append(callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10))

    # if tensor_board == True:
    #     callbacks_list.append(callbacks.TensorBoard(log_dir='log_dir', histogram_freq=1))

    return callbacks_list


def fit_model(model, train_generator, validation_generator, callbacks_list, nb_train_step, nb_val_step):
    '''
    Обучение сети
    :param model: модель сети
    :param train_generator:
    :param validation_generator:
    :param callbacks_list: список callbacks
    :param nb_train_step: количество шагов обучения - train
    :param nb_val_step: количество шагов проверки - validation
    :return: история обучения
    '''
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=nb_train_step,
                                  epochs=NB_EPOCH,
                                  validation_data=validation_generator,
                                  callbacks=callbacks_list,
                                  validation_steps=nb_val_step)
    return history


def save_model_and_weights(model, save_model, version_model, save_weights, version_weights):
    '''
    Сохранение модели сети и ее весовых коэффициентов
    :param model: модель сети
    :param save_model: сохранять ли модель?
    :param version_model: версия модели
    :param save_weights: сохранять ли веса?
    :param version_weights: версия весов
    :return:
    '''
    if save_model == True:
        json_file = open(MODEL_NAME.format(version_model) + '.json', 'w')
        json_file.write(model.to_json())
        json_file.close()

    if save_weights == True:
        model.save(MODEL_NAME.format(version_model) + '_weights_v{}.h5'.format(version_weights))


def show_results(history):
    '''
    Отображение графиков обучения
    :param history: история (создается в 'fit_model')
    :return:
    '''
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


def network_configuration():
    '''
    Сохранение основных параметров сети в .csv документ
    :return:
    '''
    if not os.path.exists(CONFIG_DIR):
        os.mkdir(CONFIG_DIR)
    version_dir = os.path.join(CONFIG_DIR, 'v{}'.format(MODEL_VERSION))
    os.mkdir(version_dir)
    file_dir = os.path.join(version_dir, CONFIG_NAME.format(MODEL_VERSION))

    with open(file_dir, 'w', newline='\n') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=';')
        filewriter.writerow(['Network version', str(MODEL_VERSION)])
        # Network architecture
        filewriter.writerow(['Network architecture'])
        filewriter.writerow(['INPUT', '206x398x1'])
        filewriter.writerow(['CONV', '32x7x7', 'RELU'])
        filewriter.writerow(['POOL', '2x2'])
        filewriter.writerow(['CONV', '64x5x5', 'RELU'])
        filewriter.writerow(['POOL', '2x2'])
        filewriter.writerow(['CONV', '64x5x5', 'RELU'])
        filewriter.writerow(['POOL', '2x2'])
        filewriter.writerow(['CONV', '128x3x3', 'RELU'])
        filewriter.writerow(['POOL', '2x2'])
        filewriter.writerow(['CONV', '128x3x3', 'RELU'])
        filewriter.writerow(['POOL', '2x2'])
        filewriter.writerow(['DROPOUT'])
        filewriter.writerow(['DENSE', '512', 'RELU'])
        filewriter.writerow(['DENSE', '1', 'SIGMOID'])
        # Fit options
        filewriter.writerow(['Fit options'])
        filewriter.writerow(['Loss function = ' + LOSS_FUNCTION])
        filewriter.writerow(['Optimizer = RMSprop'])
        filewriter.writerow(['Learning rate = ' + str(LR)])
        filewriter.writerow(['Number epochs = ' + str(NB_EPOCH)])
        filewriter.writerow(['Batch size = ' + str(BATCH_SIZE)])
        filewriter.writerow(['Number train images = ' + str((NB_TRAIN_STEP * BATCH_SIZE) * 2)])
        filewriter.writerow(['Number validation images = ' + str((NB_VAL_STEP * BATCH_SIZE) * 2)])
        # Data augmentation
        filewriter.writerow(['Data augmentation'])
        filewriter.writerow(['rotation_range='])
        filewriter.writerow(['width_shift_range='])
        filewriter.writerow(['height_shift_range='])
        filewriter.writerow(['shear_range='])
        filewriter.writerow(['zoom_range='])
        filewriter.writerow(['horizontal_flip='])
        # Callbacks
        filewriter.writerow(['Callbacks'])
        filewriter.writerow(['EarlyStopping(---)'])
        filewriter.writerow(['ModelCheckpoint(monitor=\'val_loss\', save_best_only=True)'])
        filewriter.writerow(['ReduceLROnPlateau(monitor=\'val_loss\', factor=0.1, patience=10)'])

        filewriter.writerow(['End'])


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


model = create_model(summary=True)

train_generator, validation_generator = create_generator()

callbacks_list = create_callbacks(early_stopping=True, model_checkpoint=True,
                                  reduce_lr_on_plateau=True, tensor_board=False)

history = fit_model(model=model, train_generator=train_generator,
                    validation_generator=validation_generator, callbacks_list=callbacks_list,
                    nb_train_step=NB_TRAIN_STEP, nb_val_step=NB_VAL_STEP)

save_model_and_weights(model=model, save_model=True, version_model=MODEL_VERSION,
                       save_weights=True, version_weights=WEIGHTS_VERSION)

show_results(history=history)

network_configuration()



