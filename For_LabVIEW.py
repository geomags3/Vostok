# ОТКЛЮЧЕНИЕ GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing import image
import numpy as np
import time

MODEL_VERSION = 16
WEIGHT_VERSION = 1

# Получение пути к файлам модели и весов
dir_path = os.getcwd()  # Путь к текущей рабочей директории
MODEL_FILE_NAME = os.path.join(dir_path, 'Vostok_model_v{}')

# Загрузка модели сети и ее весов
json_file = open(MODEL_FILE_NAME.format(MODEL_VERSION) + '.json', 'r')
model = model_from_json(json_file.read())
json_file.close()
model.load_weights(MODEL_FILE_NAME.format(MODEL_VERSION) + '_weights_v{}.h5'.format(WEIGHT_VERSION))

def image_predict(array):
    '''
    Вызывается из LabVIEW!
    Функция проверки изображения нейронной сетью.
    Преобразование массива:
    1. Добавление размерности в начало [206, 398] -> [1, 206, 398]
    2. Добавление размерности в конец [1, 206, 398] -> [1, 206, 398, 1]
    LabVIEW выдает массив вида [206, 398],
    а метод '.predict' принимает массив вида [1, 206, 398, 1]

    :param array: изображение преобразованное в двумерный массив значений интенсивности
    :return: prediction[0][0] - оценка (Score)
    '''
    array = np.expand_dims(array, axis=0)  # добавить размерность в начало [206, 398] -> [1, 206, 398]
    array = np.expand_dims(array, axis=3)  # добавить размерность в конец [1, 206, 398] -> [1, 206, 398, 1]
    prediction = model.predict(array)  # предсказание сети (двумерный массив [][])
    return prediction[0][0]


def check_one_image(model, image_path):
    '''
    Проверка одного изображения сетью
    :param model: модель сети
    :param image_path: путь к изображению
    :return:
    prediction[0][0] - оценка (Score)
    time.time() - temp_time - время, затраченное на предсказание сетью
    '''
    img = image.load_img(image_path, target_size=[38, 390, 1], color_mode='grayscale')
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    temp_time = time.time()
    prediction = model.predict(x)
    return prediction[0][0], time.time() - temp_time

# PATH = 'E:/fail/fail 108.png'
# # time_arr = []
# # i_arr = []
# for i in range(50):
#     pred, time_pred = check_one_image(model, PATH)
#     print(pred, time_pred * 1000)
    # time_arr.append(time_pred * 1000)
    # print(pred, time_arr[i])
    # i_arr.append(i + 1)


