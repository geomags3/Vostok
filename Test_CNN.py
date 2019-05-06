# ОТКЛЮЧЕНИЕ GPU
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import time, os, csv


IMAGE_SIZE = [38, 390, 1]  # 206, 398, 1
MODEL_VERSION = 16
WEIGHT_VERSION = 1

REPORT_VERSION = 0

MODEL_FILE_NAME = 'Vostok_model_v{}'

ONE_IMAGE_PATH = 'E:/Only gap/pass/pass 142.png'

TEST_DIR = 'E:/Only gap'

CONFIG_DIR = 'C:/Users/geoma/Documents/GitHub/Vostok/configuration/v{}'

FORMAT_REPORT_NAME = 'report_all_image_v{}'


def loading_model_from_file(version_model, version_weights):
    '''
    Загрузка модели и ее весов из файла
    :param version_model: версия модели
    :param version_weights: версия весов
    :return: загруженную модели
    '''
    print('\n...Нейронная сеть загружается из файла\n')
    json_file = open(MODEL_FILE_NAME.format(version_model) + '.json', 'r')
    loading_model = model_from_json(json_file.read())
    json_file.close()
    loading_model.load_weights(MODEL_FILE_NAME.format(version_model)
                               + '_weights_v{}.h5'.format(version_weights))
    print('\n...Загрузка сети завершена\n')
    return loading_model


def check_one_image(model, image_path):
    '''
    Проверка одного изображения сетью
    :param model: модель сети
    :param image_path: путь к изображению
    :return:
    '''
    img = image.load_img(image_path, target_size=IMAGE_SIZE, grayscale=True)
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)

    plt.imshow(img)
    plt.show()

    print('Predicted: ', prediction[0][0])


def check_test_images_and_generate_report(model, show_fail, show_pass, generate_report):
    '''
    Проверка сетью изображений из папок 'fail' и 'pass' в 'TEST_DIR'.
    Создание отчета.
    :param model: модель сети
    :param show_fail: показывать изображения бракованных крышек?
    :param show_pass: показывать изображения годных крышек?
    :param generate_report: генерировать отчет?
    :return:
    '''
    # ПРОВЕРКА FAIL ИЗОБРАЖЕНИЙ
    index = 1
    fail_prediction_list = []
    img_names = os.listdir(TEST_DIR + '/fail')
    number_fail_images = len(img_names)
    for img_name in img_names:
        index += 1
        img = image.load_img(TEST_DIR + '/fail/' + img_name,
                             target_size=IMAGE_SIZE, grayscale=True)
        x = image.img_to_array(img)
        x /= 255
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)
        if show_fail == True:
            plt.title(img_name)
            plt.imshow(img)
            plt.show()
        fail_prediction_list.append([TEST_DIR + '/fail/' + img_name,
                                     str(prediction[0][0])])
    print('\n...Всего проверено ' + str(number_fail_images) + ' FAIL изображений')

    # ПРОВЕРКА PASS ИЗОБРАЖЕНИЙ
    index = 1
    sum_time = 0
    pass_prediction_list = []
    img_names = os.listdir(TEST_DIR + '/pass')
    number_pass_images = len(img_names)
    for img_name in img_names:
        index += 1
        img = image.load_img(TEST_DIR + '/pass/' + img_name,
                             target_size=IMAGE_SIZE, grayscale=True)
        x = image.img_to_array(img)
        x /= 255
        x = np.expand_dims(x, axis=0)
        time_1 = time.time()
        prediction = model.predict(x)
        time_2 = time.time()
        sum_time += time_2 - time_1
        if show_pass == True:
            plt.title(img_name)
            plt.imshow(img)
            plt.show()
        pass_prediction_list.append([TEST_DIR + '/pass/' + img_name,
                                     str(prediction[0][0])])
    print('\n...Всего проверено ' + str(number_pass_images) + ' PASS изображений')

    time_prediction_single_image = (sum_time / number_pass_images) * 1000
    print('\n...Среднее время распознавания одного изображения: %.3s'
          % (str(time_prediction_single_image)) + ' мс')

    # СОЗДАНИЕ ОТЧЕТА
    if generate_report == True:
        report_name = FORMAT_REPORT_NAME.format(REPORT_VERSION)
        full_report_name = CONFIG_DIR.format(MODEL_VERSION) + '/' + report_name + '.csv'

        with open(full_report_name, 'w', newline='\n') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=';')
            filewriter.writerow(['FAIL', str(number_fail_images)])
            filewriter.writerow(['PASS', str(number_pass_images)])
            filewriter.writerow(['Time', '%.3s' % str(time_prediction_single_image)])

            filewriter.writerow(['FAIL images'])
            for curr_image in fail_prediction_list:
                filewriter.writerow([curr_image[0], curr_image[1]])
            filewriter.writerow(['PASS images'])
            for curr_image in pass_prediction_list:
                filewriter.writerow([curr_image[0], curr_image[1]])


model = loading_model_from_file(version_model=MODEL_VERSION, version_weights=WEIGHT_VERSION)
# check_one_image(model=model, image_path=ONE_IMAGE_PATH)
check_test_images_and_generate_report(model=model, show_fail=False, show_pass=False, generate_report=True)



