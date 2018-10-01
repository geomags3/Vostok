from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import time, os, csv

IMAGE_SIZE = [206, 398, 1]
THRESHOLD = 0.98
MODEL_VERSION = 1
REPORT_VERSION = 3

MODEL_FILE_NAME = 'Vostok_model_v{}.json'
WEIGHTS_FILE_NAME = 'Vostok_weights_v{}.h5'
# ONE_IMAGE_PATH_PC = '/home/user/Рабочий стол/VostokCNN/Image_for_CNN_4000_image/test/pass/pass 103.png'
# ONE_IMAGE_PATH_DELL = '.../Image_for_CNN_4000_image/test/pass/pass 103.png'
TEST_DIR = '/media/user/TOSHIBA EXT/Test Wrapper Result/Dataset_Train_and_Test/test'

CONFIG_DIR = '/home/user/Рабочий стол/Python/Vostok/configuration/v{}'
FORMAT_REPORT_NAME = 'report_v{}_%.2f'
# NAME_IMAGE_LIST = 'list_wrong_image.csv'



def loading_model_from_file(version_model, version_weights):
    print('\n...Нейронная сеть загружается из файла\n')
    json_file = open(MODEL_FILE_NAME.format(version_model), 'r')
    loading_model = model_from_json(json_file.read())
    json_file.close()
    loading_model.load_weights(WEIGHTS_FILE_NAME.format(version_weights))
    print('\n...Загрузка сети завершена\n')
    return loading_model

def check_one_image(model, image_path):
    img = image.load_img(image_path, target_size=IMAGE_SIZE, grayscale=True)
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    if prediction >= THRESHOLD:
        prediction_label = 'PASS'
    else: prediction_label = 'FAIL'

    if image_path.find('pass') != -1:
        correct_label = 'PASS'
    else: correct_label = 'FAIL'

    plt.title('Предсказание сети: ' + prediction_label + str(prediction) +
              '\nПравильный ответ: ' + correct_label)
    plt.imshow(img)
    plt.show()

def check_test_images_and_generate_report(model, number_fail_images, number_pass_images,
                                          show_fail, show_pass, generate_report):
    n_fail = 0
    index = 1
    wrong_fail_prediction_list = []
    # img_names = [TEST_DIR + '/fail/fail {}.png'.format(i)
    #             for i in range(1, number_fail_images + 1)]
    img_names = os.listdir(TEST_DIR + '/fail')
    for img_name in img_names:
        index += 1
        img = image.load_img(TEST_DIR + '/fail/' + img_name,
                             target_size=IMAGE_SIZE, grayscale=True)
        x = image.img_to_array(img)
        x /= 255
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)
        if prediction > THRESHOLD:
            if show_fail == True:
                plt.title('Предсказание сети: PASS' + str(prediction) +
                          '\nПравильный ответ: FAIL' +
                          '\nНазвание файла: ' + img_name)
                plt.imshow(img)
                plt.show()
            n_fail += 1
            wrong_fail_prediction_list.append(TEST_DIR + '/fail/' + img_name)

    visual_check = int(input('\n...Сколько изображений с изначально неправильной классификацией FAIL?\n'
                             + '>>> '))
    wrong_fail_prediction = n_fail - visual_check
    print('\n...Количество неправильных классификаций FAIL: ', wrong_fail_prediction)
    fail_accuracy = ((number_fail_images - wrong_fail_prediction) / number_fail_images) * 100
    print('\n...Точность распознавания FAIL изображений: %.4s' % (str(fail_accuracy)) + '%')

    n_pass = 0
    index = 1
    sum_time = 0
    wrong_pass_prediction_list = []
    # img_names = [TEST_DIR + '/pass/pass {}.png'.format(i)
    #              for i in range(1, number_pass_images + 1)]
    img_names = os.listdir(TEST_DIR + '/pass')
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
        if prediction <= THRESHOLD:
            if show_pass == True:
                plt.title('Предсказание сети: FAIL' + str(prediction) +
                          '\nПравильный ответ: PASS' +
                          '\nНазвание файла: ' + img_name)
                plt.imshow(img)
                plt.show()
            n_pass += 1
            wrong_pass_prediction_list.append(TEST_DIR + '/pass/' + img_name)

    visual_check = int(input('\n...Сколько изображений с изначально неправильной классификацией PASS?\n'
                             + '>>> '))
    wrong_pass_prediction = n_pass - visual_check
    print('\n...Количество неправильных классификаций PASS: ', wrong_pass_prediction)
    pass_accuracy = ((number_pass_images - wrong_pass_prediction) / number_pass_images) * 100
    print('\n...Точность распознавания PASS изображений: %.4s' % (str(pass_accuracy)) + '%')

    full_network_accuracy = (fail_accuracy + pass_accuracy) / 2

    print('\n...Полная точность сети: %.4s' % (str(full_network_accuracy)) + '%')

    time_prediction_single_image = (sum_time / number_pass_images) * 1000

    print('\n...Среднее время распознавания одного изображения: %.3s'
          % (str(time_prediction_single_image)) + ' мс')

    if generate_report == True:
        report_name = FORMAT_REPORT_NAME.format(REPORT_VERSION) % (THRESHOLD)
        full_report_name = CONFIG_DIR.format(MODEL_VERSION) + '/' + report_name + '.csv'

        with open(full_report_name, 'w', newline='\n') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=';')
            filewriter.writerow(['FAIL', str(number_fail_images), str(n_fail)])
            filewriter.writerow(['PASS', str(number_pass_images), str(n_pass)])
            filewriter.writerow(['Time', '%.3s' % str(time_prediction_single_image)])

            filewriter.writerow(['FAIL images'])
            for wrong_image in wrong_fail_prediction_list:
                filewriter.writerow([wrong_image])
            filewriter.writerow(['PASS images'])
            for wrong_image in wrong_pass_prediction_list:
                filewriter.writerow([wrong_image])
        # with open(full_name_image_list, 'w', newline='\n') as csvfile:
        #     filewriter = csv.writer(csvfile, delimiter=';')
        #     filewriter.writerow(['FAIL images'])
        #     for wrong_image in wrong_fail_prediction_list:
        #         filewriter.writerow([wrong_image])
        #     filewriter.writerow(['PASS images'])
        #     for wrong_image in wrong_pass_prediction_list:
        #         filewriter.writerow([wrong_image])


model = loading_model_from_file(version_model=1, version_weights=1)
# check_one_image(model, ONE_IMAGE_PATH_PC)
check_test_images_and_generate_report(model, 9000, 9000, show_fail=False, show_pass=False, generate_report=True)



