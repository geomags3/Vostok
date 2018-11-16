from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import time, os, csv
import cv2
from tensorflow.python.keras import backend as K
from vis.visualization import visualize_cam
from vis.utils import utils

IMAGE_SIZE = [206, 398, 1]  # 206, 398, 1
MODEL_VERSION = 12
WEIGHT_VERSION = 1

THRESHOLD = 0  #0.9856
REPORT_VERSION = 0

MODEL_FILE_NAME = 'Vostok_model_v{}'
WEIGHTS_FILE_NAME = 'Vostok_weights_v{}.h5'
ONE_IMAGE_PATH = 'H:/Test Wrapper Result/Dataset_Train_and_Test_v2/test/pass/pass 11384.png'  # fail 333.png pass 235


# TEST_DIR = 'H:/Test Wrapper Result/Dataset_Train_and_Test_v2/test'
TEST_DIR = 'H:/Test Wrapper Result/Dataset_Train_and_Test_v2/test'

# CONFIG_DIR = 'C:/Users/Geomags/Documents/GitHub/Vostok/configuration/v{}'
CONFIG_DIR = 'C:/Users/geoma/Documents/GitHub/Vostok/configuration/v{}'
FORMAT_REPORT_NAME = 'full_report_v{}_%.2f'



def loading_model_from_file(version_model, version_weights):
    print('\n...Нейронная сеть загружается из файла\n')
    json_file = open(MODEL_FILE_NAME.format(version_model) + '.json', 'r')
    loading_model = model_from_json(json_file.read())
    json_file.close()
    loading_model.load_weights(MODEL_FILE_NAME.format(version_model)
                               + '_weights_v{}.h5'.format(version_weights))
    print('\n...Загрузка сети завершена\n')
    return loading_model

def check_one_image(model, image_path):
    img = image.load_img(image_path, target_size=IMAGE_SIZE, grayscale=True)
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    # if prediction >= THRESHOLD:
    #     prediction_label = 'PASS'
    # else: prediction_label = 'FAIL'
    #
    # if image_path.find('pass') != -1:
    #     correct_label = 'PASS'
    # else: correct_label = 'FAIL'
    #
    # plt.title('Предсказание сети: ' + prediction_label + '\n' + str(prediction[0][0]) +
    #           '\nПравильный ответ: ' + correct_label)
    plt.imshow(img)
    # plt.show()

    print('Predicted: ', prediction)
    print(str(np.argmax(prediction[0])))

    cap_output = model.output[:, 0]
    last_conv_layer = model.get_layer('conv2d_4')

    grads = K.gradients(cap_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(128):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    # ret, heatmap = cv2.threshold(heatmap, 100, 255, cv2.THRESH_TOZERO)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # superimposed_img = heatmap + img
    superimposed_img = cv2.addWeighted(heatmap, 0.4, rgb_img, 1, 0)
    # cv2.namedWindow('Heatmap')
    cv2.imshow('Heatmap', superimposed_img)
    cv2.waitKey(0)

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
        if True:
        # if np.argmax(prediction[0]) == 1:
        # if prediction > THRESHOLD:
            if show_fail == True:
                plt.title('Предсказание сети: PASS' + str(prediction) +
                          '\nПравильный ответ: FAIL' +
                          '\nНазвание файла: ' + img_name)
                plt.imshow(img)
                plt.show()
            n_fail += 1
            wrong_fail_prediction_list.append([TEST_DIR + '/fail/' + img_name,
                                               str(prediction[0][0])])

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
        if True:
        # if np.argmax(prediction[0]) == 0:
        # if prediction <= THRESHOLD:
            if show_pass == True:
                plt.title('Предсказание сети: FAIL' + str(prediction) +
                          '\nПравильный ответ: PASS' +
                          '\nНазвание файла: ' + img_name)
                plt.imshow(img)
                plt.show()
            n_pass += 1
            wrong_pass_prediction_list.append([TEST_DIR + '/pass/' + img_name,
                                               str(prediction[0][0])])

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
                filewriter.writerow([wrong_image[0], wrong_image[1]])
            filewriter.writerow(['PASS images'])
            for wrong_image in wrong_pass_prediction_list:
                filewriter.writerow([wrong_image[0], wrong_image[1]])
        # with open(full_name_image_list, 'w', newline='\n') as csvfile:
        #     filewriter = csv.writer(csvfile, delimiter=';')
        #     filewriter.writerow(['FAIL images'])
        #     for wrong_image in wrong_fail_prediction_list:
        #         filewriter.writerow([wrong_image])
        #     filewriter.writerow(['PASS images'])
        #     for wrong_image in wrong_pass_prediction_list:
        #         filewriter.writerow([wrong_image])


model = loading_model_from_file(version_model=MODEL_VERSION, version_weights=WEIGHT_VERSION)
# check_one_image(model=model, image_path=ONE_IMAGE_PATH)
check_test_images_and_generate_report(model=model, number_fail_images=3910, number_pass_images=3910,
                                      show_fail=False, show_pass=False, generate_report=True)



