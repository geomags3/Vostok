from tensorflow.python.keras.models import model_from_json
import numpy as np
# from tensorflow.python.keras.preprocessing import image
# import time
# THRESHOLD = 0.9856
# REPORT_VERSION = 1
# IMAGE_SIZE = [206, 398, 1]

MODEL_VERSION = 7
WEIGHT_VERSION = 1

MODEL_FILE_NAME = 'C:/Users/geoma/Documents/GitHub/Vostok/Vostok_model_v{}'
WEIGHTS_FILE_NAME = 'C:/Users/geoma/Documents/GitHub/Vostok/Vostok_weights_v{}.h5'

# ONE_IMAGE_PATH_DELL = '.../Image_for_CNN_4000_image/test/pass/pass 103.png'

# TEST_DIR = 'H:/Test Wrapper Result/Dataset_Train_and_Test_v2/test'
# TEST_DIR = 'H:/Test Wrapper Result'

# CONFIG_DIR = 'C:/Users/Geomags/Documents/GitHub/Vostok/configuration/v{}'
# CONFIG_DIR = 'C:/Users/geoma/Documents/GitHub/Vostok/configuration/v{}'
# FORMAT_REPORT_NAME = 'report_v{}_%.2f'

json_file = open(MODEL_FILE_NAME.format(MODEL_VERSION) + '.json', 'r')
model = model_from_json(json_file.read())
json_file.close()
model.load_weights(MODEL_FILE_NAME.format(MODEL_VERSION) + '_weights_v{}.h5'.format(WEIGHT_VERSION))

def image_predict(array):
    # # model = loading_model_from_file(version_model=MODEL_VERSION, version_weights=WEIGHT_VERSION)
    # img = image.load_img(image_path, target_size=IMAGE_SIZE, grayscale=True)
    # x = image.img_to_array(img)
    # x /= 255
    array = np.expand_dims(array, axis=0)  # добавить размерность в начало [206, 398] -> [1, 206, 398]
    array = np.expand_dims(array, axis=3)  # добавить размерность в конец [1, 206, 398] -> [1, 206, 398, 1]
    # time_1 = time.time()
    # prediction = model.predict(x)
    prediction = model.predict(array)  # предсказание сети (двумерный массив [][])
    # time_2 = time.time()
    # out = []
    # out.append(prediction[0][0])
    # out.append((time_2 - time_1) * 1000)
    return prediction[0][0]

# IMAGE_PATH = 'C:/Users/geoma/Desktop/test/fail 2.png'
# out = check_one_image(IMAGE_PATH)
# print('>>> '+ str(out[0]) + ' --- ' + str(out[1]))

# a = check_one_image(IMAGE_PATH)
# print(a)