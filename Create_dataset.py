import os, shutil
import random

FAIL_DIR = 'E:/Only gap/fail'
PASS_DIR = 'E:/Only gap/pass'

# БАЗОВАЯ ДИРЕКТОРИЯ
BASE_DIR = 'E:/Only gap/Dataset/'

def create_directories(create_train_dir, create_val_dir, create_test_dir):
    '''
    Создать папки 'train', 'validation' и 'test',
    а в них подпапки 'fail', 'pass'
    :param create_train_dir: создавать папку 'train'?
    :param create_val_dir: создавать папку 'validation'?
    :param create_test_dir: создавать папку 'test'?
    :return:
    '''
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)

    if create_train_dir == True:
        train_dir = os.path.join(BASE_DIR, 'train')
        os.mkdir(train_dir)

        train_fail_dir = os.path.join(train_dir, 'fail')
        os.mkdir(train_fail_dir)
        train_pass_dir = os.path.join(train_dir, 'pass')
        os.mkdir(train_pass_dir)

    if create_val_dir == True:
        validation_dir = os.path.join(BASE_DIR, 'validation')
        os.mkdir(validation_dir)

        validation_fail_dir = os.path.join(validation_dir, 'fail')
        os.mkdir(validation_fail_dir)
        validation_pass_dir = os.path.join(validation_dir, 'pass')
        os.mkdir(validation_pass_dir)

    if create_test_dir == True:
        test_dir = os.path.join(BASE_DIR, 'test')
        os.mkdir(test_dir)

        test_fail_dir = os.path.join(test_dir, 'fail')
        os.mkdir(test_fail_dir)
        test_pass_dir = os.path.join(test_dir, 'pass')
        os.mkdir(test_pass_dir)

def images_distributions(number_fail_image, number_pass_image, fail_set,
                         pass_set, in_train, in_val, in_test):
    '''
    Из общего количества FAIL изображений случайным образом формирует выборки:
    'BASE_DIR/train/fail'
    'BASE_DIR/validation/fail'
    'BASE_DIR/test/fail'
    Из общего количества PASS изображений случайным образом формирует выборки:
    'BASE_DIR/train/pass'
    'BASE_DIR/validation/pass'
    'BASE_DIR/test/pass'

    :param number_fail_image: общее количество FAIL изображений
    :param number_pass_image: общее количество PASS изображений
    :param fail_set: массив количества FAIL изображений создаваемых выборок [train, validation, test]
    :param pass_set: массив количества PASS изображений создаваемых выборок [train, validation, test]
    :param in_train: создавать выборки train?
    :param in_val: создавать выборки validation?
    :param in_test: создавать выборки test?
    :return:
    '''
    # FAIL
    random_image_list = list(range(1, number_fail_image + 1))
    random.shuffle(random_image_list)

    if in_train == True:
        fnames = ['fail {}.png'.format(random_image_list[i]) for i in range(fail_set[0])]  # 1, 1601
        for fname in fnames:
            src = os.path.join(FAIL_DIR, fname)
            dst = os.path.join(BASE_DIR + 'train/fail', fname)
            shutil.copyfile(src, dst)
        print('...train/fail: ', len(os.listdir(BASE_DIR + 'train/fail')))

    if in_val == True:
        fnames = ['fail {}.png'.format(random_image_list[i]) for i in range(fail_set[0],  # 1601, 2001
                                                                            fail_set[0] + fail_set[1])]
        for fname in fnames:
            src = os.path.join(FAIL_DIR, fname)
            dst = os.path.join(BASE_DIR + 'validation/fail', fname)
            shutil.copyfile(src, dst)
        print('...validation/fail: ', len(os.listdir(BASE_DIR + 'validation/fail')))

    if in_test == True:
        fnames = ['fail {}.png'.format(random_image_list[i]) for i in range(fail_set[0] + fail_set[1],  # 2001, 2501
                                                                            fail_set[0] + fail_set[1] + fail_set[2])]
        for fname in fnames:
            src = os.path.join(FAIL_DIR, fname)
            dst = os.path.join(BASE_DIR + 'test/fail', fname)
            shutil.copyfile(src, dst)
        print('...test/fail: ', len(os.listdir(BASE_DIR + 'test/fail')))

    # PASS
    random_image_list = list(range(1, number_pass_image + 1))
    random.shuffle(random_image_list)

    if in_train == True:
        fnames = ['pass {}.png'.format(random_image_list[i]) for i in range(pass_set[0])]  # 1, 1601
        for fname in fnames:
            src = os.path.join(PASS_DIR, fname)
            dst = os.path.join(BASE_DIR + 'train/pass', fname)
            shutil.copyfile(src, dst)
        print('...train/pass: ', len(os.listdir(BASE_DIR + 'train/pass')))

    if in_val == True:
        fnames = ['pass {}.png'.format(random_image_list[i]) for i in range(pass_set[0],  # 1601, 2001
                                                                            pass_set[0] + pass_set[1])]
        for fname in fnames:
            src = os.path.join(PASS_DIR, fname)
            dst = os.path.join(BASE_DIR + 'validation/pass', fname)
            shutil.copyfile(src, dst)
        print('...validation/pass: ', len(os.listdir(BASE_DIR + 'validation/pass')))

    if in_test == True:
        fnames = ['pass {}.png'.format(random_image_list[i]) for i in range(pass_set[0] + pass_set[1],  # 2001, 2501
                                                                            pass_set[0] + pass_set[1] +
                                                                            pass_set[2])]
        for fname in fnames:
            src = os.path.join(PASS_DIR, fname)
            dst = os.path.join(BASE_DIR + 'test/pass', fname)
            shutil.copyfile(src, dst)
        print('...test/pass: ', len(os.listdir(BASE_DIR + 'test/pass')))

create_directories(create_train_dir=True, create_val_dir=True, create_test_dir=True)
images_distributions(number_fail_image=4193, number_pass_image=4239, fail_set=[2920, 620, 640],
                     pass_set=[2920, 620, 640], in_train=True, in_val=True, in_test=True)