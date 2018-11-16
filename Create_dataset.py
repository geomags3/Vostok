import os, shutil
import random

FAIL_DIR = 'H:/Test Wrapper Result/Images/all gap/fail'
PASS_DIR = 'H:/Test Wrapper Result/Images/all gap/pass'

# БАЗОВАЯ ДИРЕКТОРИЯ ДЛЯ КОМПЬЮТЕРА
# BASE_DIR = '/home/user/Рабочий стол/VostokCNN/Image_for_CNN_4000_image/'
# БАЗОВАЯ ДИРЕКТОРИЯ ДЛЯ НОУТБУКА
# BASE_DIR = 'C:/Users/Geomags/Desktop/VostokCNN/Dataset_Train_and_Test/'
BASE_DIR = 'H:/Test Wrapper Result/Dataset_Train_and_Test_v3_all_gap/'

def create_directories(create_train_dir, create_val_dir, create_test_dir):
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
    # FAIL
    # random_image_list = list(range(1, fail_set[0] + fail_set[1] + fail_set[2] + 1))
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
    # random_image_list = list(range(1, pass_set[0] + pass_set[1] + pass_set[2] + 1))
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
images_distributions(number_fail_image=21116, number_pass_image=13910, fail_set=[8500, 1500, 3910],
                     pass_set=[8500, 1500, 3910], in_train=True, in_val=True, in_test=True)