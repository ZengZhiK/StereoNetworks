import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(flying_dir):
    train_left_img = []
    train_right_img = []
    train_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    train_left_img_dir = sorted(os.listdir(flying_dir + '/train/image_clean/left/'))
    for img in train_left_img_dir:
        if is_image_file(flying_dir + '/train/image_clean/left/' + img):
            train_left_img.append(flying_dir + '/train/image_clean/left/' + img)
            train_left_disp.append(flying_dir + '/train/disparity/left/' + img.split(".")[0] + '.pfm')
            train_right_img.append(flying_dir + '/train/image_clean/right/' + img)

    test_left_img_dir = sorted(os.listdir(flying_dir + '/val/image_clean/left/'))
    for img in test_left_img_dir:
        if is_image_file(flying_dir + '/val/image_clean/left/' + img):
            test_left_img.append(flying_dir + '/val/image_clean/left/' + img)
            test_left_disp.append(flying_dir + '/val/disparity/left/' + img.split(".")[0] + '.pfm')
            test_right_img.append(flying_dir + '/val/image_clean/right/' + img)

    return train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp
