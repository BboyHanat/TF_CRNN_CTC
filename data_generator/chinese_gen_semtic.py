"""
Name : chinese_gen_semtic.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-08-30 00:14
Desc:
"""

import sys
from multiprocessing import Pool

sys.path.append("..")
import numpy as np
import cv2
import random
import datetime
import shutil
import os
from PIL import Image, ImageDraw, ImageFont
from local_utils.establish_char_dict import *
from fontTools.fontBuilder import TTFont
import glog as logger

img_format = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
font_format = ["ttf", 'eot', 'fon', 'font', 'woff', 'woff2', 'otf', 'ttc', 'TTF', 'TTC', 'OTF', 'EOT', 'FONT', 'FON', 'WOFF', 'WOFF2']


class CharacterGen:
    def __init__(self, character_seq, len_range=(2, 15)):
        """

        :param character_seq:
        :param batch_size:
        """
        self.character_seq = character_seq
        #random.shuffle(self.character_seq)
        self.len_range = len_range
        self.get_next = self._get_next_batch()
        self.corpus_length = len(self.character_seq)

    @staticmethod
    def random_add_blank(character_seq):
        character_seq = character_seq.replace('\0x00', '')
        character_seq = character_seq.replace('\n', '')
        character_seq = character_seq.replace('\r', '')
        character_seq = character_seq.replace('\t', '')
        character_seq = character_seq.replace(' ', '')
        character_seq = list(character_seq)
        len_seq = len(character_seq)
        if len_seq < 2:
            return None
        if random.randint(0, 6) >= 4 and 7 < len_seq < 13:
            character_seq.insert(random.randint(2, len_seq - 1), ' ')

        # len_seq = len(character_seq)

        # top = random.randint(0, 15 - len_seq)
        # bottom = 15 - len_seq - top
        # for i in range(top):
        #     character_seq.insert(0, ' ')
        # for i in range(bottom):
        #     character_seq.insert(len_seq - top, ' ')
        return character_seq

    def _get_next_batch(self):
        """

        :return:
        """
        while True:
            seek = random.randint(0, self.corpus_length - 1)
            char_seq = self.random_add_blank(self.character_seq[seek])
            if char_seq == None:
                continue
            yield char_seq


class ImageGen:
    def __init__(self, images_path):
        """

        :param character_seq:
        :param batch_size:
        """
        self.images_path = images_path
        random.shuffle(self.images_path)
        self.get_next = self.get_next_batch()

    def get_next_batch(self):
        """
        get next image
        :return:
        """
        seek = 0
        while True:
            if len(self.images_path) - seek <= 1:
                random.shuffle(self.images_path)
                seek = 0
            try:
                image = cv2.imread(self.images_path[seek])
                image = Image.fromarray(image)
            except:
                continue
            yield image
            seek += 1


class FontGen:
    def __init__(self, font_path):
        """

        :param character_seq:
        :param batch_size:
        """
        self.font_path = font_path
        random.shuffle(self.font_path)
        self.get_next = self.get_next_batch()

    def get_next_batch(self):
        """
        get next image
        :return:
        """
        seek = 0
        while True:
            if len(self.font_path) - seek <= 1:
                random.shuffle(self.font_path)
                seek = 0
            try:
                font = self.font_path[seek]
            except:
                continue
            yield font
            seek += 1


def random_region(image, width_limit=100, height_limit=100):
    """

    :param image:
    :return:
    """
    w, h = image.size
    if random.randint(0, 6) >= 0:
        width_limit_new = random.randint(width_limit, int(width_limit * 1.2))
        height_limit_new = random.randint(height_limit, int(height_limit * 1.2))
        if width_limit_new < w and height_limit_new < h:
            x1 = random.randint(0, w - width_limit_new)
            y1 = random.randint(0, h - height_limit_new)
        else:
            image = image.resize((width_limit_new, height_limit_new), Image.BICUBIC)
            x1 = 0
            y1 = 0
        x2 = x1 + width_limit_new
        y2 = y1 + height_limit_new
        roi = image.crop((x1, y1, x2, y2))
        w, h = roi.size
        x1 = random.randint(0, w - width_limit)
        y1 = random.randint(0, h - height_limit)
        x2 = x1 + width_limit
        y2 = y1 + height_limit

    else:
        if width_limit < w and height_limit < h:
            x1 = random.randint(0, w - width_limit)
            y1 = random.randint(0, h - height_limit)
        else:
            image = image.resize((width_limit, height_limit), Image.BICUBIC)
            x1 = 0
            y1 = 0
        x2 = x1 + width_limit
        y2 = y1 + height_limit
        roi = image.crop((x1, y1, x2, y2))
        x1 = 0
        y1 = 0
        x2 = width_limit
        y2 = height_limit

    return roi, x1, y1, x2, y2


def get_fontcolor(image):
    """
    get font color by mean
    :param image:
    :return:
    """
    # image_pil = Image.fromarray(image)
    # image_pil.show()
    image = np.asarray(image)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    bg = lab_image[:, :, 0]
    l_mean = np.mean(bg)

    new_l = random.randint(0, 127 - 80) if l_mean > 127 else random.randint(127 + 80, 255)
    new_a = random.randint(0, 255)
    new_b = random.randint(0, 255)

    lab_rgb = np.asarray([[[new_l, new_a, new_b]]], np.uint8)
    rbg = cv2.cvtColor(lab_rgb, cv2.COLOR_Lab2RGB)

    r = rbg[0, 0, 0]
    g = rbg[0, 0, 1]
    b = rbg[0, 0, 2]

    return (r, g, b)


def random_get_font(font_lsit):
    """
    get font by lsit
    :param font_lsit:255
    :return:
    """
    font = random.randint(0, len(font_lsit))
    return font


def random_shuffle_synthetic(synthetic):
    """

    :param synthetic:
    :return:
    """
    random.shuffle(synthetic)
    return synthetic


def copy_have_yen_font(src_fonts_path, have_yen_path):
    """
    copy have yen font to have_yen_path
    :param src_fonts_path:
    :param have_yen_path:
    :return:
    """
    logger.info(have_yen_path)
    os.makedirs(have_yen_path, exist_ok=True)
    font_path = [os.path.join(src_fonts_path, font) for font in os.listdir(src_fonts_path) if font.split(".")[-1] in font_format and ('.DS' not in font)]
    for font in font_path:
        try:
            ttf = TTFont(font)
            uni_list = ttf['cmap'].tables[0].ttFont.getGlyphOrder()
            rmb = 'yen'
            name = font.split('/')[-1]
            if rmb in uni_list:
                save_path = os.path.join(have_yen_path, name)
                shutil.copy(font, save_path)
        except Exception as e:
            logger.info(font, e)
            continue


def get_font_image(font: str, char_list, font_size_range):
    """

    :param font_gen:
    :param font_have_yen_gen:
    :param char_list:
    :param font_size_range:
    :return:
    """

    def random_rotate(bg, angle_range=(-2, 2)):
        """

        :param bg: pil rgba mode
        :param angle:
        :return:
        """
        w, h = bg.size
        if random.randint(0, 10) > 5:
            angle = random.randint(angle_range[0], angle_range[1])
            bg = bg.rotate(angle, center=(w // 2, h // 2), expand=True, fillcolor=(0, 0, 0, 0))
        return bg

    def random_add_board(draw, x, y, char_str, fnt):
        b = random.randint(0, 2)
        if b < 1:
            draw.text((x - 2, y), char_str, font=fnt, fill=(0, 0, 0, 255))
            draw.text((x + 2, y), char_str, font=fnt, fill=(0, 0, 0, 255))
            draw.text((x, y - 2), char_str, font=fnt, fill=(0, 0, 0, 255))
            draw.text((x, y + 2), char_str, font=fnt, fill=(0, 0, 0, 255))

            # thicker border
            draw.text((x - 2, y - 2), char_str, font=fnt, fill=(0, 0, 0, 255))
            draw.text((x + 2, y - 2), char_str, font=fnt, fill=(0, 0, 0, 255))
            draw.text((x - 2, y + 2), char_str, font=fnt, fill=(0, 0, 0, 255))
            draw.text((x + 2, y + 2), char_str, font=fnt, fill=(0, 0, 0, 255))
            draw.text((x, y), char_str, font=fnt, fill=(255, 255, 255, 255))
        else:
            draw.text((x - 2, y - 2), char_str, fill=(255, 255, 255, 255), font=fnt)

    aspect_arr = ['h', 'h', 'h']
    a = random.randint(0, 2)
    aspect = aspect_arr[a]
    font_size = random.randint(font_size_range[0], font_size_range[1])
    char_list_aspect = []
    for char in char_list:
        char_list_aspect.append(char)
        if aspect == 'v':
            char_list_aspect.append('\n')
    fnt = ImageFont.truetype(font, font_size)
    if fnt is None:
        return None, -1
    char_str = "".join(char_list_aspect)
    spacing = random.randint(0, 2)
    print(char_str, len(char_str))
    if aspect == 'v':
        size = fnt.getsize_multiline(char_str, spacing=spacing)
    else:
        size = fnt.getsize(char_str)
    font_bg = Image.new("RGBA", (size[0], size[1]), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(font_bg)
    random_add_board(draw, 2, 2, char_str, fnt)
    # if aspect == 'h':
    # draw.text((0, 0), char_str, fill=(255, 255, 255, 255), font=fnt)
    # else:
    #     draw.multiline_text((0, 0), char_str, fill=(255, 255, 255, 255), spacing=spacing, font=fnt)
    font_bg = random_rotate(font_bg)
    return font_bg, aspect


def random_paste(image, font_bg, x1, y1, x2, y2):
    """
    random paste font to image
    :param image:
    :param font_bg:
    :re turn:
    """
    image = np.array(image)
    font_bg = np.array(font_bg)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    image_roi = image[y1:y2, x1:x2, :]
    r, g, b = get_fontcolor(image_roi)
    font_bg[:, :, 0] = font_bg[:, :, 0] // 255 * r
    font_bg[:, :, 1] = font_bg[:, :, 1] // 255 * g
    font_bg[:, :, 2] = font_bg[:, :, 2] // 255 * b
    alpha = round(random.uniform(0.8, 1), 1)
    alpha_channle = np.asarray(font_bg[:, :, -1], np.float32)
    alpha_channle = alpha_channle * alpha / 255.0
    image_roi[:, :, 0] = image_roi[:, :, 0] * (1 - alpha_channle) + font_bg[:, :, 0] * alpha_channle
    image_roi[:, :, 1] = image_roi[:, :, 1] * (1 - alpha_channle) + font_bg[:, :, 1] * alpha_channle
    image_roi[:, :, 2] = image_roi[:, :, 2] * (1 - alpha_channle) + font_bg[:, :, 2] * alpha_channle
    image_roi[:, :, 3] = image_roi[:, :, 3] * (1 - alpha_channle) + font_bg[:, :, 3] * alpha_channle

    image[y1:y2, x1:x2, :] = image_roi
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = Image.fromarray(image)
    return image


def image_normlization(image):
    w, h = image.size
    image = np.asarray(image)
    image = cv2.resize(image, (int(w * (32 / h)), 32), cv2.INTER_CUBIC)
    w, h = image.shape[1], image.shape[0]
    if w < 456:
        left = (456 - w) // 2
        right = 456 - (left + w)
        image = np.pad(image, ((0, 0), (left, right), (0, 0)), mode='constant', constant_values=0)
    elif w > 456:
        image = cv2.resize(image, (456, 32), cv2.INTER_CUBIC)
    image = Image.fromarray(image)
    return image


######################################################

path_chinese_synthetic = "./sentence.txt"
path_img = '/hanat/ocr_background_img'
path_font = '/data/User/hanat/TF_CRNN_CTC/data/new_fonts'
path_have_yen_path = '/data/User/hanat/TF_CRNN_CTC/data/have_yen_fonts'  # '/data/User/hanat/TF_CRNN_CTC/data/have_yen_fonts'
path_save = '/hanat/data4/image_data'
annotation_file = '/hanat/data4/data_'
font_size_range = (30, 50)
process_num = 24
batch_r = 14458

# path_chinese_synthetic = "./sentence.txt"
# path_img = '/Users/aidaihanati/TezignProject/PSD/替换素材(1)/images/background'
# path_font = '/Users/aidaihanati/TezignProject/TF_CRNN_CTC/data/fonts'
# path_have_yen_path = '/Users/aidaihanati/TezignProject/TF_CRNN_CTC/data/fonts'
# path_save = './hanat/train'
# annotation_file = './hanat/data_'
# font_size_range = (30, 50)
# process_num = 4
# batch_r = 10

os.makedirs(path_save, exist_ok=True)
fp = open(path_chinese_synthetic, "r")
chinese_synth = fp.readlines()
fp.close()

img_path = [os.path.join(path_img, img) for img in os.listdir(path_img) if img.split(".")[-1] in img_format and ('.DS' not in img)]
font_path = [(index+84, os.path.join(path_font, font)) for index, font in enumerate(os.listdir(path_font)) if font.split(".")[-1] in font_format and ('.DS' not in font)]
font_have_yen_path = [os.path.join(path_have_yen_path, font) for font in os.listdir(path_have_yen_path) if font.split(".")[-1] in font_format and ('.DS' not in font)]


def ocr_data_thread(font_info):
    """

    :param font_path:
    :return:
    """

    index = font_info[0]
    font = font_info[1]
    fp_txt = open(annotation_file + str(index % process_num) + '.txt', "a+")
    batch_size = 10
    chinese_synth_list = list(chinese_synth)
    character_gen = CharacterGen(chinese_synth_list)
    batch_repeat = batch_r  # if batch_size < shuffle_limmit else len(chinese_synth) // 2 // 5
    for repeat_times in range(batch_repeat):
        logger.info("font name is {}, font index {}, batch size is {} step is {}".format(font.split("/")[-1], str(index), str(batch_size), str(repeat_times)))
        char_list = character_gen.get_next.__next__()
        image = None
        while image is None:
            img_seek = random.randint(0, len(img_path) - 1)
            try:
                image = cv2.imread(img_path[img_seek])
                image = Image.fromarray(image)
            except:
                print("afafafa")
                continue
        #try:
        if '¥' in char_list:
            ttf = TTFont(font)
            uni_list = ttf['cmap'].tables[0].ttFont.getGlyphOrder()
            rmb = u'yen'
            if rmb not in uni_list:
                font_have_yen = random.randint(0, len(font_have_yen_path) - 1)
                font = font_have_yen_path[font_have_yen]

        font_image, aspect = get_font_image(font, char_list, font_size_range)
        size = font_image.size
        bg_pil, x1, y1, x2, y2 = random_region(image, size[0], size[1])
        bg_pil = random_paste(bg_pil, font_image, x1, y1, x2, y2)
        bg_pil = image_normlization(bg_pil)
        image_name = str(index) + "_" + str(repeat_times) + ".jpg"
        image_save = os.path.join(path_save, image_name)
        bg_pil.save(image_save)
        if os.path.exists(image_save):
            if os.path.getsize(image_save) > 1:
                char_list = ''.join(char_list)
                char_list = char_list.replace(' ', '')
                annotation_info = image_name + "^" + char_list + "\n"
                fp_txt.write(annotation_info)
            else:
                os.remove(image_save)

        # except:
        #     print("FUCK!!!!!!!", chinese_synth_list)
        #     continue
    fp_txt.close()


# for font_info in font_path:
#     ocr_data_thread(font_info)



start = datetime.datetime.now()
pool = Pool(process_num)
print(len(font_path))
for font_info in font_path:
    pool.apply_async(ocr_data_thread, (font_info,))
pool.close()
pool.join()

end = datetime.datetime.now()
print(end - start)
