"""
Name : chinese_synthetic_gen.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-08-07 10:15
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


def random_region(image, width_limit=100, height_limit=100):
    """

    :param image:
    :return:
    """
    w, h = image.size
    if width_limit < w - 1 and height_limit < h - 1:
        x1 = random.randint(0, w - width_limit - 1)
        y1 = random.randint(0, h - height_limit - 1)
    else:
        image = image.resize((width_limit + 1, height_limit + 1), Image.BICUBIC)
        x1 = 0
        y1 = 0
    x2 = x1 + width_limit
    y2 = y1 + height_limit
    roi = image.crop((x1, y1, x2, y2))
    #roi = Image.new("RGB", (width_limit, height_limit), color=(0, 0, 0))
    return roi


def get_fontcolor(image):
    """
    get font color by mean
    :param image:
    :return:
    """
    # image_pil = Image.fromarray(image)
    # image_pil.show()
    image = np.asarray(image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
    h_mean = int(np.mean(image_hsv[:, :, 0]))
    s_mean = int(np.mean(image_hsv[:, :, 1]))
    v_mean = int(np.mean(image_hsv[:, :, 2]))

    h_new = (random.randint(100, 155)+h_mean) % 255
    s_new = (random.randint(30, 225)+s_mean) % 255
    v_new = (random.randint(70, 185)+v_mean) % 255
    hsv_rgb = np.asarray([[[h_new,s_new,v_new]]], np.uint8)
    rbg = cv2.cvtColor(hsv_rgb, cv2.COLOR_HSV2RGB_FULL)
    r = rbg[0, 0, 0]
    g = rbg[0, 0, 1]
    b = rbg[0, 0, 2]
    return (r, g, b)


def random_get_font(font_lsit):
    """
    get font by lsit
    :param font_lsit:
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


class CharacterGen:
    def __init__(self, character_seq, batch_size=5):
        """

        :param character_seq:
        :param batch_size:
        """
        self.character_seq = character_seq
        self.batch_size = batch_size
        self.get_next = self.get_next_batch()

    def get_next_batch(self):
        """

        :return:
        """
        seek = 0
        while True:
            if len(self.character_seq) - seek < self.batch_size:
                for i in range(len(self.character_seq) - seek):
                    self.character_seq.insert(0, self.character_seq[-1])
                    del self.character_seq[-1]
                random.shuffle(self.character_seq)
                seek = 0
            yield self.character_seq[seek:seek + self.batch_size]
            seek += self.batch_size


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


######################################################
path_chinese_synthetic = "./chinese_synthetic.txt"
path_img = '/data/User/李佳楠/data/ocr_background_img'
path_font = '/data/User/hanat/TF_CRNN_CTC/data/fonts'
path_have_yen_path = '/data/User/hanat/TF_CRNN_CTC/data/have_yen_fonts'     #'/data/User/hanat/TF_CRNN_CTC/data/have_yen_fonts'
path_save = '/hanat/data1/image_data'
annotation_file = '/hanat/data1/data_'
font_size_range = (30, 100)
process_num = 16

os.makedirs(path_save, exist_ok=True)
fp = open(path_chinese_synthetic, "r")
chinese_synth = fp.readline()
fp.close()

img_path = [os.path.join(path_img, img) for img in os.listdir(path_img) if img.split(".")[-1] in img_format and ('.DS' not in img)]
font_path = [(index, os.path.join(path_font, font)) for index,font in enumerate(os.listdir(path_font)) if font.split(".")[-1] in font_format and ('.DS' not in font)]
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
    random.shuffle(chinese_synth_list)
    character_gen = CharacterGen(chinese_synth_list, batch_size=batch_size)
    try:
        fnt = ImageFont.truetype(font, 32)
    except:
        return
    batch_repeat = 38250  # if batch_size < shuffle_limmit else len(chinese_synth) // 2 // 5
    for repeat_times in range(batch_repeat):
        fnt1 = fnt
        logger.info("font name is {}, font index {}, batch size is {} step is {}".format(font.split("/")[-1], str(index), str(batch_size), str(repeat_times)))
        char_list = character_gen.get_next.__next__()
        image = None
        font_size = random.randint(font_size_range[0], font_size_range[1])
        fnt1.size = font_size
        while image is None:
            img_seek = random.randint(0, len(img_path) - 1)
            try:
                image = cv2.imread(img_path[img_seek])
                image = Image.fromarray(image)
            except:
                continue
        try:
            if '¥' in char_list:
                ttf = TTFont(font)
                uni_list = ttf['cmap'].tables[0].ttFont.getGlyphOrder()
                rmb = u'yen'
                if rmb not in uni_list:
                    font_have_yen = random.randint(0, len(font_have_yen_path)-1)
                    font = font_have_yen_path[font_have_yen]
                    fnt1 = ImageFont.truetype(font, font_size)

            char_list = "".join(char_list)
            size = fnt1.getsize(char_list)
            bg_pil = random_region(image, size[0], size[1])
            draw = ImageDraw.Draw(bg_pil)
            color = get_fontcolor(bg_pil)
            draw.text((0, 0), char_list, fill=color, font=fnt1)
            image_name = str(index) + "_" + str(repeat_times) + ".jpg"
            image_save = os.path.join(path_save, image_name)
            bg_pil.save(image_save)
            if os.path.exists(image_save):
                if os.path.getsize(image_save) > 1:
                    annotation_info = image_name + "^" + char_list + "\n"
                    fp_txt.write(annotation_info)
                else:
                    os.remove(image_save)
        except:
            print("FUCK!!!!!!!", chinese_synth_list)
            continue
    fp_txt.close()

start = datetime.datetime.now()
pool = Pool(process_num)
print(len(font_path))
for font_info in font_path:
    pool.apply_async(ocr_data_thread, (font_info, ))
pool.close()
pool.join()

end = datetime.datetime.now()
print(end-start)

