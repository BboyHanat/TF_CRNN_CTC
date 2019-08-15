"""
Name : chinese_synthetic_gen.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-08-07 10:15
Desc:
"""
import sys

sys.path.append("..")
import numpy as np
import cv2
import random
import datetime
import shutil
import os
from PIL import Image, ImageDraw, ImageFont
from local_utils.establish_char_dict import *

img_format = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
font_format = ["ttf", 'eot', 'fon', 'font', 'woff', 'woff2', 'otf', 'ttc', 'TTF', 'TTC', 'OTF', 'EOT', 'FONT', 'FON', 'WOFF', 'WOFF2']
symbol = [',', '.', '。', '?', '¥', '$', '#', '+', '(', ')', '/', '-', '*', '@', '%', '~']


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
    roi = Image.new("RGB", (width_limit, height_limit), color=(0, 0, 0))
    return roi


def get_fontcolor(image):
    """
    get font color by mean
    :param image:
    :return:
    """
    image = np.asarray(image)
    b_mean = int(np.mean(image[:, :, 0]))
    g_mean = int(np.mean(image[:, :, 1]))
    r_mean = int(np.mean(image[:, :, 2]))
    b_range = (0, b_mean - 50) if b_mean > 127 else (b_mean + 50, 255)
    g_range = (0, g_mean - 50) if g_mean > 127 else (g_mean + 50, 255)
    r_range = (0, r_mean - 50) if r_mean > 127 else (r_mean + 50, 255)

    b = random.randint(b_range[0], b_range[1])
    g = random.randint(g_range[0], g_range[1])
    r = random.randint(r_range[0], r_range[1])
    return (r, g, b)


def random_get_font(font_lsit):
    """
    get font by lsit
    :param font_lsit:
    :return:
    """
    font = random.randint(0, len(font_lsit))
    return font


def random_shuffle_add_symbol(synthetic, two_symbol_limit=7):
    """
    random shuffle and random add symbol
    :param synthetic:
    :return:
    """
    random.shuffle(synthetic)
    a = random.randint(0, 10)
    if a >= 5 and a < 7:
        which_symbol = random.randint(0, len(symbol))
        loc = random.randint(0, len(synthetic))
        synthetic.insert(loc, symbol[which_symbol])
    elif a >= 7 and len(synthetic) > two_symbol_limit:
        which_symbol1 = random.randint(0, len(symbol))
        which_symbol2 = random.randint(0, len(symbol))
        loc1 = random.randint(0, len(synthetic))
        synthetic.insert(loc1, symbol[which_symbol1])
        loc2 = random.randint(0, len(synthetic))
        synthetic.insert(loc2, symbol[which_symbol2])
    return synthetic


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


def ocr_data_create(path_chinese_synthetic, path_img, path_font, path_save, annotation_file, char_dict_path, ord_map_path, shuffle_limmit=5, shuffle_repeat=5, font_size_range=(30, 100)):
    """
    生成ocr数据
    1 生成文字ocr图像数据
    2 生成txt文件 内容每一行（*_*.jpg 杭啊哎哦发卷发）
    3 生成json 文字与编码映射文件
    4 生成json 对编码做索引
    :param path_chinese_synthetic:
    :param path_img:
    :param path_font:
    :param path_save:
    :param annotation_file:
    :param char_dict_path:
    :param ord_map_path:
    :param shuffle_limmit:
    :param shuffle_repeat:
    :param font_size_range:
    :return:
    """
    fp = open(path_chinese_synthetic, "r")
    chinese_synth = fp.readline()
    # char_builder = CharDictBuilder()
    # char_builder.write_char_dict(list(chinese_synth), char_dict_path)
    # char_builder.map_ord_to_index(list(chinese_synth), ord_map_path)

    img_path = [os.path.join(path_img, img) for img in os.listdir(path_img) if img.split(".")[-1] in img_format and ('.DS' not in img)]
    font_path = [os.path.join(path_font, font) for font in os.listdir(path_font) if font.split(".")[-1] in font_format and ('.DS' not in font)]

    fp = open(annotation_file, "w")
    epoch = 0
    for f_index, font in enumerate(font_path):
        # fnt_bytes = open(font,'r')
        for batch_size in range(15, 16):
            chinese_synth_list = list(chinese_synth)
            random.shuffle(chinese_synth_list)
            character_gen = CharacterGen(chinese_synth_list, batch_size=batch_size)

            print("font name is {}, font index {}, generate epoch {}, batch size is {}".format(font.split("/")[-1], str(f_index), str(epoch), str(batch_size)))
            batch_repeat = len(chinese_synth) // 2 + 1  # if batch_size < shuffle_limmit else len(chinese_synth) // 2 // 5
            for repeat_times in range(batch_repeat):
                char_list = character_gen.get_next.__next__()
                image = None
                while image is None:
                    img_seek = random.randint(0, len(img_path) - 1)
                    try:
                        image = cv2.imread(img_path[img_seek])
                        image = Image.fromarray(image)
                    except:
                        continue
                font_size = random.randint(font_size_range[0], font_size_range[1])
                try:
                    fnt = ImageFont.truetype(font, font_size)

                    if len(char_list) > shuffle_limmit:
                        for shuffle_times in range(shuffle_repeat):
                            char_list_shuffled = random_shuffle_synthetic(char_list)
                            char_list_shuffled = "".join(char_list_shuffled)
                            size = fnt.getsize(char_list_shuffled)
                            bg_pil = random_region(image, size[0], size[1])
                            draw = ImageDraw.Draw(bg_pil)
                            color = get_fontcolor(bg_pil)
                            draw.text((0, 0), char_list_shuffled, fill=color, font=fnt)
                            now = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                            image_name = now + "_" + str(shuffle_times) + "_" + str(repeat_times) + ".jpg"
                            image_save = os.path.join(path_save, image_name)
                            bg_pil.save(image_save)
                            if os.path.exists(image_save):
                                if os.path.getsize(image_save) > 1:
                                    annotation_info = image_name + "^" + char_list_shuffled + "\n"
                                    fp.write(annotation_info)
                                else:
                                    os.remove(image_save)
                    else:
                        char_list = "".join(char_list)
                        size = fnt.getsize(char_list)
                        bg_pil = random_region(image, size[0], size[1])
                        draw = ImageDraw.Draw(bg_pil)
                        color = get_fontcolor(bg_pil)
                        draw.text((0, 0), char_list, fill=color, font=fnt)
                        now = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                        image_name = now + "_" + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + "_" + str(repeat_times) + ".jpg"
                        image_save = os.path.join(path_save, image_name)
                        bg_pil.save(image_save)
                        if os.path.exists(image_save):
                            if os.path.getsize(image_save) > 1:
                                annotation_info = image_name + "^" + char_list + "\n"
                                fp.write(annotation_info)
                            else:
                                os.remove(image_save)
                except:
                    print("FUCK!!!!!!!", chinese_synth_list)
                    continue
            epoch += 1


if __name__ == "__main__":
    path_chinese_synthetic = "./chinese_127.txt"
    path_img = '/data/User/hanat/data/bg_image'
    path_font = '/data/User/hanat/CRNN_Tensorflow/data/fonts'
    path_save = '../data/train'
    txt_save = '../data/train.txt'
    char_dict_path = '../data/char_dict/char_dict.json'
    ord_map_path = '../data/char_dict/ord_map.json'
    ocr_data_create(path_chinese_synthetic, path_img, path_font, path_save, txt_save, char_dict_path, ord_map_path)
