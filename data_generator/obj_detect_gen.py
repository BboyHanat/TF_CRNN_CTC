"""
Name : obj_detect_gen.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-08-20 14:03
Desc:
"""
import numpy as np
import cv2
import random
import datetime
import shutil
import os
import pprint
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
from PIL import Image, ImageDraw, ImageFont
from local_utils.establish_char_dict import *
from fontTools.fontBuilder import TTFont
from fontTools import log
import glog as logger
import logging
import warnings
warnings.filterwarnings("ignore")
log.setLevel(logging.ERROR)

img_format = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
font_format = ["ttf", 'eot', 'fon', 'font', 'woff', 'woff2', 'otf', 'TTF', 'OTF', 'EOT', 'FONT', 'FON', 'WOFF', 'WOFF2']    # not support '.ttc' or '.TTC' font file


class CharacterGen:
    def __init__(self, character_seq, len_range=(2, 16)):
        """

        :param character_seq:
        :param batch_size:
        """
        self.character_seq = character_seq
        random.shuffle(self.character_seq)
        self.len_range = len_range
        self.get_next = self._get_next_batch()

    def _get_next_batch(self):
        """

        :return:
        """
        seek = 0
        while True:
            length = random.randint(self.len_range[0], self.len_range[1])
            if len(self.character_seq) - seek < length:
                for i in range(len(self.character_seq) - seek):
                    self.character_seq.insert(0, self.character_seq[-1])
                    del self.character_seq[-1]
                random.shuffle(self.character_seq)
                seek = 0
            yield self.character_seq[seek:seek + length]
            seek += length


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
    v_new = (random.randint(30, 225)+v_mean) % 255
    hsv_rgb = np.asarray([[[h_new, s_new, v_new]]], np.uint8)
    rbg = cv2.cvtColor(hsv_rgb, cv2.COLOR_HSV2RGB_FULL)
    r = rbg[0, 0, 0]
    g = rbg[0, 0, 1]
    b = rbg[0, 0, 2]
    print(r, g, b)


    # b_mean = int(np.mean(image[:, :, 0]))
    # g_mean = int(np.mean(image[:, :, 1]))
    # r_mean = int(np.mean(image[:, :, 2]))
    # b_range = (0, b_mean - 80) if b_mean > 127 else (b_mean + 80, 255)
    # g_range = (0, g_mean - 80) if g_mean > 127 else (g_mean + 80, 255)
    # r_range = (0, r_mean - 80) if r_mean > 127 else (r_mean + 80, 255)
    #
    # b = random.randint(b_range[0], b_range[1])
    # g = random.randint(g_range[0], g_range[1])
    # r = random.randint(r_range[0], r_range[1])
    return (r, g, b)


def random_region(image, width_limit=100, height_limit=100):
    """

    :param image:
    :return:
    """
    w, h = image.size
    if width_limit < w - 1 and height_limit < h - 1:
        x1 = random.randint(0, w - width_limit - 1)
        y1 = random.randint(0, h - height_limit - 1)
        x2 = x1 + width_limit
        y2 = y1 + height_limit
    else:
        x1 = -1
        y1 = -1
        x2 = -1
        y2 = -1

    return x1, y1, x2, y2


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


def get_font_image(font_gen:FontGen, font_have_yen_gen:FontGen, char_list, font_size_range):
    """

    :param font_gen:
    :param font_have_yen_gen:
    :param char_list:
    :param font_size_range:
    :return:
    """
    def get_fnt(font_size, char_list):
        """
        get font
        :param font:
        :param font_size:
        :param char_list:
        :param font_have_yen_path:
        :return:
        """
        font = font_gen.get_next.__next__()
        index = len(char_list) - 1
        for char in reversed(char_list):
            unicode_char = char.encode("unicode_escape")
            if unicode_char < b'\\u4e00' and char != '¥':
                continue
            # if unicode_char > b'\\u9fcb':
            #     logger.info('delete char {} who > 9fcb'.format(char_list[index]))
            #     del char_list[index]
            #     index -= 1
            #     continue

            if char != '¥':
                utf_8_char = unicode_char.decode('utf-8').split('\\')[-1].strip('u')
                utf_8_char_lower = 'uni' + utf_8_char
                utf_8_char_upper ='uni' + utf_8_char.upper()
            else:
                utf_8_char_lower = 'yen'
                utf_8_char_upper = 'YEN'
                font = font_have_yen_gen.get_next.__next__()

            ttf = TTFont(font)
            lower = ttf.getGlyphSet().get(utf_8_char_lower)
            upper = ttf.getGlyphSet().get(utf_8_char_upper)
            if lower is None and upper is None:
                logger.info('delete char {}'.format(char_list[index]))
                del char_list[index]
            elif lower is not None:
                if lower._glyph.numberOfContours == 0:
                    logger.info('delete char {}'.format(char_list[index]))
                    del char_list[index]
            elif upper is not None:
                if upper._glyph.numberOfContours == 0:
                    logger.info('delete char {}'.format(char_list[index]))
                    del char_list[index]
            else:
                del char_list[index]

            index -= 1
        if len(char_list) < 2:
            return None, char_list
        else:
            try:
                return ImageFont.truetype(font, font_size), char_list
            except:
                logger.warning("can't load font file!")
                return None, char_list

    def random_rotate(bg, angle_range=(-6, 6)):
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

    aspect_arr = ['v', 'h', 'h']
    a = random.randint(0, 2)
    aspect = aspect_arr[a]
    font_size = random.randint(font_size_range[0], font_size_range[1])
    char_list_aspect = []
    for char in char_list:
        char_list_aspect.append(char)
        if aspect == 'v':
            char_list_aspect.append('\n')
    fnt, char_list_aspect = get_fnt(font_size, char_list_aspect)
    if fnt is None:
        return None, -1
    char_str = "".join(char_list_aspect)
    spacing = random.randint(0, 2)
    if aspect == 'v':
        size = fnt.getsize_multiline(char_str, spacing=spacing)
    else:
        size = fnt.getsize(char_str)
    font_bg = Image.new("RGBA", (size[0], size[1]), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(font_bg)
    if aspect == 'h':
        draw.text((0, 0), char_str, fill=(255, 255, 255, 255), font=fnt)
    else:
        draw.multiline_text((0, 0), char_str, fill=(255, 255, 255, 255), spacing=spacing, font=fnt)
    font_bg = random_rotate(font_bg)
    return font_bg, aspect


def is_covered_any(coords, coord, aspect, iou_ratio, distance):
    """

    :param coords: a list of coordinate
    :param coord:
    :return: return True if have releation
    """

    def judge_releations(coord1, coord2):
        y_intersec = min(coord1[3], coord2[3]) - max(coord1[1], coord2[1])
        x_intersec = min(coord1[2], coord2[2]) - max(coord1[0], coord2[0])
        if y_intersec * x_intersec > 0:
            return True

        if aspect == 'h':
            union = max(coord1[3], coord2[3]) - min(coord1[1], coord2[1])
            iou = y_intersec / union
            if iou > iou_ratio:
                return True if distance > max(coord1[0], coord2[0]) - min(coord1[1], coord2[1]) else False
            else:
                return False
        elif aspect == 'v':
            union = max(coord1[2], coord2[2]) - min(coord1[0], coord2[0])
            iou = x_intersec / union
            if iou > iou_ratio:
                return True if distance > max(coord1[1], coord2[1]) - min(coord1[3], coord2[3]) else False
            else:
                return False

    if coord[0] < 0:
        return False

    for c in coords:
        if judge_releations(c, coord):
            return False

    return True


def xml_writer(save_dir, image_name, w, h, coord_list, label_list, image_format='jpg'):
    """

    :param image_name:
    :param coord_list:
    :param label_list:
    :return:
    """
    node_root = Element('annotation')
    '''folder'''
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
    '''filename'''
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = 'image_name'
    '''source'''
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'The VOC2007 Database'
    node_annotation = SubElement(node_source, 'annotation')
    node_annotation.text = 'PASCAL VOC2007'
    node_image = SubElement(node_source, 'image')
    node_image.text = 'flickr'
    '''size'''
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(w)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(h)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    '''segmented'''
    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'
    '''object coord and label'''
    for i, coord in enumerate(coord_list):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = label_list[i] + "_text"
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(coord[0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(coord[1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(coord[2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(coord[3])

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    dom = parseString(xml)

    save_xml = os.path.join(save_dir, image_name.replace(image_format, 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)


def obj_data_creator(path_chinese_synthetic, path_to_img, path_to_font, font_have_yen_path, path_save_img, path_save_xml, repeat_times=200):
    """

    :param path_chinese_synthetic:
    :param path_to_img:
    :param path_to_font:
    :param font_have_yen_path:
    :param path_save_img:
    :param path_save_xml:
    :param font_repeat_times:
    :return:
    """
    os.makedirs(path_save_img, exist_ok=True)
    os.makedirs(path_save_xml, exist_ok=True)
    fp = open(path_chinese_synthetic, "r")
    chinese_synth = fp.readline()

    img_path = [os.path.join(path_to_img, img) for img in os.listdir(path_to_img) if img.split(".")[-1] in img_format and ('.DS' not in img)]
    font_path = [os.path.join(path_to_font, font) for font in os.listdir(path_to_font) if font.split(".")[-1] in font_format and ('.DS' not in font)]
    font_have_yen_path = [os.path.join(font_have_yen_path, font) for font in os.listdir(font_have_yen_path) if font.split(".")[-1] in font_format and ('.DS' not in font)]
    fp.close()

    chinese_synth_list = list(chinese_synth)
    image_gen = ImageGen(img_path)
    character_gen = CharacterGen(chinese_synth_list)
    font_gen = FontGen(font_path)
    font_have_yen_gen = FontGen(font_have_yen_path)
    repeat = 0
    while True:
        image_bg = image_gen.get_next.__next__()
        image_w, image_h = image_bg.size
        image_repeat = 0
        image_repeat_maximum = random.randint(2, 6)
        side = image_w if image_w < image_h else image_h
        font_size_range = (int(side * 0.05), int(side * 0.09))
        coord_list = []
        label_list = []
        while True:
            char_list = character_gen.get_next.__next__()

            font_bg, aspect = get_font_image(font_gen, font_have_yen_gen, char_list, font_size_range)
            if font_bg is None or font_bg.size[0] > image_w or font_bg.size[1] > image_h:
                continue
            font_bg_w, font_bg_h = font_bg.size
            x1, y1, x2, y2 = random_region(image_bg, font_bg_w, font_bg_h)
            distance = image_w * 0.1 if aspect == 'h' else image_h * 0.1
            if is_covered_any(coord_list, [x1, y1, x2, y2], aspect, 0.8, distance=distance):
                """paste image"""
                print(char_list)
                image_bg = random_paste(image_bg, font_bg, x1, y1, x2, y2)
                coord_list.append([x1, y1, x2, y2])
                label_list.append(aspect)

                image_repeat += 1
                if image_repeat >= image_repeat_maximum:
                    break

        """image xml generate"""
        if len(label_list) > 0:
            now = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            image_name = now + "_" + str(image_repeat_maximum) + "_" + str(repeat_times) + "_" + str(repeat) + ".jpg"    # 时间戳_目标数量_某一个字体第几次生成的样本_字体索引.jpg
            image_save = os.path.join(path_save_img, image_name)
            image_bg.save(image_save)
            if os.path.exists(image_save):
                if os.path.getsize(image_save) > 1:
                    logger.info("generate images num {}".format(str(repeat)))
                    xml_writer(path_save_xml, image_name, image_w, image_h, coord_list, label_list)
                else:
                    os.remove(image_save)
        repeat += 1
        if repeat > repeat_times:
            break


if __name__ == "__main__":
    path_chinese_synthetic = "./chinese_synthetic.txt"      # 字库文件路径
    path_to_img = '/Users/aidaihanati/TezignProject/PSD/替换素材(1)/images/background'      # 背景图所在文件夹路径  机器学习服务器对应的路径 '/data/User/hanat/data/bg_image'
    path_to_font = '/Users/aidaihanati/TezignProject/Font'      # 字体文件所在文件夹的路径  机器学习服务器对应的路径 '/data/User/hanat/TF_CRNN_CTC/data/font'
    font_have_yen_path = '/Users/aidaihanati/TezignProject/Font'    # 具有符号 '¥' 的字体库文件 机器学习服务器对应的路径 '/data/User/hanat/TF_CRNN_CTC/data/have_yen_fonts'
    path_save_img = '../data/voc_test/image'    # 图片生成文件夹路径
    path_save_xml = '../data/voc_test/xml'      # 每一张图对应的xml 文件所在路径
    repeat_times = 5000     # 样本生成的数量 相当于生成5000个样本
    obj_data_creator(path_chinese_synthetic, path_to_img, path_to_font, font_have_yen_path, path_save_img, path_save_xml, repeat_times)
