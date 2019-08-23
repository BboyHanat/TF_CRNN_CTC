"""
Name : select_font_full.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-08-23 15:01
Desc:
"""

import os
import shutil
from fontTools.fontBuilder import TTFont
from fontTools import log
from glog import logger
import logging
import warnings
warnings.filterwarnings("ignore")
log.setLevel(logging.ERROR)

img_format = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
font_format = ["ttf", 'eot', 'fon', 'font', 'woff', 'woff2', 'otf', 'TTF', 'OTF', 'EOT', 'FONT', 'FON', 'WOFF', 'WOFF2']    # not support '.ttc' or '.TTC' font file

char_dict = {'0': 'zero',
             '1': 'one',
             '2': 'two',
             '3': 'three',
             '4': 'four',
             '5': 'five',
             '6': 'six',
             '7': 'seven',
             '8': 'eight',
             '9': 'nine',
             'A': 'A',
             'B': 'B',
             'C': 'C',
             'D': 'D',
             'E': 'E',
             'F': 'F',
             'G': 'G',
             'H': 'H',
             'I': 'I',
             'J': 'J',
             'K': 'K',
             'L': 'L',
             'M': 'M',
             'N': 'N',
             'O': 'O',
             'P': 'P',
             'Q': 'Q',
             'R': 'R',
             'S': 'S',
             'T': 'T',
             'U': 'U',
             'V': 'V',
             'W': 'W',
             'X': 'X',
             'Y': 'Y',
             'Z': 'Z',
             'a': 'a',
             'b': 'b',
             'c': 'c',
             'd': 'd',
             'e': 'e',
             'f': 'f',
             'g': 'g',
             'h': 'h',
             'i': 'i',
             'j': 'j',
             'k': 'k',
             'l': 'l',
             'm': 'm',
             'n': 'n',
             'o': 'o',
             'p': 'p',
             'q': 'q',
             'r': 'r',
             's': 's',
             't': 't',
             'u': 'u',
             'v': 'v',
             'w': 'w',
             'x': 'x',
             'y': 'y',
             'z': 'z',
             ',': 'comma',
             '.': 'period',
             '?': 'question',
             '$': 'dollar',
             '+': 'plus',
             '-': 'hyphen',
             '*': 'asterisk',
             '/': 'slash',
             '%': 'percent',
             '@': 'at',
             '(': 'parenleft',
             ')': 'parenright',
             ':': 'colon',
             '~': 'asciitilde'
             }


def check_fonts(char_list, font):
    """
    check fonts
    :param char_list:
    :param font:
    :return: True or False
    """

    for char in reversed(char_list):
        unicode_char = char.encode("unicode_escape")
        utf_8_char = unicode_char.decode('utf-8').split('\\')[-1]
        utf_8_char = utf_8_char if len(utf_8_char)==1 else utf_8_char[1:]
        if utf_8_char in char_dict.keys():
            utf_8_char_check = char_dict[utf_8_char]

        elif char != '¥':
            utf_8_char = unicode_char.decode('utf-8').split('\\')[-1].strip('u')
            utf_8_char_check = 'uni' + utf_8_char.upper()
        else:
            continue
        try:
            ttf = TTFont(font)
            lower = ttf.getGlyphSet().get(utf_8_char_check)
            if lower is None:
                logger.info('1char {} is not in font'.format(char))
                return False
            else:
                if lower._glyph.numberOfContours == 0:
                    logger.info('2char {} is not in font'.format(char))
                    return False
                else:
                    continue
        except:
            logger.info('3char {} is not in font'.format(char))
            return False
    return True


if __name__ == '__main__':
    path_chinese_synthetic = './chinese_synthetic.txt'
    fp = open(path_chinese_synthetic, "r")
    chinese_synth = fp.readline()
    char_list = list(chinese_synth)
    path_to_font = '/data/User/hanat/TF_CRNN_CTC/data/fonts'
    mv_path = "/data/User/hanat/TF_CRNN_CTC/data/fonts_bak"
    os.makedirs(mv_path, exist_ok=True)
    font_path = [os.path.join(path_to_font, font) for font in os.listdir(path_to_font) if font.split(".")[-1] in font_format and ('.DS' not in font)]
    print(len(font_path))
    for index, font in enumerate(font_path):
        print(index)
        if not check_fonts(char_list, font):
            font_name = font.split('/')[-1]
            print(font_name)
            save_path = os.path.join(mv_path, font_name)
            print(save_path)
            shutil.move(font, save_path)
