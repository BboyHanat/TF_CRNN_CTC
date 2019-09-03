"""
Name : merge_txt.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-09-02 10:53
Desc:
"""

import os


def get_gt(txt_path):
    """
    get ground truth
    :param txt_path:
    :return:
    """
    fp = open(txt_path, "r")
    chinese_synth = fp.readline()
    chinese_synth = list(chinese_synth)
    return chinese_synth


def get_text(root_path):
    """
    get text
    :param root_path:
    :return:
    """
    text_path = [os.path.join(root_path, txt_P) for txt_P in os.listdir(root_path) if os.path.isdir (os.path.join(root_path,txt_P)) and '.DS' not in txt_P]
    txt_list = []
    for tp in text_path:
        txt_list[len(txt_list):] = [os.path.join(tp, txt) for txt in os.listdir(tp) if txt.endswith('.txt') and '.DS' not in txt]

    synth_list = []
    for index, txt in enumerate(txt_list):
        fp = open(txt, "r")
        synth_lines = fp.readlines()
        for line in synth_lines:
            if not line == '':
                line = line.replace('\n', '')
                line = line.replace('\t','')
                line = line.replace('\r','')
                line = line.replace('---', '')
                line = line.replace(' ','')
                line = line.replace('\u3000', '')
                line = list(line)
                if not line == []:
                    print(line)
                    synth_list.append(line)
            else:
                break
    return synth_list


def filter_text(synth_list, chinese_synth):
    """

    :param synth_list:
    :param chinese_synth:
    :return:
    """
    print(chinese_synth)
    synth_list_filtered = []
    for synth in synth_list:
        for index in range(len(synth) - 1, 0, -1):
            if synth[index] not in chinese_synth or synth[index] in [' ']:
                del synth[index]

        if not synth == []:
            synth_list_filtered.append(synth)

    return synth_list_filtered


def write_line(synth_list_filtered, save_path, txt_save_name):
    """
    write line
    :param synth_list_filtered:
    :param save_path:
    :return:
    """
    os.makedirs(save_path, exist_ok=True)
    file = os.path.join(save_path, txt_save_name)
    fp = open(file, 'w')
    output_str = []
    for synth in synth_list_filtered:
        if len(synth) > 15:
            len_sentence = len(synth)
            list_len = len_sentence // 15 if len_sentence // 15 == len_sentence / 15 else len_sentence // 15 + 1
            for i in range(list_len - 1):
                last_shift = (i + 1) * 15 if len_sentence - (i + 1) * 15 >= 15 else len_sentence - (i + 1) * 15
                sentence = ''.join(synth[i * 15:last_shift])
                sentence = sentence.strip('---')
                sentence = sentence.strip('\n')
                sentence = sentence + '\n'
                if sentence not in ['\n', '\t', '\r'] and len(sentence) > 1:
                    output_str.append(sentence)
        else:
            sentence = ''.join(synth)
            sentence = sentence.strip('\n')
            sentence = sentence + '\n'
            if len(sentence) > 1:
                output_str.append(sentence)
    print(len(output_str))
    fp.writelines(output_str)


if __name__ == "__main__":
    chinese_synth_path = './chinese_synthetic.txt'
    synth_path = './Sougou_reduced'
    save_path = './'
    txt_save_name = 'sentence.txt'
    chinese_synth = get_gt(chinese_synth_path)
    synth_list = get_text(synth_path)
    synth_list_filtered = filter_text(synth_list, chinese_synth)
    write_line(synth_list_filtered, save_path, txt_save_name)

