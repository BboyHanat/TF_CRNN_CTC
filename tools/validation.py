"""
Name : validation.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-08-19 10:15
Desc:
"""

import argparse
import os
import tensorflow as tf
from crnn_model.crnn_net import ChineseCrnnNet
from config.global_config import cfg as CFG
from data_provider import shadownet_data_feed_pipline
from data_provider import tf_io_pipline_fast_tools

# define training parameter
hidden_nums = CFG.ARCH.HIDDEN_UNITS
hidden_layers = CFG.ARCH.HIDDEN_LAYERS
num_classes = CFG.ARCH.NUM_CLASSES
input_size = CFG.ARCH.INPUT_SIZE
seq_len = CFG.ARCH.SEQ_LENGTH
batch_size = CFG.TRAIN.BATCH_SIZE
epochs = CFG.TRAIN.EPOCHS
need_decode = CFG.TRAIN.NEED_DECODE
show_step = CFG.TRAIN.SHOW_STEP
val_times = CFG.VAL.VAL_TIMES
val_step = CFG.VAL.VAL_STEP
learning_rate = CFG.TRAIN.LEARNING_RATE
lr_decay_steps = CFG.TRAIN.LR_DECAY_STEPS
lr_decay_rate = CFG.TRAIN.LR_DECAY_RATE
lr_staircase = CFG.TRAIN.LR_STAIRCASE


def init_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', type=str, default='../data/',
                        help='Directory containing train_features.tfrecords')
    parser.add_argument('-w', '--weights_path', type=str, default='../ckpt/chinese_ocr/chinese_crnn_1000.ckpt',
                        help='Path to pre-trained weights to continue training')
    return parser.parse_args()


def validation_data(dataset_dir, weights_path):
    """

    :param dataset_dir:
    :param weights_path:
    :param train_data_num:
    :return:
    """

    sess = tf.Session()
    val_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
        dataset_dir=dataset_dir,
        flags='val'
    )
    val_images, val_labels, val_images_paths = val_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE
    )
    decoder = tf_io_pipline_fast_tools.FeatureDecoder(lexicon_path=os.path.join(dataset_dir + "lexicon.txt"))
    chinese_crnn = ChineseCrnnNet(hidden_nums=hidden_nums,
                                  layers_nums=hidden_layers,
                                  num_classes=num_classes,
                                  pretrained_model=weights_path,
                                  sess=sess,
                                  feature_decoder=decoder,
                                  learning_rate=learning_rate,
                                  lr_decay_steps=lr_decay_steps,
                                  lr_decay_rate=lr_decay_rate,
                                  lr_staircase=lr_staircase
                                  )
    chinese_crnn.validation(val_images,
                            val_labels,
                            sql_len=seq_len,
                            batch_size=batch_size,
                            val_times=val_times,
                            name="chinese_crnn"
                            )


if __name__ == '__main__':
    arg_init = init_args()
    validation_data(dataset_dir=arg_init.dataset_dir,
                    weights_path=arg_init.weights_path,
                    )
