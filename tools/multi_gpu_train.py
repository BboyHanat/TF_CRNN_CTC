"""
Name : multi_gpu_train.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-08-26 14:19
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

    parser.add_argument('-d', '--dataset_dir', type=str, default='/hanat/data1/',
                        help='Directory containing train_features.tfrecords')
    parser.add_argument('-w', '--weights_path', type=str, default='../ckpt/chinese_ocr_multi/chinese_crnn_20340.ckpt',
                        help='Path to pre-trained weights to continue training')
    parser.add_argument('-tm', '--train_data_num', type=int, default=3471510,
                        help='Path to pre-trained weights to continue training')
    parser.add_argument('-g', '--gpu_num', type=int, default=4,
                        help='Path to pre-trained weights to continue training')

    return parser.parse_args()


def train(dataset_dir, weights_path, train_data_num, gpu_num):
    """
    train
    :param dataset_dir:
    :param weights_path:
    :param train_data_num:
    :return:
    """
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    # sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess = tf.Session(config=sess_config)
    # prepare dataset
    train_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
        dataset_dir=dataset_dir,
        flags='train'
    )
    val_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
        dataset_dir=dataset_dir,
        flags='val'
    )
    train_images, train_labels, train_images_paths = train_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE
    )
    val_images, val_labels, val_images_paths = val_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE
    )
    train_epochs = train_data_num // (batch_size*gpu_num) * 100
    val_epochs = train_data_num // (batch_size*gpu_num)
    save_epochs = train_data_num // (batch_size*gpu_num)
    show_epochs = 100
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

    chinese_crnn.multi_gpu_train(gpu_num=gpu_num,
                                 train_input_data=train_images,
                                 train_label=train_labels,
                                 val_input_data=val_images,
                                 val_label=val_labels,
                                 sql_len=seq_len,
                                 batch_size=batch_size,
                                 name="chinese_crnn",
                                 val_epochs=val_epochs,
                                 train_epochs=train_epochs,
                                 save_epochs=save_epochs,
                                 show_epochs=show_epochs
                                 )


if __name__ == '__main__':
    arg_init = init_args()
    train(dataset_dir=arg_init.dataset_dir,
          weights_path=arg_init.weights_path,
          train_data_num=arg_init.train_data_num,
          gpu_num=arg_init.gpu_num
          )
