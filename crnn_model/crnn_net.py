#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import time
from os import path as osp
from tensorflow.contrib import rnn
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib import slim
from crnn_model import cnn_basenet
from crnn_model.vgg import vgg_a


class ChineseCrnnNet:
    """
        Implement the crnn model for squence recognition
    """

    def __init__(self,
                 hidden_nums,
                 layers_nums,
                 num_classes,
                 pretrained_model,
                 sess,
                 feature_decoder,
                 learning_rate,
                 lr_decay_steps,
                 lr_decay_rate,
                 lr_staircase
                 ):
        """

        :param phase: 'Train' or 'Test'
        :param hidden_nums: Number of hidden units in each LSTM cell (block)
        :param layers_nums: Number of LSTM cells (blocks)
        :param num_classes: Number of classes (different symbols) to detect
        """
        self._hidden_nums = hidden_nums
        self._layers_nums = layers_nums
        self._num_classes = num_classes
        self.sess = sess
        self.pretrained_model = pretrained_model
        self.feature_decoder = feature_decoder
        self.learning_rate = learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.lr_staircase = lr_staircase
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, 32, None, 3], name='data_input')
        self.labels = tf.sparse_placeholder(dtype=tf.int32, name='label_input')

    def _map_to_sequence(self, inputdata, name):
        """
        Implements the map to sequence part of the network.
        This is used to convert the CNN feature map to the sequence used in the stacked LSTM layers later on.
        Note that this determines the length of the sequences that the LSTM expects
        :param inputdata:tensor[batch, h, w, c]
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            shape = inputdata.get_shape().as_list()
            print(shape)
            assert shape[1] == 1  # H of the feature map must equal to 1

            ret = tf.squeeze(inputdata, axis=1, name='squeeze')

        return ret

    def _sequence_label(self, inputdata, name):
        """
        Implements the sequence label part of the network
        :param inputdata: tensor[batch, h, w, c]
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # construct stack lstm rcnn layer
            # forward lstm cell
            fw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for
                            nh in [self._hidden_nums] * self._layers_nums]
            # Backward direction cells
            bw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for
                            nh in [self._hidden_nums] * self._layers_nums]

            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_list, bw_cell_list, inputdata,
                dtype=tf.float32
            )
            stack_lstm_layer = tf.nn.dropout(stack_lstm_layer, 0.5, name='sequence_drop_out')
            [_, _, hidden_nums] = inputdata.get_shape().as_list()  # [batch, width, 2*n_hidden]
            # shape = tf.shape(stack_lstm_layer)
            # rnn_reshaped = tf.reshape(stack_lstm_layer, [shape[0] * shape[1], shape[2]])
            logits = slim.fully_connected(stack_lstm_layer, self._num_classes, activation_fn=None)

            # # w = tf.get_variable(
            # #     name='w',
            # #     shape=[hidden_nums, self._num_classes],
            # #     initializer=tf.truncated_normal_initializer(stddev=0.02),
            # #     trainable=True
            # # )
            # # Doing the affine projection
            # logits = tf.matmul(rnn_reshaped, w, name='logits')
            #logits = tf.reshape(logits, [shape[0], shape[1], self._num_classes], name='logits_reshape')
            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')
            # Swap batch and batch axis
            rnn_out = tf.transpose(logits, [1, 0, 2], name='transpose_time_major')  # [width, batch, n_classes]
        return rnn_out, raw_pred

    def inference(self, name, reuse=False):
        """
        Main routine to construct the network
        :param inputdata:tensor[batch, h, w, c]
        :param name:
        :param reuse:
        :return:
        """
        # first apply the cnn feature extraction stage
        cnn_out = vgg_a(self.input_data)
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # second apply the map to sequence stage
            sequence = self._map_to_sequence(
                inputdata=cnn_out, name='map_to_sequence_module'
            )
            # third apply the sequence label stage
            net_out, raw_pred = self._sequence_label(
                inputdata=sequence, name='sequence_rnn_module'
            )
        return net_out

    def decode_sequence(self, rnn_output, seq_len, batch_size):
        """
        decode output sequence from network
        :param net_out:
        :return:
        """
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            rnn_output,
            seq_len * np.ones(batch_size),
            merge_repeated=False
        )
        return decoded, log_prob

    def compute_loss(self, rnn_output, seq_len, batch_size):
        """
        compute network loss
        :param inputdata:tensor[batch, h, w, c]
        :param labels:tensor[batch, [sparse label sequence]]
        :param name:
        :param reuse:
        :return:
        """
        loss = tf.reduce_mean(
            tf.nn.ctc_loss(
                labels=self.labels, inputs=rnn_output,
                sequence_length=seq_len * np.ones(batch_size),
            ),
            name='ctc_loss'
        )
        return loss

    def graph_optimizer(self, loss, global_step):
        """
        optimizer the graph
        :param inputdata:tensor[batch, h, w, c]
        :param labels:
        :return:
        """
        # learning_rate = tf.train.exponential_decay(
        #     learning_rate=self.learning_rate,
        #     global_step=global_step,
        #     decay_steps=self.lr_decay_steps,
        #     decay_rate=self.lr_decay_rate,
        #     staircase=self.lr_staircase)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     optimizer = tf.train.MomentumOptimizer(
        #         learning_rate=learning_rate, momentum=0.9).minimize(
        #         loss=loss, global_step=global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=loss, global_step=global_step)

        return optimizer, self.learning_rate

    def compute_accuracy(self, ground_truth, decode_sequence, display=False, mode='per_char'):
        """
        Computes accuracy
        :param ground_truth:
        :param decode_sequence: decoded feature sequence
        :param display: if you want to show accuary,you need to set this parameter to True
        :param mode: full_sequence or per char to compute accuracy
        :return: avg accuracy
        """
        str_lists, number_lists = self.feature_decoder.sparse_tensor_to_str(decode_sequence)
        number_lists = list(number_lists)
        if mode == 'per_char':
            accuracy = []
            for index, label in enumerate(ground_truth):
                prediction = list(number_lists[index])
                total_count = len(label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(label):
                        try:
                            if tmp == prediction[i]:
                                correct_count += 1
                        except:
                            continue
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / total_count)
                    except ZeroDivisionError:
                        if len(prediction) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)
            avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
        elif mode == 'full_sequence':
            try:
                correct_count = 0
                for index, label in enumerate(ground_truth):
                    prediction = list(number_lists[index])
                    if prediction == label:
                        correct_count += 1
                avg_accuracy = correct_count / len(ground_truth)
            except ZeroDivisionError:
                if len(number_lists) == 0:
                    avg_accuracy = 1
                else:
                    avg_accuracy = 0
        else:
            raise NotImplementedError('Other accuracy compute mode has not been implemented')
        if display:
            print('Mean accuracy is {:5f}'.format(avg_accuracy))
        return avg_accuracy

    def load_pretrained_model(self):
        """
        load pretrained model from file path
        this is current mothod, you can take this code to another project with very few modifications
        :return:
        """

        def get_variables_in_checkpoint_file(file_name):
            try:
                reader = pywrap_tensorflow.NewCheckpointReader(file_name)
                var_to_shape_map = reader.get_variable_to_shape_map()
                return var_to_shape_map
            except Exception as e:  # pylint: disable=broad-except
                print(str(e))
                if "corrupted compressed block contents" in str(e):
                    print("It's likely that your checkpoint file has been compressed "
                          "with SNAPPY.")

        def get_variables_to_restore(variables, var_keep_dic):
            variables_to_restore = []
            for v in variables:
                # exclude
                if v.name.split(':')[0] in var_keep_dic:
                    print('Variables restored: %s' % v.name)
                    variables_to_restore.append(v)
                else:
                    print('Variables restored: %s' % v.name)
            return variables_to_restore

        variables = tf.global_variables()
        self.sess.run(tf.variables_initializer(variables, name='init'))
        print("variables initilized ok")
        # Get dictionary of model variable
        if self.pretrained_model is not None:
            var_keep_dic = get_variables_in_checkpoint_file(self.pretrained_model)
            # # Get the variables to restore
            variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(self.sess, self.pretrained_model)

    def train(self,
              train_input_data,
              train_label,
              val_input_data,
              val_label,
              sql_len,
              batch_size,
              name,
              train_data_num,
              val_times,
              train_epochs=1000,
              show_step=1000,
              val_step=3000,
              need_decode=True,
              tboard_save_dir='../tboard/crnn_chinese_ocr',
              model_save_dir='../ckpt/chinese_ocr'
              ):
        """
        train network, and validation networrk every (val_step) step training
        :param train_input_data:
        :param train_label:
        :param val_input_data:
        :param val_label:
        :param batch_size:
        :param name:
        :param train_data_num:
        :param val_times:
        :param train_epochs:
        :param show_step:
        :param val_step:
        :param need_decode:
        :param tboard_save_dir:
        :param model_save_dir:
        :return:
        """

        # define crnn network and optimizer
        global_step = tf.Variable(0, dtype=tf.int32, name='g_step', trainable=False)
        inference_ret = self.inference(name=name, reuse=False)
        decode, log_prob = self.decode_sequence(inference_ret, sql_len, batch_size)
        loss = self.compute_loss(inference_ret,sql_len, batch_size)
        optimizer, learning_rate = self.graph_optimizer(loss, global_step)

        # define tensorflow summary
        tf.summary.scalar(name='train_ctc_loss', tensor=loss)
        tf.summary.scalar(name='learning_rate', tensor=learning_rate)
        merge_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(tboard_save_dir)
        summary_writer.add_graph(self.sess.graph)
        saver = tf.train.Saver()

        # load pretrained model if the path had been declared
        self.load_pretrained_model()

        epoch = 0
        while epoch < train_epochs:
            for i in range(train_data_num // batch_size):
                train_seq_data, train_seq_labels = self.sess.run([train_input_data, train_label])
                train_feed_dict = {self.input_data: train_seq_data, self.labels: train_seq_labels}
                _, step = self.sess.run([optimizer, global_step], train_feed_dict)
                if step % show_step == 0 and step >= show_step:
                    # train part
                    if need_decode:
                        train_ctc_loss_value, decoded_train_predictions, merge_summary = self.sess.run([loss, decode, merge_summary_op], train_feed_dict)
                        print('epoch {} step {} loss {}'.format(str(epoch), str(step), str(train_ctc_loss_value)))
                        str_lists, number_lists = self.feature_decoder.sparse_tensor_to_str(decoded_train_predictions[0])
                        print("decoded string list is :", str_lists)
                    else:
                        train_ctc_loss_value, merge_summary = self.sess.run([loss, merge_summary_op], train_feed_dict)
                        print('epoch {} step {} loss {}'.format(str(epoch), str(step), str(train_ctc_loss_value)))

                    if step % val_step == 0 and step >= val_step:
                        accuary_per_char_list = []
                        accuary_full_sequence_list = []
                        for j in range(val_times):
                            val_seq_data, val_seq_labels = self.sess.run([val_input_data, val_label])
                            val_feed_dict = {self.input_data: val_seq_data, self.labels: val_seq_labels}
                            train_ctc_loss_value, decoded_train_predictions, step, merge_summary = self.sess.run([loss, decode, global_step, merge_summary_op], val_feed_dict)
                            accuary_per_char_list.append(self.compute_accuracy(val_seq_labels, decoded_train_predictions[0]))
                            accuary_full_sequence_list.append(self.compute_accuracy(val_seq_labels, decoded_train_predictions[0], mode='full_sequence'))
                        accuary_per_char = np.mean(np.asarray(accuary_per_char_list))
                        accuary_full_sequence = np.mean(np.asarray(accuary_full_sequence_list))
                        print('per character accuary {} , full sequence accuary {} \n epoch {} step {} loss {}'.format(str(accuary_per_char),
                                                                                                                       str(accuary_full_sequence),
                                                                                                                       str(epoch),
                                                                                                                       str(step),
                                                                                                                       str(train_ctc_loss_value)))
            epoch += 1
            model_name = 'chinese_crnn_{}.ckpt'.format(str(epoch))
            model_save_path = osp.join(model_save_dir, model_name)
            saver.save(sess=self.sess, save_path=model_save_path)

    def forward(self):
        pass
