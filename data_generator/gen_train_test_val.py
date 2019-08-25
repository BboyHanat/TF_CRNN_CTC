"""
Name : gen_train_test_val.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-08-23 17:50
Desc:
"""

import os
import random

data_fp = open("/hanat/data/data.txt",'r')
train_fp = open("/hanat/data/train.txt", 'w')
val_fp = open("/hanat/data/val.txt", 'w')
test_fp = open("/hanat/data/test.txt", 'w')

val_num_interval = 9
test_num_interval = 10
count = 0
while True:
    line = data_fp.readline()
    if not line:
        break
    i = random.randint(0, 10)
    if count % val_num_interval == 0 and count >= val_num_interval:
        val_fp.write(line)
    elif count % test_num_interval == 0 and count >= test_num_interval:
        test_fp.write(line)
    else:
        train_fp.write(line)
    count += 1

data_fp.close()
train_fp.close()
val_fp.close()
test_fp.close()




