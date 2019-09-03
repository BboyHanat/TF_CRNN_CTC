"""
Name : gen_train_test_val.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-08-23 17:50
Desc:
"""

import os
import random

num_thread = 16
data_file = [open("/hanat/data3/data_"+str(index)+'.txt', 'r') for index in range(num_thread)]

train_fp = open("/hanat/data3/train.txt", 'w')
val_fp = open("/hanat/data3/val.txt", 'w')
test_fp = open("/hanat/data3/test.txt", 'w')

val_num_interval = 9
test_num_interval = 10
count = 0
for i in range(num_thread):
    data_fp = data_file[i]
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

for i in range(num_thread):
    data_file[i].close()

print("File num is {}".format(str(count)))
train_fp.close()
val_fp.close()
test_fp.close()




