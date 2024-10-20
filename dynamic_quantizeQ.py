import torch
import numpy as np

# Q = [15, 14, 13, 13, 14, 11]


def quantization(x, fixed_bit, decimal_bit):
    mini = -2 ** (fixed_bit - decimal_bit - 1)
    maxi = -mini - 2 ** (-decimal_bit)
    fixedpoint_truevalue = np.floor(x * 2 ** decimal_bit) / 2 ** decimal_bit
    fixedpoint_truevalue = np.clip(fixedpoint_truevalue, mini, maxi)
    return fixedpoint_truevalue


state_dict = torch.load("retrain_model67.pt")
Q_list = []
num_bits = 16
for i in state_dict:
    print(i)
    weight = state_dict[i].data
    error_sum = 2. ** 16
    Q_temp = 0
    for j in range(11, num_bits):
        print(j)
        error_temp = 0.
        for k in range(len(np.asarray(weight.cpu()).reshape(1, -1)[0])):
            num = np.asarray(weight.cpu()).reshape(1, -1)[0][k]
            error_temp += (num - quantization(num, num_bits, j)) ** 2
        if error_sum > error_temp:
            error_sum = error_temp
            Q_temp = j
    Q_list.append(Q_temp)
    print("state:%s  error:%.7f  Q:%d" % (i, error_sum, Q_temp))
    conv_weight = None
f = open("Q_value.txt", "w")
f.writelines(str(Q_list))
f.close()
