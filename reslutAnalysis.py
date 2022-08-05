import os

with open('./results.txt', 'r') as f:
    a = f.readlines()

INN_IoU_list = []
ANN_IoU_list = []

for i in a:
    if len(INN_IoU_list) < 200 and 'mean IoU' in i:
        INN_IoU_list.append(i[-5:-1])
    elif len(INN_IoU_list) == 200 and 'mean IoU' in i:
        ANN_IoU_list.append(i[-5:-1])

INN_IoU = max(INN_IoU_list)
ANN_IoU = max(ANN_IoU_list)
