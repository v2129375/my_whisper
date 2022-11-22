import os

import pandas
from tqdm import tqdm
import re
from opencc import OpenCC

local_path = "/Data/dataset/ASR_dataset/aishell/data_aishell"
train_path = "aishell/train.csv"
dev_path = "aishell/dev.csv"
test_path = "aishell/test.csv"

transcript_path = f"{local_path}/transcript/aishell_transcript_v0.8.txt"

# 讀取train和dev檔案路徑
train_files_path = []
dev_files_path = []
test_files_path = []


def recur_train(rootdir):
    for root, dirs, files in tqdm(os.walk(rootdir)):
        for file in files:
            if 'DS_Store' in file:
                continue
            train_files_path.append(os.path.join(root, file))
        for dir in dirs:
            recur_train(dir)


def recur_dev(rootdir):
    for root, dirs, files in tqdm(os.walk(rootdir)):
        for file in files:
            if 'DS_Store' in file:
                continue
            dev_files_path.append(os.path.join(root, file))
        for dir in dirs:
            recur_dev(dir)


def recur_test(rootdir):
    for root, dirs, files in tqdm(os.walk(rootdir)):
        for file in files:
            if 'DS_Store' in file:
                continue
            test_files_path.append(os.path.join(root, file))
        for dir in dirs:
            recur_test(dir)


recur_train(f"{local_path}/wav/train")
recur_dev(f"{local_path}/wav/dev")
recur_test(f"{local_path}/wav/test")

# 讀取文本檔案轉成字典
cc = OpenCC('s2t') # 簡體轉繁體
_d = {}
with open(transcript_path, encoding='utf-8') as f:
    data = f.readlines()
    for i in tqdm(data):
        k, v = re.split('\s+', i, 1)
        _d[k.strip()] = cc.convert(v.replace('\n', '').replace('\t', '').replace(' ', ''))
# print(_d)

# 整合資訊並輸出成csv檔案
res_train = []
for file in tqdm(train_files_path):
    file_name = file.split('/')[-1][:-4]
    if file_name in _d:
        res_train.append((file, _d[file_name]))
res_dev = []
for file in tqdm(dev_files_path):
    file_name = file.split('/')[-1][:-4]
    if file_name in _d:
        res_dev.append((file, _d[file_name]))
res_test = []
for file in tqdm(test_files_path):
    file_name = file.split('/')[-1][:-4]
    if file_name in _d:
        res_test.append((file, _d[file_name]))

pandas.DataFrame(res_train,columns=["path","text"]).to_csv(train_path,index=False)
pandas.DataFrame(res_dev,columns=["path","text"]).to_csv(dev_path,index=False)
pandas.DataFrame(res_test,columns=["path","text"]).to_csv(test_path,index=False)