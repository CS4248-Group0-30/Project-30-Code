# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:29:57 2023

@author: Ark

Jieba Chinese tokenization testing

"""

import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET

import jieba

#jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
#for str in strs:
    #seg_list = jieba.cut(str,use_paddle=True) # 使用paddle模式
    #print("Paddle Mode: " + '/'.join(list(seg_list)))

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))



import xmltodict

os.chdir(r'E:\OneDrive\NUS\CS4248 NLP\Project\zh-en/')

xml_file = 'IWSLT17.TED.dev2010.zh-en.zh.xml'
with open(xml_file, 'r') as file:
    xml_file_string = file.read()

xml_file_dict = xmltodict.parse(xml_file_string)

# example corpus
print(dir(xml_file_dict))

i = 0
corpus_dict = xml_file_dict['mteval']['srcset']['doc'][i]['seg']
corpus_df = pd.DataFrame.from_dict(corpus_dict)

for n in corpus_df.index:
    line = corpus_df.loc[n,'#text']
    seg_list = jieba.cut(line, cut_all=False)
    seg_line = ", ".join(seg_list)
    corpus_df.loc[n,'tokenized'] = seg_line

corpus_df.to_csv('example_from_'+xml_file+'_.csv')



# full translation
os.chdir(r'E:\National University of Singapore\Lin Yuan Xun, Caleb - CS4248/')
corpus_df = pd.read_csv('tokenized_train_data.csv')

for n in corpus_df.index:
    line = corpus_df.loc[n,'zh']
    seg_list = jieba.cut(line, cut_all=False)
    seg_line = ", ".join(seg_list)
    corpus_df.loc[n,'tokens_zh'] = seg_line
    if n % 1000 == 0:
        print(n)

corpus_df.to_csv('tokenized_train_data_en_zh.csv')











