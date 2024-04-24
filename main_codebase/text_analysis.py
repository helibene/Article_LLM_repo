# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 01:52:05 2024

@author: Alexandre
"""

from newspaper import Article, ArticleException
import nltk
import os
import time
from pygooglenews import GoogleNews
import json
import pandas as pd
from dateparser import parse as parse_date
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.sentiments import PatternAnalyzer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings('ignore')
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MIN_NUM_SENTENCES = 2
MAX_NUM_SENTENCES = 100 # max 600

MIN_SENTENCE_LEN = 2
MAX_SENTENCE_LEN = 100
MAX_SENTENCE_CHAR_LEN = 1000


# vs : vaderSentiment
# tb : textblob
# ts :transformers
class text_analysis :
    def __init__(self): 
        self.vs_analyzer = SentimentIntensityAnalyzer()
        self.tf_sentiment_pipeline = pipeline("text-classification",return_all_scores =True,function_to_apply="sigmoid", model = MODEL_ID) #softmax#sentiment-analysis
        self.tb_NaiveBayes = NaiveBayesAnalyzer()
        self.tb_PatternAnalyzer = PatternAnalyzer()
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # self.config = AutoConfig.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    
    def test_sent(self, text) :
        print(len(str(text)))
        dict_out={}
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        dict_out["ts.neg"] = scores[0]
        dict_out["ts.neu"] = scores[1]
        dict_out["ts.pos"] = scores[2]
        return dict_out
    def lenStats(self,text,dict_out={}) :
        textblob = TextBlob(text)
        dict_out["tb.sent"] = len(textblob.sentences)
        dict_out["tb.noun"] = len(textblob.noun_phrases)
        dict_out["tb.word"] = len(textblob.words)
        dict_out["tb.char"] = len(text)
        return dict_out
        
    def subpolStats(self,text,dict_out={}) :
        textblob_pa = TextBlob(text, analyzer=self.tb_PatternAnalyzer)
        tb_pola = textblob_pa.sentiment.polarity
        tb_sub = textblob_pa.sentiment.subjectivity
        dict_out["tb.pol"] = tb_pola
        dict_out["tb.sub"] = tb_sub
        dict_out["tb.polaj"] = (tb_pola+1)/2
        return dict_out
    def sentimentStats(self,text,dict_out={},ret_list=[True,True,True,True,True]) :
        if ret_list[0] :
            textblob_nba = TextBlob(text, analyzer=self.tb_NaiveBayes)
            tp_pos = textblob_nba.sentiment.p_pos
            tp_neg = textblob_nba.sentiment.p_neg
            dict_out["tb.pos"] = tp_pos/(tp_pos+tp_neg)
            dict_out["tb.neg"] = tp_neg/(tp_pos+tp_neg)
        if ret_list[1] :
            vs = self.vs_analyzer.polarity_scores(text)
            vs = dict(vs)
            dict_out["vs.pos"] = vs["pos"]
            dict_out["vs.neu"] = vs["neu"]
            dict_out["vs.neg"] = vs["neg"]
            dict_out["vs.comp"] = vs["compound"]
        if ret_list[2] :
            dict_out = dict_out | self.test_sent(text)
            # print(len(text))
            # sp = self.tf_sentiment_pipeline(text)
            # dict_out["ts.neg"] = sp[0][0]['score']
            # dict_out["ts.pos"] = sp[0][1]['score']
        if ret_list[0] and ret_list[1] and ret_list[2] and ret_list[3] :
            dict_out["tb.class"] = bool(dict_out["tb.neg"]<dict_out["tb.pos"])
            dict_out["vs.class"] = bool(dict_out["vs.neg"]<dict_out["vs.pos"])
            dict_out["ts.class"] = bool(dict_out["ts.neg"]<dict_out["ts.pos"])
        if ret_list[0] and ret_list[1] and ret_list[2] and ret_list[3] and ret_list[4] :
            dict_out["al.pos"] = float(float(int(dict_out["tb.class"])+int(dict_out["vs.class"])+int(dict_out["ts.class"]))/3)#float(dict_out["tb.polaj"])*
            dict_out["al.neg"] = 1-dict_out["al.pos"]
        return dict_out
    
    def analyseText2(self,text,noLen=False) :
        if noLen:
            return self.subpolStats(text) | self.sentimentStats(text)
        else :
            return self.lenStats(text) | self.subpolStats(text) | self.sentimentStats(text)
    
    def analyseArticle(self, text) :
        disp = True
        # article_status = ""
        textblob = TextBlob(text)
        sentences_list = textblob.sentences
        dict_list = []
        len_list = []
        sent_len = 0
        sent_count = 0
        if sent_count < MAX_NUM_SENTENCES : #and sent_count > MIN_NUM_SENTENCES
            #happends
            for sentence in sentences_list :
                sent_len = len(sentence.words)
                if sent_len < MAX_SENTENCE_LEN and sent_len > MIN_SENTENCE_LEN :
                    #happends
                    big_word_valid = True
                    for w in sentence.words :
                        if len(w) < MAX_SENTENCE_CHAR_LEN :
                            big_word_valid = False
                    if big_word_valid :
                        len_list.append(sent_len)
                        new_dict = self.analyseText2(str(sentence),True)
                        if "tb.class" in new_dict.keys() :
                            del new_dict["tb.class"]
                            del new_dict["vs.class"]
                            del new_dict["ts.class"]
                        dict_list.append(new_dict) # | self.lenStats(text)
                        sent_count = sent_count + 1
                else :
                    pass
                    # print("lv2_"+str(sent_len))
        else :
            pass
            # print("lv1_"+str(sent_len))
        #     # lenStats()
        #     return self.subpolStats(text)
        # else:
        #     for sentence in sentences_list :
        #         sent_len = len(sentence.words)
        #         if sent_len < MAX_SENTENCE_LEN and sent_len > MIN_SENTENCE_LEN :
        #             big_word_valid=True
        #             # for w in sentence.words :
        #                 # if len(w) > MAX_SENTENCE_CHAR_LEN :
        #                 #     big_word_valid = False
        #                     # print("big_word_valid FALSE")
        #                     #print(w)
                    
        #         else :
        #             pass
        #             # if disp :
        #             #     print("MAX_SENTENCE_LEN_or_MIN_SENTENCE_LEN WARNNING")
        #     else :
        #         pass
        #         # if disp :
        #         #     print("LOW_NUM_SENTENCES_or_MAX_NUM_SENTENCES ERROR")
        # print(100*sent_count/len(sentences_list))
        out_dict = self.weigthAverage(dict_list,len_list)
        return out_dict | self.lenStats(text)
    
    
        # if len(sentences_list)<MIN_NUM_SENTENCES :
        #     article_status = article_status + "_LOW_NUM_SENTENCES_"
        # if len(sentences_list)>MAX_NUM_SENTENCES :
        #     article_status = article_status + "_HIGH_NUM_SENTENCES_"
            
    def weigthAverage(self,dict_list,len_list):
        if len(dict_list) != 0 and len(len_list) != 0 and sum(len_list) !=0 :
            sum_dict = {}
            key_list = dict_list[0].keys()
            len_list_sum = sum(len_list)
            for key in key_list :
                sum_dict[key] = 0
            for key in key_list :
                sum_num = 0
                for i in range(len(dict_list)) :
                    sum_num = sum_num + float(dict_list[i][key])*float(len_list[i])
                sum_dict[key] = sum_num/len_list_sum
            return sum_dict
        else :
            return {}

print("IMPORT : text_analysis")