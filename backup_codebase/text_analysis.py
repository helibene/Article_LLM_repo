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

from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_SENTENCE_LEN = 50
# vs : vaderSentiment
# tb : textblob
# ts :transformers
class text_analysis :
    def __init__(self): 
        self.vs_analyzer = SentimentIntensityAnalyzer()
        self.tf_sentiment_pipeline = pipeline("text-classification",return_all_scores =True,function_to_apply="sigmoid", model = MODEL_ID) #softmax#sentiment-analysis
        self.tb_NaiveBayes = NaiveBayesAnalyzer()
        self.tb_PatternAnalyzer = PatternAnalyzer()

    def analyseText(self, text) :
        dict_out = {}
        textblob = TextBlob(text)
        textblob_nba = TextBlob(text, analyzer=self.tb_NaiveBayes)
        textblob_pa = TextBlob(text, analyzer=self.tb_PatternAnalyzer)
        
        vs = self.vs_analyzer.polarity_scores(text)
        vs = dict(vs)
        
        sp = self.tf_sentiment_pipeline(text)
        #print(textblob.sentences)
        dict_out["tb.sentences"] = len(textblob.sentences)
        dict_out["tb.noun_phrases"] = len(textblob.noun_phrases)
        dict_out["tb.words"] = len(textblob.words)
        
        tb_pola = textblob_pa.sentiment.polarity
        tb_sub = textblob_pa.sentiment.subjectivity
        dict_out["tb.polarity"] = tb_pola # [-1.0, 1.0].
        dict_out["tb.subjectivity"] = tb_sub #[0.0, 1.0]
        dict_out["tb.subjectivity_aj"] = (tb_pola+1)/2 #[0.0, 1.0]
        if tb_sub == 0 :
            dict_out["tb.pol_div_sub"] = 1
        else :
            dict_out["tb.pol_div_sub"] = ((tb_pola+1)/2)/tb_sub #[0.0, 1.0]
        if tb_pola == -1 :
            dict_out["tb.sub_div_pol"] = 1
        else :
            dict_out["tb.sub_div_pol"] = tb_sub/((tb_pola+1)/2) #[0.0, 1.0]
        
        tp_pos = textblob_nba.sentiment.p_pos
        tp_neg = textblob_nba.sentiment.p_neg
        dict_out["tb.p_pos"] = tp_pos/(tp_pos+tp_neg)
        dict_out["tb.p_neg"] = tp_neg/(tp_pos+tp_neg)
        dict_out["tb.p_class"] = bool(tp_neg<tp_pos)
    
        
        dict_out["vs.neg"] = vs["neg"]
        dict_out["vs.neu"] = vs["neu"]
        dict_out["vs.pos"] = vs["pos"]
        dict_out["vs.compound"] = vs["compound"]
        dict_out["vs.p_class"] = bool(vs["neg"]<vs["pos"])
        
        dict_out["ts.neg"] = sp[0][0]['score']
        dict_out["ts.pos"] = sp[0][1]['score']
        dict_out["ts.class"] = bool(dict_out["ts.neg"] <dict_out["ts.pos"])
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
        dict_out["tb.pol"] = tb_pola # [-1.0, 1.0].
        dict_out["tb.sub"] = tb_sub #[0.0, 1.0]
        dict_out["tb.polaj"] = (tb_pola+1)/2 #[0.0, 1.0]
        return dict_out
    def sentimentStats(self,text,dict_out={},ret_list=[True,True,True]) :
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
            sp = self.tf_sentiment_pipeline(text)
            dict_out["ts.neg"] = sp[0][0]['score']
            dict_out["ts.pos"] = sp[0][1]['score']
        return dict_out
    
    def analyseText2(self, text,noLen=False) :
        if noLen:
            return self.subpolStats(text) | self.sentimentStats(text)
        else :
            return self.lenStats(text) | self.subpolStats(text) | self.sentimentStats(text)
    
    def analyseArticle(self, text) :
        textblob = TextBlob(text)
        sentences_list = textblob.sentences
        dict_list = []
        len_list = []
        for sentences in sentences_list :
            sent_len = len(sentences)
            if sent_len < MAX_SENTENCE_LEN :
                len_list.append(sent_len)
                new_dict = self.analyseText2(str(sentences),True)
                dict_list.append(new_dict | self.lenStats(text))
        out_dict = self.weigthAverage(dict_list,len_list)
        return out_dict
        
    def weigthAverage(self,dict_list,len_list):
        if len(dict_list) != 0 :
            sum_dict = {}
            key_list = dict_list[0].keys()
            len_list_sum = sum(len_list)
            for key in key_list :
                sum_dict[key] = 0
            for key in key_list :
                sum_num = 0
                for i in range(len(dict_list)) :
                    sum_num = sum_num + dict_list[i][key]*len_list[i]
                sum_dict[key] = sum_num/len_list_sum
            return sum_dict
        else :
            return {"nlp_error":True}
            
# test_list = ["""Kyiv, Ukraine
# CNN
#   — 
# Russian forces deported Bohdan Yermokhin from the occupied Ukrainian city of Mariupol in the spring of 2022, flew him to Moscow on a government plane and placed him into a foster family. He was sent to a patriotic camp near the capital where flag-waving staff praised Russian President Vladimir Putin and tried to teach him nationalistic songs.

# The Ukrainian teenager was given a Russian passport and sent to a Russian school. And then, in the fall of 2023, not long before his 18th birthday, he received a summons from a Russian military recruitment office.

# Yermokhin, who’s now back in Ukraine and recovering from his ordeal in Kyiv, told CNN he believed this was the last step in Russia’s attempt to bully him into submission – a bid to sign him up as a soldier to fight against his own people.

# “(I was told that) Ukraine was losing, that children were used for organ donations there, and that I would be sent to war right away. I told them that if I was sent to the war, at least I would fight for my own country, not for them,” he said.

# Yermokhin was part of a group of children known as the “Mariupol 31,” who were taken to Russia. Ukrainian authorities estimate that 20,000 children have been forcibly transported to Russia since Moscow launched its full-scale invasion of the country in February 2022. More than 2,100 children remain missing, according to official statistics, but the government says the real number could be much higher."""]
# # test_list = ["Amazing"]
# ta = text_analysis()
# list_out = []
# for string in test_list :
#     out_dict = ta.analyseArticle(string)
#     list_out.append(out_dict)
#     #print(string,"   ",out_dict)
# df = pd.DataFrame(list_out) # ,columns = list(list_out[0].keys)




# test_list = ['technology', 'file', 'advantages', 'things', 'know', 'right', 'cofounder', 'think', 'competitive', 'aesthetics', 'adobe', 'postscript', 'going', 'wanted', 'john', 'wharton', 'warnock', 'point']
# test_list = ['committee', 'technology', 'services', 'plans', 'president', 'information', 'purdue', 'director', 'streamline', 'vice', 'sustaining', 'student', 'plan', 'synergies']
# test_list = ['nuclear', 'chemical', 'mit', 'gates', 'wins', 'students', 'research', 'substrate', 'engineering', 'chris', 'scholarship', 'boyce', 'work']
# test_list = ['happy hello', 'sad', "work", "lofe","trump"]
# test_list = ['happy hello']
# test_list = ['Fuck you', 'file', 'advantages', 'things', 'know', 'right', 'cofounder', 'think', 'competitive', 'aesthetics', 'adobe', 'postscript', 'going', 'wanted', 'john', 'wharton', 'warnock', 'point']
# test_list = ['youre', 'united', 'sea', 'role', 'safer', 'risk', 'technology', 'dangerous', 'drivers', 'jobs', 'risks']
print("IMPORT : text_analysis")