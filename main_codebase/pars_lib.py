# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:25:29 2024

@author: Alexandre
"""
import main_var
mv = main_var.main_var()
from newspaper import Article, ArticleException
from utils_art import openDFcsv,saveDFcsv,deleteUnnamed,saveSTRtxt,display_df,openSTRtxt
import hashlib
import pandas as pd
import text_analysis
import warnings
warnings.filterwarnings('ignore')
ts = text_analysis.text_analysis()
from transformers import logging
logging.set_verbosity_error()
# import nltk
# import os
# import time
# from dateparser import parse as parse_date
# from datetime import date, datetime, timedelta
# from pathlib import Path
# import numpy as np
# _SELECTION = 3

# All columns : ["html", "title", "authors", "publish_date", "text", "top_image", "images", "movies", "keywords", "summary"]
_PARSING_COL_SELECTION = ["title", "authors","keywords", "text"] #,"summary" ######, "publish_date"
_LOAD_NLP = True
output_fields = ['url', 'pk', 'hash_key', 'title', 'authors', 'valid', 'text_len', 'keywords_list'] + ["tb.sent", "tb.noun", "tb.word", "tb.char", "tb.pol", "tb.sub", "tb.polaj", "tb.pos", "tb.neg", "vs.pos", "vs.neu", "vs.neg","vs.comp","ts.pos","ts.neg"]#, "summary" ####'publish_date',


def urlToDict(url) :
    article = urlToArticle(url,nlp=_LOAD_NLP)
    if type(article) != type(None) :
        out_dict = articleToDict(article)
        return out_dict
    else :
        return None

def urlToArticle(url, display=False,nlp=False) :
    article = Article(url)
    try:
        article.download()
        article.parse()
        if nlp :
            article.nlp()
        return article
    except ArticleException:
        if display :
            print("WARNING : url could not parse :",url)
        pass
        return None
    except KeyboardInterrupt:
        pass
        return None
    except ValueError:
        if display :
            print("WARNING : url issue value error :",url)
        pass
        return None

def articleToDict(article) :
    out_dict = {}
    list_parse_main = _PARSING_COL_SELECTION
    for parse_str in list_parse_main :
        if parse_str == "html" :
            out_dict[parse_str] = article.html
        if parse_str == "title" :
            out_dict[parse_str] = article.title
        if parse_str == "authors" :
            out_dict[parse_str] = article.authors
        if parse_str == "publish_date" :
            out_dict[parse_str] = article.publish_date
        if parse_str == "text" :
            out_dict[parse_str] = article.text
        if parse_str == "top_image" :
            out_dict[parse_str] = article.top_image
        if parse_str == "images" :
            out_dict[parse_str] = article.images
        if parse_str == "movies" :
            out_dict[parse_str] = article.movies
        if parse_str == "keywords" :
            out_dict[parse_str] = article.keywords
        if parse_str == "summary" :
            out_dict[parse_str] = article.summary
    return out_dict

def getStartArticle(article_dict,length=300) :
    text = article_dict["text"]
    if len(text) > length :
        return text[0:length]
    else :
        return text[0:len(text)]
    

def readStatsFromURL(url, saveArticle=False, display=False,increment=0,add_nlp=1) :
    display_hashpk_for_debug = False
    valid = True
    nlp_valid = True
    pk = url.split("articles/")[1].replace("?oc=5","")
    hash_key = hashlib.shake_256(str(pk).encode()).hexdigest(20)
    out_dict = {"url":url,"pk":pk,"hash_key":hash_key}
    if display_hashpk_for_debug :
        print(" - START - Read article online and get text and keyword into dict #"+str(increment)+"  ("+str(hash_key)+")")
    ar_dict = urlToDict(url)
    if display_hashpk_for_debug :
        print(" - DONE - Read article online and get text and keyword into dict  #"+str(increment)+"  ("+str(hash_key)+")")
    display_text = " - Loadind Article #"+str(increment)+"  '"+str(hash_key)+"' ("
    if type(ar_dict) != type(None) :
        text = ar_dict['text']
        text_len = len(text)
        if text_len != 0:
            # if 'publish_date' in ar_dict.keys() and False :
            #     if type(ar_dict['publish_date']) == type(None) :
            #         publish_date = ""
            #     else :
            #         if type(parse_date(str(ar_dict['publish_date']))) == type(None) :
            #             publish_date = ""
            #         else :
            #             publish_date = str(parse_date(str(ar_dict['publish_date'])).date().strftime('%Y-%m-%d'))
            #     out_dict["publish_date"] = publish_date
            # else :
            #     out_dict["publish_date"] = None
            # out_dict["publish_date"] = "xxx"
            if display_hashpk_for_debug :
                print(" - STOP - Generate NLP and length indicators using  'text_analysis' module  #"+str(increment)+"  ("+str(hash_key)+")")
            analysis_nlp_dict = {}
            if add_nlp == 2:
                analysis_nlp_dict = ts.analyseArticle(text)
            if add_nlp == 1:
                analysis_nlp_dict = ts.lenStats(text)
            out_dict = out_dict | ar_dict
            # out_dict["valid"] = True
            out_dict["text_len"] = text_len
            out_dict["keywords_list"] = out_dict["keywords"]
            # out_dict["summary"] = out_dict["summary"]
            out_dict = out_dict | analysis_nlp_dict
            if display_hashpk_for_debug :
                print(" - STOP - Generate NLP and length indicators using  'text_analysis' module  #"+str(increment)+"  ("+str(hash_key)+")")
            if saveArticle :
                dict_for_save = {"hash_key":hash_key,"text":text}
                articleDictToFile(dict_for_save, mv.article_path)
                del text
        else :
            valid = False
            nlp_valid = False
            display_text = display_text+"Not valid : Text empty)"
    else :
        valid = False
        nlp_valid = False
        display_text = display_text+"Not valid : Dict could not be read)"
    if valid:
        if display :
            print(display_text+"Valid Article)")
            out_dict["valid"] = valid
            out_dict["nlp_valid"] = nlp_valid
        return out_dict
    else :
        if display :
            print(display_text)
        out_dict["valid"] = valid
        out_dict["nlp_valid"] = False
        return out_dict
        

def readArticleFileTable(index_from=0,index_to=99999999,save_articles=True,save_final=True,save_steps=False,display_data=False,step_pct=0.1,add_nlp=1,filtered_input_df=False): ####, "publish_date"
    stat_field_selection0 = ["url", "pk", "hash_key", "title", "authors", "keywords_list","summary", "text_len","valid"]
    stat_field_selection1 = ['url', 'pk', 'hash_key', "title", "authors", 'keywords_list', 'text_len', 'valid', 'tb.sent', 'tb.noun', 'tb.word', 'tb.char']
    stat_field_selection2 = ["url", "pk", "hash_key", "title", "authors", "keywords_list", "text_len", "valid", "tb.sent", "tb.noun", "tb.word", "tb.char", "tb.pol", "tb.sub", "tb.pos", "tb.neg", "vs.pos", "vs.neu", "vs.neg","vs.comp","ts.pos","ts.neu","ts.neg","al.pos","al.neg"]#,"tb.class","vs.class","vs.class","al.pos","al.neg"], "tb.polaj"
    if add_nlp == 0 :
        stat_field = stat_field_selection0
    if add_nlp == 1 :
        stat_field = stat_field_selection1
    if add_nlp == 2 :
        stat_field = stat_field_selection2
    df = pd.DataFrame([], columns = stat_field)
    df_input = openDFcsv(mv.query_path, mv.query_filename)
    index_to = min(index_to,df_input.shape[0])
    df_input_url = df_input["link"][index_from:index_to]
    art_count = 0
    for url_entry in df_input_url:
        ar_list = readStatsFromURL(url_entry,save_articles,display_data,art_count,add_nlp)
        # ar_list["publish_date"] = "ccc"
        df = addDictToDF(df,ar_list)
        art_count = art_count + 1
        if (art_count%int((index_to-index_from)*step_pct) == 0 or art_count==len(df_input_url)) and save_steps :
            df_out = df[stat_field]
            df_out = deleteUnnamed(df_out,"hash_key")
            saveDFcsv(df_out, mv.scarp_path, mv.scarp_filename+"_"+str(art_count)) # , mode="w"
        
    if save_final :
        df = df[stat_field]
        df = deleteUnnamed(df,"hash_key")
        saveDFcsv(df, mv.scarp_path, mv.scarp_filename,True) # , mode="w"
        print("Final file saved here :",mv.scarp_path)
    if display_data :
        display_df(df)
    return df

def fillDFwithListList(df, ar_list):
    out_list = []
    for i in range(len(ar_list)) :
        out_list.append(ar_list[i])
    # print(len(out_list))    
    df.loc[len(df)] = out_list
    return df

def addDictToDF(df, ar_dict):
    df_add = pd.DataFrame([ar_dict], columns = ar_dict.keys())#.reset_index(drop=False)#.reset_index(inplace=True, drop=True)
    # df = df.reset_index(inplace=True, drop=True)
    if type(df_add) != type(None) and type(df) != type(None) :
        df = pd.concat([df,df_add])#.reset_index(drop=False)#.reset_index(inplace=True, drop=True)#
    return df

def articleDictToFile(ar_dict, path) :  #requires pk & text
    length_limit = 160
    di_keys = ar_dict.keys()
    if "text" in di_keys and "hash_key" in di_keys :
        if len(ar_dict["hash_key"])<length_limit :
            # cwd = os.getcwd()  # Get the current working directory (cwd)
            # files = os.listdir(cwd)
            saveSTRtxt(ar_dict["text"],path,ar_dict["hash_key"])
            return True
        else :
            print("WARNING : hash_key is too long (longer than " + str(length_limit) + " caracters)")
            return False
    else :
        print("ERROR : articleDictToFile could not find 'text' or 'hash_key' in the dict provided")
        return False

def calculateNLP(index_from=0,index_to=9999999999,step_pct=0.0001,save_steps=True,save_final=True):
    #df = pd.DataFrame([], columns = stat_field)
    df_input = openDFcsv(mv.scarp_path, mv.scarp_filename)
    df_input = df_input[df_input["valid"]==True]
    quant=0.1
    before_len = df_input.shape[0]
    df_input=clearPercentile(df_input,"text_len",quant)
    after_len = df_input.shape[0]
    index_to = min(index_to,after_len)
    df_hash = df_input["hash_key"][index_from:index_to]
    base_df = pd.DataFrame([], columns = list(df_input.columns))#.reset_index(drop=False)#.reset_index(inplace=True, drop=True)
    count=0
    for hash_key in df_hash :
        text = openSTRtxt(mv.article_path, hash_key)
        analysis_nlp_dict = ts.analyseArticle(str(text))
        dict_entry = df_input[df_input["hash_key"]==hash_key].to_dict(orient='records')[0]
        add_dict = dict_entry|analysis_nlp_dict
        base_df=addDictToDF(base_df,add_dict)
        if (count%int((index_to-index_from)*step_pct) == 0 or count==len(df_hash)) and save_steps :
            base_df_temp = deleteUnnamed(base_df,"hash_key")
            saveDFcsv(base_df_temp, mv.scarp_path, mv.scarp_filename+"_nlp_"+str(count)) # , mode="w"
            print("Saved #"+str(count)+"  to "+str(index_to-index_from))
        count=count+1
    if save_final:
        saveDFcsv(base_df, mv.scarp_path, mv.scarp_filename+"_nlp")

def clearPercentile(df,col,quant=0.1):
    df_pctl1 = int(df[[col]].quantile(quant,0).iloc[0].tolist())
    df_pctl2 = int(df[[col]].quantile(1-quant,0).iloc[0].tolist())
    df = df[df[col]>df_pctl1]
    df = df[df[col]<df_pctl2]
    return df
#calculateNLP()
print("IMPORT : pars_lib")
