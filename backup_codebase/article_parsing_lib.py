# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:25:29 2024

@author: Alexandre
"""
import main_var
env = "test/"
mv = main_var.main_var(env=env)

from newspaper import Article, ArticleException
import nltk
import os
import time
from utils_art import openDFcsv,openSTRtxt,openDFxlsx,saveDFcsv,saveSTRtxt,openConfFile
import hashlib
from dateparser import parse as parse_date
import pandas as pd
from datetime import date, datetime, timedelta
import text_analysis
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')
ts = text_analysis.text_analysis()

_SELECTION = 3
_LOAD_NLP = True

open_path = mv.query_path  #  "C:/Users/User/OneDrive/Desktop/article/files_3/1_1_query_main/"+env
filename_input = mv.query_filename # "query_main_final"#"query_large_2010_to_2023"#"query_main_final"

save_path = mv.scarp_path  # "C:/Users/User/OneDrive/Desktop/article/files_3/1_2_scarp_main/"+env #parssing_df_main/
filename_out = mv.scarp_filename

save_path_article = mv.article_path   #   "C:/Users/User/OneDrive/Desktop/article/files_3/1_3_article_main/"+env #article_download_main_2

# output_fields = ["url", "pk", "hash_key", "title", "authors", "publish_date", "keywords_list", "text_len","valid"]# + ["tb.sent", "tb.noun", "tb.word", "tb.char", "tb.pol", "tb.sub", "tb.polaj", "tb.pos", "tb.neg", "vs.pos", "vs.neu", "vs.neg","vs.comp","ts.pos","ts.neg"], "summary"
output_fields = ['url', 'pk', 'hash_key', 'publish_date', 'title', 'authors', 'valid', 'text_len', 'keywords_list'] + ["tb.sent", "tb.noun", "tb.word", "tb.char", "tb.pol", "tb.sub", "tb.polaj", "tb.pos", "tb.neg", "vs.pos", "vs.neu", "vs.neg","vs.comp","ts.pos","ts.neg"]#, "summary"


def urlToDict(url, selection=_SELECTION) :
    article = urlToArticle(url,nlp=_LOAD_NLP)
    if type(article) != type(None) :
        out_dict = articleToDict(article, selection)
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

def articleToDict(article, selection=_SELECTION) :
    list_parse_main = []
    list_parse_total = ["html", "title", "authors", "publish_date", "text", "top_image", "images", "movies", "keywords", "summary"]
    list_parse_simple = ["title", "authors", "publish_date", "text", "keywords"]
    list_parse_stats = ["title", "authors", "keywords", "text"]
    list_parse_stats_text = ["title", "authors","keywords","summary", "text"]
    out_dict = {}
    if selection == 0 :
        list_parse_main = list_parse_total
    if selection == 1 :
        list_parse_main = list_parse_simple
    if selection == 2 :
        list_parse_main = list_parse_stats
    if selection == 3 :
        list_parse_main = list_parse_stats_text
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
    valid = True
    pk = url.split("articles/")[1].replace("?oc=5","")
    hash_key = hashlib.shake_256(str(pk).encode()).hexdigest(20)
    out_dict = {"url":url,"pk":pk,"hash_key":hash_key}
    ar_dict = urlToDict(url,_SELECTION)
    display_text = " - Loadind Article #"+str(increment)+"   ("
    if type(ar_dict) != type(None) :
        text = ar_dict['text']
        text_len = len(text)
        if text_len != 0:
            if 'publish_date' in ar_dict.keys() and False :
                if type(ar_dict['publish_date']) == type(None) :
                    publish_date = ""
                else :
                    if type(parse_date(str(ar_dict['publish_date']))) == type(None) :
                        publish_date = ""
                    else :
                        publish_date = str(parse_date(str(ar_dict['publish_date'])).date().strftime('%Y-%m-%d'))
                out_dict["publish_date"] = publish_date
            else :
                out_dict["publish_date"] = None
            analysis_nlp_dict = {}
            if add_nlp == 2:
                analysis_nlp_dict = ts.analyseArticle(text)
            if add_nlp == 1:
                analysis_nlp_dict = ts.lenStats(text)
            out_dict = out_dict | ar_dict
            out_dict["valid"] = True
            out_dict["text_len"] = text_len
            out_dict["keywords_list"] = out_dict["keywords"]
            # out_dict["summary"] = out_dict["summary"]
            out_dict = out_dict | analysis_nlp_dict
            if saveArticle :
                dict_for_save = {"hash_key":hash_key,"text":text}
                articleDictToFile(dict_for_save, save_path_article)
                del text
        else :
            valid = False
            display_text = display_text+"Not valid : Text empty)"
    else :
        valid = False
        display_text = display_text+"Not valid : Dict could not be read)"
    if valid:
        if display :
            print(display_text+"Valid Article)")
        return out_dict
    else :
        if display :
            print(display_text)
        out_dict["valid"] = False
        return out_dict
        

def readArticleFileTable(index_from=0,index_to=1000,save_articles=True,save_final=True,save_steps=False,display_df=False,step_pct=0.1,add_nlp=1,filtered_input_df=False):
    #stat_field_selection = ["url", "pk", "hash_key", "title", "authors", "publish_date", "keywords_list","summary", "text_len","valid"]# + ["tb.sent", "tb.noun", "tb.word", "tb.char", "tb.pol", "tb.sub", "tb.polaj", "tb.pos", "tb.neg", "vs.pos", "vs.neu", "vs.neg","vs.comp","ts.pos","ts.neg"]
    # stat_field_selection = ["url", "pk", "hash_key", "title", "authors", "publish_date", "keywords_list", "text_len","valid"]# + ["tb.sent", "tb.noun", "tb.word", "tb.char", "tb.pol", "tb.sub", "tb.polaj", "tb.pos", "tb.neg", "vs.pos", "vs.neu", "vs.neg","vs.comp","ts.pos","ts.neg"]
    stat_field_selection0 = ["url", "pk", "hash_key", "title", "authors", "publish_date", "keywords_list","summary", "text_len","valid"]
    stat_field_selection1 = ['url', 'pk', 'hash_key', 'publish_date', 'title', 'authors', 'text', 'keywords', 'valid', 'text_len', 'keywords_list', 'tb.sent', 'tb.noun', 'tb.word', 'tb.char']
    stat_field_selection2 = ["url", "pk", "hash_key", "title", "authors", "publish_date", "keywords_list", "text_len","valid", "tb.sent", "tb.noun", "tb.word", "tb.char", "tb.pol", "tb.sub", "tb.polaj", "tb.pos", "tb.neg", "vs.pos", "vs.neu", "vs.neg","vs.comp","ts.pos","ts.neg"]
    if add_nlp == 0 :
        stat_field = stat_field_selection0
    if add_nlp == 1 :
        stat_field = stat_field_selection1
    if add_nlp == 2 :
        stat_field = stat_field_selection2
    df = pd.DataFrame([], columns = stat_field)
    # suffix = ""
    # if filtered_input_df :
    #     suffix = "_final"
    # else :
    #     suffix = "_cap"
    # df_input = openDFcsv(open_path, filename_input+suffix)
    df_input = openDFcsv(open_path, filename_input)
    df_input_url = df_input["link"][index_from:min(index_to,df_input.shape[0])]
    art_count = 0
    for url_entry in df_input_url:
        ar_list = readStatsFromURL(url_entry,save_articles,display_df,art_count,add_nlp)
        # print(ar_list.keys())
        df = addDictToDF(df,ar_list)
        art_count = art_count + 1
        if (art_count%int((min(index_to,df_input.shape[0])-index_from)*step_pct) == 0 or art_count==len(df_input_url)) and save_steps :
            df_out = df[output_fields]
            saveDFcsv(df_out, save_path, filename_out+"_"+str(art_count)) # , mode="w"
        df = df[output_fields]
        
    if save_final :
        df = df.drop[["Unnamed: 0_q","Unnamed: 0.1]"]]
        saveDFcsv(df, save_path, filename_out+"_backup") # , mode="w"
        saveDFcsv(df, save_path, filename_out,True) # , mode="w"
        print("Final file saved here :",save_path)
    if display_df :
        display(df)
    return df

def fillDFwithListList(df, ar_list):
    out_list = []
    for i in range(len(ar_list)) :
        out_list.append(ar_list[i])
    # print(len(out_list))    
    df.loc[len(df)] = out_list
    return df

def addDictToDF(df, ar_dict):
    df_add = pd.DataFrame([ar_dict], columns = ar_dict.keys())
    df = pd.concat([df,df_add]).reset_index(drop=True)
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

def generateNLPonKeywords(df_main,index_from=0,index_to=500,display_log=True):
    if index_to == -1 :
        index_to = df_main.shape[0]
    df_np = df_main["word_combined_all"].apply(np.array).to_numpy()[0:index_to]
    df_hash = df_main["hash_key"].apply(np.array).to_numpy()[0:index_to]
    mat_index = []
    for i in range(index_from,index_to) :
        if display_log :
            print(" - Generate NLP for article #"+str(i)+"  (char:"+str(len(df_np[i]))+")")
        nlp_dict = ts.analyseText2(str(df_np[i]),True)
        mat_index.append(nlp_dict|{"hash_key":df_hash[i]})
    df = pd.DataFrame(mat_index, columns = list(mat_index[0].keys())) 
    df.set_index("hash_key", inplace=True)
    df_main.set_index("hash_key", inplace=True)
    display(df)
    display(df_main)
    df = df_main.join(df,on='hash_key', how="inner",lsuffix='_k')
    return df

print("IMPORT : article_parsing_lib ")