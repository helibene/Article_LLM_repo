# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:06:14 2024

@author: Alexandre
"""

import main_var
mv = main_var.main_var()

from utils_art import openDFcsv,saveDFcsv,display_df,deleteUnnamed

import pandas as pd
import numpy as np

from IPython.core.display import Image as image
from PIL import Image
from IPython.display import Image
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm



import text_analysis
import warnings
import re
warnings.filterwarnings('ignore')
ts = text_analysis.text_analysis()

TAG_SELECTION_LIST = ["NNP","NN","JJ"]

folder_path_embd_df = mv.embdedding_path  # "C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embd_df/"
filename_embd_df = mv.embdedding_filename
folder_path_embd_raw = mv.embdedding_path_raw
filename_embd_raw_save = mv.embdedding_filename_raw

join1_path = mv.join1_path  # "C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embd_df/"
join1_filename = mv.join1_filename
keyword_path = mv.keyword_path
keyword_filename = mv.keyword_filename

### Embedding Functions

def dfColumnToMatrix(df,data_col="o_data",max_index=9999,index_col="hash_key",matToDf=True) :
    max_index = min(max_index,df.shape[0])
    mat_data = df[data_col].apply(np.array).to_numpy()[0:max_index]
    mat_index = []
    if index_col != "" :
        mat_index = df[index_col].apply(np.array).to_numpy()[0:max_index]
    mat_out = []
    for i in range(max_index) :
        split_list = mat_data[i].strip('[]').split(',')
        if index_col != "" :
            mat_out.append([mat_index[i]] + split_list)
        else :
            mat_out.append([i] + split_list)
    if matToDf :
        mat_out_df = pd.DataFrame(mat_out)
        mat_out_df.set_index(0, inplace=True)
        if index_col != "" :
            mat_out_df.index.rename(index_col, inplace=True)
        else :
            mat_out_df.index.rename('index', inplace=True)
        return mat_out_df
    else :
        return mat_out
    
def extractEmbedding(df,number_entries=99999999999): #51372
    print(df.dtypes)
    df_out = dfColumnToMatrix(df,data_col="o_data",max_index=number_entries,index_col="hash_key",matToDf=True) #
    return df_out

def extractEmbeddingFromFile(number_entries=10000,display_stats=True): #51372
    df_main = openDFcsv(folder_path_embd_df,filename_embd_df)
    print(folder_path_embd_df+filename_embd_df)
    emb_mat = extractEmbedding(df_main,number_entries)
    if display_stats :
        print("df_main : ",df_main)
        display_df(df_main)
        print("emb_mat : ",emb_mat)
        display_df(emb_mat)
    saveDFcsv(emb_mat, folder_path_embd_raw, filename_embd_raw_save,True)
    return emb_mat
### Keyword Functions

def parse_keywords_list(df, column_name="keywords_list",titleFlag=False,entry_limit=1000,output_df=False):#51400
    out = df[column_name].to_numpy()[0:entry_limit]
    union_list = [[] for i in range(entry_limit)]
    count = 0
    for entry in out :
        if not titleFlag :
            parsed_list = decomposeKeywordList(entry)#.replace("'","").lower().strip("][").split(', ') 
        else :
            parsed_list = decomposeTitle(entry)
        union_list[count] = parsed_list
        count = count +1
    if output_df :
        return union_list #
    else :
        return union_list

def flattenMatrix(mat):
    out_list = []
    for li in mat :
        out_list = out_list + li
    return out_list
def getMostCommunKeywords(union_list_np, source_limit=10000,return_word_list=True,display_stats=True) :
    df = pd.DataFrame(union_list_np, columns=['keyword'])
    df = df['keyword'].value_counts().to_frame("count").sort_values(by=['count'],ascending=True)
    if display_stats :
        display_df(df)
    count_sum_before = df["count"].sum()
    entry_sum_before = df.shape[0]
    source_limiy_count = int(df.iloc[[int(-min(entry_sum_before,source_limit))]]["count"].tolist()[0])
    df = df[df['count'].between(source_limiy_count, source_limit)]
    count_sum_after = df["count"].sum()
    entry_sum_after = df.shape[0]
    if display_stats :
        print("Unique keywords :",entry_sum_before,"  Sum of occurences :",count_sum_before,"  ("+str(round(count_sum_before/entry_sum_before,2))+" avg)")
        print("Unique keywords :",entry_sum_after,"  Sum of occurences :",count_sum_after,"  ("+str(round(count_sum_after/entry_sum_after,2))+" avg)")
    if return_word_list :
        df2 = df.index.to_numpy()
        return df2
    else :
        return df

def cleanString(string) :
    #~out_str = string.strip("][”“|’><%—–//").replace("'", "").replace("\\d", "");
    out_str = "".join(re.findall("[a-zA-Z]+", string))
    if len(out_str)>3:
        return out_str
    else :
        return ""

def cleanString(string) :
    #~out_str = string.strip("][”“|’><%—–//").replace("'", "").replace("\\d", "");
    out_str = "".join(re.findall("[a-zA-Z]+", string))
    if len(out_str)>3:
        return out_str
    else :
        return ""

def decomposeTitle(string, getList=True) :
    decomposed = "=".join(re.findall("[a-zA-Z]+", string.lower()))
    if getList :
        decomposed = decomposed.split('=')
        # if type(decomposed) != type(None) :
        #     for i in range(len(decomposed)) :
        #         if len(decomposed[i])<3:
        #             decomposed = decomposed.remove(decomposed[i])
    return decomposed

def decomposeKeywordList(string,getList=True) :
    decomposed = "&".join(re.findall("[a-zA-Z]+", string.lower())) #.split(', ')
    if getList :
        decomposed = decomposed.split('&')
        # if type(decomposed) != type(None) :
        #     for i in range(len(decomposed)) :
        #         if len(decomposed[i])<3:
        #             decomposed = decomposed.remove(decomposed[i])
    return decomposed

def df_setup(par_list) :
    df = pd.DataFrame(par_list)#
    col_count = df.shape[1]
    df.rename(columns=lambda x: str(x), inplace=True)#
    df['word_count'] = df.apply(lambda x: x.count(), axis=1)#
    df = df.fillna(value="")#df = df.fillna(value="#")
    df["0"] = df["0"].astype(str)
    df['word_combined'] = df[[str(i) for i in range(col_count)]].agg(' '.join, axis=1)
    #df['word_combined'].replace(" #","", inplace=True)
    # display_df(df)
    # print(df[["word_count"]].describe())
    return df



def genrateKeywordExtract(df_main, entry_limit=10, common_word_max = 10, column_name="keywords_list", isTitle = False):
    entry_limit = min(df_main.shape[0],entry_limit)
    par_list = parse_keywords_list(df_main,column_name,isTitle,entry_limit)
    df = df_setup(par_list)
    par_list = flattenMatrix(par_list)
    out = getMostCommunKeywords(par_list,common_word_max)
    # print(out)
    out = set(out)
    row_list = []
    for i in range(entry_limit) :
        row_list.append(list(out & set(df.loc[i, :].values.flatten().tolist())))
    df_min = df_setup(row_list)
    df = df.join(df_min, how="inner", lsuffix='_f', rsuffix='_s')
    #df["word_combined_f"].replace(" @","")
    #df["word_combined_s"].replace(" @","")
    #df['word_combined_s'] = df['word_combined_s'].str.replace(' @','')
    #df['word_combined_f'] = df['word_combined_s'].str.replace(' @','')
    df['word_combined_f'] = df['word_combined_f'].str.replace(r'\s+', ' ', regex=True)
    df['word_combined_s'] = df['word_combined_s'].str.replace(r'\s+', ' ', regex=True)
    return df

def generateNLPonKeywords(df_main,index_from=0,index_to=500,nlp_source_col="word_combined_all",display_log=True,display_res=True):
    if index_to == -1 :
        index_to = df_main.shape[0]
    df_np = df_main[nlp_source_col].apply(np.array).to_numpy()[0:index_to]
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
    df_out = df_main.join(df,on='hash_key', how="inner",rsuffix='_k')
    if display_res :
        print("Input DF shape :",df_main.shape)
        display_df(df_main.head(3))
        print("Generated DF shape :",df.shape)
        display_df(df.head(3))
        print("Output DF shape :",df_out.shape)
        display_df(df_out.head(3))
    return df_out

def mainKeywordWF(entry_limit=9999999,common_word_max=500,add_nlp_stats=True,nlp_source_col="word_combined_all") :
    df_main = openDFcsv(join1_path,join1_filename)
    df_main = df_main[["hash_key","title_quer","keywords_list"]]
    df1 = genrateKeywordExtract(df_main,entry_limit,common_word_max,"title_quer",True)
    df2 = genrateKeywordExtract(df_main,entry_limit,common_word_max,"keywords_list",False)
    print(df1.dtypes)
    print("stop")
    df = df1.join(df2, how="inner", lsuffix='_t', rsuffix='_k')
    #df = df[["0_f_t","1_f_t","3_f_t","word_count_f_t","word_combined_f_t","0_f_k","1_f_k","3_f_k","word_count_f_k","word_combined_f_k","0_s_t","1_s_t","3_s_t","word_count_s_t","word_combined_s_t","0_s_k","1_s_k","3_s_k","word_count_s_k","word_combined_s_k"]]
    df = df[["word_count_f_t","word_combined_f_t","word_count_f_k","word_combined_f_k","word_count_s_t","word_combined_s_t","word_count_s_k","word_combined_s_k"]]#,"0_s_k","0_s_t"]]
    df['word_combined_all'] = df[["word_combined_f_t","word_combined_f_k"]].agg(' '.join, axis=1)
    df['word_count_all'] = df["word_count_f_t"]+df["word_count_f_k"]
    df['word_combined_all_sel'] = df[["word_combined_s_t","word_combined_s_k"]].agg(' '.join, axis=1)
    df['word_count_all_sel'] = df["word_count_s_t"]+df["word_count_s_k"]
    #df = df_main.join(df, how="inner", rsuffix='_new')  
    # df = df_main.join(df, how="inner", rsuffix='_new1')  
    df = df_main.join(df, how="inner")
    if add_nlp_stats :
        df = generateNLPonKeywords(df,0,df.shape[0],nlp_source_col,True,True)
    # df = df_main.join(df, how="inner", rsuffix='_new2')
    df = deleteUnnamed(df,"hash_key")
    saveDFcsv(df, keyword_path, keyword_filename,True)
    return df

print("IMPORT : embedding_keyword_module_lib")
# df_keyword = mainKeywordWF(entry_limit=1000,
#               common_word_max=700,
#               add_nlp_stats=True,
#               nlp_source_col="word_combined_all")