# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 00:46:29 2024

@author: Alexandre
"""
import main_var
mv = main_var.main_var()
import pandas as pd
from utils_art import *

def joinQuerryAndParse(save=True,remove_invalid=True,display=True,filtered_input_df=False) :
    # rename_dict = {"title_q":"title_quer","title_p":"title_par","published_q":"published","year_q":"year","year_month_q":"year_month","source_url_q":"source_url","url_list_q":"url_list","url_TLD_q":"url_TLD","source_title_q":"source_title","category_q":"category","authors":"authors","keywords_list":"keywords_list","text_len_p":"text_len","tb.sentences":"tb.sentences","tb.noun_phrases":"tb.noun_phrases","tb.words":"tb.words","tb.polarity":"tb.polarity","tb.subjectivity":"tb.subjectivity","tb.p_pos":"tb.p_pos","tb.p_neg":"tb.p_neg","vs.neg":"vs.neg","vs.neu":"vs.neu","vs.pos":"vs.pos","vs.compound":"vs.compound","valid":"valid","link_q":"link","pk_q":"pk",}
    # rename_dict = {"title_q":"title_quer","title_p":"title_par","published_q":"published","year_q":"year","year_month_q":"year_month","source_url_q":"source_url","url_list_q":"url_list","url_TLD_q":"url_TLD","source_title_q":"source_title","category_q":"category","authors":"authors","keywords_list":"keywords_list","text_len_p":"text_len","tb.sentences":"sentences","tb.noun_phrases":"noun_phrases","tb.words":"words","tb.polarity":"polarity","tb.subjectivity":"subjectivity","tb.p_pos":"tb.pos","tb.p_neg":"tb.neg","vs.neg":"vs.neg","vs.neu":"vs.neu","vs.pos":"vs.pos","vs.compound":"vs.comp","valid":"valid","link_q":"link","pk_q":"pk"}
    #rename_dict = {"pk_q":"pk"}
    
    rename_dict = {"title_q":"title_quer","title_p":"title_par","pk_q":"pk","hash_key_q":"hash_key"}
    del_col_list = ["pk_p","hash_key_p","publish_date"]
    df_q = openDFcsv(mv.query_path,mv.query_filename)
    df_q.sort_values(by=['hash_key'],ascending=False)
    #df_q = deleteUnnamed(df_q,"hash_key")
    df_q_len = df_q.shape[0]
    if display :
        print("QUERRY dataset loaded from ",mv.query_path)
        print("QUERRY dataset has entry length of :",df_q_len,"\n")
    df_p = openDFcsv(mv.scarp_path,mv.scarp_filename)
    # df_p = deleteUnnamed(df_p,"hash_key")
    df_p_len = df_p.shape[0]
    if display :
        print("PARSSING dataset loaded from ",mv.scarp_path)
        print("PARSSING dataset has entry length of :",df_p_len," ("+calculatePct(df_p_len,df_q_len)+"% of querry data)\n")
    # df = df_q.join(df_p, how="inner", lsuffix='_q', rsuffix='_p') #,on='hash_key'
    df_q.hash_key = df_q.hash_key.astype('str') 
    df_p.hash_key = df_p.hash_key.astype('str') 
    print(type(df_q))
    print(type(df_p))
    print(df_q.dtypes)
    print(df_p.dtypes)
    df = df_q.join(df_p, how="inner", lsuffix='_q', rsuffix='_p') #,on='hash_key'
    df = df.rename(columns=rename_dict)
    df = df.drop_duplicates(subset=['pk'])
    for col in del_col_list :
        if col in list(df.columns) :
            del df[col]
    # df = df.set_index('hash_key')
    join_df_len = df.shape[0]
    if display :
        print("JOINED dataset has entry length of :",join_df_len," ("+calculatePct(join_df_len,df_p_len)+"% of parssing data)")
    if remove_invalid :
        df = df.loc[(df['valid'] == True)]
        join_df_valid_len = df.shape[0]
        if display :
            print("JOINED dataset VALID entries :",join_df_valid_len," ("+calculatePct(join_df_valid_len,join_df_len)+"% of joined data)")
            print("JOINED dataset INVALID entries :",join_df_len-join_df_valid_len," ("+calculatePct(join_df_valid_len,join_df_len,ajust_for_denom=1)+"% of joined data)\n")
        join_df_len = join_df_valid_len
    if display :
        print("TOTAL yield : from",df_q_len," to ",join_df_len,"("+calculatePct(join_df_len,df_q_len)+"% yeald)\n")
    if save :
        df = deleteUnnamed(df,"hash_key")
        saveDFcsv(df,mv.join1_path,mv.join1_filename)
        if display :
            print("JOINED dataset saved here :",mv.join1_path+mv.join1_filename+".csv")
    return df

def joinAllDF() :
    df_join1 = openDFcsv(mv.join1_path,mv.join1_filename)
    df_join1 = df_join1.set_index('hash_key')
    df_embdedding = openDFcsv(mv.embdedding_path,mv.embdedding_filename)
    df_embdedding = df_embdedding.set_index('hash_key')
    df_keyword = openDFcsv(mv.keyword_path,mv.keyword_filename)
    df_keyword = df_keyword.set_index('hash_key')
    df = df_join1.join(df_embdedding, how="inner",on='hash_key', rsuffix='_e')
    df = df.join(df_keyword, how="inner",on='hash_key', rsuffix='_k')
    # df = deleteUnnamed(df,"hash_key")
    saveDFcsv(df,mv.join2_path,mv.join2_filename)
    print(mv.join2_path,mv.join2_filename+"test")
    return df