# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 00:46:29 2024

@author: Alexandre
"""
import main_var
mv = main_var.main_var()
import pandas as pd
from utils_art import calculatePct, openDFcsv, deleteUnnamed, saveDFcsv, display_df

def joinQuerryAndParse(save=True,remove_invalid=True,display=True,filtered_input_df=False, union_df=None) :
    rename_dict = {"title_q":"title_quer","title_p":"title_par","pk_q":"pk","hash_key_q":"hash_key"}
    del_col_list = ["pk_p","hash_key_p","publish_date"]
    df_q = openDFcsv(mv.query_path,mv.query_filename)
    df_q.sort_values(by=['hash_key'],ascending=False)
    df_q_len = df_q.shape[0]
    if display :
        print("QUERRY dataset loaded from ",mv.query_path)
        print("QUERRY dataset has entry length of :",df_q_len,"\n")
    df_p = openDFcsv(mv.scarp_path,mv.scarp_filename)
    df_p_len = df_p.shape[0]
    if display :
        print("PARSSING dataset loaded from ",mv.scarp_path)
        print("PARSSING dataset has entry length of :",df_p_len," ("+calculatePct(df_p_len,df_q_len)+"% of querry data)\n")
    df_q.hash_key = df_q.hash_key.astype('str') 
    df_p.hash_key = df_p.hash_key.astype('str') 
    df = df_q.join(df_p, how="inner", lsuffix='_q', rsuffix='_p') #,on='hash_key'
    df = df.rename(columns=rename_dict)
    df = df.drop_duplicates(subset=['pk'])
    for col in del_col_list :
        if col in list(df.columns) :
            del df[col]
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
    if type(union_df)!=type(None):
        if str(union_df.dtypes)==str(df.dtypes) :
            df = pd.concat([df,union_df])
    if display :
        print("TOTAL yield : from",df_q_len," to ",join_df_len,"("+calculatePct(join_df_len,df_q_len)+"% yeald)\n")
    if save :
        df = deleteUnnamed(df,"hash_key")
        saveDFcsv(df,mv.join1_path,mv.join1_filename)
        if display :
            print("JOINED dataset saved here :",mv.join1_path+mv.join1_filename+".csv")
    return df

def joinAllDF(save=True, display_stats=True,display_data=True, union_df=None) :
    df_join1 = openDFcsv(mv.join1_path,mv.join1_filename)
    df_join1 = df_join1.set_index('hash_key')
    df_join1_len = df_join1.shape[0]

    df_embdedding = openDFcsv(mv.embdedding_path,mv.embdedding_filename)
    df_embdedding = df_embdedding.set_index('hash_key')
    df_embdedding_len = df_embdedding.shape[0]

    df_keyword = openDFcsv(mv.keyword_path,mv.keyword_filename)
    df_keyword = df_keyword.set_index('hash_key')
    df_keyword_len = df_keyword.shape[0]

    df = df_join1.join(df_embdedding, how="inner",on='hash_key', rsuffix='_e')
    df = df.join(df_keyword, how="inner",on='hash_key', rsuffix='_k')
    df_second_join_len = df.shape[0]
    df = deleteUnnamed(df,"hash_key")
    if display_stats :
        print("JOIN1 dataset loaded from ",mv.join1_path)
        print("JOIN1 dataset has entry length of :",df_join1_len,"\n")
        print("EMBEDDING dataset loaded from ",mv.embdedding_path)
        print("EMBEDDING dataset has entry length of :",df_embdedding_len," ("+calculatePct(df_embdedding_len,df_join1_len)+"% of querry data)\n")
        print("KEYWORD dataset loaded from ",mv.keyword_path)
        print("KEYWORD dataset has entry length of :",df_keyword_len," ("+calculatePct(df_keyword_len,df_join1_len)+"% of querry data)\n")
        print("TOTAL yield : from",df_join1_len," to ",df_second_join_len,"("+calculatePct(df_second_join_len,df_join1_len)+"% yeald)\n")
    
    if type(union_df)!=type(None):
        if str(union_df.dtypes)==str(df.dtypes) :
            df = pd.concat([df,union_df])
    if save :
        saveDFcsv(df,mv.join2_path,mv.join2_filename)
    if display_data :
        display_df(df)
    return df