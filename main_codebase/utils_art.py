# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 00:11:13 2024

@author: Alexandre
"""

import pandas as pd
import os


## Conf variables
conf_file_path="C:/Users/User/OneDrive/Desktop/article/file_2/.code_control/"
conf_file_filename="code_conf_excel"

## Open Files

def openDFcsv(path, filename) :
    return pd.read_csv(path+filename+".csv")

def openDFplk(path, filename) :
    return pd.read_pickle(path+filename+".pkl")

def openSTRtxt(path, filename,returnList=True) :
    if "desktop.ini" not in filename :
        file = open(path+filename+".txt", "r", encoding="utf-8")
        if returnList :
            out_str = file.readlines()
        else :
            out_str = file.read()
        file.close()
    else :
        out_str = ""
    return out_str

def openDFxlsx(path, filename,sheet_name=None,header=1,index_col=None) :
    df = pd.read_excel(path+filename+".xlsx", sheet_name=sheet_name,header=header,index_col=index_col)
    return df

## Save Files

def saveDFcsv(df, path, filename="default_filename",override=True) : # ,index=False,index_label=""
    if not override :
        file_exist = os.path.isfile(path+filename+".csv")
        if file_exist :
            print("WARNING :",filename,"alredy exist so dataframe could not be saved.")
            return False
    df.to_csv(path+filename+".csv", encoding="utf-8",mode='w') # , index=False) index=index, index_label=index_label ,mode=mode
    return True

def saveDFplk(df, path, filename) :
    df.to_pickle(path+filename+".pkl") 
    
def saveSTRtxt(string, path, filename) :
    with open(path+filename+".txt", "w", encoding="utf-8") as file:
        file.write(string)
    file.close()
    

def openConfFile(sheet_name="gpt_models") :
    df = openDFxlsx(conf_file_path,conf_file_filename,sheet_name)
    # out_dict = df.to_dict('split')
    return df



def cfn_index(sheet_name="gpt_models",index=0,column_name="model_name") :#
    return cfn_field(sheet_name,"index",index,column_name)

def cfn_field(sheet_name="gpt_models",column_search="model_name",value_search="gpt-4",column_name="token_limit",max_return=1) :#
    df = openConfFile(sheet_name)
    if column_search not in list(df.columns) or ((column_name not in list(df.columns)) and  (column_name != "")):
        return None
    series = df.loc[(df[column_search] == value_search)]
    if series.shape[0] == 0:
        return None
    if column_name != "" :
        return list(series[column_name])[0:max_return]
    else :
        return series.to_dict("records")[0:max_return] #,index=False
    # return df

def deleteUnnamed(df,set_index="") :
    #un_col_list = df.columns.str.contains('unnamed', case=False)
    un_col_list = df.columns.str.contains('^Unnamed', case=False)
    if len(list(un_col_list))>0:
        df.drop(df.columns[df.columns.str.contains('^Unnamed', case=False)], axis=1, inplace=True)
    # df_q = df_q.loc[:, ~df_q.columns.str.contains('^Unnamed')]
    if set_index != "" :
        df = df.set_index(set_index)
    return df
    
def getOpenAIKey(path,filename) :
    return str(openSTRtxt(path,filename)[0])

def display_df(df):
    display(df)
    
print("IMPORT : utils_art")
