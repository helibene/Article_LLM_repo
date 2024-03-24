# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 00:11:13 2024

@author: Alexandre
"""

import pandas as pd
from pathlib import Path
import os
import xlrd

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

# def cfn_field(sheet_name="gpt_models",column_search="model_name",value_search="gpt-4") :
#     df = openConfFile(sheet_name)
#     # df = list(df.loc[(df['index'] == index)][column_name])[0]
#     return df

#     for entry in conf_dict["data"] :
#         if entry[0] == field :
#             return str(conf_dict["data"][0][1])+str(entry[1])
        
# def cfn_field(sheet_name="gpt_models",index=0,filename="",addRoot=True) :
#     conf_dict = openConfFile(sheet_name)
#     for entry in conf_dict["data"] :
#         if entry[0] == field :
#             return str(conf_dict["data"][0][1])+str(entry[1])


print("imported 'utils_art'")
#df = openDFxlsx(conf_file_path,conf_file_filename,"paths_list")
# out1 = cfn_index(index=2,column_name="")
# print(out1)
# out2 = cfn_index(column_name="model_name")
# print(out2)
#out3 = cfn_field(column_name="")
#print(out3)

#out1 = cfn_field("prompts","index",0,"prompt_value")
#print(out1)
#out1 = cfn_index("prompts",0,"prompt_value")
#print(out1)


# def openSTRtxtList(path, filename) :
#     file = open(path+filename+".txt", "r", encoding="utf-8")
#     out_str = file.readlines()
#     file.close()
#     return out_str

# def openSTRtxt(path, filename) :
#     file = open(path+filename+'.txt') # "r", encoding="utf-8"
#     out_str = file.readlines()
#     file.close()
#     return out_str
# file = open()
# file.write(string)
#file.close()
# df = openDFcsv("C:/Users/User/OneDrive/Desktop/article/file_2/.code_control/","excel_path_table_test_raw_csv")
# print(df["path_name"])
# filepath = Path(path+filename+".txt", "w")  
# filepath.parent.mkdir(parents=True, exist_ok=True)  
# df.to_csv(filepath)

# with open(path+filename+".txt") as file: # , "w"
#    file.write(string)

# saveSTRtxt("test","C:/Users/User/OneDrive/Desktop/article/files/","article_scrap_num_XXXXXX")
# display(df)

# data = pd.read_excel("C:/Users/User/OneDrive/Desktop/article/file_2/.code_control/code_conf_excel.xlsx", sheet_name='gpt_models', header=1,index_col="index")
# print(data.head())
# print(data.info())
# print(type(data))
# display(data)

# def openConfFile(full_path) :
#     csv_df = pd.read_csv(full_path)
#     out_dict = csv_df.to_dict('split')
#     return out_dict

# def cfn(field="join2_folder_out",filename="",addRoot=True) :
#     conf_dict = openConfFile(conf_file_path)
#     for entry in conf_dict["data"] :
#         if entry[0] == field :
#             return str(conf_dict["data"][0][1])+str(entry[1])

# def cfn_index(sheet_name="gpt_models",index=1,column_name="model_name") :#
#     df = openConfFile(sheet_name)
#     if 'index' not in list(df.columns) or ((column_name not in list(df.columns)) and  (column_name != "")):
#         return None
#     series = df.loc[(df['index'] == index)]
#     if series.shape[0] == 0:
#         return None
#     if column_name != "" :
#         return list(series[column_name])[0]
#     else :
#         return series.to_dict("records")[0] #,index=False
#     # return df

# def cfn_field(sheet_name="gpt_models",column_search="model_name",value_search="gpt-4",column_name="token_limit",max_return=2) :#
#     df = openConfFile(sheet_name)
#     series = df.loc[(df[column_search] == value_search)]
#     if series.shape[0] == 0:
#         return None
#     if column_name != "" :
#         return list(series[column_name])[0]
#     else :
#         return series.to_dict("records")[0] #,index=False
#     # return df