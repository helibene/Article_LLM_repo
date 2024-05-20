# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 00:11:13 2024

@author: Alexandre
"""

import pandas as pd
import os

## Conf variables
conf_file_path="C:/Users/User/OneDrive/Desktop/Article_LLM/main_files/0_1_bin/"
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

# Conf file 

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

# Other 
def deleteUnnamed(df,set_index="") :
    un_col_list = df.columns.str.contains('^Unnamed', case=False)
    if len(list(un_col_list))>0:
        df.drop(df.columns[df.columns.str.contains('^Unnamed', case=False)], axis=1, inplace=True)
    if set_index != "" :
        if set_index in list(df.columns) :
            df = df.set_index(set_index)
    return df
    
def getOpenAIKey(path,filename) :
    return str(openSTRtxt(path,filename)[0])

def display_df(df):
    display(df)
    
def calculateRatio(n1,n2,stringReturn=True,round_num=2) :
    our_num = round(float(n1)/max(float(n2),1),round_num)
    if stringReturn :
        if n2 != 0 :
            return str(our_num)
        else :
            "error_div_0"
    else :
        return float(our_num)
     
def calculatePct(n1,n2,stringReturn=True,round_num=2,ajust_for_denom=0) :
    our_num = round((100*(-(float(n2)*ajust_for_denom)+float(n1)))/max(float(n2),1),round_num)
    if stringReturn :
        if n2 != 0 :
            return str(our_num)
        else :
            "error_div_0"
    else :
        return float(our_num)
    
def unionNfiles(pathList=[],filenameList=[],savePath="",saveFilename="") :
    file_out = openDFcsv(pathList[0],filenameList[0])
    for i in range(len(pathList)-1) :
        filenew = openDFcsv(pathList[i+1],filenameList[i+1])
        file_out = file_out.union(filenew)
    if savePath!="" and saveFilename!="":
        saveDFcsv(file_out,savePath,saveFilename)
    return file_out
        
def unionFiles(path1="",filename1="",path2="",filename2="",savePath="",saveFilename="") :
    file1 = openDFcsv(path1,filename1)
    file2 = openDFcsv(path2,filename2)
    file_out = pd.concat([file1,file2])
    file_out = deleteUnnamed(file_out,"hash_key")
    if savePath!="" and saveFilename!="":
        saveDFcsv(file_out,savePath,saveFilename)
    return file_out


def select_articles(directory_path,min_char=0,max_char=600,str_lookup="your activity and behavior on this site made us think that you are a bot",max_sel=99999999999): # "have permission to edit" "AI assistant is fun to use"
    out_list = []
    try:
        files = os.listdir(directory_path)
        count=0
        valid=True
        while valid and count<len(files):
            file=files[count]
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                #print(file)
                file_raw = file.replace(".txt","")
                data = openSTRtxt(directory_path,file_raw)
                data = str(data)
                if (len(data)>min_char and len(data)<max_char) or str_lookup in data :
                    out_list.append(file_raw)
            count=count+1
            if count>max_sel :
                valid=False
    except OSError:
        print("Error occurred while reading files.")
    print(out_list)
    print("Articles lookedup :",count,"  Article Selected :",len(out_list))
    return out_list

print("IMPORT : utils_art")

