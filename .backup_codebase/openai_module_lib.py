# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 00:13:11 2024

@author: Alexandre
"""
import main_var
env = "test/"
mv = main_var.main_var(env=env)

from utils_art import openDFcsv,openSTRtxt,openDFxlsx,saveDFcsv,saveSTRtxt,openConfFile,cfn_index,cfn_field
import openai
import os
from os import *
from pathlib import Path
from openai import OpenAI
import tiktoken
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
import hashlib
import pandas as pd


# Chat GPT
openai.api_key = "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"
model_list = ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-16k","gpt-4-0125-preview"]
role_list = ["user","system", "assistant", "tool"]
max_token=100
llm_client = OpenAI(api_key=openai.api_key)
open_path = mv.article_path  # "C:/Users/User/OneDrive/Desktop/article/files_3/1_3_article_main/arc/"
save_path = mv.embdedding_path  # "C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embd_df/"
filename_save = mv.embdedding_filename # "embd_out_main_test"
select_fields_comp = ['o_id','i_content', 'o_content', 'i_role', 'i_model', 'i_temperature', 'i_token_max', 'i_n', 'i_seed', 'i_name', 'i_frequency_penalty', 'i_presence_penalty','o_system_fingerprint', 'o_logprobs', 'o_model', 'o_object', 'o_created', 'o_finish_reason', 'o_index', 'o_role', 'o_token_output', 'o_token_input', 'o_token_total', "valid"]
select_fields_emb = ['o_index','i_text', 'i_model', 'o_object', 'o_object_list', 'i_encoding_format', 'i_dimensions', 'o_data', 'o_token_input', 'o_token_total', "valid"]
selected_fields_comp = ["hash_key",'o_id', 'i_role', 'i_model', 'i_temperature', 'i_token_max', 'i_frequency_penalty', 'i_presence_penalty', 'o_created','i_content','o_content', 'o_token_output', 'o_token_input', 'o_token_total',"valid"]
selected_fields_emp = ["hash_key",'i_model', 'i_dimensions', 'i_encoding_format', 'i_user', 'o_data', 'o_token_input', 'o_token_total','valid']


# LMM Querries

def apply_completions(input_dict,display=False):
    chat_completion = llm_client.chat.completions.create(
        messages=[
            {
                "role": input_dict["i_role"],
                "content": input_dict["i_content"],
                "name": input_dict["i_name"],
            }
        ],
        model=input_dict["i_model"],
        temperature=input_dict["i_temperature"],
        max_tokens=input_dict["i_token_max"],
        n=input_dict["i_n"],
        seed=input_dict["i_seed"],
        frequency_penalty=input_dict["i_frequency_penalty"],
        presence_penalty=input_dict["i_presence_penalty"]
    )
    if display :
        print(chat_completion)
    return chat_completion

def apply_embeddings(input_dict,display=False):
    text_embeddings = llm_client.embeddings.create(
        input=input_dict["i_text"],
        model=input_dict["i_model"],
        encoding_format=input_dict["i_encoding_format"],
        dimensions=input_dict["i_dimensions"],
        user=input_dict["i_user"])
    if display :
        print(text_embeddings)
    return text_embeddings

## Ceate Input Conf

def llmInputConfCompletion(content,role_num=0,model_num=0,temperature=1,max_tokens=2000,num_answer=1,seed=0, hash_key=None) :
    return {"i_content":content,
            "i_role":role_list[role_num],
            "i_model":model_list[model_num],
            "i_temperature":temperature,
            "i_token_max":max_tokens,
            "i_n":num_answer,
            "i_seed":seed,
            "i_name":"name_test",
            "i_frequency_penalty":0,
            "i_presence_penalty":0,
            "hash_key":hash_key}

def llmInputConfEmbeddings(content, model="text-embedding-3-small", encoding_format="float", dimensions=10, hash_key=None) :
    return {"i_text":content,
            "i_model":model,
            "i_encoding_format":encoding_format,
            "i_dimensions":dimensions,
            "i_user":"name_test",
            "hash_key":hash_key}

## LLM Querry output Parsing

def outputDictParseCompletion(output,display=False) :
    out_dict = {}
    gpt_dict = dict(output)
    out_dict["o_id"] = gpt_dict["id"]
    out_dict["o_system_fingerprint"] = gpt_dict["system_fingerprint"]
    out_dict["o_logprobs"] = dict(gpt_dict["choices"][0])["logprobs"]
    out_dict["o_model"] = gpt_dict["model"]
    out_dict["o_object"] = gpt_dict["object"]
    out_dict["o_created"] = gpt_dict["created"]
    out_dict["o_finish_reason"] = dict(gpt_dict["choices"][0])["finish_reason"]
    out_dict["o_index"] = dict(gpt_dict["choices"][0])["index"]
    out_dict["o_content"] = dict(dict(gpt_dict["choices"][0])["message"])["content"]
    out_dict["o_role"] = dict(dict(gpt_dict["choices"][0])["message"])["role"]
    out_dict["o_object"] = gpt_dict["object"]
    out_dict["o_token_output"] = dict(gpt_dict["usage"])["completion_tokens"]
    out_dict["o_token_input"] = dict(gpt_dict["usage"])["prompt_tokens"]
    out_dict["o_token_total"] = dict(gpt_dict["usage"])["total_tokens"]
    if display :
        print(out_dict)
    return out_dict
        
def outputDictParseEmbeddings(output,display=False) :
    out_dict = {}
    gpt_dict = dict(output)
    out_dict["o_data"] = dict(gpt_dict["data"][0])["embedding"]
    out_dict["o_index"] = dict(gpt_dict["data"][0])["index"]
    out_dict["o_object"] = dict(gpt_dict["data"][0])["object"]
    out_dict["o_model"] = gpt_dict["model"]
    out_dict["o_object_list"] = gpt_dict["object"]
    out_dict["o_object_list"] = gpt_dict["object"]
    out_dict["o_token_input"] = dict(gpt_dict["usage"])["prompt_tokens"]
    out_dict["o_token_total"] = dict(gpt_dict["usage"])["total_tokens"]
    if display :
        print(out_dict)
    return out_dict

def parseList(list_par) :
    output_str = ""
    if type(list_par) == type([]) :
        for i in list_par :
            output_str = output_str + str(i)
    elif type(list_par) == type("") :
        output_str = list_par
    return str(output_str)


def textListToText(text_list) :
    out_list = ""
    for text in text_list :
        out_list = out_list + text
    return out_list

# def llmInputConfArticle(article_text,llm_prompt) :
#     context_prompt = "\nHere is the article :\n"
#     final_prompt = str(llm_prompt)+str(context_prompt)+article_text
#     return llmInputConf(final_prompt)

    
def num_tokens_from_string(text="", encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


# def saveNP(data,fmt='%f'): #path,
#     np.savetxt("C:/Users/User/OneDrive/Desktop/article/file_2/test_llm_output/test_save.txt",data, fmt=fmt)

# def loadNP(): #path
#     return np.loadtxt('C:/Users/User/OneDrive/Desktop/article/file_2/test_llm_output/test_save.txt', dtype=float)

def plot3Dpn(np_data):
    fig = px.scatter_3d(x=np_data[:, 0], y=np_data[:, 1], z=np_data[:, 2],color=np_data[:, 3], opacity=0.8)
    fig.show()

def plotTSNE(data,n_components=2,perplexity=3,random_state=10):
    tsne = TSNE(n_components=n_components,perplexity=perplexity,random_state=random_state) # , random_state=100
    X_tsne = tsne.fit_transform(data)
    print(tsne.kl_divergence_)
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1]) #, color=np.array(range(69))
    fig.update_layout(
        title="t-SNE visualization of Custom Classification dataset",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
    )
    fig.show()




def loadArticleFolderList(folder_path="",cutoff=99999999) :
    root_path = Path(folder_path)
    file_list = os.listdir(root_path)
    file_list = file_list[:cutoff]
    return file_list

def getNumberOfArticles(folder_path="", display=True) :
    root_path = Path(folder_path)
    file_list = os.listdir(root_path)
    file_list_len = len(file_list)
    if display : 
        print("In folder : ",folder_path," found ",file_list_len," article files.")
    return 

def loadListArticleHash(folder_path="",list_hash=[]) :
    text_dict_list = []
    for i in list_hash:
        hash_name = i.replace(".txt","")
        text = openSTRtxt(folder_path+"/",hash_name)
        text_loaded_list_len = len(text)
        dict_entry = {"hash_key":hash_name,"text":str(textListToText(text))} #textListToText(
        text_dict_list.append(dict_entry)
    return text_dict_list

def getStatsOnArticleText(article_text_list) :
    out_dict = {}
    out_dict["line_num"] = len(article_text_list)
    article_text = textListToText(article_text_list)
    out_dict["char_num"] =len(article_text)
    char_list = ["\n", ".","?","!",'"',",","“","”",":","–","-",";","http://","https://","$","€","|"]
    for char in char_list :
        out_dict[char] = article_text.count(char)
    return out_dict

def getDataToQuerryListLLM(max_prompt=5,articTquestF=True) :
    out_dict_List = []
    if articTquestF :
        getNumberOfArticles(open_path)
        filename_list = loadArticleFolderList(mv.article_path,max_prompt) # ["fa897c02295f34ce2e15f602769edf204ea00be7.txt"]
        out_dict_List = loadListArticleHash(mv.article_path,filename_list)
    else :
        prompt_list = cfn_field("prompts","prompt_type","content","prompt_value",max_prompt)
        for prompt in prompt_list:
            hash_key = hashlib.shake_256(str(prompt).encode()).hexdigest(20)
            out_dict_List.append({"hash_key":hash_key,"text":prompt})
    return out_dict_List


def dictSelectKeyList(input_dict,selected_fields) :
    out_dict = {}
    for key, value in input_dict.items():
        if key in selected_fields :
            out_dict[key] = value
    return out_dict

def getStandardDfnumColumn(num=10):
    return pd.DataFrame([], columns=[""+str(x) for x in range(num)])

def addDictToDF(df=None, ar_dict={},selected_fields=[]):
    if selected_fields == [] :
        selected_fields = ar_dict.keys()
    else :
        ar_dict = dictSelectKeyList(ar_dict,selected_fields)
    if type(df) == type(None) :
        df = pd.DataFrame([], columns = selected_fields) 
    df_add = pd.DataFrame([ar_dict], columns = selected_fields)
    if True : # list(df_add.columns) == (df.columns) :
        df = pd.concat([df,df_add]).reset_index(drop=True)
    else :
        print("WARNING : df could not be added because the columns list is different")
    return df








def mainGeneration(articleTRUEquestionFALSE=True,completionTRUEembedding=False,dimension=10,max_prompt=1000000,token_max_emb=7500,cara_max_emb=500,save_final=True,display_df=True,save_steps=True,step_pct=0.01):
    model_list = [0] # [0,1,2]
    temperature_list = [0.5] # [0,0.25,0.5,0.75,1]
    df=None
    set_index_key = "hash_key" #'o_created' #"hash_key"
    prompt_list = getDataToQuerryListLLM(max_prompt,articleTRUEquestionFALSE)
    # prompt_list = prompt_list[0:100]
    # prompt_list = cfn_field("prompts","prompt_type","content","prompt_value",max_prompt) #
    count = 0
    for prompt in prompt_list:
        for model_n in model_list:
            for temperature_n in temperature_list :
                valid_dict = {"valid":"VALID"}
                input_dict = ()
                if articleTRUEquestionFALSE :
                    input_dict = llmInputConfCompletion(prompt["text"],model_num=model_n,temperature=temperature_n,hash_key=prompt["hash_key"])
                    out_raw = apply_completions(input_dict)
                    out_dict = outputDictParseCompletion(out_raw)
                    selected_fields = selected_fields_comp
                else :
                    num_tokens = num_tokens_from_string(prompt["text"])
                    if num_tokens > token_max_emb :
                        valid_dict = {"valid":"WARNING"}
                        prompt["text"] = prompt["text"][0:cara_max_emb]
                    print(" - #"+str(count),"- ",valid_dict,"-",num_tokens,"-",len(prompt["text"]),"-",prompt["hash_key"])
                    input_dict = llmInputConfEmbeddings(prompt["text"],dimensions=dimension,hash_key=prompt["hash_key"])
                    print(input_dict)
                    print(type(input_dict))
                    out_raw = apply_embeddings(input_dict)
                    out_dict = outputDictParseEmbeddings(out_raw)
                    selected_fields = selected_fields_emp
                final_dict = input_dict | out_dict  # | valid_dict
                df = addDictToDF(df,final_dict,selected_fields)
                if (count%int(float(min(max_prompt,len(prompt_list)))*step_pct) == 0  and count != 0) and save_steps:
                    saveDFcsv(df.set_index(set_index_key), save_path, filename_save+"_"+str(count),True)
                count = count + 1
    if display_df :
        display(df.head(3))
    if save_final :
        saveDFcsv(df.set_index(set_index_key), save_path, filename_save,True)
    return df
























print("IMPORT : openai_module_lib")