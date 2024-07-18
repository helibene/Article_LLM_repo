# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 00:13:11 2024

@author: Alexandre
"""


import main_var
mv = main_var.main_var()

from utils_art import openDFcsv,openSTRtxt,openDFxlsx,saveDFcsv,saveSTRtxt,openConfFile,cfn_index,cfn_field,getOpenAIKey
import openai
import os
#from os import *
from pathlib import Path
from openai import OpenAI
import tiktoken
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
import hashlib
import pandas as pd
from utils_art import deleteUnnamed,display_df
import requests
import shutil
from PIL import Image 


# Chat GPT
openai.api_key = getOpenAIKey('C:/Users/User/OneDrive/Desktop/Article_LLM/api_key/','api_key')
llm_client = OpenAI(api_key=openai.api_key)

model_list_comp = ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-16k","gpt-3.5-turbo-instruct","gpt-4-0125-preview","gpt-4-turbo","gpt-4"]
model_list_embd = ["text-embedding-3-small", "text-embedding-3-large","text-embedding-ada-002"]
model_list_img = ["dall-e-2", "dall-e-3"]
role_list = ["user","system", "assistant", "tool"]
user_list = ["u1","u2","u3"]
model_num_comp=0
model_num_embd=0
model_num_img=1
role_num=0
max_token=100

open_path = mv.article_path  # "C:/Users/User/OneDrive/Desktop/article/files_3/1_3_article_main/arc/"
save_path = mv.embdedding_path  # "C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embd_df/"
filename_save = mv.embdedding_filename # "embd_out_main_test"
embdedding_path = mv.embdedding_path
embdedding_filename = mv.embdedding_filename
completion_path = mv.completion_path
completion_filename = mv.completion_filename
image_path = mv.image_path
image_filename = mv.image_filename



fields_comp1 = ['o_id','i_content', 'o_content', 'i_role', 'i_model', 'i_temperature', 'i_token_max', 'i_n', 'i_seed', 'i_name', 'i_frequency_penalty', 'i_presence_penalty','o_system_fingerprint', 'o_logprobs', 'o_model', 'o_object', 'o_created', 'o_finish_reason', 'o_index', 'o_role', 'o_token_output', 'o_token_input', 'o_token_total', "valid"]
fields_emp1 = ['o_index','i_text', 'i_model', 'o_object', 'o_object_list', 'i_encoding_format', 'i_dimensions', 'o_data', 'o_token_input', 'o_token_total', "valid"]
fields_comp2 = ["hash_key",'o_id', 'i_role', 'i_model', 'i_temperature', 'i_token_max', 'i_frequency_penalty', 'i_presence_penalty', 'o_created','i_content','o_content', 'o_token_output', 'o_token_input', 'o_token_total',"valid"]
fields_emp2 = ["hash_key",'i_model', 'i_dimensions', 'i_encoding_format', 'i_user', 'o_data', 'o_token_input', 'o_token_total','valid']
fields_all_comp = ["i_content","i_role","i_user","i_model","i_temperature","i_token_max","i_n","i_seed","i_frequency_penalty","i_presence_penalty","hash_key","o_id","o_system_fingerprint","o_logprobs","o_model","o_object","o_created","o_finish_reason","o_index","o_content","o_role","o_token_output","o_token_input","o_token_total"]
fields_all_emb = ["i_text","i_user","i_model","i_encoding_format","i_dimensions","hash_key","o_data","o_index","o_object","o_model","o_object_list","o_token_input","o_token_total"]
fields_all_img = ["i_prompt","i_model","i_size","i_quality","i_number","i_response_format","i_style","i_user","hash_key","o_created","o_url","o_revised_prompt"]

selected_fields_comp = fields_all_comp
selected_fields_emp = fields_all_emb
selected_fields_img = fields_all_img
# LMM Querries

def apply_completions(input_dict,display=False):
    chat_completion = llm_client.chat.completions.create(
        messages=[
            {
                "role": input_dict["i_role"],
                "content": input_dict["i_content"],
            }
        ],
        model=input_dict["i_model"],
        temperature=input_dict["i_temperature"],
        max_tokens=input_dict["i_token_max"],
        n=input_dict["i_n"],
        seed=input_dict["i_seed"],
        frequency_penalty=input_dict["i_frequency_penalty"],
        presence_penalty=input_dict["i_presence_penalty"],
        user=input_dict["i_user"]
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

def apply_image_gen(input_dict,display=False):
    image_generation = llm_client.images.generate(
        prompt=input_dict["i_prompt"],
        model=input_dict["i_model"],
        size=input_dict["i_size"],
        quality=input_dict["i_quality"],
        user=input_dict["i_user"],
        response_format=input_dict["i_response_format"],
        style=input_dict["i_style"],
        n=input_dict["i_number"])
    if display :
        print(image_generation)
    return image_generation
## Ceate Input Conf

def llmInputConfCompletion(content,role_num=0,model_num=0,user_num=0,temperature=1,max_tokens=2000,num_answer=1,seed=0, hash_key=None) :
    return {"i_content":content,
            "i_user":user_list[user_num],
            "i_role":role_list[role_num],
            "i_model":model_list_comp[model_num],
            "i_temperature":temperature,
            "i_token_max":max_tokens,
            "i_n":num_answer,
            "i_seed":seed,
            "i_frequency_penalty":0,
            "i_presence_penalty":0,
            "hash_key":hash_key}

def llmInputConfEmbeddings(content,user_num=0, model_num=0, encoding_format="float", dimensions=10,hash_key=None) :
    return {"i_text":content,
            "i_user":user_list[user_num],
            "i_model":model_list_embd[model_num],
            "i_encoding_format":encoding_format,
            "i_dimensions":dimensions,
            "hash_key":hash_key}

def llmInputConfImage(prompt="A photograph of a white Siamese cat.", model_num=0,user_num=0, size="1792x1024", quality="hd",number=1, response_format="url",style='vivid',hash_key=None) : # vivid  natural standard
    return {"i_prompt":prompt,
            "i_model":model_list_img[model_num],
            "i_size":size,
            "i_quality":quality,
            "i_number":number,
            "i_response_format":response_format,
            "i_style":style,
            "i_user":user_list[user_num],
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
    out_dict["o_token_input"] = dict(gpt_dict["usage"])["prompt_tokens"]
    out_dict["o_token_total"] = dict(gpt_dict["usage"])["total_tokens"]
    if display :
        print(out_dict)
    return out_dict

def outputDictParseImages(output,display=False) :
    out_dict = {}
    gpt_dict = dict(output)
    out_dict["o_created"] = gpt_dict["created"]
    out_dict["o_url"] = dict(gpt_dict["data"][0])["url"]
    out_dict["o_revised_prompt"] = dict(gpt_dict["data"][0])["revised_prompt"]
    
    if display :
        print(out_dict)
    return out_dict

# def parseList(list_par) :
#     output_str = ""
#     if type(list_par) == type([]) :
#         for i in list_par :
#             output_str = output_str + str(i)
#     elif type(list_par) == type("") :
#         output_str = list_par
#     return str(output_str)

def textListToText(text_list) :
    out_list = ""
    for text in text_list :
        out_list = out_list + text
    return out_list

def num_tokens_from_string(text="", encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def embdApplyMaxToken(prompt_text, token_max_emb, valid_dict, cara_max_emb=1000) :
    num_tokens = num_tokens_from_string(prompt_text)
    if num_tokens > token_max_emb :
        valid_dict = {"valid":"WARNING"}
        prompt_text = prompt_text[0:cara_max_emb]
    return prompt_text,valid_dict
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

# def getDataToQuerryListLLM(max_prompt=5,articTquestF=True) :
#     out_dict_List = []
#     if articTquestF :
#         getNumberOfArticles(mv.article_path)
#         filename_list = loadArticleFolderList(mv.article_path,max_prompt) # ["fa897c02295f34ce2e15f602769edf204ea00be7.txt"]
#         out_dict_List = loadListArticleHash(mv.article_path,filename_list)
#     else :
#         prompt_list = cfn_field("prompts","prompt_type","content","prompt_value",max_prompt)
#         for prompt in prompt_list:
#             hash_key = hashlib.shake_256(str(prompt).encode()).hexdigest(20)
#             out_dict_List.append({"hash_key":hash_key,"text":prompt})
#     return out_dict_List


def getDataToQuerryListLLM(max_prompt=5,input_selection="art",promt_type="image") : #content
    out_dict_List = []
    if input_selection=="art" :
        getNumberOfArticles(mv.article_path)
        filename_list = loadArticleFolderList(mv.article_path,max_prompt) # ["fa897c02295f34ce2e15f602769edf204ea00be7.txt"]
        out_dict_List = loadListArticleHash(mv.article_path,filename_list)
    elif input_selection=="que" :
        prompt_list = cfn_field("prompts","prompt_type",promt_type,"prompt_value",max_prompt)
        # print(promt_type)
        # print(prompt_list)
        for prompt in prompt_list:
            hash_key = hashlib.shake_256(str(prompt).encode()).hexdigest(20)
            out_dict_List.append({"hash_key":hash_key,"text":prompt})
    else :
        print("ERROR could not find mentioned Input Type : ",input_selection)
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

def scrapImage(final_dict):
    import urllib3
    url = final_dict["o_url"]
    file_name = str(str(save_path)+str("img")+str("/")+str(final_dict["hash_key"])+"xxx"+str(".png"))
    data = requests.get(url).content 
    f = open(file_name,'wb') 
    f.write(data) 
    f.close() 
    # res = requests.get(url, stream = True)
    # urllib3.urlretrieve(url, file_name)
    # print(type(res.status_code))
    # if res.status_code == 200:
    #     print("lol")
    #     print(res.raw)
    #     print(type(res.raw))
    #     #print(res.json())
    #     # print(res.data)
    #     # print(res.content)


    #     with open(file_name,'wb') as f:
            
    #         shutil.copyfileobj(res.raw, f)
    #         pass
    # else :
    #     final_dict["valid"] = "ERROR"
    #     print("Could not download image '"+final_dict["hash_key"]+"' (error code :"+str(res.status_code)+")")
    return final_dict

def getAlgoTypeVars(algo_type="emb"):
    selected_fields = []
    selected_save_path = ""
    selected_save_file = ""
    if algo_type=="cmp" :
        selected_fields = selected_fields_comp
        selected_save_path = completion_path
        selected_save_file = completion_filename
    elif algo_type=="emb":
        selected_fields = selected_fields_emp
        selected_save_path = embdedding_path
        selected_save_file = embdedding_filename
    elif algo_type=="img":
        selected_fields = selected_fields_img
        selected_save_path = image_path
        selected_save_file = image_filename
    else:
        print("ERROR could not find mentioned Algorithm Type to find save path : ",algo_type)
    return selected_fields, selected_save_path, selected_save_file
def mainGeneration(input_data="art",algo_type="emb",dimension=10,max_prompt=1000000,token_max_emb=7500,cara_max_emb=1000,save_final=True,save_steps=True,display_df_var=False,display_steps=True,step_pct=0.01):
    #input_data : "art" Article, "que" Preset Questions, "img" Image
    #algo_type : "emb" Embeddings, "cmp" Completion, "img" Image Genereation
    df=None
    temperature=0.5
    max_tokens=300
    set_index_key = "hash_key" #'o_created' #"hash_key"
    promt_type=""
    if input_data == "que" :
        promt_type = "image"#"content"
    if input_data == "art" :
        promt_type = "instructions"
    elif input_data == "img" :
        promt_type = "image"
    prompt_list = getDataToQuerryListLLM(max_prompt,input_data,promt_type)
    #print(prompt_list)
    count = 0
    for prompt in prompt_list:
        valid_dict = {"valid":"VALID"}
        input_dict = {}
        if algo_type=="cmp" :
            input_dict = llmInputConfCompletion(prompt["text"],model_num=model_num_comp,temperature=temperature,max_tokens=max_tokens,hash_key=prompt["hash_key"])
            out_raw = apply_completions(input_dict)
            out_dict = outputDictParseCompletion(out_raw)
            selected_fields = selected_fields_comp
        elif algo_type=="emb":
            prompt["text"],valid_dict=embdApplyMaxToken(prompt["text"], token_max_emb, valid_dict)
            input_dict = llmInputConfEmbeddings(prompt["text"],model_num=model_num_embd,dimensions=dimension,hash_key=prompt["hash_key"])
            out_raw = apply_embeddings(input_dict)
            out_dict = outputDictParseEmbeddings(out_raw)
            selected_fields = selected_fields_emp
        
        elif algo_type=="img":
            input_dict = llmInputConfImage(prompt["text"],model_num=model_num_img,hash_key=prompt["hash_key"],quality="hd",size="1024x1024")
            out_raw = apply_image_gen(input_dict)
            out_dict = outputDictParseImages(out_raw)
            selected_fields = selected_fields_img
        else:
            print("ERROR could not find mentioned Algorithm Type : ",algo_type)
            valid_dict = {"valid":"ERROR"}
        final_dict = input_dict | out_dict | valid_dict
        if algo_type=="img":
            final_dict = scrapImage(final_dict)
        if display_steps :
            print(" - #"+str(count),"- ",valid_dict,"-",len(prompt["text"]),"-",prompt["hash_key"])
        selected_fields, selected_save_path, selected_save_file = getAlgoTypeVars(algo_type)
        df = addDictToDF(df,final_dict,selected_fields)
        if (count%max(1,int(float(min(max_prompt,len(prompt_list)))*step_pct)) == 0  and count != 0) and save_steps:
            df_int = deleteUnnamed(df,set_index_key)
            saveDFcsv(df_int, selected_save_path, selected_save_file+"_"+str(count),True)
        count = count + 1
    if display_df_var :
        display_df(df.head(3))
    if save_final :
        df = deleteUnnamed(df,set_index_key)
        saveDFcsv(df, selected_save_path, selected_save_file,True)
    return df
mainGeneration(input_data="que",algo_type="img",step_pct=1,max_prompt=20)
#scrapImage({"hash_key":"test"})
print("IMPORT : openai_module_lib")