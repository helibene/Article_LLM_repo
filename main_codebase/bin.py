# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:05:32 2024

@author: Alexandre
"""



"""IMPORTS"""
# import json
# from pandas import *
# from datetime import date, datetime, timedelta
# import utils
# import test
# import xlrd
# from pathlib import Path
# from sklearn.datasets import make_classification
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from sklearn.decomposition import IncrementalPCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.manifold import Isomap
# from sklearn.neighbors import KNeighborsTransformer
# from sklearn.pipeline import make_pipeline
# from sklearn.datasets import make_regression
# from textblob import TextBlob
# import tempfile


# np.random.seed(0)





"""MAIN_VAR"""
# ####   article_scraping
# env = "arc/"
# main_path = "C:/Users/User/OneDrive/Desktop/article/files_3/1_1_query_main/"+env
# filename = "reduction_test"
# path_parse_stat = "C:/Users/User/OneDrive/Desktop/article/files_3/1_2_scarp_main/"+env
# path_union_stat = "C:/Users/User/OneDrive/Desktop/article/files_3/1_4_join_main/"+env
# filename_union_stat = "join_main_test-filter"
# llm_join_out = "C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/"
# llm_join_out_filename = "article_stats_embedding"

# ####   article_parsing
# env = "arc/"
# open_path = "C:/Users/User/OneDrive/Desktop/article/files_3/1_1_query_main/"+env
# filename_input = "main_scrap"#"query_large_2010_to_2023"#"query_main_final"
# save_path = "C:/Users/User/OneDrive/Desktop/article/files_3/1_2_scarp_main/"+env #parssing_df_main/
# filename_out = "test_vol___________XXXXXXXX"
# save_path_article = "C:/Users/User/OneDrive/Desktop/article/files_3/1_3_article_main/"+env #article_download_main_2

# ####   openai_module
# open_path = "C:/Users/User/OneDrive/Desktop/article/files_3/1_3_article_main/arc/"
# save_path = "C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embd_df/"
# filename_save = "embd_out_main_test"
# llm_result = "C:/Users/User/OneDrive/Desktop/article/file_2/test_llm_output/new_embedding/"
# ####   embedding_keyword_module
# folder_path_input = "C:/Users/User/OneDrive/Desktop/article/files_3/1_4_join_main/arc/"
# filename_input = "join_main_big"
# folder_path_embd_df="C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embd_df/"
# folder_path_embd_join="C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embd_join/"
# folder_path_embd_raw="C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/test/"
# filename_embd_df="embd_out_main_testfinal"
# filename_embd_join="embd_all_main_testfinal"
# filename_embd_raw_save="embedding_matrix_max"
# filename_embd_raw_load="embedding_matrix_XXXX"
# folder_path_keyword_out="C:/Users/User/OneDrive/Desktop/article/files_3/2_2_keyword_main/out/"
# folder_path_keyword_out_nlp="C:/Users/User/OneDrive/Desktop/article/files_3/2_2_keyword_main/out_nlp/"
# filename_keyword_out="keywords_df_main_test"
# filename_keyword_out_nlp="keywords_df_nlp_main_test"
# # # folder_path="C:/Users/User/OneDrive/Desktop/article/file_2/test_llm_output/run2/"
# # folder_path_embd="C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/embd/only_embd/"
# # folder_path_keyword="C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/keyword/"
# # folder_path_llm="C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/llm/"
# # folder_path_graph="C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/graphs/"
# # folder_path_viz="C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/viz/"
# # \files_3\1_4_join_main\arc
# # filename="article_stats_embedding"
# # filename_embd="embedding_matrix_1000"#"embedding_matrix_main"#"article_stats_embedding_smain" #""
# # filename_keyw="article_keyword_main"
# # open_path_keyw="C:/Users/User/OneDrive/Desktop/article/file_2/join_1_df/"
# ####   visualization_module
# folder_path_input_df="C:/Users/User/OneDrive/Desktop/article/files_3/3_1_join_main/arc/"
# folder_path_input_embd="C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embf_raw_viz/"
# filename_input_df="keyword_with_nlp_viz_col1"
# filename_input_embd="embedding_matrix_20000"

# folder_path_graph="C:/Users/User/OneDrive/Desktop/article/files_3/3_2_visu_main/"



"""UTIL_ART"""
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

"""ARTICLE_SCRAPING"""
# print(df[~ser_source_low])
# init_size = df.shape[0]
# init_uni_size = len(ser_source)
# if display_stats :
#     print("START       - ",{"Articles sum :":df.shape[0],"Number of unique sources :":len(ser_source),"Average number of article per source :":round(float(df.shape[0])/float(len(ser_source)),2)})

# ser_source_high = ser_source[ser_source > thd_high]
# ser_source_low = ser_source[ser_source < thd_low]
# ser_source_med = ser_source[(ser_source > thd_low) & (ser_source < thd_high)]
# print("ser_source_low",ser_source_low)
# ser_source_low2 = ser_source_low['source_title'].value_counts().to_frame("count").sort_values(by=['count'],ascending=True)
# print("ser_source_low2",ser_source_low)
# list_source_low = list(ser_source_low.keys())
# df_low = df['source_title'].isin(list_source_low)
# df = df[~df_low]
# #df_low_len = df_low.value_counts().to_frame("count").sum()
# df_low_len = len(list(df.value_counts().to_frame("count")))
# if display_stats :
#     print("TOO LOW     - ",{"Articles sum :":df.shape[0],"Number of unique sources :":df_low_len,"Average number of article per source :":round(float(df_low.shape[0])/float(df_low_len),2)})
#     print("TOO LOW     - ","Articles removed :",df_low.shape[0],'  ('+str(round(100*df_low.shape[0]/df.shape[0],2))+'%)')

# # ser_source = df['source_title'].value_counts()
# # ser_source_high = ser_source[ser_source > thd_high]
# list_source_high = list(ser_source_high.keys())
# start_df = getStandardDfInput(list(df.columns))
# for source_high in list_source_high :
#     newdf = df.loc[(df['source_title'] == source_high)].sort_values(by=["hash_key"],ascending=False).head(thd_high)
#     start_df = pd.concat([start_df,newdf]).reset_index(drop=True)

# df_high = df['source_title'].isin(list_source_high)
# df_high_len = len(df_high.value_counts())
# init_size2 = df.shape[0]
# df = df[~df_high]
# df = pd.concat([df,start_df]).reset_index(drop=True)
# ser_source = df['source_title'].value_counts()
# end_size = df.shape[0]
# end_uni_size = len(ser_source)
# df["hash_key"].drop_duplicates()
# if display_stats :
#     print("TOO HIGH    - ",{"Articles sum :":df_high.shape[0],"Number of unique sources :":df_high_len,"Average number of article per source :":round(df_high.shape[0]/df_high_len)})
#     print("TOO HIGH    - ","Articles removed :",init_size2-df.shape[0],'  ('+str(round(100*df_high.shape[0]/df.shape[0],2))+'%)')

#     print("FINAL DF    - ",{"Articles sum :":df.shape[0],"Number of unique sources :":end_uni_size,"Average number of article per source :":round(df.shape[0]/end_uni_size)})
#     # print("len loss :",float(100*(init_size-end_size)/init_size),"%","     uni_len loss :",float(100*(init_uni_size-end_uni_size)/init_uni_size),"%")
#     # print("MAX :",max(df['source_title'].value_counts()),"  MIN :",min(df['source_title'].value_counts()))

# if display_end_stats :
#     plotDFstatisticsQuerry(df,onlyYear=True)
# if save :
#     saveDFcsv(df,main_path,filename+"_cap")
# return df
# ssource_limiy_count = int(ser_source.iloc[[int(thd_low)]]["count"].tolist()[0])
# ser_source = ser_source[ser_source['count'].between(source_limiy_count, 1000000)]


# df = openDFcsv("C:/Users/User/OneDrive/Desktop/Article_LLM/main_files/1_1_query_main/arc/","query_medium_2010_to_2023")
# display(df)
# df = randomSampleSelection(df,30)
# display(df)


# def mainJoin() :
#     #join1_path,embdedding_path,embdedding_path_raw,keyword_path,join1_filename,embdedding_filename,embdedding_filename_raw,keyword_filename
#     # ekml.mainKeywordWF(9999999,500)
#     join1_path_i = join1_path
#     embdedding_path_i = embdedding_path
#     embdedding_path_raw_i = embdedding_path_raw
#     keyword_filename_i = keyword_filename
#     join1_filename_i = join1_filename
#     embdedding_filename_i = embdedding_filename
#     embdedding_filename_raw_i = embdedding_filename_raw
#     embdedding_path_raw_i = embdedding_path_raw
#     keyword_path_i = keyword_path
#     join1 = openDFcsv(join1_path_i,join1_filename_i)
#     embdedding = openDFcsv(embdedding_path_i,embdedding_filename_i)
#     embdedding_raw = openDFcsv(embdedding_path_raw_i,embdedding_filename_raw_i)
#     keyword = openDFcsv(keyword_path_i,keyword_filename_i)
#     join1 = join1.set_index('hash_key')
#     embdedding = embdedding.set_index('hash_key')
#     keyword = keyword.set_index('hash_key')
#     print("join1 col")
#     print(join1.dtypes)
#     print(join1.shape[0])
#     print("embdedding col")
#     print(embdedding.dtypes)
#     print(embdedding.shape[0])
#     print("keyword col")
#     print(keyword.dtypes)
#     print(keyword.shape[0])
#     # embdedding =embdedding[["hash_key"]]
#     df = join1.join(embdedding, how="inner",on='hash_key',lsuffix='_a', rsuffix='_b') #
#     df = df.join(keyword, how="inner",on='hash_key',lsuffix='_c', rsuffix='_d') #
#     # df = join1.join(keyword, how="inner",on='hash_key',lsuffix='_z', rsuffix='_a')
#     print("out : ",join2_path,join2_filename)
#     saveDFcsv(df,join2_path,join2_filename)
#     # return df

# join1_path+join1_filename
# embdedding_filename+embdedding_filename
# embdedding_path_raw+embdedding_filename_raw
# keyword_path
# keyword_filename

# def joinArticleStatsAndLLM(df_art,save=True,display=True) :
#     df_llm = loadFromFolder(llm_result,save=False) # ,"","".set_index('hash_key')
#     df_llm = df_llm.set_index('hash_key')
#     df = df_art.join(df_llm, how="inner",on='hash_key',lsuffix='', rsuffix='_llm')#, 
#     # df = df[union_fields2]
#     join2_df_len = df.shape[0]
#     df_art_len = df_art.shape[0]
#     if display :
#         print("JOINED2 dataset has entry length of :",join2_df_len," ("+calculatePct(join2_df_len,df_art_len)+"% of JOINED1 data)")
#     if save :
#         df = deleteUnnamed(df,"hash_key")
#         saveDFcsv(df,llm_join_out,llm_join_out_filename)
#     return df




"""ARTICLE_PARSING"""
# def generateNLPonKeywords(df_main,index_from=0,index_to=500,display_log=True):
#     if index_to == -1 :
#         index_to = df_main.shape[0]
#     df_np = df_main["word_combined_all"].apply(np.array).to_numpy()[0:index_to]
#     df_hash = df_main["hash_key"].apply(np.array).to_numpy()[0:index_to]
#     mat_index = []
#     for i in range(index_from,index_to) :
#         if display_log :
#             print(" - Generate NLP for article #"+str(i)+"  (char:"+str(len(df_np[i]))+")")
#         nlp_dict = ts.analyseText2(str(df_np[i]),True)
#         mat_index.append(nlp_dict|{"hash_key":df_hash[i]})
#     df = pd.DataFrame(mat_index, columns = list(mat_index[0].keys())) 
#     df.set_index("hash_key", inplace=True)
#     df_main.set_index("hash_key", inplace=True)
#     display(df)
#     display(df_main)
#     df = df_main.join(df,on='hash_key', how="inner",lsuffix='_k')
#     return df

#print("IMPORT : article_parsing_lib ")



"""OPENAI"""
# def saveNP(data,fmt='%f'): #path,
#     np.savetxt("C:/Users/User/OneDrive/Desktop/article/file_2/test_llm_output/test_save.txt",data, fmt=fmt)

# def loadNP(): #path
#     return np.loadtxt('C:/Users/User/OneDrive/Desktop/article/file_2/test_llm_output/test_save.txt', dtype=float)

# def llmInputConfArticle(article_text,llm_prompt) :
#     context_prompt = "\nHere is the article :\n"
#     final_prompt = str(llm_prompt)+str(context_prompt)+article_text
#     return llmInputConf(final_prompt)


"""TEXT_ANALYSIS"""
            
# test_list = ["""Kyiv, Ukraine
# CNN
#   — 
# Russian forces deported Bohdan Yermokhin from the occupied Ukrainian city of Mariupol in the spring of 2022, flew him to Moscow on a government plane and placed him into a foster family. He was sent to a patriotic camp near the capital where flag-waving staff praised Russian President Vladimir Putin and tried to teach him nationalistic songs.

# The Ukrainian teenager was given a Russian passport and sent to a Russian school. And then, in the fall of 2023, not long before his 18th birthday, he received a summons from a Russian military recruitment office.

# Yermokhin, who’s now back in Ukraine and recovering from his ordeal in Kyiv, told CNN he believed this was the last step in Russia’s attempt to bully him into submission – a bid to sign him up as a soldier to fight against his own people.

# “(I was told that) Ukraine was losing, that children were used for organ donations there, and that I would be sent to war right away. I told them that if I was sent to the war, at least I would fight for my own country, not for them,” he said.

# Yermokhin was part of a group of children known as the “Mariupol 31,” who were taken to Russia. Ukrainian authorities estimate that 20,000 children have been forcibly transported to Russia since Moscow launched its full-scale invasion of the country in February 2022. More than 2,100 children remain missing, according to official statistics, but the government says the real number could be much higher."""]
# # test_list = ["Amazing"]
# ta = text_analysis()
# list_out = []
# for string in test_list :
#     out_dict = ta.analyseArticle(string)
#     list_out.append(out_dict)
#     #print(string,"   ",out_dict)
# df = pd.DataFrame(list_out) # ,columns = list(list_out[0].keys)




# test_list = ['technology', 'file', 'advantages', 'things', 'know', 'right', 'cofounder', 'think', 'competitive', 'aesthetics', 'adobe', 'postscript', 'going', 'wanted', 'john', 'wharton', 'warnock', 'point']
# test_list = ['committee', 'technology', 'services', 'plans', 'president', 'information', 'purdue', 'director', 'streamline', 'vice', 'sustaining', 'student', 'plan', 'synergies']
# test_list = ['nuclear', 'chemical', 'mit', 'gates', 'wins', 'students', 'research', 'substrate', 'engineering', 'chris', 'scholarship', 'boyce', 'work']
# test_list = ['happy hello', 'sad', "work", "lofe","trump"]
# test_list = ['happy hello']
# test_list = ['Fuck you', 'file', 'advantages', 'things', 'know', 'right', 'cofounder', 'think', 'competitive', 'aesthetics', 'adobe', 'postscript', 'going', 'wanted', 'john', 'wharton', 'warnock', 'point']
# test_list = ['youre', 'united', 'sea', 'role', 'safer', 'risk', 'technology', 'dangerous', 'drivers', 'jobs', 'risks']


"""EMBD_KEYWD"""

# def decomposeKeywordList(string,getList=True) :
#     decomposed = "&".join(re.findall("[a-zA-Z]+", string.lower())) #.split(', ')
#     if getList :
#         decomposed = decomposed.split('&')
#     return decomposed
# #     blob = TextBlob(str(string))
# #     word_list = []
# #     for wordT in blob.tags :
# #         if wordT[1] in TAG_SELECTION_LIST :
# #             string2 = str(cleanString(wordT[0].lower()))
# #             if string2 != "" :
# #                 word_list.append(string2)
# #     return word_list


