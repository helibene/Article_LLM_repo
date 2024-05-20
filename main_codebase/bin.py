# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:05:32 2024

@author: Alexandre
"""


# def loadFromFolder(folder_path="",force_schema=[],save=False, unicity_key="hash_key",cast_date_col="published"):
#     if folder_path == "" :
#         folder_path = main_path
#     root_path = os.Path(folder_path)
#     file_list = os.listdir(root_path)
#     first_flag = True
#     for filename_entry in file_list:
#         if ".csv" in filename_entry :
#             df = openDFcsv(folder_path,filename_entry.replace(".csv",""))
#             if first_flag :
#                 first_flag = False
#                 df_out = df
#             else :
#                 df_out = pd.concat([df_out, df], ignore_index=True)
#     if "Unnamed: 0" in list(df_out.columns) :
#         df_out = df_out.rename(columns={"Unnamed: 0":"index"})
#     if force_schema!=[] :
#         df_out = df_out[force_schema]
#     if unicity_key != "" and unicity_key in list(df_out.columns) :
#         df_out = df_out.drop_duplicates(subset=['hash_key'])
#     if cast_date_col != "" and cast_date_col in list(df_out.columns) :
#         df_out[cast_date_col+"_date_type"] = pd.to_datetime(df_out[cast_date_col])
#     if save :
#         df_out = df_out.set_index('hash_key')
#         saveDFcsv(df_out, folder_path, filename+"_all_aggregated")
#     return df_out


# #M1
# df = loadFromFolder(path_union_stat) #
# col_list = ["category","source_title","year_month","url_TLD","year"]
# df_list = calculateStatsColList(df,col_list,"len",display_df=True)
# df_list = calculateStatsColList(df,col_list,"nlp",display_df=True,display_stats=False) #,out_raw=False

# #M2
# df_pol = df[["subjectivity"]].round(2).value_counts().to_frame("count").sort_values(by=['subjectivity'],ascending=True)
# df_pol.plot.barh(y='count',use_index=True, rot=0, figsize=(7, 30), title="Words Cnt") # , logx=True

# #M3
# source_list = ["The New York Times","The Nation","Business Insider","Business Insider"]
# out_df = selectOnDf(df,source_list=source_list)
# display(out_df)

# #M4
# plotDFstatisticsQuerry(df)

# #M5
# displayStats(df)
# #saveDFcsv(out_df, "C:/Users/User/OneDrive/Desktop/article/file_2/.bin/", "viz_test3")

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

# def mainJoinOut() :
#     df = openDFcsv(keyword_path,keyword_filename)
#     df_out = ekml.generateNLPonKeywords(df,0,df.shape[0],True)
#     # df_out = deleteUnnamed(df_out,"hash_key")
#     saveDFcsv(df_out, join2_path,join2_filename)
#     return df_out

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

# 'title'
# 'title_detail'
#     'type'
#     'language'
#     'base'
#     'value'
# 'link'
# 'links' (list)
#     'rel'
#     'type'
#     'href'
# 'id'
# 'guidislink'
# 'published'
# 'published_parsed'
# 'summary'
# 'summary_detail'
#     'type'
#     'language'
#     'base'
#     'value'
# 'source'
#     'href'
#     'title'
# 'sub_articles' (list)


# main_path = mv.query_path  # "C:/Users/User/OneDrive/Desktop/article/files_3/1_1_query_main/"+env
# filename = mv.query_filename  #  "reduction_test"

# scarp_path = mv.scarp_path  #  "C:/Users/User/OneDrive/Desktop/article/files_3/1_2_scarp_main/"+env
# scarp_filename = mv.scarp_filename

# join1_path = mv.join1_path
# join1_filename = mv.join1_filename

# embdedding_path=mv.embdedding_path
# embdedding_filename=mv.embdedding_filename

# embdedding_path_raw = mv.embdedding_path_raw
# embdedding_filename_raw = mv.embdedding_filename_raw

# keyword_path = mv.keyword_path
# keyword_filename = mv.keyword_filename

# join2_path = mv.join2_path
# join2_filename = mv.join2_filename



# path_union_stat = join1_path"C:/Users/User/OneDrive/Desktop/article/files_3/1_4_join_main/"+env
# filename_union_stat = join1_filename"join_main_test-filter"

# llm_result = "C:/Users/User/OneDrive/Desktop/article/file_2/test_llm_output/new_embedding/"

# llm_join_out = "C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/"
# llm_join_out_filename = "article_stats_embedding"


# df = getStandardDf(_STANDARD_SCRAPPING_FIELDS)
# df2 = fullWorkflow(gn,df,True)
# display(df2)

# def calculateStatsNLP(df,groupping,display_data=True,display_stats=True,out_raw=False,only_keyword_nlp=False,sort_by_groupping=True):
#     # rename_dict = {"tb.pol":"Polarity","tb.sub":"Subjectivity","pos1":"Positivity","neu1":"Neutrality","neg1":"Negativity","pos2":"Positivity2","neg2":"Negativity2","compound":"Compound"}
#     # Ajusting NLP values (0 to 1)
#     nlp_col_list = ["tb.pol","tb.sub","tb.pos","tb.neg","vs.pos","vs.neu","vs.neg","vs.comp","ts.pos","ts.neu","ts.neg","al.pos","al.neg","tb.pol_k","tb.sub_k","tb.polaj","tb.pos_k","tb.neg_k","vs.pos_k","vs.neu_k","vs.neg_k","vs.comp_k","ts.neg_k","ts.neu_k","ts.pos_k","al.pos_k","al.neg_k","0_tsne","1_tsne","2_tsne","0_pca","1_pca","2_pca","0_ipca","1_ipca","2_ipca","0_tnn","1_tnn","2_tnn"]

#     input_col_list = list(df.columns)
#     column_list_ajusted = []
#     nlp_col_list_val = []
#     for nlp_col in nlp_col_list :
#         if nlp_col in input_col_list :
#             if display_stats :
#                 print(nlp_col," ",getStatsFromCol(df,nlp_col))
#             df[nlp_col+"_aju"] = (df[nlp_col]-getStatsFromCol(df,nlp_col)[0])/getStatsFromCol(df,nlp_col)[2]
#             column_list_ajusted.append(nlp_col+"_aju")
#             nlp_col_list_val.append(nlp_col)
    
#     if out_raw :
#         df_main = df
#     else :
#         column_list = nlp_col_list_val + column_list_ajusted
#         df_group = df[[groupping]+column_list].groupby(groupping).sum(column_list)
#         df_count = df[groupping].value_counts().to_frame("count")#
#         display_df(df_count)
#         display_df(df_group)
#         df_group=df_group.sort_values(by=[groupping],ascending=True)
#         df_count=df_count.sort_values(by=[groupping],ascending=True)
#         df_main = df_group.join(df_count, how="inner",on=groupping).sort_values(by=['count'],ascending=True)
#         df_main=df_main.sort_values(by=[groupping],ascending=True)
#         list_field_count = []
#         for field in column_list :
#             list_field_count.append(field+"_avg")
#         df_main = df_main[column_list].div(df_main['count'], axis=0).astype(float)#[list_field_count]
#         df_main=df_main.sort_values(by=[groupping],ascending=True)
#         df_main = df_main.join(df_count, how="inner",on=groupping,lsuffix="_old",rsuffix='_new')
#         if sort_by_groupping :
#             df_main = df_main.sort_values(by=[groupping],ascending=True)
#         else:
#             df_main = df_main.sort_values(by=["count"],ascending=True)
#     for col in column_list_ajusted :
#         df_main[col] = (df_main[col]-getStatsFromCol(df_main,col)[0])/getStatsFromCol(df_main,col)[2]
#     if display_data :
#         print("Dataframe Statistics NLP Column :'"+groupping+"'")
#         display_df(df_main)
#     return df_main
# def calculateStatsLength(df,groupping,display_data=True):
#     #rename_dict = {"text_len":"char_n","sentences":"sentence_n","noun_phrases":"noun_n","words":"words_n"}
#     rename_dict = {"tb.char":"char_n","tb.sent":"sentence_n","tb.noun":"noun_n","tb.word":"words_n"}
#     df = df.rename(columns=rename_dict)
#     list_of_len_fields=list(rename_dict.values())
#     # df_group = df[[groupping,"char_n","sentence_n","noun_n","words_n"]].groupby(groupping).sum(["char_n","sentence_n","noun_n","words_n"])
#     #                                        df_group = df[[groupping]+list_of_len_fields].groupby(groupping).sum(list_of_len_fields)
#     df_group = df[groupping+list_of_len_fields].groupby(groupping).sum(list_of_len_fields)
#     df_count = df[groupping].value_counts().to_frame("count")#
#     df_main = df_group.join(df_count, how="inner",on=groupping)                 #.sort_values(by=['count'],ascending=True)
#     df_main[["char_per_count","sentence_per_count","noun_per_count","word_per_count"]] = df_main[list_of_len_fields].div(df_main['count'], axis=0).astype(float)
#     df_main[["char_per_sentence","noun_per_sentence","word_per_sentence"]] = df_main[["char_n","noun_n","words_n"]].div(df_main["sentence_n"], axis=0).astype(float)
#     df_main[["char_per_word"]] = df_main[["char_n"]].div(df_main["words_n"], axis=0).astype(float)
#     #                        df_main = df_main.sort_values(by=["count"],ascending=True)
#     df_main = df_main.sort_values(by=[groupping],ascending=True)
#     df_main = df_main.reset_index()
#     if display_data : 
#         print("Dataframe Statistics Length Column :'"+str(groupping)+"'")
#         display_df(df_main)
#     return df_main

# def calculateStatsLength(df,groupping,display_df=True):
#     #rename_dict = {"text_len":"char_n","sentences":"sentence_n","noun_phrases":"noun_n","words":"words_n"}
#     rename_dict = {"tb.char":"char_n","tb.sent":"sentence_n","tb.noun":"noun_n","tb.word":"words_n"}
#     df = df.rename(columns=rename_dict)
#     list_of_len_fields=list(rename_dict.values())
#     # df_group = df[[groupping,"char_n","sentence_n","noun_n","words_n"]].groupby(groupping).sum(["char_n","sentence_n","noun_n","words_n"])
#     df_group = df[[groupping]+list_of_len_fields].groupby(groupping).sum(list_of_len_fields)
#     df_count = df[groupping].value_counts().to_frame("count")#
#     df_main = df_group.join(df_count, how="inner",on=groupping).sort_values(by=['count'],ascending=True)
#     df_main[["char_per_count","sentence_per_count","noun_per_count","word_per_count"]] = df_main[list_of_len_fields].div(df_main['count'], axis=0).astype(float)
#     df_main[["char_per_sentence","noun_per_sentence","word_per_sentence"]] = df_main[["char_n","noun_n","words_n"]].div(df_main["sentence_n"], axis=0).astype(float)
#     df_main[["char_per_word"]] = df_main[["char_n"]].div(df_main["words_n"], axis=0).astype(float)
#     df_main = df_main.sort_values(by=["count"],ascending=True)
#     if display_df :
#         print("Dataframe Statistics Length Column :'"+groupping+"'")
#         display_df(df_main)
#     return df_main

# def calculateStatsNLP(df,groupping,display_df=True,display_stats=False,out_raw=False):
#     # rename_dict = {"tb.pol":"Polarity","tb.sub":"Subjectivity","pos1":"Positivity","neu1":"Neutrality","neg1":"Negativity","pos2":"Positivity2","neg2":"Negativity2","compound":"Compound"}
#     rename_dict = {"tb.polaj":"polarity","tb.sub":"subjectivity","al.pos":"positivity","al.neg":"negativity"}
#     df = df.rename(columns=rename_dict)
#     column_list = list(rename_dict.values())
#     column_list_ajusted = []
#     for field_count in column_list :
#         if display_stats :
#             print(field_count," ",getStatsFromCol(df,field_count))
#         df[field_count+"_aj"] = (df[field_count]-getStatsFromCol(df,field_count)[0])/getStatsFromCol(df,field_count)[2]
#         column_list_ajusted.append(field_count+"_aj")
#     if out_raw :
#         df_main = df
#     else :
#         column_list = column_list + column_list_ajusted
#         df_group = df[[groupping]+column_list].groupby(groupping).sum(column_list)
#         df_count = df[groupping].value_counts().to_frame("count")#
#         df_main = df_group.join(df_count, how="inner",on=groupping).sort_values(by=['count'],ascending=True)
#         list_field_count = []
#         for field in column_list :
#             list_field_count.append(field+"_per_count")
#         df_main[list_field_count] = df_main[column_list].div(df_main['count'], axis=0).astype(float)
#         # df_main = df_main.sort_values(by=["count"],ascending=True)
#         df_main = df_main.sort_values(by=[groupping],ascending=True)
#     if display_df :
#         print("Dataframe Statistics NLP Column :'"+groupping+"'")
#         display_df(df_main)
#     return df_main

# def getStatsFromCol(df, column) :
#     min_val = df[column].min()
#     max_val = df[column].max()
#     return min_val,max_val,max_val-min_val

# def calculateStatsColList(df, column_list=[],stat_type="len",display_df=True,display_stats=False,out_raw=False):# ,stat_type="nlp"
#     df_list_out = []
#     for col in column_list :
#         if stat_type=="len":
#             df_app = calculateStatsLength(df,col,display_df)
#             df_list_out.append(df_app)
#         if stat_type=="nlp":
#             df_app = calculateStatsNLP(df,col,display_df,display_stats,out_raw)
#             df_list_out.append(df_app)
#     return df_list_out

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

#open_path = mv.query_path  #  "C:/Users/User/OneDrive/Desktop/article/files_3/1_1_query_main/"+env
#filename_input = mv.query_filename # "query_main_final"#"query_large_2010_to_2023"#"query_main_final"

# save_path = mv.scarp_path  # "C:/Users/User/OneDrive/Desktop/article/files_3/1_2_scarp_main/"+env #parssing_df_main/
#filename_out = mv.scarp_filename

#save_path_article = mv.article_path   #   "C:/Users/User/OneDrive/Desktop/article/files_3/1_3_article_main/"+env #article_download_main_2
# output_fields = ["url", "pk", "hash_key", "title", "authors", "publish_date", "keywords_list", "text_len","valid"]# + ["tb.sent", "tb.noun", "tb.word", "tb.char", "tb.pol", "tb.sub", "tb.polaj", "tb.pos", "tb.neg", "vs.pos", "vs.neu", "vs.neg","vs.comp","ts.pos","ts.neg"], "summary"



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

    # def analyseText(self, text) :
    #     dict_out = {}
    #     textblob = TextBlob(text)
    #     textblob_nba = TextBlob(text, analyzer=self.tb_NaiveBayes)
    #     textblob_pa = TextBlob(text, analyzer=self.tb_PatternAnalyzer)
        
    #     vs = self.vs_analyzer.polarity_scores(text)
    #     vs = dict(vs)
        
    #     sp = self.tf_sentiment_pipeline(text)
    #     #print(textblob.sentences)
    #     dict_out["tb.sentences"] = len(textblob.sentences)
    #     dict_out["tb.noun_phrases"] = len(textblob.noun_phrases)
    #     dict_out["tb.words"] = len(textblob.words)
        
    #     tb_pola = textblob_pa.sentiment.polarity
    #     tb_sub = textblob_pa.sentiment.subjectivity
    #     dict_out["tb.polarity"] = tb_pola # [-1.0, 1.0].
    #     dict_out["tb.subjectivity"] = tb_sub #[0.0, 1.0]
    #     dict_out["tb.subjectivity_aj"] = (tb_pola+1)/2 #[0.0, 1.0]
    #     if tb_sub == 0 :
    #         dict_out["tb.pol_div_sub"] = 1
    #     else :
    #         dict_out["tb.pol_div_sub"] = ((tb_pola+1)/2)/tb_sub #[0.0, 1.0]
    #     if tb_pola == -1 :
    #         dict_out["tb.sub_div_pol"] = 1
    #     else :
    #         dict_out["tb.sub_div_pol"] = tb_sub/((tb_pola+1)/2) #[0.0, 1.0]
        
    #     tp_pos = textblob_nba.sentiment.p_pos
    #     tp_neg = textblob_nba.sentiment.p_neg
    #     dict_out["tb.p_pos"] = tp_pos/(tp_pos+tp_neg)
    #     dict_out["tb.p_neg"] = tp_neg/(tp_pos+tp_neg)
    #     dict_out["tb.p_class"] = bool(tp_neg<tp_pos)
    
        
    #     dict_out["vs.neg"] = vs["neg"]
    #     dict_out["vs.neu"] = vs["neu"]
    #     dict_out["vs.pos"] = vs["pos"]
    #     dict_out["vs.compound"] = vs["compound"]
    #     dict_out["vs.p_class"] = bool(vs["neg"]<vs["pos"])
        
    #     dict_out["ts.neg"] = sp[0][0]['score']
    #     dict_out["ts.pos"] = sp[0][1]['score']
    #     dict_out["ts.class"] = bool(dict_out["ts.neg"] <dict_out["ts.pos"])
    #     return dict_out
            # print(len(text))
            # sp = self.tf_sentiment_pipeline(text)
            # dict_out["ts.neg"] = sp[0][0]['score']
            # dict_out["ts.pos"] = sp[0][1]['score']
            
    # def analyseArticle(self, text) :
    #     disp = True
    #     # article_status = ""
    #     textblob = TextBlob(text)
    #     sentences_list = textblob.sentences
    #     dict_list = []
    #     len_list = []
    #     sent_len = 0
    #     sent_count = 0
    #     if sent_count < MAX_NUM_SENTENCES : #and sent_count > MIN_NUM_SENTENCES
    #         #happends
    #         for sentence in sentences_list :
    #             sent_len = len(sentence.words)
    #             if sent_len < MAX_SENTENCE_LEN and sent_len > MIN_SENTENCE_LEN :
    #                 #happends
    #                 big_word_valid = True
    #                 for w in sentence.words :
    #                     if len(w) > MAX_SENTENCE_CHAR_LEN :
    #                         big_word_valid = False
    #                 if big_word_valid : #
    #                     len_list.append(sent_len)
    #                     new_dict = self.analyseText2(str(sentence),True)
    #                     if "tb.class" in new_dict.keys() :
    #                         del new_dict["tb.class"]
    #                         del new_dict["vs.class"]
    #                         del new_dict["ts.class"]
    #                     dict_list.append(new_dict) # | self.lenStats(text)
    #                     sent_count = sent_count + 1
    #             else :
    #                 pass
    #                 # print("lv2_"+str(sent_len))
    #     else :
    #         pass
    #         # print("lv1_"+str(sent_len))
    #     #     # lenStats()
    #     #     return self.subpolStats(text)
    #     # else:
    #     #     for sentence in sentences_list :
    #     #         sent_len = len(sentence.words)
    #     #         if sent_len < MAX_SENTENCE_LEN and sent_len > MIN_SENTENCE_LEN :
    #     #             big_word_valid=True
    #     #             # for w in sentence.words :
    #     #                 # if len(w) > MAX_SENTENCE_CHAR_LEN :
    #     #                 #     big_word_valid = False
    #     #                     # print("big_word_valid FALSE")
    #     #                     #print(w)
                    
    #     #         else :
    #     #             pass
    #     #             # if disp :
    #     #             #     print("MAX_SENTENCE_LEN_or_MIN_SENTENCE_LEN WARNNING")
    #     #     else :
    #     #         pass
    #     #         # if disp :
    #     #         #     print("LOW_NUM_SENTENCES_or_MAX_NUM_SENTENCES ERROR")
    #     # print(100*sent_count/len(sentences_list))
    #     out_dict = self.weigthAverage(dict_list,len_list)
    #     return out_dict | self.lenStats(text)
    
    
    #     # if len(sentences_list)<MIN_NUM_SENTENCES :
    #     #     article_status = article_status + "_LOW_NUM_SENTENCES_"
    #     # if len(sentences_list)>MAX_NUM_SENTENCES :
    #     #     article_status = article_status + "_HIGH_NUM_SENTENCES_"
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
# df_keyword = mainKeywordWF(entry_limit=1000,
#               common_word_max=700,
#               add_nlp_stats=True,
#               nlp_source_col="word_combined_all")
    #df["word_combined_f"].replace(" @","")
    #df["word_combined_s"].replace(" @","")
    #df['word_combined_s'] = df['word_combined_s'].str.replace(' @','')
    #df['word_combined_f'] = df['word_combined_s'].str.replace(' @','')
    
    
    # def cleanString(string) :
    #     #~out_str = string.strip("][”“|’><%—–//").replace("'", "").replace("\\d", "");
    #     out_str = "".join(re.findall("[a-zA-Z]+", string))
    #     if len(out_str)>3:
    #         return out_str
    #     else :
    #         return ""

    # def decomposeTitle(string, getList=True) :
    #     decomposed = "=".join(re.findall("[a-zA-Z]+", string.lower()))
    #     if getList :
    #         decomposed = decomposed.split('=')
    #         # if type(decomposed) != type(None) :
    #         #     for i in range(len(decomposed)) :
    #         #         if len(decomposed[i])<3:
    #         #             decomposed = decomposed.remove(decomposed[i])
    #     return decomposed

    # def decomposeKeywordList(string,getList=True) :
    #     decomposed = "&".join(re.findall("[a-zA-Z]+", string.lower())) #.split(', ')
    #     if getList :
    #         decomposed = decomposed.split('&')
    #         # if type(decomposed) != type(None) :
    #         #     for i in range(len(decomposed)) :
    #         #         if len(decomposed[i])<3:
    #         #             decomposed = decomposed.remove(decomposed[i])
    #     return decomposed
"""VISU"""
# def addVisuColumns(df) :
#     if "url_TLD" in df.columns :
#         df["bool_url"] = np.where(df['url_TLD'] == "com", True, False)
#         #df["bool_url"] = df["url_TLD"] == "com"
#     df["bool_tb_pos"] = np.where(df['tb.pos'] > df['tb.neg'], "y", "n")
#     df["bool_vs_pos"] = np.where(df['vs.pos'] > df['vs.neg'], "y", "n")
#     df["bool_ts_pos"] = np.where(df['ts.pos'] > df['ts.neg'], "y", "n")
#     df["bool_vs_neu"] = np.where(df['vs.neu'] > 0, "y", "n")
#     df["bool_vs_comp"] = np.where(df['vs.comp'] > 0, "y", "n")
#     df['bool_sent_all'] = df[["bool_tb_pos","bool_vs_pos","bool_ts_pos","bool_vs_neu","bool_vs_comp"]].agg(''.join, axis=1)
#     return df



# def calculateRatio(n1,n2,stringReturn=True,round_num=2) :
#     our_num = round(float(n1)/max(float(n2),1),round_num)
#     if stringReturn :
#         if n2 != 0 :
#             return str(our_num)
#         else :
#             "error_div_0"
#     else :
#         return float(our_num)
     

# def calculatePct(n1,n2,stringReturn=True,round_num=2,ajust_for_denom=0) :
#     our_num = round((100*(-(float(n2)*ajust_for_denom)+float(n1)))/max(float(n2),1),round_num)
#     if stringReturn :
#         if n2 != 0 :
#             return str(our_num)
#         else :
#             "error_div_0"
#     else :
#         return float(our_num)
    

# def joinQuerryAndParse(save=True,remove_invalid=True,display=True,filtered_input_df=False) :
#     # rename_dict = {"title_q":"title_quer","title_p":"title_par","published_q":"published","year_q":"year","year_month_q":"year_month","source_url_q":"source_url","url_list_q":"url_list","url_TLD_q":"url_TLD","source_title_q":"source_title","category_q":"category","authors":"authors","keywords_list":"keywords_list","text_len_p":"text_len","tb.sentences":"tb.sentences","tb.noun_phrases":"tb.noun_phrases","tb.words":"tb.words","tb.polarity":"tb.polarity","tb.subjectivity":"tb.subjectivity","tb.p_pos":"tb.p_pos","tb.p_neg":"tb.p_neg","vs.neg":"vs.neg","vs.neu":"vs.neu","vs.pos":"vs.pos","vs.compound":"vs.compound","valid":"valid","link_q":"link","pk_q":"pk",}
#     # rename_dict = {"title_q":"title_quer","title_p":"title_par","published_q":"published","year_q":"year","year_month_q":"year_month","source_url_q":"source_url","url_list_q":"url_list","url_TLD_q":"url_TLD","source_title_q":"source_title","category_q":"category","authors":"authors","keywords_list":"keywords_list","text_len_p":"text_len","tb.sentences":"sentences","tb.noun_phrases":"noun_phrases","tb.words":"words","tb.polarity":"polarity","tb.subjectivity":"subjectivity","tb.p_pos":"tb.pos","tb.p_neg":"tb.neg","vs.neg":"vs.neg","vs.neu":"vs.neu","vs.pos":"vs.pos","vs.compound":"vs.comp","valid":"valid","link_q":"link","pk_q":"pk"}
#     #rename_dict = {"pk_q":"pk"}
#     rename_dict = {"title_q":"title_quer","title_p":"title_par","pk_q":"pk"}
#     del_col_list = ["pk_p"]
#     df_q = openDFcsv(main_path,filename)
#     df_q = deleteUnnamed(df_q,"hash_key")
#     df_q_len = df_q.shape[0]
#     if display :
#         print("QUERRY dataset loaded from ",main_path)
#         print("QUERRY dataset has entry length of :",df_q_len,"\n")
#     df_p = openDFcsv(scarp_path,scarp_filename)
#     df_p = deleteUnnamed(df_p,"hash_key")
#     df_p_len = df_p.shape[0]
#     if display :
#         print("PARSSING dataset loaded from ",scarp_path)
#         print("PARSSING dataset has entry length of :",df_p_len," ("+calculatePct(df_p_len,df_q_len)+"% of querry data)\n")
#     df = df_q.join(df_p, how="inner",on='hash_key', lsuffix='_q', rsuffix='_p')
#     df = df.rename(columns=rename_dict)
#     df = df.drop_duplicates(subset=['pk'])
#     for col in del_col_list :
#         if col in list(df.columns) :
#             del df[col]
#     # df = df.set_index('hash_key')
#     # df = df[list(rename_dict.values())]
#     join_df_len = df.shape[0]
#     if display :
#         print("JOINED dataset has entry length of :",join_df_len," ("+calculatePct(join_df_len,df_p_len)+"% of parssing data)")
#     if remove_invalid :
#         df = df.loc[(df['valid'] == True)]
#         join_df_valid_len = df.shape[0]
#         if display :
#             print("JOINED dataset VALID entries :",join_df_valid_len," ("+calculatePct(join_df_valid_len,join_df_len)+"% of joined data)")
#             print("JOINED dataset INVALID entries :",join_df_len-join_df_valid_len," ("+calculatePct(join_df_valid_len,join_df_len,ajust_for_denom=1)+"% of joined data)\n")
#         join_df_len = join_df_valid_len
#     if display :
#         print("TOTAL yield : from",df_q_len," to ",join_df_len,"("+calculatePct(join_df_len,df_q_len)+"% yeald)\n")
#     if save :
#         # df = deleteUnnamed(df,"hash_key")
#         saveDFcsv(df,join1_path,join1_filename)
#         if display :
#             print("JOINED dataset saved here :",join1_path+join1_filename+".csv")
#     return df


# from article_scraping_lib import *
# from utils_art import *
# import article_scraping_lib
# import pandas as pd
# from textblob import TextBlob
# test_text = "This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (initializing a BertForSequenceClassification model from a BertForPreTraining model). This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
# textblob = TextBlob(test_text)
# sentences_list = textblob.sentences
# df = getStandardDf(article_scraping_lib._STANDARD_SCRAPPING_FIELDS)
# df2, test = fullWorkflow(gn,df,True)
# display(df2)
# print(test)
# print(df2.dtypes)
# print(len(sentences_list))
# print(sentences_list)
# Members = {"John": "Male", "Kat": "Female", "Doe": "Female","Clinton": "Male"}

# del Members["lol"]
# print(Members)
# print(len(sentences_list[0].words))

    # base = {"x":1,"y":2,"z":3,"c":"category","size":"words","hover_name":"source_title","custom_data":["title_quer"],
    #                 "custom_data":None,"browser":True,"facet_row":None,"facet_col":None,"facet_row_spacing":0.02,"facet_col_spacing":0.02,
    #                 "animation_frame":None,"animation_group":None,"marginal_x":MARGINAL_LIST[0],"marginal_y":MARGINAL_LIST[0],
    #                 "log_x":False,"log_y":False,"log_z":False,"render_mode":PLOT_RENDER,"size_max":30,"opacity":0.85}

    # base = {"x":1,"y":2,"z":3,"c":"category","size":"tb.word","hover_name":"source_title","custom_data":["title_quer"], #"custom_data":None,
    #                 "browser":browser,"facet_row":None,"facet_col":None,"facet_row_spacing":0.02,"facet_col_spacing":0.02,
    #                 "animation_frame":None,"animation_group":None,"marginal_x":MARGINAL_LIST[0],"marginal_y":MARGINAL_LIST[0],
    #                 "log_x":False,"log_y":False,"log_z":False,"render_mode":PLOT_RENDER,"size_max":30,"opacity":0.85}
    
"""AGREFATION_MODULE"""
    # rename_dict = {"title_q":"title_quer","title_p":"title_par","published_q":"published","year_q":"year","year_month_q":"year_month","source_url_q":"source_url","url_list_q":"url_list","url_TLD_q":"url_TLD","source_title_q":"source_title","category_q":"category","authors":"authors","keywords_list":"keywords_list","text_len_p":"text_len","tb.sentences":"tb.sentences","tb.noun_phrases":"tb.noun_phrases","tb.words":"tb.words","tb.polarity":"tb.polarity","tb.subjectivity":"tb.subjectivity","tb.p_pos":"tb.p_pos","tb.p_neg":"tb.p_neg","vs.neg":"vs.neg","vs.neu":"vs.neu","vs.pos":"vs.pos","vs.compound":"vs.compound","valid":"valid","link_q":"link","pk_q":"pk",}
    # rename_dict = {"title_q":"title_quer","title_p":"title_par","published_q":"published","year_q":"year","year_month_q":"year_month","source_url_q":"source_url","url_list_q":"url_list","url_TLD_q":"url_TLD","source_title_q":"source_title","category_q":"category","authors":"authors","keywords_list":"keywords_list","text_len_p":"text_len","tb.sentences":"sentences","tb.noun_phrases":"noun_phrases","tb.words":"words","tb.polarity":"polarity","tb.subjectivity":"subjectivity","tb.p_pos":"tb.pos","tb.p_neg":"tb.neg","vs.neg":"vs.neg","vs.neu":"vs.neu","vs.pos":"vs.pos","vs.compound":"vs.comp","valid":"valid","link_q":"link","pk_q":"pk"}
    #rename_dict = {"pk_q":"pk"}
        
    
"""DIM_RED"""

# def testDimReduct(df,i) :
#     # df = dfNormalize(df)
#     df_tsne = dimReduction_TSNE(df)
#     plot2Dpandas(df_tsne,title="TSNE (T-distributed Stochastic Neighbor Embedding)",save=True,path=folder_path_graph+"TSNE_",savecount=i)
    
#     df_pca = dimReduction_PCA(df)
#     plot2Dpandas(df_pca,title="PCA (Principal Component Analysis)",save=True,path=folder_path_graph+"PCA_",savecount=i)
    
#     df_ipca = dimReduction_IPCA(df)
#     plot2Dpandas(df_ipca,title="IPCA (Incremental Principal Component Analysis)",save=True,path=folder_path_graph+"IPCA_",savecount=i)
    
#     df_nnt = dimReduction_NNT(df)
#     plot2Dpandas(df_nnt,title="NNT (Nearest Neighbors Transformer)",save=True,path=folder_path_graph+"IPCA_",savecount=i)

# def calculateStatsNLP(df,groupping,display_data=True,display_stats=False,out_raw=False,only_keyword_nlp=False):
#     # rename_dict = {"tb.pol":"Polarity","tb.sub":"Subjectivity","pos1":"Positivity","neu1":"Neutrality","neg1":"Negativity","pos2":"Positivity2","neg2":"Negativity2","compound":"Compound"}
    
#     if only_keyword_nlp :
#         rename_dict = {"tb.polaj_k":"polarity","tb.sub_k":"subjectivity","ts.pos_k":"positivity","ts.neg_k":"negativity"}
#     else :
#         rename_dict = {"tb.polaj":"polarity","tb.sub":"subjectivity","al.pos":"positivity","al.neg":"negativity"}
#     df = df.rename(columns=rename_dict)
#     column_list = list(rename_dict.values())
#     column_list_ajusted = []
#     for field_count in column_list :
#         if display_stats :
#             print(field_count," ",getStatsFromCol(df,field_count))
#         df[field_count+"_aj"] = (df[field_count]-getStatsFromCol(df,field_count)[0])/getStatsFromCol(df,field_count)[2]
#         column_list_ajusted.append(field_count+"_aj")
#     if out_raw :
#         df_main = df
#     else :
#         column_list = column_list + column_list_ajusted
#         df_group = df[[groupping]+column_list].groupby(groupping).sum(column_list)
#         df_count = df[groupping].value_counts().to_frame("count")#
#         df_main = df_group.join(df_count, how="inner",on=groupping).sort_values(by=['count'],ascending=True)
#         list_field_count = []
#         for field in column_list :
#             list_field_count.append(field+"_per_count")
#         df_main[list_field_count] = df_main[column_list].div(df_main['count'], axis=0).astype(float)
#         # df_main = df_main.sort_values(by=["count"],ascending=True)
#         df_main = df_main.sort_values(by=[groupping],ascending=True)
#     if display_data :
#         print("Dataframe Statistics NLP Column :'"+groupping+"'")
#         display_df(df_main)
#     return df_main
