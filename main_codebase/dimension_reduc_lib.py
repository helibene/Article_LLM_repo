# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:52:42 2024

@author: Alexandre
"""


import main_var
mv = main_var.main_var()


from utils_art import *

from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsTransformer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

import pandas as pd
import numpy as np
from textblob import TextBlob
import tempfile
import time
import datetime
# np.random.seed(0)

from IPython.core.display import Image as image
from PIL import Image
from IPython.display import Image
import plotly.express as px
import plotly
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def matrixToDf(mat,label="def",normalise=True):
    col_count = len(mat[0])
    col_list=[]
    for i in range(col_count) :
        col_list.append(str(i)+"_"+label)
    out_df = pd.DataFrame(mat,columns=col_list)
    if normalise :
        out_df = dfNormalize(out_df)
    return out_df

def dfNormalize(df) :
    return ((df - df.mean()) / df.std())

def dimReduction_TSNE(df, n_components=2, perplexity=30, early_exaggeration =4.0,learning_rate=1000,n_iter=1000,verbose=0,random_state=0,norm_output=False):
    tsne = TSNE(n_components=n_components, perplexity=perplexity,early_exaggeration = early_exaggeration,learning_rate =learning_rate,n_iter=n_iter,verbose=verbose,random_state=random_state)
    out_df = tsne.fit_transform(df)
    out_df = matrixToDf(out_df,"tsne")
    return out_df

def dimReduction_PCA(df, n_components=2, svd_solver="auto",tol=0.0,whiten=False,random_state=0,norm_output=False):
    pca = PCA(n_components=n_components,svd_solver=svd_solver,tol=tol,random_state=random_state)
    out_df = pca.fit_transform(df)
    out_df = matrixToDf(out_df,"pca")
    return out_df

def dimReduction_IPCA(df, n_components=2,whiten=False, batch_size=100,norm_output=False):
    ipca = IncrementalPCA(n_components=n_components,batch_size=batch_size)
    out_df = ipca.fit_transform(df)
    out_df = matrixToDf(out_df,"ipca")
    return out_df

def dimReduction_NNT(df,n_components=2,n_neighbors=5,mode='distance',algorithm='auto',leaf_size=30,p=2,eigen_solver='auto',tol=0.0,metric='minkowski',n_jobs=None,norm_output=False):
    #KTN mode : 'distance' 'connectivity'
    #KTN algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    #BOTH n_jobs : None -1
    #BOTH metric = ['minkowski',  'manhattan','cityblock','l1',  'euclidean','l2',  'cosine',  'haversine',  'nan_euclidean'] 'precomputed' ?
    #ISO eigen_solver : ['auto', 'arpack', 'dense']
    cache_path = tempfile.gettempdir()
    knt = KNeighborsTransformer(mode=mode,n_neighbors=n_neighbors,algorithm=algorithm,leaf_size=leaf_size,metric=metric,p=p,n_jobs=n_jobs)
    iso = Isomap(n_components=n_components,n_neighbors=n_neighbors,eigen_solver=eigen_solver,metric=metric,tol=tol,p=p,n_jobs=n_jobs)# 
    nnt = make_pipeline(knt,iso,memory=cache_path)
    out_df = nnt.fit_transform(df)
    out_df = matrixToDf(out_df,"nnt")
    return out_df

def generateDirRed(mat_emb,df_main,n_components=2,norm_output=True,active_sel=[True,True,True,True],return_list=False,display_stats=True) :
    out_list = []
    out_list_label = []
    if display_stats :
        print("Input dataframe shape :",df_main.shape)
    if active_sel[0] :
        df_tsne = dimReduction_TSNE(mat_emb,n_components,norm_output=norm_output)
        df_main = df_main.join(df_tsne, how="inner", lsuffix='_l',rsuffix='_r')
        # out_list.append(df_tsne_j)
        out_list_label.append("TSNE")
    if active_sel[1] :
        df_pca = dimReduction_PCA(mat_emb,n_components,norm_output=norm_output)
        df_main = df_main.join(df_pca, how="inner", lsuffix='_l',rsuffix='_r')
        # out_list.append(df_pca_j)
        out_list_label.append("PCA")
    if active_sel[2] :
        df_ipca = dimReduction_IPCA(mat_emb,n_components,norm_output=norm_output)
        df_main = df_main.join(df_ipca, how="inner", lsuffix='_l',rsuffix='_r')
        # out_list.append(df_ipca_j)
        out_list_label.append("IPCA")
    if active_sel[3] :
        df_nnt = dimReduction_NNT(mat_emb,n_components,norm_output=norm_output)
        df_main = df_main.join(df_nnt, how="inner", lsuffix='_l',rsuffix='_r')
        # out_list.append(df_nnt_j)
        out_list_label.append("NNT")
    if return_list :
        return out_list, out_list_label
    else :
        return df_main, out_list_label

def generateDimReducedDF(n_components=3, norm_output=True, active_sel=[True,True,True,True]):
    df_embd_input = openDFcsv(mv.embdedding_path,mv.embdedding_filename)
    mat_emb_input = openDFcsv(mv.embdedding_path_raw,mv.embdedding_filename_raw)
    df_embd_input_backup = df_embd_input.copy()
    if "o_data" in list(df_embd_input.columns) :
        df_embd_input.drop(["o_data"], axis=1,inplace=True)
    mat_emb_input.drop(["hash_key"], axis=1,inplace=True)
    df_embd_output, out_list_label = generateDirRed(mat_emb_input, df_embd_input,n_components=n_components, norm_output=norm_output, active_sel=active_sel)
    df_embd_input_backup = deleteUnnamed(df_embd_input_backup,"hash_key")
    df_embd_output = deleteUnnamed(df_embd_output,"hash_key")
    saveDFcsv(df_embd_input_backup, mv.embdedding_path, mv.embdedding_filename+"_before_dim_reduct")
    saveDFcsv(df_embd_output, mv.embdedding_path, mv.embdedding_filename)
    return df_embd_output




#### Caculate stats

def calculateStatsLength(df,groupping,display_df=True):
    #rename_dict = {"text_len":"char_n","sentences":"sentence_n","noun_phrases":"noun_n","words":"words_n"}
    rename_dict = {"tb.char":"char_n","tb.sent":"sentence_n","tb.noun":"noun_n","tb.word":"words_n"}
    df = df.rename(columns=rename_dict)
    list_of_len_fields=list(rename_dict.values())
    # df_group = df[[groupping,"char_n","sentence_n","noun_n","words_n"]].groupby(groupping).sum(["char_n","sentence_n","noun_n","words_n"])
    df_group = df[[groupping]+list_of_len_fields].groupby(groupping).sum(list_of_len_fields)
    df_count = df[groupping].value_counts().to_frame("count")#
    df_main = df_group.join(df_count, how="inner",on=groupping).sort_values(by=['count'],ascending=True)
    df_main[["char_per_count","sentence_per_count","noun_per_count","word_per_count"]] = df_main[list_of_len_fields].div(df_main['count'], axis=0).astype(float)
    df_main[["char_per_sentence","noun_per_sentence","word_per_sentence"]] = df_main[["char_n","noun_n","words_n"]].div(df_main["sentence_n"], axis=0).astype(float)
    df_main[["char_per_word"]] = df_main[["char_n"]].div(df_main["words_n"], axis=0).astype(float)
    df_main = df_main.sort_values(by=["count"],ascending=True)
    if display_df : 
        print("Dataframe Statistics Length Column :'"+groupping+"'")
        display(df_main)
    return df_main

def calculateStatsNLP(df,groupping,display_df=True,display_stats=False,out_raw=False,only_keyword_nlp=False):
    # rename_dict = {"tb.pol":"Polarity","tb.sub":"Subjectivity","pos1":"Positivity","neu1":"Neutrality","neg1":"Negativity","pos2":"Positivity2","neg2":"Negativity2","compound":"Compound"}
    
    if only_keyword_nlp :
        rename_dict = {"tb.polaj_k":"polarity","tb.sub_k":"subjectivity","ts.pos_k":"positivity","ts.neg_k":"negativity"}
    else :
        rename_dict = {"tb.polaj":"polarity","tb.sub":"subjectivity","al.pos":"positivity","al.neg":"negativity"}
    df = df.rename(columns=rename_dict)
    column_list = list(rename_dict.values())
    column_list_ajusted = []
    for field_count in column_list :
        if display_stats :
            print(field_count," ",getStatsFromCol(df,field_count))
        df[field_count+"_aj"] = (df[field_count]-getStatsFromCol(df,field_count)[0])/getStatsFromCol(df,field_count)[2]
        column_list_ajusted.append(field_count+"_aj")
    if out_raw :
        df_main = df
    else :
        column_list = column_list + column_list_ajusted
        df_group = df[[groupping]+column_list].groupby(groupping).sum(column_list)
        df_count = df[groupping].value_counts().to_frame("count")#
        df_main = df_group.join(df_count, how="inner",on=groupping).sort_values(by=['count'],ascending=True)
        list_field_count = []
        for field in column_list :
            list_field_count.append(field+"_per_count")
        df_main[list_field_count] = df_main[column_list].div(df_main['count'], axis=0).astype(float)
        # df_main = df_main.sort_values(by=["count"],ascending=True)
        df_main = df_main.sort_values(by=[groupping],ascending=True)
    if display_df :
        print("Dataframe Statistics NLP Column :'"+groupping+"'")
        display(df_main)
    return df_main

def getStatsFromCol(df, column) :
    min_val = df[column].min()
    max_val = df[column].max()
    return min_val,max_val,max_val-min_val

def calculateStatsColList(df, column_list=[],stat_type="len",display_df=True,display_stats=False,out_raw=False):# ,stat_type="nlp"
    df_list_out = []
    for col in column_list :
        if stat_type=="len":
            df_app = calculateStatsLength(df,col,display_df)
            df_list_out.append(df_app)
        if stat_type=="nlp":
            df_app = calculateStatsNLP(df,col,display_df,display_stats,out_raw)
            df_list_out.append(df_app)
    return df_list_out

print("IMPORT : dimension_reduc ")

def saveStats(col_list=[],length=True,NLP=True) :
    df = openDFcsv(mv.join1_path,mv.join1_filename)
    if col_list == [] :
        col_list = ["category","source_title","year_month","url_TLD","year"]
    if length :
        df_list_len = calculateStatsColList(df,col_list,"len",display_df=True)
        count=0
        for col in col_list :
            saveDFcsv(df_list_len[count],mv.join1_path,mv.join1_filename+"_len_"+str(col))
            count = count + 1
    if NLP :
        df_list_npl = calculateStatsColList(df,col_list,"nlp",display_df=True,display_stats=False) 
        count=0
        for col in col_list :
            saveDFcsv(df_list_npl[count],mv.join1_path,mv.join1_filename+"_nlp_"+str(col))
            count = count + 1
# saveStats()
# df = generateDimReducedDF(n_components=3, norm_output=True, active_sel=[True,True,True,True])
# print(df)