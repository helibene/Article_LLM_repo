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

def generateDirRed(mat_emb,df_main,n_components=2,norm_output=True,active_sel=[True,False,False,False],return_list=False,display_stats=True) :
    out_list = []
    out_list_label = []
    if display_stats :
        print("Input dataframe shape :",df_main.shape)
    if active_sel[0] :
        df_tsne = dimReduction_TSNE(mat_emb,n_components,norm_output=norm_output)
        df_main = df_main.join(df_tsne, how="inner")
        # out_list.append(df_tsne_j)
        out_list_label.append("TSNE")
    if active_sel[1] :
        df_pca = dimReduction_PCA(mat_emb,n_components,norm_output=norm_output)
        df_main = df_main.join(df_pca, how="inner")
        # out_list.append(df_pca_j)
        out_list_label.append("PCA")
    if active_sel[2] :
        df_ipca = dimReduction_IPCA(mat_emb,n_components,norm_output=norm_output)
        df_main = df_main.join(df_ipca, how="inner")
        # out_list.append(df_ipca_j)
        out_list_label.append("IPCA")
    if active_sel[3] :
        df_nnt = dimReduction_NNT(mat_emb,n_components,norm_output=norm_output)
        df_main = df_main.join(df_nnt, how="inner")
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

print("IMPORT : dimension_reduc ")

# df = generateDimReducedDF()
# print(df)