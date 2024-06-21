# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:52:42 2024

@author: Alexandre
"""


import main_var
mv = main_var.main_var()
from utils_art import openDFcsv, saveDFcsv, display_df, deleteUnnamed
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsTransformer
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import tempfile
np.random.seed(0)
import warnings
warnings.filterwarnings('ignore')
import copy
# from sklearn.datasets import make_classification
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.datasets import make_regression
PCA_TYPE_LIST = ["auto", "full", "arpack", "randomized"]
NNT_MODE_LIST = ['distance', 'connectivity']
NNT_ALGO_LIST = ['auto', 'ball_tree', 'kd_tree', 'brute']
NNT_METRIC_LIST = ['minkowski',  'manhattan','cityblock','l1',  'euclidean','l2',  'cosine',  'haversine',  'nan_euclidean']# 'precomputed' ?
NNT_ISO_EIGEN_LIST = ['auto', 'arpack', 'dense']

"""Embedding reduction"""
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

def dimReduction_PCA(df, n_components=2, svd_solver=PCA_TYPE_LIST[0],tol=0.0,whiten=False,random_state=0,norm_output=False):
    pca = PCA(n_components=n_components,svd_solver=svd_solver,tol=tol,random_state=random_state)
    out_df = pca.fit_transform(df)
    out_df = matrixToDf(out_df,"pca")
    return out_df

def dimReduction_IPCA(df, n_components=2,whiten=False, batch_size=100,norm_output=False):
    ipca = IncrementalPCA(n_components=n_components,batch_size=batch_size)
    out_df = ipca.fit_transform(df)
    out_df = matrixToDf(out_df,"ipca")
    return out_df

def dimReduction_NNT(df,n_components=2,n_neighbors=50,mode=NNT_MODE_LIST[0],algorithm=NNT_ALGO_LIST[1],leaf_size=30,p=2,eigen_solver=NNT_ISO_EIGEN_LIST[2],tol=0.0,metric=NNT_METRIC_LIST[0],n_jobs=-1,norm_output=False):
    cache_path = tempfile.gettempdir()
    knt = KNeighborsTransformer(mode=mode,n_neighbors=n_neighbors,algorithm=algorithm,leaf_size=leaf_size,metric=metric,p=p,n_jobs=n_jobs)
    iso = Isomap(n_components=n_components,n_neighbors=n_neighbors,neighbors_algorithm=algorithm,eigen_solver=eigen_solver,metric=metric,tol=tol,p=p,n_jobs=n_jobs)# 
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
        df_tsne = dimReduction_TSNE(mat_emb, n_components, norm_output=norm_output)
        df_main = df_main.join(df_tsne, how="inner", lsuffix='_l',rsuffix='_r')
        out_list.append(df_tsne)
        out_list_label.append("TSNE")
    if active_sel[1] :
        df_pca = dimReduction_PCA(mat_emb, n_components, svd_solver=PCA_TYPE_LIST[0], norm_output=norm_output)
        df_main = df_main.join(df_pca, how="inner", lsuffix='_l',rsuffix='_r')
        out_list.append(df_pca)
        out_list_label.append("PCA")
    if active_sel[2] :
        df_ipca = dimReduction_IPCA(mat_emb, n_components,norm_output=norm_output)
        df_main = df_main.join(df_ipca, how="inner", lsuffix='_l',rsuffix='_r')
        out_list.append(df_ipca)
        out_list_label.append("IPCA")
    if active_sel[3] :
        df_nnt = dimReduction_NNT(mat_emb,n_components,norm_output=norm_output)
        df_main = df_main.join(df_nnt, how="inner", lsuffix='_l',rsuffix='_r')
        out_list.append(df_nnt)
        out_list_label.append("NNT")
    if return_list :
        return out_list, out_list_label
    else :
        return df_main, out_list_label

def generateDimReducedDF(n_components=3, norm_output=True, active_sel=[True,True,True,True]):
    df_embd_input = openDFcsv(mv.embdedding_path,mv.embdedding_filename)#.head(10000)
    mat_emb_input = openDFcsv(mv.embdedding_path_raw,mv.embdedding_filename_raw)#.head(10000)
    df_embd_input_backup = df_embd_input.copy()
    if "o_data" in list(df_embd_input.columns) :
        df_embd_input.drop(["o_data"], axis=1,inplace=True)
    mat_emb_input.drop(["hash_key"], axis=1,inplace=True)
    df_embd_output, out_list_label = generateDirRed(mat_emb_input, df_embd_input,n_components=n_components, norm_output=norm_output, active_sel=active_sel)
    df_embd_input_backup = deleteUnnamed(df_embd_input_backup,"hash_key")
    df_embd_output = deleteUnnamed(df_embd_output,"hash_key")
    saveDFcsv(df_embd_input_backup, mv.embdedding_path, mv.embdedding_filename+"_before_dim_reduct")#+"_before_dim_reduct_1"
    saveDFcsv(df_embd_output, mv.embdedding_path, mv.embdedding_filename)#+"_after_dim_reduct_3"
    return df_embd_output


#### Caculate stats (pivot)

def normalizeColumnList(df,col_list,suffix="_aju") :
    out_col_list = []
    for col in col_list :
        if col in list(df.columns):
            #df.loc[:,col+suffix] = (df.loc[:,col]-getStatsFromCol(df,col)[0])/getStatsFromCol(df,col)[2]
            df[col+suffix] = (df[col]-getStatsFromCol(df,col)[0])/getStatsFromCol(df,col)[2]
            out_col_list.append(col+suffix)
        else:
            print("ERROR : did not find col to normalize in df")
    return df, out_col_list

def groupStats(df,groupping,other_cols,transfo_type="sum",sort_by_groupping=True):
    if type(groupping)!=type(list([])):
        groupping_li = [groupping]
    else :
        groupping_li = groupping
    df = df[groupping_li+other_cols]
    df_group = df[groupping_li+other_cols].groupby(groupping).sum(other_cols).astype(np.float64)#transform(transfo_type#set_index(groupping)[groupping]+
    df_count = df[groupping].value_counts().to_frame("count")
    df_main = df_group.join(df_count, how="inner",on=groupping)
    sortByGroupping(df_main,groupping,sort_by_groupping)
    return df_main,df_count

def sortByGroupping(df,groupping,sort_by_groupping) :
    if type(groupping)!=type(list([])):
        groupping_li = [groupping]
    else :
        groupping_li = groupping
    if sort_by_groupping :
        if len(groupping_li)==1 :
            df = df.sort_values(by=groupping,ascending=True)
        else:
            df = df.sort_values(by=groupping_li[0],ascending=True)
    else :
        df = df.sort_values(by=["count"],ascending=False)
    return df

def divByCount(df,col_list,suffix=""):
    col_suff=[]
    for col in col_list :
        col_suff.append(col+suffix)
    df[col_suff] = df[col_list].div(df['count'], axis=0).astype(np.float64)
    return df#.join(df_count, how="inner")

def limitSel(df,source_limit,groupping):
    if type(groupping)==type([]):
        if len(groupping)>1:
            groupping_str=str(groupping[1])
    else :
        groupping_str=str(groupping)
    if groupping_str in list(df.columns) :
        df2 = copy.deepcopy(df)[groupping_str].value_counts().to_frame("count").sort_values(by=['count'],ascending=False)
        if df2.shape[0]>source_limit :
            select_val_list = list(set(df2.head(source_limit).reset_index()[groupping_str].tolist()))
            df = df[df[groupping_str].isin(select_val_list)]
    return df

def calculateStatsLength(df,groupping,display_data=False,sort_by_groupping=True,flatten_index=False,limit_sel=15):
    df = df[~df["tb.pol"].isnull()]
    cols = ["tb.char","tb.sent","tb.noun","tb.word"]
    df = limitSel(df,limit_sel,groupping)
    df_main,df_count = groupStats(df,groupping,cols,sort_by_groupping=sort_by_groupping)
    #df_main[["tb.char_pc","tb.sent_pc","tb.noun_pc","tb.word_pc"]] = df_main[cols].div(df_main['count'], axis=0).astype(float)
    df_main[["tb.char_ps","tb.noun_ps","tb.word_ps"]] = df_main[["tb.char","tb.noun","tb.word"]].div(df_main["tb.sent"], axis=0).astype(float)
    df_main[["tb.char_pw"]] = df_main[["tb.char"]].div(df_main["tb.word"], axis=0).astype(float)
    df_main=divByCount(df_main,["tb.char","tb.sent","tb.noun","tb.word"],"_pa").astype(np.float64)
    df = sortByGroupping(df,groupping,sort_by_groupping)
    if flatten_index:
        df_main = df_main.reset_index()
    if display_data : 
        print("Dataframe Statistics Length Column :'"+str(groupping)+"'")
        display_df(df_main)
    return df_main

def calculateStatsNLP(df,groupping,display_data=False,display_stats=True,out_raw=False,only_keyword_nlp=False,sort_by_groupping=True,flatten_index=False,limit_sel=15):
    nlp_col_list = ["tb.pol","tb.sub","tb.pos","tb.neg","vs.pos","vs.neu","vs.neg","vs.comp","ts.pos","ts.neu","ts.neg","al.pos","al.neg","tb.pol_k","tb.sub_k","tb.pos_k","tb.neg_k","vs.pos_k","vs.neu_k","vs.neg_k","vs.comp_k","ts.neg_k","ts.neu_k","ts.pos_k","al.pos_k","al.neg_k","0_tsne","1_tsne","2_tsne","0_pca","1_pca","2_pca","0_ipca","1_ipca","2_ipca","0_tnn","1_tnn","2_tnn"]
    nlp_col_list = columnListinDf(df,nlp_col_list)
    df = df[~df[nlp_col_list[0]].isnull()]
    df = limitSel(df,limit_sel,groupping)
    df, out_col_list = normalizeColumnList(df,nlp_col_list,"_aj")
    if not out_raw :
        column_list = nlp_col_list+out_col_list
        df_main,df_count = groupStats(df,groupping,column_list,sort_by_groupping=sort_by_groupping)
        #df_main = limitSel(df_main,limit_sel,groupping)
        # if groupping in column_list :
        #     column_list.remove(groupping)
        df_main=divByCount(df_main,column_list)
        df, out_col_list = normalizeColumnList(df_main,out_col_list,"")
        
    else :
        df_main = df
    df = sortByGroupping(df,groupping,sort_by_groupping)
    if flatten_index:
        df_main = df_main.reset_index()
    if display_data :
        print("Dataframe Statistics NLP Column :'"+str(groupping)+"'")
        display_df(df_main)
    return df_main

def columnListinDf(df,col_list,suffix=""):
    out_list = []
    for col in col_list :
        if col in list(df.columns) :
            out_list.append(col+suffix)
    return out_list

def getStatsFromCol(df, column) :
    min_val = df[column].min()
    max_val = df[column].max()
    return min_val,max_val,max_val-min_val

def calculateStatsColList(df, column_list=[],display_df=True,display_stats=False,out_raw=False,save=True):# ,stat_type="nlp"
    df_list_out = []
    for col in column_list :
        df_len = calculateStatsLength(df,col,display_df)
        df_nlp = calculateStatsNLP(df,col,display_df,display_stats,out_raw)
        df_stats = df_nlp.join(df_len, how="inner",on=col,lsuffix="_nlp",rsuffix='_len')
        df_stats["count_equal"] =np.where(df_stats['count_nlp'] ==df_stats['count_len'], True,False)#df_stats[df_stats["count_nlp"]==df_stats["count_len"]]
        df_list_out.append(df_stats)
        if save :
            saveDFcsv(df_stats,mv.visu_path,mv.visu_filename+"_"+str(col))
    return df_list_out
    
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

def extractFromKey(df,charNum=1):
    df["key_extract"] = df["hash_key"].str.slice(0,charNum)
    return df

# print("IMPORT : dimension_reduc ")

# df_main = openDFcsv(mv.join2_path,mv.join2_filename)
# display(extractFromKey(df_main)["key_extract"])

# agg_list = ["category","year","source_title",["year","category"],["year","source_title"]]#,["year","category"],["year","source_title"]]
# field_list = [False,True,False,True,True]#,["year","category"],["year","source_title"]]
# df_list=calculateStatsColList(df_main,agg_list,display_df=True,display_stats=True,out_raw=False,save=True)
 