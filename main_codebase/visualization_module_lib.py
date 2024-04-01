# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:08:04 2024

@author: Alexandre
"""

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

PLOT_WIDTH=1800
PLOT_HEIGHT=1000
PLOT_AUTOSIZE=True
PCA_TYPE_LIST = ["auto", "full", "arpack", "randomized"]
PLOT_RENDER_LIST = ["png","browser","svg"]
PLOT_RENDER = PLOT_RENDER_LIST[1]
TAG_SELECTION_LIST = ["NNP","NN","JJ"]
MARGINAL_LIST = [None,'rug', 'box', 'violin','histogram']
TRENDLINE_LIST = [None,'ols', 'lowess', 'rolling', 'expanding','ewm']
DO_NOT_RENDER_PLOT=False



import ipywidgets as widgets


def dfNormalize(df) :
    return ((df - df.mean()) / df.std())

def generateTestSet() :
    X, y = make_classification(
        n_features=24,
        n_classes=5,
        n_samples=1500,
        n_informative=3,
        random_state=5,
        n_clusters_per_class=1,
    )
    return X, y

## Dimention reduction

def dimReduction_TSNE(df, n_components=2, perplexity=30, early_exaggeration =4.0,learning_rate=1000,n_iter=1000,verbose=0,random_state=0,norm_output=False):
    tsne = TSNE(n_components=n_components, perplexity=perplexity,early_exaggeration = early_exaggeration,learning_rate =learning_rate,n_iter=n_iter,verbose=verbose,random_state=random_state)
    out_df = tsne.fit_transform(df)
    out_df = pd.DataFrame(out_df)
    if norm_output :
        out_df = dfNormalize(out_df)
    return out_df

def dimReduction_PCA(df, n_components=2, svd_solver="auto",tol=0.0,whiten=False,random_state=0,norm_output=False):
    pca = PCA(n_components=n_components,svd_solver=svd_solver,tol=tol,random_state=random_state)
    out_df = pca.fit_transform(df)
    out_df = pd.DataFrame(out_df)
    if norm_output :
        out_df = dfNormalize(out_df)
    return out_df

def dimReduction_IPCA(df, n_components=2,whiten=False, batch_size=100,norm_output=False):
    ipca = IncrementalPCA(n_components=n_components,batch_size=batch_size)
    out_df = ipca.fit_transform(df)
    out_df = pd.DataFrame(out_df)
    if norm_output :
        out_df = dfNormalize(out_df)
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
    out_df = pd.DataFrame(out_df)
    if norm_output :
        out_df = dfNormalize(out_df)
    return out_df

def generateDirRed(mat_emb,df_main,n_components=2,norm_output=False,active_sel=[True,False,False,False]) :
    out_list = []
    out_list_label = []
    if active_sel[0] :
        df_tsne = dimReduction_TSNE(mat_emb,n_components,norm_output=norm_output)
        df_tsne_j = df_main.join(df_tsne, how="inner")
        out_list.append(df_tsne_j)
        out_list_label.append("TSNE")
    if active_sel[1] :
        df_pca = dimReduction_PCA(mat_emb,n_components,norm_output=norm_output)
        df_pca_j = df_main.join(df_pca, how="inner")
        out_list.append(df_pca_j)
        out_list_label.append("PCA")
    if active_sel[2] :
        df_ipca = dimReduction_IPCA(mat_emb,n_components,norm_output=norm_output)
        df_ipca_j = df_main.join(df_ipca, how="inner")
        out_list.append(df_ipca_j)
        out_list_label.append("IPCA")
    if active_sel[3] :
        df_nnt = dimReduction_NNT(mat_emb,n_components,norm_output=norm_output)
        df_nnt_j = df_main.join(df_nnt, how="inner")
        out_list.append(df_nnt_j)
        out_list_label.append("NNT")

    return out_list, out_list_label


## Visualization using "plotly"

def renderAllOptions(df,BD=True,confList=[],label="") :
    plt_list = []
    for conf in confList :
        main_conf = createdefconfdict()|conf
        if label != "" :
            main_conf['title'] = main_conf['title'] + "  -  " + label
        if (BD) :
            plt = plot2D(df,main_conf)
        else :
            plt = plot3D(df,main_conf)
        plt_list.append(plt)
    return plt_list

# 2D plots : 

def plot2D(df,conf_dict={}) : # Scatter
    fig = px.scatter(df, x=conf_dict["x"], y=conf_dict["y"], color=conf_dict["c"],size=conf_dict["size"],
        symbol=conf_dict["symbol"],text=conf_dict["text"], 
        hover_name=conf_dict["h_name"], hover_data=conf_dict["h_data"], custom_data=conf_dict["c_data"], 
        facet_row=conf_dict["facet_r"], facet_col=conf_dict["facet_c"],facet_col_wrap=conf_dict["facet_cw"],
        facet_row_spacing=conf_dict["facet_rs"], facet_col_spacing=conf_dict["facet_cs"],
        animation_frame=conf_dict["animation_frame"], animation_group=conf_dict["animation_group"],
        marginal_x=conf_dict["marginal_x"], marginal_y=conf_dict["marginal_y"],
        trendline=conf_dict["trendline"], log_x=conf_dict["log_x"],log_y=conf_dict["log_y"], render_mode=conf_dict["render_mode"],
        size_max=conf_dict["size_max"], opacity=conf_dict["opacity"]) 
    # fig.update_annotations("bgcolor":rgba(13,57,F2,0),captureevents:test_val)
    fig.update_layout(title=conf_dict["title"], xaxis_title=conf_dict["xtitle"], yaxis_title=conf_dict["ytitle"],width=PLOT_WIDTH,height=PLOT_HEIGHT,autosize=PLOT_AUTOSIZE)
    if not DO_NOT_RENDER_PLOT :
        if type(conf_dict["browser"]) == type(None) :
            fig.show()
        else :
            fig.show(renderer="browser")
            # widgets.IntSlider()
    return fig


def plot3D(df,conf_dict={}) :
    fig = px.scatter_3d(df, x=conf_dict["x"], y=conf_dict["y"], z=conf_dict["z"], color=conf_dict["c"],size=conf_dict["size"],
        symbol=conf_dict["symbol"],text=conf_dict["text"],
        hover_name=conf_dict["h_name"], hover_data=conf_dict["h_data"],custom_data=conf_dict["c_data"],
        animation_frame=conf_dict["animation_frame"], animation_group=conf_dict["animation_group"],
        log_x=conf_dict["log_x"],log_y=conf_dict["log_y"],log_z=conf_dict["log_z"],
        size_max=conf_dict["size_max"], opacity=conf_dict["opacity"])
    fig.update_layout(title=conf_dict["title"], xaxis_title=conf_dict["xtitle"], yaxis_title=conf_dict["ytitle"],width=PLOT_WIDTH,height=PLOT_HEIGHT,autosize=PLOT_AUTOSIZE) #, zaxis_title=conf_dict["ztitle"]
    if not DO_NOT_RENDER_PLOT :
        if type(conf_dict["browser"]) == type(None) :
            fig.show()
        else :
            fig.show(renderer="browser")
    return fig

# ["x","y","z","c","size","symbol","text","h_name","h_data","c_data","title","xtitle","ytitle"]
def createdefconfdict() :
    ret_dict = {}
    val_list = ["x","y","z","c","size","symbol","h_name","h_data","c_data","text","facet_r","facet_c","facet_cw","facet_rs","facet_cs","title","xtitle","ytitle","ztitle","browser","animation_frame","animation_group","marginal_x","marginal_y","trendline","log_x","log_y","log_z","render_mode","size_max","opacity"]
    for val in val_list :
        ret_dict[val] = None
    return ret_dict

## Visualization using "pandas"

def plot2Dpandas(df,size=3,color="#a98d19",width=15,height=15,title="default",xl="x",yl="y",save=False,path="",savecount=0) :
    res = df.plot.scatter(x = 0, y = 1, s=size, c="o_token_input",figsize=(width,height),title=title,xlabel=xl,ylabel=yl).get_figure(); 
    if save :
        res.savefig(path+"fig_"+str(savecount)+".png")
        
## Visualization using "matplotlib"

def plot2Dmatplotlib(df) :
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df[0], df[1], 0.5,"#a98d19")#, c=df['color']
    plt.show()
    
    
##
def testDimReduct(df,i) :
    # df = dfNormalize(df)
    df_tsne = dimReduction_TSNE(df)
    plot2Dpandas(df_tsne,title="TSNE (T-distributed Stochastic Neighbor Embedding)",save=True,path=folder_path_graph+"TSNE_",savecount=i)
    
    df_pca = dimReduction_PCA(df)
    plot2Dpandas(df_pca,title="PCA (Principal Component Analysis)",save=True,path=folder_path_graph+"PCA_",savecount=i)
    
    df_ipca = dimReduction_IPCA(df)
    plot2Dpandas(df_ipca,title="IPCA (Incremental Principal Component Analysis)",save=True,path=folder_path_graph+"IPCA_",savecount=i)
    
    df_nnt = dimReduction_NNT(df)
    plot2Dpandas(df_nnt,title="NNT (Nearest Neighbors Transformer)",save=True,path=folder_path_graph+"IPCA_",savecount=i)



def update_point(trace, points, selector):
    c = list(scatter.marker.color)
    s = list(scatter.marker.size)
    for i in points.point_inds:
        c[i] = '#bae2be'
        s[i] = 20
        with f.batch_update():
            scatter.marker.color = c
            scatter.marker.size = s

def getRenderLists(experimental=True,basic=True,test=True) :
    # standardDict = {"x":1,"y":2,"z":3,"c":"category","size":"words","h_name":"source_title","c_data":["title_quer"],
    #                 "c_data":None,"browser":True,'facet_r':None,"facet_c":None,"facet_rs":0.02,"facet_cs":0.02,
    #                 "animation_frame":None,"animation_group":None,"marginal_x":MARGINAL_LIST[0],"marginal_y":MARGINAL_LIST[0],
    #                 "log_x":False,"log_y":False,"log_z":False,"render_mode":PLOT_RENDER,"size_max":30,"opacity":0.85}

    standardDict = {"x":1,"y":2,"z":3,"c":"category","size":"tb.word","h_name":"source_title","c_data":["title_quer"],
                    "c_data":None,"browser":True,'facet_r':None,"facet_c":None,"facet_rs":0.02,"facet_cs":0.02,
                    "animation_frame":None,"animation_group":None,"marginal_x":MARGINAL_LIST[0],"marginal_y":MARGINAL_LIST[0],
                    "log_x":False,"log_y":False,"log_z":False,"render_mode":PLOT_RENDER,"size_max":30,"opacity":0.85}

    
    _2d_embd_exp = {"x":0,"y":1,'c':"category","animation_frame":"year_month","title":"test_year_month","browser":True}
              
    _3d_embd_exp = {"x":0,"y":1,"z":2,'c':"category","animation_frame":"year_month","title":"test_year_month","browser":True}
    _2dList_exp = [_2d_embd_exp]
    _3dList_exp = [_3d_embd_exp]
    
    _2d_embd_basic = standardDict|{"x":0,"y":1,'c':"category","title":"Scatter plot of dimention reduced embeding data from articles","xtitle":"Component 1","ytitle":"Component 2"}
    _3d_embd_basic = standardDict|{"x":0,"y":1, 'z':2,'c':"category","title":"Scatter plot of dimention reduced embeding data from articles","xtitle":"Component 1","ytitle":"Component 2","ztitle":"Component 3"}
    _2d_sent_basic = standardDict|{"x":"tb.polaj","y":"tb.sub",'c':"category","title":"Scatter plot of the 'polarity' and 'objectivity' of article data","xtitle":"Polarity (0-1)","ytitle":"Subjectivity (0-1)","marginal_x":MARGINAL_LIST[3],"marginal_y":MARGINAL_LIST[4]}
    _3d_sent_basic = standardDict|{"x":"tb.polaj","y":"tb.sub","z":"ts.pos",'c':"category","title":"Scatter plot of the 'polarity', 'objectivity' and 'positivity' of article data","xtitle":"Polarity (0-1)","ytitle":"Subjectivity (0-1)","ztitle":"Positivity (0-1)"}
    _2d_len_basic = standardDict|{"x":"tb.sent","y":"tb.noun",'c':"category","title":"Scatter plot of text nature and volume data from articles","xtitle":"Number of Sentences","ytitle":"Number of Nouns"}
    _3d_len_basic = standardDict|{"x":"tb.sent","y":"tb.noun", 'z':"tb.word",'c':"category","title":"Scatter plot of text nature and volume data from articles","xtitle":"Number of Sentences","ytitle":"Number of Nouns","ztitle":"Number of Words"}
    _2dList_basic = [_2d_embd_basic ,_2d_sent_basic ,_2d_len_basic]
    _3dList_basic = [_3d_embd_basic ,_3d_sent_basic, _3d_len_basic]
    
    _2d_embd_test = standardDict|{"x":0,"y":1,'c':"category","title":"Scatter plot of dimention reduced embeding data from articles","xtitle":"Component 1","ytitle":"Component 2"}
    _3d_embd_test = standardDict|{"x":0,"y":1,"z":2,'c':"category","title":"Scatter plot of dimention reduced embeding data from articles","xtitle":"Component 1","ytitle":"Component 2"}
    
    _2dList_test = [_2d_embd_test]
    _3dList_test = [_3d_embd_test]
    
    _2dList_out = []
    _3dList_out = [] 
    if experimental :
        _2dList_out = _2dList_out + _2dList_exp
        _3dList_out = _3dList_out + _3dList_exp
    if basic :
        _2dList_out = _2dList_out + _2dList_basic
        _3dList_out = _3dList_out + _3dList_basic
    if test :
        _2dList_out = _2dList_out + _2dList_test
        _3dList_out = _3dList_out + _3dList_test
    return _2dList_out, _3dList_out


def savePlotList(plot_list,path,label="default") :
    current_timestamp = datetime.datetime.now()
    current_timestamp = str(current_timestamp.strftime("%Y,%m,%d;%H,%M,%S"))
    for i in range(len(plot_list)) :
        plotSave(plot_list[i],path,label+"_n="+str(i)+"_d="+current_timestamp)
def plotSave(plot,path,filename) :
    plot.write_html(path+filename+".html")
    # # x = None
    # # y = None
    # # z = None
    # category = "category"#""bool_sent_all"#"category"
    # size = "words"#"text_len"  #"text_len"#"words"  #"text_len"#
    # symbol = None#  "" #None #  "url_TLD"
    # h_name = "source_title"  #"title_quer" #None #
    # h_data = []#"title_quer","subjectivity","polarity","pos1","neg1","0_s_k","0_s_t"] # None# "title_quer"#None #"source_title"
    # c_data = None  #"word_combined_s_k"
    # browser = True 
    # facet_r = None #"year_month" #"category"
    # facet_c = None
    # facet_rs =0.03
    # facet_cs =0.03
    # animation_frame = "year_month"#"year_month"
    # animation_group = None#"hash_key"
    # marginal_x = MARGINAL_LIST[0]
    # marginal_y = MARGINAL_LIST[0]
    # trendline = TRENDLINE_LIST[0]
    # log_x =False
    # log_y = False
    # log_z = False
    # render_mode = PLOT_RENDER
    # size_max = 30
    # opacity=0.9
    
# def linegraph(df) :
#     # df = px.data.gapminder().query("country=='Canada'")
#     fig = px.line(df, x='year', y='words', color='category')#_month , symbol="url_TLD"_mont
#     fig.show()
# #     fig = px.line(df, x="year_month", title='TEST',color='category')
# #     fig.show()