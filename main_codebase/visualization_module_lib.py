# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:08:04 2024

@author: Alexandre
"""

from utils_art import *
import main_var
mv = main_var.main_var()
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import cm
from random import randint
import random
import colorsys
import copy
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from IPython.display import display, HTML
from PIL import Image, ImageFont, ImageDraw, ImageOps
# from dimension_reduc_lib import calculateStatsNLP
# import matplotlib.cm as cm
# import ipywidgets as widgets
# from IPython.core.display import Image as image
# from PIL import Image
# from IPython.display import Image
# from textblob import TextBlob
# import tempfile
# import time
# import plotly

PLOT_WIDTH=1500
PLOT_HEIGHT=1000
PLOT_AUTOSIZE=False
PLOT_RENDER_LIST = ["png","browser","svg"]
PLOT_RENDER = PLOT_RENDER_LIST[1]
TAG_SELECTION_LIST = ["NNP","NN","JJ"]
MARGINAL_LIST = [None,"rug", "box", "violin","histogram"]
TRENDLINE_LIST = [None,"ols", "lowess", "rolling", "expanding","ewm"]
DO_NOT_RENDER_PLOT=False
np.random.seed(0)

all_conf_dict = {'data_frame': None, 'color': None, 'height': None, 'width': None, 'template': None, 'title': None, 'labels': None, 'hover_name': None, 'hover_data': None, 'custom_data': None, 'color_discrete_sequence': None, 'color_discrete_map': None, 'x': None, 'y': None, 'log_x': None, 'log_y': None, 'range_x': None, 'range_y': None, 'error_x': None, 'error_x_minus': None, 'error_y': None, 'error_y_minus': None, 'facet_row': None, 'facet_col': None, 'facet_col_wrap': 0, 'facet_row_spacing': None, 'facet_col_spacing': None, 'symbol': None, 'size': None, 'text': None, 'animation_frame': None, 'animation_group': None, 'category_orders': None, 'orientation': None, 'color_continuous_scale': None, 'range_color': None, 'color_continuous_midpoint': None, 'symbol_sequence': None, 'symbol_map': None, 'opacity': None, 'size_max': None, 'marginal_x': None, 'marginal_y': None, "marginal": None, 'trendline': None, 'trendline_options': None, 'trendline_color_override': None, 'trendline_scope': None, 'render_mode': None, 'z': None, 'log_z': None, 'range_z': None, 'error_z': None, 'error_z_minus': None, 'names': None, 'values': None, 'hole': None, 'pattern_shape_sequence': None, 'pattern_shape_map': None, 'base': None, 'pattern_shape': None, 'barmode': None, 'text_auto': None, 'line_dash': None, 'line_group': None, 'line_dash_sequence': None, 'line_dash_map': None, 'line_close': None, 'line_shape': None, 'barnorm': None, 'markers': None, 'groupnorm': None, 'parents': None, 'path': None, 'ids': None, 'branchvalues': None, 'maxdepth': None, 'a': None, 'b': None, 'c': None, 'dimensions': None, 'browser':None, 'title': None, 'xtitle': None, 'ytitle': None, 'ztitle': None, 'histnorm':None, "histfunc": None, "cumulative": None, "nbins": 0, "direction": "clockwise", "start_angle": 90, "range_r": None, "range_theta": None, "log_r": None}
_TOPIC_LIST = ["TOP"]#,"WORLD","NATION","BUSINESS","TECHNOLOGY","ENTERTAINMENT","SCIENCE","SPORTS","HEALTH"]
_FONT = ImageFont.truetype("FONTS/arial.ttf", 30)

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

def calculateStatsNLP(df):
    print("DF has",df.shape[0],"rows and",df.shape[1],"columns")
    print("DF data is from",df["published"].min(),"to",df["published"].max()," (unique years:",df["year"].nunique()," unique months:",str(df["year_month"].nunique())+")")
    print("DF has this number of unique sources:",df["source_title"].nunique())
    col_list = ["tb.sent","tb.noun","tb.word","tb.char","tb.pol","tb.sub","tb.pos","tb.neg","vs.pos","vs.neu","vs.neg","vs.comp","ts.pos","ts.neu","ts.neg","al.pos","al.neg","0_tsne","1_tsne","2_tsne","0_pca","1_pca","2_pca","0_ipca","1_ipca","2_ipca","word_count_f_t","word_count_f_k","word_count_s_t","word_count_s_k","word_count_all","word_count_all_sel","tb.pol_k","tb.sub_k","tb.polaj","tb.pos_k","tb.neg_k","vs.pos_k","vs.neu_k","vs.neg_k","vs.comp_k","ts.neg_k","ts.neu_k","ts.pos_k","al.pos_k","al.neg_k"]
    display(HTML(df[col_list].describe().to_html()))
    createWordCloud(df,save_plots=True)

def flattenStringList(mat):
    out_list = ""
    for li in mat :
        out_list = out_list + li
    return out_list

def generateWordCloud(df,column,category="",year="",add_label="",display=True):
    if category!="":
        df=df[df["category"]==category] #'word_combined_all_sel'stopwords=stopwords, 
    if year!="":
        df=df[df["year"]==year]
    text=df[column].values
    text = flattenStringList(text)
    list_word = text.split(" ")
    word_list=getMostCommunKeywords(list_word, source_limit=20,return_word_list=True,display_stats=True)
    for word in word_list:
        text =  text.replace(" "+word+" ", " ")
    title="Commun Words '"+str(column)+"' "+(" category: '"+str(category)+"' ")*(category!="")+(" year: '"+str(year)+"' ")*(year!="")+" ("+str(df.shape[0])+" entries) "+str(add_label)
    wordcloud = WordCloud(background_color="white", max_words=10000,min_font_size=15, max_font_size= 150,  width=1000, height=1000,prefer_horizontal=1,scale=1.5,font_step=3,relative_scaling=0.3,repeat=True,min_word_length=3).generate_from_text(str(text))
    img=ImageOps.expand(wordcloud.to_image(), border=40, fill=(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((5, 5),title,(0,0,0),font=_FONT)
    if display :
        plt.figure(figsize=(20,20))
        plt.imshow(img, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    return img

def nlpFilter(df,index=0,volumepct=0.1):
    top=True
    col_list = ["tb.pol","tb.sub","tb.pos","vs.pos","ts.pos"]
    col_list_label = ["Polarity","Subjectivity","Positivity (TB)","Positivity (VS)","Positivity (TS)"]
    if index%2!=0:
        top=False
    col = col_list[int(index/2)%5]
    col_label = col_list_label[int(index/2)%5]
    size=df.shape[0]
    df_out=df.sort_values(col,ascending=not top)[:int(size*volumepct)]
    return df_out, str(str("Top"*top)+str("Bottom"*(not top))+" of '"+col_label+"' index  ("+str(volumepct*100)+"%)")

def createWordCloud(df,display_plots=True,save_plots=True):
    from visualization_module_lib import generateWordCloud
    graph_folder=word_cloud_folder
    _TOPIC_LIST = df["category"].unique()#["TOP","WORLD","NATION"]#,"BUSINESS","TECHNOLOGY","ENTERTAINMENT","SCIENCE","SPORTS","HEALTH"]
    _YEAR_LIST = df["year"].unique()
    for cat in _TOPIC_LIST:
        out=generateWordCloud(df,"word_combined_all",category=cat,display=display_plots)
        if save_plots :
            out.save(mv.visu_path+graph_folder+mv.visu_filename+"_"+cat+".png","PNG")
    for year in _YEAR_LIST:
        out=generateWordCloud(df,"word_combined_all",year=year,display=display_plots)
        if save_plots :
            out.save(mv.visu_path+graph_folder+mv.visu_filename+"_"+str(year)+".png","PNG")
    for i in range(10):
        df_out,label = nlpFilter(df,i)
        out=generateWordCloud(df_out,"word_combined_all",add_label=label,display=display_plots)
        if save_plots :
            out.save(mv.visu_path+graph_folder+mv.visu_filename+"_"+str(i)+".png")



    
def generateKeywordFrequency(df,column):
    text = df[column].values
    plt.figure(figsize=(20,20))
    plot = text.plot.barh(y='count',use_index=True, legend=False, fontsize= 8, title="Volume month/year") #, figsize=(5, 5)
## Visualization using "plotly"

# def renderAllOptions(df,BD=True,confList=[],label="") :
#     plt_list = []
#     for conf in confList :
#         main_conf = createdefconfdict()|conf
#         if label != "" :
#             main_conf["title"] = main_conf["title"] + "  -  " + label
#         if (BD) :
#             plt = plot2D(df,main_conf)
#         else :
#             plt = plot3D(df,main_conf)
#         plt_list.append(plt)
#     return plt_list

def renderAllOptions(df,confList=[]) :
    plt_list = []
    for conf in confList :
        main_conf = createdefconfdict()|conf
        main_conf["data_frame"]=df.sort_values(by=['year_month'],ascending=True)
        #print(main_conf)
        plt = plotWithConf(main_conf)
        plt_list.append(plt)
    return plt_list
# 2D plots : 

# def plot2D(df,conf_dict={}) : # Scatter
#     fig = px.scatter(df, x=conf_dict["x"], y=conf_dict["y"], color=conf_dict["c"],size=conf_dict["size"],
#         symbol=conf_dict["symbol"],text=conf_dict["text"], 
#         hover_name=conf_dict["hover_name"], hover_data=conf_dict["hover_data"], custom_data=conf_dict["custom_data"], 
#         facet_row=conf_dict["facet_row"], facet_col=conf_dict["facet_col"],facet_col_wrap=conf_dict["facet_col_wrap"],
#         facet_row_spacing=conf_dict["facet_row_spacing"], facet_col_spacing=conf_dict["facet_col_spacing"],
#         animation_frame=conf_dict["animation_frame"], animation_group=conf_dict["animation_group"],
#         marginal_x=conf_dict["marginal_x"], marginal_y=conf_dict["marginal_y"],
#         trendline=conf_dict["trendline"], log_x=conf_dict["log_x"],log_y=conf_dict["log_y"], render_mode=conf_dict["render_mode"],
#         size_max=conf_dict["size_max"], opacity=conf_dict["opacity"],width=500) 
#     fig.update_layout(title=conf_dict["title"], xaxis_title=conf_dict["xtitle"], yaxis_title=conf_dict["ytitle"],width=PLOT_WIDTH,height=PLOT_HEIGHT,autosize=PLOT_AUTOSIZE,showlegend=True)
#     if not DO_NOT_RENDER_PLOT :
#         if type(conf_dict["browser"]) == type(None) :
#             fig.show()
#         else :
#             fig.show(renderer="browser")
#     return fig


# def plot3D(df,conf_dict={}) :
#     fig = px.scatter_3d(df, x=conf_dict["x"], y=conf_dict["y"], z=conf_dict["z"], color=conf_dict["c"],size=conf_dict["size"],
#         symbol=conf_dict["symbol"],text=conf_dict["text"],
#         hover_name=conf_dict["hover_name"], hover_data=conf_dict["hover_data"],custom_data=conf_dict["custom_data"],
#         animation_frame=conf_dict["animation_frame"], animation_group=conf_dict["animation_group"],
#         log_x=conf_dict["log_x"],log_y=conf_dict["log_y"],log_z=conf_dict["log_z"],
#         size_max=conf_dict["size_max"], opacity=conf_dict["opacity"])
#     fig.update_layout(title=conf_dict["title"],width=PLOT_WIDTH,height=PLOT_HEIGHT,autosize=PLOT_AUTOSIZE,scene = dict(bgcolor="#FFFFFF",xaxis=dict(title=conf_dict["xtitle"], color="#000000", gridcolor="#888888",gridwidth=5),yaxis=dict(title=conf_dict["ytitle"], color="#000000", gridcolor="#888888",gridwidth=5),zaxis=dict(title=conf_dict["ztitle"], color="#000000", gridcolor="#888888",gridwidth=5)))
#     if not DO_NOT_RENDER_PLOT :
#         if type(conf_dict["browser"]) == type(None) :
#             fig.show()
#         else :
#             fig.show(renderer="browser")
#     return fig

# def bar(df,conf_dict={}) :
#     fig = px.bar(df, x=conf_dict["x"], y=conf_dict["y"], color=conf_dict["c"],
#         pattern_shape=conf_dict["pattern_shape"],barmode=conf_dict["barmode"],facet_row=conf_dict["facet_row"], facet_col=conf_dict["facet_col"],facet_col_wrap=conf_dict["facet_col_wrap"],
#         facet_row_spacing=conf_dict["facet_row_spacing"], facet_col_spacing=conf_dict["facet_col_spacing"],
#         hover_name=conf_dict["hover_name"], hover_data=conf_dict["hover_data"],custom_data=conf_dict["custom_data"],
#         text=conf_dict["text"],base=conf_dict["base"],labels=conf_dict["labels"],
#         animation_frame=conf_dict["animation_frame"], animation_group=conf_dict["animation_group"],
#         log_x=conf_dict["log_x"],log_y=conf_dict["log_y"], opacity=conf_dict["opacity"])
#     fig.update_layout(title=conf_dict["title"], xaxis_title=conf_dict["xtitle"], yaxis_title=conf_dict["ytitle"],width=PLOT_WIDTH,height=PLOT_HEIGHT,autosize=PLOT_AUTOSIZE,text_auto=conf_dict["text_auto"]) #, zaxis_title=conf_dict["ztitle"]
#     if not DO_NOT_RENDER_PLOT :
#         if type(conf_dict["browser"]) == type(None) :
#             fig.show()
#         else :
#             fig.show(renderer="browser")
#     return fig

# ["x","y","z","c","size","symbol","text","hover_name","hover_data","custom_data","title","xtitle","ytitle"]
# def createdefconfdict() :
#     standard_dict = {}
#     ret_dict = {}
#     val_list = ["x","y","z","c","size","symbol","hover_name","hover_data","custom_data","text","facet_row","facet_col","facet_col_wrap","facet_row_spacing","facet_col_spacing","title","xtitle","ytitle","ztitle","browser","animation_frame","animation_group","marginal_x","marginal_y","trendline","log_x","log_y","log_z","render_mode","size_max","opacity"]
#     for val in val_list :
#         ret_dict[val] = None
#     return ret_dict
def createdefconfdict() :
    return all_conf_dict
all_conf_dict
## Visualization using "pandas"

def plot2Dpandas(df,size=3,color="#a98d19",width=15,height=15,title="default",xl="x",yl="y",save=False,path="",savecount=0) :
    res = df.plot.scatter(x = 0, y = 1, s=size, c="o_token_input",figsize=(width,height),title=title,xlabel=xl,ylabel=yl).get_figure(); 
    if save :
        res.savefig(path+"fig_"+str(savecount)+".png")
        
## Visualization using "matplotlib"

def plot2Dmatplotlib(df) :
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df[0], df[1], 0.5,"#a98d19")#, c=df["color"]
    plt.show()
    
def plotDFstatisticsQuerry(df, source_limit=50,onlyYear=False) :
    if onlyYear :
        time_field = "year"
    else :
        time_field = "year_month"
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 15))

    df_category = df['category'].value_counts().to_frame("count").sort_values(by=['count'],ascending=True)
    
    df_source = df['source_title'].value_counts().to_frame("count").sort_values(by=['count'],ascending=True)
    source_limit = min(source_limit,len(df_source))
    source_limiy_count = int(df_source.iloc[[int(-source_limit)]]["count"].tolist()[0])
    df_source = df_source[df_source['count'].between(source_limiy_count, 1000000)]

    df_date_year_month = df[time_field].value_counts().to_frame("count").sort_values(by=[time_field],ascending=True)
    axes[1].tick_params(labelcolor='black', labelright=True, labelleft=False)
    axes[1].invert_yaxis()

    plot = df_source.plot.barh(y='count',use_index=True,ax=axes[0], legend=False, ylabel="", fontsize= 8, title="Sources") #, figsize=(5, 5)
    plot = df_date_year_month.plot.barh(y='count',use_index=True,ax=axes[1], legend=False, ylabel="", fontsize= 8, title="Volume month/year") #, figsize=(5, 5)
    plot = df_category.plot.pie(y='count',use_index=True, legend=False, ylabel="", title="Distribution of categories") #, figsize=(5, 5)
    
    return plot
##



# def update_point(trace, points, selector):
#     c = list(scatter.marker.color)
#     s = list(scatter.marker.size)
#     for i in points.point_inds:
#         c[i] = "#bae2be"
#         s[i] = 20
#         with f.batch_update():
#             scatter.marker.color = c
#             scatter.marker.size = s

def getRenderLists(dimentions=[True,True],emmbeding=[False,False,False,False],sentType=[False,False],posType=[False,False],length=False,exp=True,browser=True) :
    dimentions=[True,True]
    emmbeding=[True,True,True,False]
    sentType=[True,True]
    posType=[True,True]
    length=True
    base = {"x":None,"y":None,"z":None,"c":None,"size":"tb.sent","hover_name":None,"custom_data":None, #"custom_data":None,
                    "browser":browser,"facet_row":None,"facet_col":None,"facet_row_spacing":None,"facet_col_spacing":None,
                    "log_x":False,"log_y":False,"log_z":False,"render_mode":PLOT_RENDER,"size_max":None,"opacity":None,"text_auto":True}
    _2d_test_0={"empty":True}
    _2d_test_1={"empty":True}
    _2d_test_2={"empty":True}
    _2d_test_3={"empty":True}
    _2d_test_4={"empty":True}
    _2d_test_5={"empty":True}
    _2d_test_6={"empty":True}
    _2d_test_7={"empty":True}
    _2d_test_8={"empty":True}
    _2d_test_9={"empty":True}
    _3d_test_0={"empty":True}
    _3d_test_1={"empty":True}
    _3d_test_2={"empty":True}
    _3d_test_3={"empty":True}
    _3d_test_4={"empty":True}
    _3d_test_5={"empty":True}
    _3d_test_6={"empty":True}
    _3d_test_7={"empty":True}
    _3d_test_8={"empty":True}
    _3d_test_9={"empty":True}
    
    std2dScatter = {"plot_type":"scatter","trendline":"ols","opacity":0.9,"size":"tb.sent"} #'ols', 'lowess', 'rolling', 'expanding' or 'ewm',"size":"tb.sent" ,"size":"count"
    std3dScatter = {"plot_type":"scatter_3d","size":"tb.sent","opacity":0.9}
    std2dPie = {"plot_type":"pie","opacity":0.9,"hole":0.2}#,"hover_name":"category"
    std2dBar = {"plot_type":"bar","opacity":0.9, "text_auto":True}
    std2dScatterPolar = {"plot_type":"scatter_polar"}
    std2dLinePolar = {"plot_type":"line_polar"}
    std2dBarPolar = {"plot_type":"bar_polar"}
    std2dLine = {"plot_type":"line"}
    std2dArea = {"plot_type":"area"}
    std2dSunburst = {"plot_type":"sunburst"}
    std2dScatterTernary = {"plot_type":"scatter_ternary"}
    std2dScatterMatrix = {"plot_type":"scatter_matrix"}
    std3dLine = {"plot_type":"line_3d"}
    std3dHistogram = {"plot_type":"histogram"}
    std_cat = {"color":"category","animation_frame":"year","text":"category","animation_group":"category"}#,"range_x":[-0.15,1.15],"range_y":[-0.15,1.15],"range_z":[-0.15,1.15]} #"parents":"category" url_TLD   #
    std_st = {"color":"source_title","animation_frame":"year","text":"source_title","animation_group":"source_title"}#,"range_x":[-0.15,1.15],"range_y":[-0.15,1.15],"range_z":[-0.15,1.15]} #"parents":"category" url_TLD#,"range_x":[-1,2],"range_y":[-1,2],"range_z":[-1,2]

    # _2d_test_0 = base|std2dLine|{"x":"year","y":"tb.sent_pa","color":"category" ,"title":"line y = vs.comp_aj","xtitle":"time","ytitle":"volume","histnorm":'percent',"histfunc":'avg',"barmode":'group',"trendline":"ols","text":"category"} ## "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year","marginal":"rug" 
    # _2d_test_1 = base|std2dLine|{"x":"year","y":"0_tsne","color":"category" ,"title":"line y = al.pos","xtitle":"time","ytitle":"volume","histnorm":'percent',"histfunc":'avg',"barmode":'group',"trendline":"ols","text":"category"} ## "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year","marginal":"rug" 
    # _2d_test_2 = base|std2dLine|{"x":"year","y":"0_pca","color":"category" ,"title":"line y = al.neg","xtitle":"time","ytitle":"volume","histnorm":'percent',"histfunc":'avg',"barmode":'group',"trendline":"ols","text":"category"} ## "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year","marginal":"rug" 
    #_2d_test_2 = base|std2dScatter|{"x":"0_tsne","y":"1_tsne","color":"category","title":"Scatter 0_tsne","animation_frame":"year","text":"category","animation_group":"category"} #"parents":"category" url_TLD
    #_3d_test_2 = base|std3dScatter|{"x":"0_tsne","y":"1_tsne", "z":"2_tsne","color":"category","title":"Scatter plot of dimention reduced embeding data from articles- TSNE","xtitle":"Component 1","ytitle":"Component 2","ztitle":"Component 3"}
    # _2d_test_0 = base|std2dScatter|std_cat|{"x":"0_tsne_aj","y":"1_tsne_aj","title":"Category : Scatter TSNE","xtitle":"0","ytitle":"1"} #"parents":"category" url_TLD
    # _2d_test_1 = base|std2dScatter|std_cat|{"x":"0_pca_aj","y":"1_pca_aj","title":"Category : Scatter PCA","xtitle":"0","ytitle":"1"} #"parents":"category" url_TLD
    # _2d_test_2 = base|std2dScatter|std_cat|{"x":"tb.pol_aj","y":"tb.sub_aj","title":"Category : Scatter Polarity/Subjectivity","xtitle":"Polarity","ytitle":"Subjectivity"} #"parents":"category" url_TLD
    # _2d_test_3 = base|std2dScatter|std_cat|{"x":"al.pos_aj","y":"al.pos_k_aj","title":"Category : Scatter Positivity/Keyword","xtitle":"Positivity","ytitle":"Positivity from Keyword"} #"parents":"category" url_TLD
    # _2d_test_0 = base|std2dScatter|std_st|{"x":"0_tsne_aj","y":"1_tsne_aj","title":"Source Title : Scatter TSNE"} #"parents":"category" url_TLD
    # _2d_test_1 = base|std2dScatter|std_st|{"x":"0_pca_aj","y":"1_pca_aj","title":"Source Title : Scatter PCA"} #"parents":"category" url_TLD
    # _2d_test_2 = base|std2dScatter|std_st|{"x":"tb.pol_aj","y":"tb.sub_aj","title":"Source Title : Scatter Polarity/Subjectivity","xtitle":"Polarity","ytitle":"Subjectivity"} #"parents":"category" url_TLD
    # _2d_test_3 = base|std2dScatter|std_st|{"x":"al.pos_aj","y":"al.pos_k_aj","title":"Source Title : Scatter Positivity/Keyword","xtitle":"Positivity","ytitle":"Positivity from Keyword"} #"parents":"category" url_TLD

    # _3d_test_0 = base|std3dScatter|stdCat|{"x":"0_tsne_aj","y":"1_tsne_aj", "z":"2_tsne_aj","title":"Scatter plot of dimention reduced embeding data from articles- TSNE","xtitle":"Component 1","ytitle":"Component 2","ztitle":"Component 3"}
    # _3d_test_1 = base|std3dScatter|stdCat|{"x":"0_pca_aj","y":"1_pca_aj", "z":"2_pca_aj","title":"Scatter plot of dimention reduced embeding data from articles- PCA","xtitle":"Component 1","ytitle":"Component 2","ztitle":"Component 3"}
    # _3d_test_2 = base|std3dLine|{"x":"0_pca_aj","y":"1_pca_aj", "z":"year","color":"source_title","title":"std3dLine","xtitle":"Component 1","ytitle":"Component 2","ztitle":"Component 3","markers":True}

    _2d_test_0 = base|std2dLine|{"x":"year_month","y":"tb.pol_aj","color":"category" ,"title":"line y = tb.pol_aj","xtitle":"time","ytitle":"volume","histnorm":'percent',"histfunc":'avg',"barmode":'group',"trendline":"ols","text":"category","trendline_scope":"overall"} ## "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year","marginal":"rug" 
    _2d_test_1 = base|std2dLine|{"x":"year_month","y":"tb.sub_aj","color":"category" ,"title":"line y = tb.sub_aj","xtitle":"time","ytitle":"volume","histnorm":'percent',"histfunc":'avg',"barmode":'group',"trendline":"ols","text":"category","trendline_scope":"overall"} ## "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year","marginal":"rug" 
    _2d_test_2 = base|std2dLine|{"x":"year_month","y":"al.pos_aj","color":"category" ,"title":"line y = al.pos_aj","xtitle":"time","ytitle":"volume","histnorm":'percent',"histfunc":'avg',"barmode":'group',"trendline":"ols","text":"category","trendline_scope":"overall"} ## "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year","marginal":"rug" 

    # #_2d_test_0 = base|std2dSunburst|{"names":"category","title":"Sunburst of categories/source"} #"parents":"category" url_TLD
    # _2d_test_0 = base|std2dScatter|{"x":"tb.pol_aj","y":"tb.pol_k_aj","color":"category","title":"Scatter tb.pol_aj and tb.pol_k_aj"} #"parents":"category" url_TLD
    # _2d_test_1 = base|std2dScatter|{"x":"tb.sub_aj","y":"tb.sub_k_aj","color":"category","title":"Scatter tb.sub_aj and tb.sub_k_aj"} #"parents":"category" url_TLD
    # _2d_test_2 = base|std2dScatter|{"x":"tb.pos_aj","y":"tb.pos_k_aj","color":"category","title":"Scatter tb.pos_aj and tb.pos_k_aj"} #"parents":"category" url_TLD
    # _2d_test_4 = base|std2dScatter|{"x":"tb.neg_aj","y":"tb.neg_k_aj","color":"category","title":"Scatter tb.neg_aj and tb.neg_k_aj"} #"parents":"category" url_TLD
    # #_2d_test_3 = base|std2dLinePolar|{"r":"count","theta":"tb.sub_aj","color":"category","title":"Polar tb.pol_aj"} #"parents":"category" url_TLD
    # #_2d_test_3 = base|std2dSunburst|{"names":"category","title":"Sunburst of categories/source"} #"parents":"category" url_TLD
    # #_2d_test_3 =  base|std2dScatterMatrix|{"dimensions":"category","color":"category","title":"Matrix"}
    # _2d_test_3 =  base|std2dScatter|{"x":"tb.pol_aj","y":"tb.pol_k_aj","facet_row":"category","color":"category","title":"Matrix"}
   
    _2d_pie_0 = base|std2dPie|{"values":"tb.sent","names":"category","title":"Pie of categories","color_discrete_sequence":px.colors.sequential.RdBu}
    _2d_pie_1 = base|std2dPie|{"values":"tb.sent","names":"url_TLD","title":"Pie of categories","color_discrete_sequence":px.colors.sequential.RdBu}
    
    _2d_bar_0 = base|std2dBar|{"x":"source_title","y":"tb.sent","color":"category","barmode":"group","title":"Bar of categories","xtitle":"time","ytitle":"volume","log_y":True} #'group', 'overlay' or 'relative' y=count 
    _2d_bar_1 = base|std2dBar|{"x":"year_month","y":"tb.sent","color":"category","barmode":"relative","title":"Bar of categories","xtitle":"time","ytitle":"volume","log_y":True} #'group', 'overlay' or 'relative' y=count tb.sent
    _2d_bar_2 = base|std2dBar|{"x":"year_month","y":"tb.char","color":"category","title":"Bar of categories","xtitle":"time","ytitle":"volume"} #'group', 'overlay' or 'relative' y=count tb.sent
    
    _2d_histogram_0 = base|std3dHistogram|{"x":"year_month","y":"tb.neg_k","color":"category","barmode":"relative","title":"y = tb.neg","xtitle":"time","ytitle":"volume","histfunc":'avg'} #,"marginal":"rug","histfunc":'count'} #histnorm='percent' "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year"  "histfunc" :   'count' 'sum' 'avg' 'min' 'max'
    _2d_histogram_1 = base|std3dHistogram|{"x":"year_month","y":"vs.neg_k","color":"category","barmode":"relative","title":"y = vs.neg","xtitle":"time","ytitle":"volume","histfunc":'avg'} #,"marginal":"rug","histfunc":'avg'} #histnorm='percent' "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year"
    _2d_histogram_2 = base|std3dHistogram|{"x":"year_month","y":"ts.neg_k","color":"category","barmode":"relative","title":"y = ts.neg","xtitle":"time","ytitle":"volume","histfunc":'avg'} #,"marginal":"rug","histfunc":'avg'} #histnorm='percent' "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year"
    _2d_histogram_3 = base|std3dHistogram|{"x":"year_month","y":"tb.sent","color":"category","barmode":"relative","title":"y = tb.sent","xtitle":"time","ytitle":"volume","histfunc":'avg'} #,"marginal":"rug","histfunc":'avg'} #histnorm='percent' "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year"
    
    #_2d_line_0 = base|std2dLine|{"x":"year_month","y":"tb.sent","title":"line y = ts.pos","xtitle":"time","ytitle":"volume","histnorm":'percent',"histfunc":'avg',"barmode":'group'} ## "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year","marginal":"rug" ,"color":"category"
    #_2d_scatterMatrix_0 =  base|std2dScatterMatrix|{"dimensions":["tb.pol","tb.pol_k","tb.sub","tb.sub_k"]}
    _2d_emmbeding_0 = base|std2dScatter|{"x":"0_tsne","y":"1_tsne","color":"category","title":"Scatter plot of dimention reduced embeding data from articles - TSNE","xtitle":"Component 1","ytitle":"Component 2"}
    _3d_emmbeding_0 = base|std3dScatter|{"x":"0_tsne","y":"1_tsne", "z":"2_tsne","color":"category","title":"Scatter plot of dimention reduced embeding data from articles- TSNE","xtitle":"Component 1","ytitle":"Component 2","ztitle":"Component 3"}
    _2d_emmbeding_1 = base|std2dScatter|{"x":"0_pca","y":"1_pca","color":"category","title":"Scatter plot of dimention reduced embeding data from articles - PCA","xtitle":"Component 1","ytitle":"Component 2"}
    _3d_emmbeding_1 = base|std3dScatter|{"x":"0_pca","y":"1_pca", "z":"2_pca","color":"category","title":"Scatter plot of dimention reduced embeding data from articles - PCA","xtitle":"Component 1","ytitle":"Component 2","ztitle":"Component 3"}
    _2d_emmbeding_2 = base|std2dScatter|{"x":"0_ipca","y":"1_ipca","color":"category","title":"Scatter plot of dimention reduced embeding data from articles - IPCA","xtitle":"Component 1","ytitle":"Component 2"}
    _3d_emmbeding_2 = base|std3dScatter|{"x":"0_ipca","y":"1_ipca","z":"2_ipca","color":"category","title":"Scatter plot of dimention reduced embeding data from articles - PCA","xtitle":"Component 1","ytitle":"Component 2","ztitle":"Component 3"}
    _2d_emmbeding_3 = base|std2dScatter|{"x":"0_nnt","y":"1_nnt","color":"category","title":"Scatter plot of dimention reduced embeding data from articles - NNT","xtitle":"Component 1","ytitle":"Component 2"}
    _3d_emmbeding_3 = base|std3dScatter|{"x":"0_nnt","y":"1_nnt", "z":"2_nnt","color":"category","title":"Scatter plot of dimention reduced embeding data from articles - NNT","xtitle":"Component 1","ytitle":"Component 2","ztitle":"Component 3"}

    _2d_sent_0 = base|std2dScatter|{"x":"tb.pol","y":"tb.sub","color":"category","title":"Scatter plot of the 'polarity' and 'objectivity' of article data","xtitle":"Polarity (-1 to 1)","ytitle":"Subjectivity (0-1)","marginal_x":MARGINAL_LIST[3],"marginal_y":MARGINAL_LIST[4]}
    _3d_sent_0 = base|std3dScatter|{"x":"tb.pol","y":"tb.sub","z":"ts.pos","color":"category","title":"Scatter plot of the 'polarity', 'objectivity' and 'positivity' of article data","xtitle":"Polarity (0-1)","ytitle":"Subjectivity (0-1)","ztitle":"Positivity (0-1)"}
    _2d_sent_1 = base|std2dScatter|{"x":"tb.pol_k","y":"tb.sub_k","color":"category","title":"Scatter plot of the 'polarity' and 'objectivity' of keywork data","xtitle":"Polarity (-1 to 1)","ytitle":"Subjectivity (0-1)","marginal_x":MARGINAL_LIST[3],"marginal_y":MARGINAL_LIST[4]}
    _3d_sent_1 = base|std3dScatter|{"x":"tb.pol_k","y":"tb.sub_k","z":"ts.pos_k","color":"category","title":"Scatter plot of the 'polarity', 'objectivity' and 'positivity' of keywork data","xtitle":"Polarity (0-1)","ytitle":"Subjectivity (0-1)","ztitle":"Positivity (0-1)"}

    _2d_pos_0 = base|std2dScatter|{"x":"tb.pos","y":"vs.pos","color":"category","title":"Scatter plot of the positivity (TB & VS) of article data","xtitle":"Pos TextBlob","ytitle":"Pos Vader Sentiment","marginal_x":MARGINAL_LIST[3],"marginal_y":MARGINAL_LIST[4]}
    _3d_pos_0 = base|std3dScatter|{"x":"tb.pos","y":"vs.pos","z":"ts.pos","color":"category","title":"Scatter plot of the positivity (TB & VS & TS) of article data","xtitle":"Pos TextBlob","ytitle":"Pos Vader Sentiment","ztitle":"Pos Transformer"}
    _2d_pos_1 = base|std2dScatter|{"x":"tb.pos_k","y":"vs.pos_k","color":"category","title":"Scatter plot of the positivity (TB & VS) of keywork data","xtitle":"Pos TextBlob","ytitle":"Pos Vader Sentiment","marginal_x":MARGINAL_LIST[3],"marginal_y":MARGINAL_LIST[4]}
    _3d_pos_1 = base|std3dScatter|{"x":"tb.pos_k","y":"vs.pos_k","z":"ts.pos_k","color":"category","title":"Scatter plot of the positivity (TB & VS & TS) of keywork data","xtitle":"Pos TextBlob","ytitle":"Pos Vader Sentiment","ztitle":"Pos Transformer"}

    _2d_line_0 = base|std2dLine|{"x":"year_month","y":"tb.pol_aj","color":"category" ,"title":"line y = tb.pol_aj","xtitle":"time","ytitle":"volume","histnorm":'percent',"histfunc":'avg',"barmode":'group',"trendline":"ols","text":"category","trendline_scope":"overall"} ## "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year","marginal":"rug" 
    _2d_line_1 = base|std2dLine|{"x":"year_month","y":"tb.sub_aj","color":"category" ,"title":"line y = tb.sub_aj","xtitle":"time","ytitle":"volume","histnorm":'percent',"histfunc":'avg',"barmode":'group',"trendline":"ols","text":"category","trendline_scope":"overall"} ## "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year","marginal":"rug" 
    _2d_line_2 = base|std2dLine|{"x":"year_month","y":"al.pos_aj","color":"category" ,"title":"line y = al.pos_aj","xtitle":"time","ytitle":"volume","histnorm":'percent',"histfunc":'avg',"barmode":'group',"trendline":"ols","text":"category","trendline_scope":"overall"} ## "y":"tb.char" , histfunc='avg', marginal="rug" box`, `violin`  barmode 'group', 'overlay' or 'relative'  ,"nbins":1  cumulative True ,"animation_frame":"year","marginal":"rug" 

    
    _2d_len = base|std2dScatter|{"x":"tb.sent","y":"tb.noun","color":"category","title":"Scatter plot of text nature and volume data from articles","xtitle":"Number of Sentences","ytitle":"Number of Nouns"}
    _3d_len = base|std3dScatter|{"x":"tb.sent","y":"tb.noun", "z":"tb.word","color":"category","title":"Scatter plot of text nature and volume data from articles","xtitle":"Number of Sentences","ytitle":"Number of Nouns","ztitle":"Number of Words"}
    
    _2dList_out = []
    _3dList_out = [] 
    
    
    _2dList_out = _2dList_out + [_2d_emmbeding_0]*emmbeding[0]+[_2d_emmbeding_1]*emmbeding[1]+[_2d_emmbeding_2]*emmbeding[2]+[_2d_emmbeding_3]*emmbeding[3]+ [_2d_sent_0]*sentType[0]+[_2d_sent_1]*sentType[1]+ [_2d_pos_0]*posType[0]+[_2d_pos_1]*posType[1]+ [_2d_len]*length+ [_2d_line_0,_2d_line_1,_2d_line_2]
    _3dList_out = _3dList_out + [_3d_emmbeding_0]*emmbeding[0]+[_3d_emmbeding_1]*emmbeding[1]+[_3d_emmbeding_2]*emmbeding[2]+[_3d_emmbeding_3]*emmbeding[3]+ [_3d_sent_0]*sentType[0]+[_3d_sent_1]*sentType[1]+ [_3d_pos_0]*posType[0]+[_3d_pos_1]*posType[1]+ [_3d_len]*length
    # _2dList_out = _2dList_out + [_2d_emmbeding_0]*emmbeding[0]+[_2d_emmbeding_1]*emmbeding[1]+[_2d_emmbeding_2]*emmbeding[2]+[_2d_emmbeding_3]*emmbeding[3]
    # _3dList_out = _3dList_out + [_3d_emmbeding_0]*emmbeding[0]+[_3d_emmbeding_1]*emmbeding[1]+[_3d_emmbeding_2]*emmbeding[2]+[_3d_emmbeding_3]*emmbeding[3]
    # _2dList_out = _2dList_out + [_2d_sent_0]*sentType[0]+[_2d_sent_1]*sentType[1]
    # _3dList_out = _3dList_out + [_3d_sent_0]*sentType[0]+[_3d_sent_1]*sentType[1]
    # _2dList_out = _2dList_out + [_2d_pos_0]*posType[0]+[_2d_pos_1]*posType[1]
    # _3dList_out = _3dList_out + [_3d_pos_0]*posType[0]+[_3d_pos_1]*posType[1]
    # _2dList_out = _2dList_out + [_2d_len]*length+[_2d_pie_0,_2d_histogram_0,_2d_histogram_1,_2d_histogram_2,_2d_histogram_3,_2d_pie_1,_2d_line_0]*exp ##_2d_bar_0,_2d_bar_1,_2d_bar_2,
    # _3dList_out = _3dList_out + [_3d_len]*length
    if "empty" not in _2d_test_0.keys() :
        _2dList_out = _2dList_out + [_2d_test_0]
    if "empty" not in _2d_test_1.keys() :
        _2dList_out = _2dList_out + [_2d_test_1]
    if "empty" not in _2d_test_2.keys() :
        _2dList_out = _2dList_out + [_2d_test_2]
    if "empty" not in _2d_test_3.keys() :
        _2dList_out = _2dList_out + [_2d_test_3]
    if "empty" not in _2d_test_4.keys() :
        _2dList_out = _2dList_out + [_2d_test_4]
    if "empty" not in _2d_test_5.keys() :
        _2dList_out = _2dList_out + [_2d_test_5]
    if "empty" not in _2d_test_6.keys() :
        _2dList_out = _2dList_out + [_2d_test_6]
    if "empty" not in _2d_test_7.keys() :
        _2dList_out = _2dList_out + [_2d_test_7]
    if "empty" not in _2d_test_8.keys() :
        _2dList_out = _2dList_out + [_2d_test_8]
    if "empty" not in _2d_test_9.keys() :
        _2dList_out = _2dList_out + [_2d_test_9]
    if "empty" not in _3d_test_0.keys() :
        _3dList_out = _3dList_out + [_3d_test_0]
    if "empty" not in _3d_test_1.keys() :
        _3dList_out = _3dList_out + [_3d_test_1]
    if "empty" not in _3d_test_2.keys() :
        _3dList_out = _3dList_out + [_3d_test_2]
    if "empty" not in _3d_test_3.keys() :
        _3dList_out = _3dList_out + [_3d_test_3]
    if "empty" not in _3d_test_4.keys() :
        _3dList_out = _3dList_out + [_3d_test_4]
    if "empty" not in _3d_test_5.keys() :
        _3dList_out = _3dList_out + [_3d_test_5]
    if "empty" not in _3d_test_6.keys() :
        _3dList_out = _3dList_out + [_3d_test_6]
    if "empty" not in _3d_test_7.keys() :
        _3dList_out = _3dList_out + [_3d_test_7]
    if "empty" not in _3d_test_8.keys() :
        _3dList_out = _3dList_out + [_3d_test_8]
    if "empty" not in _3d_test_9.keys() :
        _3dList_out = _3dList_out + [_3d_test_9]
    # if length :
    #     _2dList_out = _2dList_out + [_2d_len]
    #     _3dList_out = _3dList_out + [_3d_len]
        
    # if experimental :
    #     _2dList_out = _2dList_out + _2dList_exp
    #     _3dList_out = _3dList_out + _3dList_exp
    # if basic :
    #     _2dList_out = _2dList_out + _2dList_basic
    #     _3dList_out = _3dList_out + _3dList_basic
    # if test :
    #     _2dList_out = _2dList_out + _2dList_test
    #     _3dList_out = _3dList_out + _3dList_test
    return _2dList_out*dimentions[0], _3dList_out*dimentions[1]


def savePlotList(plot_list,path,label="default") :
    current_timestamp = datetime.datetime.now()
    current_timestamp = str(current_timestamp.strftime("%Y,%m,%d;%H,%M,%S"))
    for i in range(len(plot_list)) :
        plotSave(plot_list[i],path,label+"_n="+str(i)+"_d="+current_timestamp)

def plotSave(plot,path,filename) :
    plot.write_html(path+filename+".html")
    
def plotWithConf(cd={}):
    if "plot_type" in cd.keys() :
        plot_type = cd["plot_type"]
    else :
        plot_type = "unknown"
    if plot_type=="scatter":
        fig = px.scatter(data_frame=cd['data_frame'], x=cd['x'], y=cd['y'], color=cd['color'], symbol=cd['symbol'], size=cd['size'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], text=cd['text'], facet_row=cd['facet_row'], facet_col=cd['facet_col'], facet_col_wrap=cd['facet_col_wrap'], facet_row_spacing=cd['facet_row_spacing'], facet_col_spacing=cd['facet_col_spacing'], error_x=cd['error_x'], error_x_minus=cd['error_x_minus'], error_y=cd['error_y'], error_y_minus=cd['error_y_minus'], animation_frame=cd['animation_frame'], animation_group=cd['animation_group'], category_orders=cd['category_orders'], labels=cd['labels'], orientation=cd['orientation'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], color_continuous_scale=cd['color_continuous_scale'], range_color=cd['range_color'], color_continuous_midpoint=cd['color_continuous_midpoint'], symbol_sequence=cd['symbol_sequence'], symbol_map=cd['symbol_map'], opacity=cd['opacity'], size_max=cd['size_max'], marginal_x=cd['marginal_x'], marginal_y=cd['marginal_y'], trendline=cd['trendline'], trendline_options=cd['trendline_options'], trendline_color_override=cd['trendline_color_override'], trendline_scope=cd['trendline_scope'], log_x=cd['log_x'], log_y=cd['log_y'], range_x=cd['range_x'], range_y=cd['range_y'], render_mode=cd['render_mode'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    elif plot_type=="scatter_3d":
        fig = px.scatter_3d(data_frame=cd['data_frame'], x=cd['x'], y=cd['y'], z=cd['z'], color=cd['color'], symbol=cd['symbol'], size=cd['size'], text=cd['text'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], error_x=cd['error_x'], error_x_minus=cd['error_x_minus'], error_y=cd['error_y'], error_y_minus=cd['error_y_minus'], error_z=cd['error_z'], error_z_minus=cd['error_z_minus'], animation_frame=cd['animation_frame'], animation_group=cd['animation_group'], category_orders=cd['category_orders'], labels=cd['labels'], size_max=cd['size_max'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], color_continuous_scale=cd['color_continuous_scale'], range_color=cd['range_color'], color_continuous_midpoint=cd['color_continuous_midpoint'], symbol_sequence=cd['symbol_sequence'], symbol_map=cd['symbol_map'], opacity=cd['opacity'], log_x=cd['log_x'], log_y=cd['log_y'], log_z=cd['log_z'], range_x=cd['range_x'], range_y=cd['range_y'], range_z=cd['range_z'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    elif plot_type=="pie":
        fig = px.pie(data_frame=cd['data_frame'], names=cd['names'], values=cd['values'], color=cd['color'], facet_row=cd['facet_row'], facet_col=cd['facet_col'], facet_col_wrap=cd['facet_col_wrap'], facet_row_spacing=cd['facet_row_spacing'], facet_col_spacing=cd['facet_col_spacing'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], category_orders=cd['category_orders'], labels=cd['labels'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'], opacity=cd['opacity'], hole=cd['hole'])
        fig.update_traces(textposition='inside', textinfo='percent+label',marker=dict(line=dict(color='#000000', width=2)))
    elif plot_type=="bar":
        fig = px.bar(data_frame=cd['data_frame'], x=cd['x'], y=cd['y'], color=cd['color'], pattern_shape=cd['pattern_shape'], facet_row=cd['facet_row'], facet_col=cd['facet_col'], facet_col_wrap=cd['facet_col_wrap'], facet_row_spacing=cd['facet_row_spacing'], facet_col_spacing=cd['facet_col_spacing'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], text=cd['text'], base=cd['base'], error_x=cd['error_x'], error_x_minus=cd['error_x_minus'], error_y=cd['error_y'], error_y_minus=cd['error_y_minus'], animation_frame=cd['animation_frame'], animation_group=cd['animation_group'], category_orders=cd['category_orders'], labels=cd['labels'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], color_continuous_scale=cd['color_continuous_scale'], pattern_shape_sequence=cd['pattern_shape_sequence'], pattern_shape_map=cd['pattern_shape_map'], range_color=cd['range_color'], color_continuous_midpoint=cd['color_continuous_midpoint'], opacity=cd['opacity'], orientation=cd['orientation'], barmode=cd['barmode'], log_x=cd['log_x'], log_y=cd['log_y'], range_x=cd['range_x'], range_y=cd['range_y'], text_auto=cd['text_auto'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    elif plot_type=="scatter_polar":
        fig = px.scatter_polar(data_frame=cd['data_frame'], r=cd['r'], theta=cd['theta'], color=cd['color'], symbol=cd['symbol'], size=cd['size'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], text=cd['text'], animation_frame=cd['animation_frame'], animation_group=cd['animation_group'], category_orders=cd['category_orders'], labels=cd['labels'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], color_continuous_scale=cd['color_continuous_scale'], range_color=cd['range_color'], color_continuous_midpoint=cd['color_continuous_midpoint'], symbol_sequence=cd['symbol_sequence'], symbol_map=cd['symbol_map'], opacity=cd['opacity'], direction=cd['direction'], start_angle=cd['start_angle'], size_max=cd['size_max'], range_r=cd['range_r'], range_theta=cd['range_theta'], log_r=cd['log_r'], render_mode=cd['render_mode'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    elif plot_type=="line_polar":
        fig = px.line_polar(data_frame=cd['data_frame'], r=cd['r'], theta=cd['theta'], color=cd['color'], line_dash=cd['line_dash'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], line_group=cd['line_group'], text=cd['text'], symbol=cd['symbol'], animation_frame=cd['animation_frame'], animation_group=cd['animation_group'], category_orders=cd['category_orders'], labels=cd['labels'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], line_dash_sequence=cd['line_dash_sequence'], line_dash_map=cd['line_dash_map'], symbol_sequence=cd['symbol_sequence'], symbol_map=cd['symbol_map'], markers=cd['markers'], direction=cd['direction'], start_angle=cd['start_angle'], line_close=cd['line_close'], line_shape=cd['line_shape'], render_mode=cd['render_mode'], range_r=cd['range_r'], range_theta=cd['range_theta'], log_r=cd['log_r'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    elif plot_type=="bar_polar":
        fig = px.bar_polar(data_frame=cd['data_frame'], r=cd['r'], theta=cd['theta'], color=cd['color'], pattern_shape=cd['pattern_shape'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], base=cd['base'], animation_frame=cd['animation_frame'], animation_group=cd['animation_group'], category_orders=cd['category_orders'], labels=cd['labels'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], color_continuous_scale=cd['color_continuous_scale'], pattern_shape_sequence=cd['pattern_shape_sequence'], pattern_shape_map=cd['pattern_shape_map'], range_color=cd['range_color'], color_continuous_midpoint=cd['color_continuous_midpoint'], barnorm=cd['barnorm'], barmode=cd['barmode'], direction=cd['direction'], start_angle=cd['start_angle'], range_r=cd['range_r'], range_theta=cd['range_theta'], log_r=cd['log_r'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    elif plot_type=="line":
        fig = px.line(data_frame=cd['data_frame'], x=cd['x'], y=cd['y'], line_group=cd['line_group'], color=cd['color'], line_dash=cd['line_dash'], symbol=cd['symbol'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], text=cd['text'], facet_row=cd['facet_row'], facet_col=cd['facet_col'], facet_col_wrap=cd['facet_col_wrap'], facet_row_spacing=cd['facet_row_spacing'], facet_col_spacing=cd['facet_col_spacing'], error_x=cd['error_x'], error_x_minus=cd['error_x_minus'], error_y=cd['error_y'], error_y_minus=cd['error_y_minus'], animation_frame=cd['animation_frame'], animation_group=cd['animation_group'], category_orders=cd['category_orders'], labels=cd['labels'], orientation=cd['orientation'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], line_dash_sequence=cd['line_dash_sequence'], line_dash_map=cd['line_dash_map'], symbol_sequence=cd['symbol_sequence'], symbol_map=cd['symbol_map'], markers=cd['markers'], log_x=cd['log_x'], log_y=cd['log_y'], range_x=cd['range_x'], range_y=cd['range_y'], line_shape=cd['line_shape'], render_mode=cd['render_mode'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    elif plot_type=="area":
        fig = px.area(data_frame=cd['data_frame'], x=cd['x'], y=cd['y'], line_group=cd['line_group'], color=cd['color'], pattern_shape=cd['pattern_shape'], symbol=cd['symbol'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], text=cd['text'], facet_row=cd['facet_row'], facet_col=cd['facet_col'], facet_col_wrap=cd['facet_col_wrap'], facet_row_spacing=cd['facet_row_spacing'], facet_col_spacing=cd['facet_col_spacing'], animation_frame=cd['animation_frame'], animation_group=cd['animation_group'], category_orders=cd['category_orders'], labels=cd['labels'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], pattern_shape_sequence=cd['pattern_shape_sequence'], pattern_shape_map=cd['pattern_shape_map'], symbol_sequence=cd['symbol_sequence'], symbol_map=cd['symbol_map'], markers=cd['markers'], orientation=cd['orientation'], groupnorm=cd['groupnorm'], log_x=cd['log_x'], log_y=cd['log_y'], range_x=cd['range_x'], range_y=cd['range_y'], line_shape=cd['line_shape'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    elif plot_type=="sunburst":
        fig = px.sunburst(data_frame=cd['data_frame'], names=cd['names'], values=cd['values'], parents=cd['parents'], path=cd['path'], ids=cd['ids'], color=cd['color'], color_continuous_scale=cd['color_continuous_scale'], range_color=cd['range_color'], color_continuous_midpoint=cd['color_continuous_midpoint'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], labels=cd['labels'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'], branchvalues=cd['branchvalues'], maxdepth=cd['maxdepth'])
    elif plot_type=="scatter_ternary":
        fig = px.scatter_ternary(data_frame=cd['data_frame'], a=cd['a'], b=cd['b'], c=cd['c'], color=cd['color'], symbol=cd['symbol'], size=cd['size'], text=cd['text'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], animation_frame=cd['animation_frame'], animation_group=cd['animation_group'], category_orders=cd['category_orders'], labels=cd['labels'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], color_continuous_scale=cd['color_continuous_scale'], range_color=cd['range_color'], color_continuous_midpoint=cd['color_continuous_midpoint'], symbol_sequence=cd['symbol_sequence'], symbol_map=cd['symbol_map'], opacity=cd['opacity'], size_max=cd['size_max'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    elif plot_type=="scatter_matrix":
        fig = px.scatter_matrix(data_frame=cd['data_frame'], dimensions=cd['dimensions'], color=cd['color'], symbol=cd['symbol'], size=cd['size'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], category_orders=cd['category_orders'], labels=cd['labels'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], color_continuous_scale=cd['color_continuous_scale'], range_color=cd['range_color'], color_continuous_midpoint=cd['color_continuous_midpoint'], symbol_sequence=cd['symbol_sequence'], symbol_map=cd['symbol_map'], opacity=cd['opacity'], size_max=cd['size_max'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    elif plot_type=="line_3d":
        fig = px.line_3d(data_frame=cd['data_frame'], x=cd['x'], y=cd['y'], z=cd['z'], color=cd['color'], line_dash=cd['line_dash'], text=cd['text'], line_group=cd['line_group'], symbol=cd['symbol'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], custom_data=cd['custom_data'], error_x=cd['error_x'], error_x_minus=cd['error_x_minus'], error_y=cd['error_y'], error_y_minus=cd['error_y_minus'], error_z=cd['error_z'], error_z_minus=cd['error_z_minus'], animation_frame=cd['animation_frame'], animation_group=cd['animation_group'], category_orders=cd['category_orders'], labels=cd['labels'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], line_dash_sequence=cd['line_dash_sequence'], line_dash_map=cd['line_dash_map'], symbol_sequence=cd['symbol_sequence'], symbol_map=cd['symbol_map'], markers=cd['markers'], log_x=cd['log_x'], log_y=cd['log_y'], log_z=cd['log_z'], range_x=cd['range_x'], range_y=cd['range_y'], range_z=cd['range_z'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    elif plot_type=="histogram":
        fig = px.histogram(data_frame=cd['data_frame'], x=cd['x'], y=cd['y'], color=cd['color'], pattern_shape=cd['pattern_shape'], facet_row=cd['facet_row'], facet_col=cd['facet_col'], facet_col_wrap=cd['facet_col_wrap'], facet_row_spacing=cd['facet_row_spacing'], facet_col_spacing=cd['facet_col_spacing'], hover_name=cd['hover_name'], hover_data=cd['hover_data'], animation_frame=cd['animation_frame'], animation_group=cd['animation_group'], category_orders=cd['category_orders'], labels=cd['labels'], color_discrete_sequence=cd['color_discrete_sequence'], color_discrete_map=cd['color_discrete_map'], pattern_shape_sequence=cd['pattern_shape_sequence'], pattern_shape_map=cd['pattern_shape_map'], marginal=cd['marginal'], opacity=cd['opacity'], orientation=cd['orientation'], barmode=cd['barmode'], barnorm=cd['barnorm'], histnorm=cd['histnorm'], log_x=cd['log_x'], log_y=cd['log_y'], range_x=cd['range_x'], range_y=cd['range_y'], histfunc=cd['histfunc'], cumulative=cd['cumulative'], nbins=cd['nbins'], text_auto=cd['text_auto'], title=cd['title'], template=cd['template'], width=cd['width'], height=cd['height'])
    else:
        print("ERROR",plot_type)
    layout_updated=False
    if "ztitle" in cd.keys():
        if type(cd["ztitle"])!=type(None)  :
            fig.update_layout(title=cd["title"],width=PLOT_WIDTH,height=PLOT_HEIGHT,autosize=PLOT_AUTOSIZE,scene = dict(bgcolor="#FFFFFF",xaxis=dict(title=cd["xtitle"], color="#000000", gridcolor="#888888",gridwidth=5),yaxis=dict(title=cd["ytitle"], color="#000000", gridcolor="#888888",gridwidth=5),zaxis=dict(title=cd["ztitle"], color="#000000", gridcolor="#888888",gridwidth=5)))
            layout_updated=True
    if not layout_updated :
        fig.update_layout(title=cd["title"], xaxis_title=cd["xtitle"], yaxis_title=cd["ytitle"],width=PLOT_WIDTH,height=PLOT_HEIGHT,autosize=PLOT_AUTOSIZE,bargap=0) #, zaxis_title=conf_dict["ztitle"]

    if not DO_NOT_RENDER_PLOT :
        if type(cd["browser"]) == type(None) :
            fig.show()
        else :
            fig.show(renderer="browser")
    return fig
    

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
#     # df = px.data.gapminder().query("country=="Canada"")
#     fig = px.line(df, x="year", y="words", color="category")#_month , symbol="url_TLD"_mont
#     fig.show()
# #     fig = px.line(df, x="year_month", title="TEST",color="category")
# #     fig.show()



rename_dict = {"hash_key":"Primary Key", "title_quer":"Main Title","link":"Google Link","published":"Date Publication","source_url":"Real Link","source_title":"Name of the source","category":"Main Category","year":"Year","year_month":"Year & Month", "url_TLD":"url_extention","keywords_list":"List Of Keywords"}


# ## Dimention reduction

# def dimReduction_TSNE(df, n_components=2, perplexity=30, early_exaggeration =4.0,learning_rate=1000,n_iter=1000,verbose=0,random_state=0,norm_output=False):
#     tsne = TSNE(n_components=n_components, perplexity=perplexity,early_exaggeration = early_exaggeration,learning_rate =learning_rate,n_iter=n_iter,verbose=verbose,random_state=random_state)
#     out_df = tsne.fit_transform(df)
#     out_df = pd.DataFrame(out_df)
#     if norm_output :
#         out_df = dfNormalize(out_df)
#     return out_df

# def dimReduction_PCA(df, n_components=2, svd_solver="auto",tol=0.0,whiten=False,random_state=0,norm_output=False):
#     pca = PCA(n_components=n_components,svd_solver=svd_solver,tol=tol,random_state=random_state)
#     out_df = pca.fit_transform(df)
#     out_df = pd.DataFrame(out_df)
#     if norm_output :
#         out_df = dfNormalize(out_df)
#     return out_df

# def dimReduction_IPCA(df, n_components=2,whiten=False, batch_size=100,norm_output=False):
#     ipca = IncrementalPCA(n_components=n_components,batch_size=batch_size)
#     out_df = ipca.fit_transform(df)
#     out_df = pd.DataFrame(out_df)
#     if norm_output :
#         out_df = dfNormalize(out_df)
#     return out_df

# def dimReduction_NNT(df,n_components=2,n_neighbors=5,mode="distance",algorithm="auto",leaf_size=30,p=2,eigen_solver="auto",tol=0.0,metric="minkowski",n_jobs=None,norm_output=False):
#     #KTN mode : "distance" "connectivity"
#     #KTN algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
#     #BOTH n_jobs : None -1
#     #BOTH metric = ["minkowski",  "manhattan","cityblock","l1",  "euclidean","l2",  "cosine",  "haversine",  "nan_euclidean"] "precomputed" ?
#     #ISO eigen_solver : ["auto", "arpack", "dense"]
#     cache_path = tempfile.gettempdir()
#     knt = KNeighborsTransformer(mode=mode,n_neighbors=n_neighbors,algorithm=algorithm,leaf_size=leaf_size,metric=metric,p=p,n_jobs=n_jobs)
#     iso = Isomap(n_components=n_components,n_neighbors=n_neighbors,eigen_solver=eigen_solver,metric=metric,tol=tol,p=p,n_jobs=n_jobs)# 
#     nnt = make_pipeline(knt,iso,memory=cache_path)
#     out_df = nnt.fit_transform(df)
#     out_df = pd.DataFrame(out_df)
#     if norm_output :
#         out_df = dfNormalize(out_df)
#     return out_df

# def generateDirRed(mat_emb,df_main,n_components=2,norm_output=False,active_sel=[True,False,False,False]) :
#     out_list = []
#     out_list_label = []
#     if active_sel[0] :
#         df_tsne = dimReduction_TSNE(mat_emb,n_components,norm_output=norm_output)
#         df_tsne_j = df_main.join(df_tsne, how="inner")
#         out_list.append(df_tsne_j)
#         out_list_label.append("TSNE")
#     if active_sel[1] :
#         df_pca = dimReduction_PCA(mat_emb,n_components,norm_output=norm_output)
#         df_pca_j = df_main.join(df_pca, how="inner")
#         out_list.append(df_pca_j)
#         out_list_label.append("PCA")
#     if active_sel[2] :
#         df_ipca = dimReduction_IPCA(mat_emb,n_components,norm_output=norm_output)
#         df_ipca_j = df_main.join(df_ipca, how="inner")
#         out_list.append(df_ipca_j)
#         out_list_label.append("IPCA")
#     if active_sel[3] :
#         df_nnt = dimReduction_NNT(mat_emb,n_components,norm_output=norm_output)
#         df_nnt_j = df_main.join(df_nnt, how="inner")
#         out_list.append(df_nnt_j)
#         out_list_label.append("NNT")

#     return out_list, out_list_label

def selectAlgoDfList(df_main,n_components=2,norm_output=False,active_sel=[True,False,False,False]) :
    algo_list = []
    out_list = []
    out_list_label = []
    df_cols = list(df_main.columns)
    if active_sel[0] :
        algo_list.append("tsne")
    if active_sel[1] :
        algo_list.append("pca")
    if active_sel[2] :
        algo_list.append("ipca")
    if active_sel[3] :
        algo_list.append("nnt")
    for algo in algo_list :
        if n_components==2 and ("0_"+algo in df_cols) and ("1_"+algo in df_cols) :
            df_sel = df_main[["0_"+algo, "1_"+algo]]
            df_ren = df_sel.rename(columns={"0_"+algo: 0, "1_"+algo: 1})
            df_ren = df_main.join(df_ren, how="inner")
        if n_components==3 and ("0_"+algo in df_cols) and ("1_"+algo in df_cols) and ("2_"+algo in df_cols) :
            df_sel = df_main[["0_"+algo, "1_"+algo, "2_"+algo]]
            df_ren = df_sel.rename(columns={"0_"+algo: 0, "1_"+algo: 1, "2_"+algo: 2})
            df_ren = df_main.join(df_ren, how="inner")
        out_list.append(df_ren)
        out_list_label.append(algo.upper())
    return out_list, out_list_label


# ALL = {data_frame,color,height,width,template,title,labels,hover_name,hover_data,custom_data,color_discrete_sequence,color_discrete_map}
# 2D = {x,y,log_x,log_y,range_x,range_y}
# ER2D = {error_x,error_x_minus,error_y,error_y_minus}
# 3D = {x,y,z,log_x,log_y,log_z,range_x,range_y,range_z}
# ER3D = {error_x,error_x_minus,error_y,error_y_minus,error_z,error_z_minus}
# POL = {r,theta,direction,start_angle,range_r,range_theta,log_r}
# FAC = {facet_row,facet_col,facet_col_wrap,facet_row_spacing,facet_col_spacing}
# LINE = {line_dash,line_group,line_dash_sequence,line_dash_map}


#    _2d_sent_1 = base|{"x":"tb.pos","y":"tb.pos_k","color":"category","title":"Scatter plot of the "polarity" and "objectivity" of keywork data","xtitle":"Polarity (-1 to 1)","ytitle":"Subjectivity (0-1)","marginal_x":MARGINAL_LIST[3],"marginal_y":MARGINAL_LIST[4]}
    # _2d_embd_exp = {"x":0,"y":1,"color":"category","animation_frame":"year_month","title":"test_year_month","browser":True}
              
    # _3d_embd_exp = {"x":0,"y":1,"z":2,"color":"category","animation_frame":"year_month","title":"test_year_month","browser":True}
    # _2dList_exp = [_2d_embd_exp]
    # _3dList_exp = [_3d_embd_exp]
    
    # _2d_sent_basic = base|{"x":"tb.pol","y":"tb.sub","color":"category","title":"Scatter plot of the "polarity" and "objectivity" of article data","xtitle":"Polarity (-1 to 1)","ytitle":"Subjectivity (0-1)","marginal_x":MARGINAL_LIST[3],"marginal_y":MARGINAL_LIST[4]}
    # _3d_sent_basic = base|{"x":"tb.pol","y":"tb.sub","z":"ts.pos","color":"category","title":"Scatter plot of the "polarity", "objectivity" and "positivity" of article data","xtitle":"Polarity (0-1)","ytitle":"Subjectivity (0-1)","ztitle":"Positivity (0-1)"}
    # _2dList_basic = [_2d_embd_basic ,_2d_sent_basic ,_2d_len_basic]
    # _3dList_basic = [_3d_embd_basic ,_3d_sent_basic, _3d_len_basic]
    
    # _2d_embd_test = base|{"x":0,"y":1,"color":"category","title":"Scatter plot of dimention reduced embeding data from articles","xtitle":"Component 1","ytitle":"Component 2"}
    # _3d_embd_test = base|{"x":0,"y":1,"z":2,"color":"category","title":"Scatter plot of dimention reduced embeding data from articles","xtitle":"Component 1","ytitle":"Component 2"}
    
    # _2dList_test = [_2d_embd_test]
    # _3dList_test = [_3d_embd_test]

# import main_var
# mv = main_var.main_var()
# from article_scraping_lib import *
# df_main = openDFcsv(mv.join2_path,mv.join2_filename+"_stats")
# # df_main = openDFcsv(mv.join2_path,mv.join2_filename)
# # print(df_main.shape)
# #df_main2 = df_main.drop(["count","category"] ,axis= 1)

# # df_main2 = df_main[["tb.pol_aj","tb.sub_aj","tb.pos_aj","vs.pos_aj","ts.pos_aj","al.pos_aj","vs.neu_aj","ts.neu_aj","vs.comp_aj","tb.pol_k_aj","tb.sub_k_aj","tb.polaj_aj","tb.pos_k_aj","vs.pos_k_aj","ts.pos_k_aj","al.pos_k_aj","vs.neu_k_aj","ts.neu_k_aj","vs.comp_k_aj"]]
# # plt_px = px.imshow(df_main2,y=df_main['category'].tolist(),color_continuous_scale='RdBu_r')#, zmin=[0.5,0.35,0.65,0.22],zmax=[0.54,0.42,0.78,0.34])
# # plt_px.show(renderer="browser")

# #calculateStatsNLP(df_main,"category")
# # plotDFstatisticsQuerry(df_main,60)
# # _2dList_out, _3dList_out = getRenderLists()
# # plt_2dList = renderAllOptions(df_main,_2dList_out)
# # plt_3dList = renderAllOptions(df_main,_3dList_out)
# #df_source = df_main2['source_title'].value_counts().to_frame("count").sort_values(by=['count'],ascending=True)


# # plotDFstatisticsQuerry(df_main)
# colors = np.array(["red","green","blue","yellow","pink","cyan","orange","purple","gray"])
# df_cat = df_main.set_index("category")#[["tb.pol"]]# ["category"].to_frame("tb.pol_aju").sort_values(by=['category'],ascending=True)
# # print(type(df_cat))
# # print(df_cat)
# fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(40, 60))
# df_cat.plot.bar(y="tb.pol_aju", edgecolor='white', linewidth=5,use_index=True,ax=ax[0,0], legend=False, ylabel="Polarity", xlabel="Category", fontsize= 0, title="",color=colors,layout=(1,1),stacked=False,width=0.85) #, figsize=(5, 5)
# df_main.plot.scatter(y="tb.pol_k_aju",x="category", s = 150,c="black",ax=ax[0,0], legend=False,layout=(2, 1))
# #fig = df_cat.plot.barh(y="tb.pol_k_aju",use_index=True,ax=ax[0,0], legend=False, ylabel="Polarity 2", xlabel="Category", fontsize= 0, title="",color=colors,layout=(1,1),stacked=False) #, figsize=(5, 5)
# # fig = df_cat.plot.bar(y="tb.sub_aju",use_index=True,ax=axes[1,1], legend=True, ylabel="Polarity", xlabel="Category", fontsize= 2, title="") #, figsize=(5, 5)
# # # # #import matplotlib.pyplot as plt

# #fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(7, 15))
# #ax[1,1].scatter(df_cat["0_tsne"], df_cat["1_tsne"], 10,c=colors)#, c=df["color"]
# df_cat.plot.scatter(x = "0_tsne", y = "1_tsne", s = 50, xlabel="TSNE 1", ylabel="TSNE 2",c=colors,ax=ax[1,1])
# plt.show()

def getColorList(length,random_col=False,second_list=True):
    color_list = []
    color_list2 = []
    for i in range(length):
        if random_col :
            color = [random.random(), random.random(), random.random()]
            #color_list.append('#%06X' % randint(0, 0xFFFFFF))
            #color_list.append([random.random(), random.random(), random.random()])
        else :
            color = colorsys.hsv_to_rgb(float(i/length),0.8,0.9)
        color_list.append(color)
        if second_list :
            color2 = np.asarray(color)*0.5
            color_list2.append(color2)
    colour_out = color_list+color_list2
    return colour_out
        

    return color_list
def displayCatStats(df_input,agg_col_input="category",filter_col=None,filter_value=None,display_col_bar = False,disp=0) :
    df = copy.deepcopy(df_input)
    agg_col_origin = copy.deepcopy(agg_col_input)
    agg_col = copy.deepcopy(agg_col_input)
    if type(filter_col)!=type(None) and type(filter_value)!=type(None) and filter_col in df.columns:
        df = df.loc[df[filter_col]==filter_value]
        df.drop([filter_col], axis=1)
        if type(agg_col)==type([]) :
            agg_col.remove(filter_col)
            agg_col = str(agg_col[0])
    df[agg_col]=df[agg_col].astype(str, copy = False)
    count_col = "count_nlp"
    total_initial_entry = df[count_col].sum()
    entry_num = df.shape[0]
    fig_size = 60
    ball_size = fig_size*20
    title_font_size=30
    field_font_size = 12#int(title_font_size/2)
    suffix="_aj"#
    algo_list = ["tsne","pca","ipca"]
    col_list_nlp_name = ["Polarity","Subjectivity","Positivity (TB)","Positivity (VS)","Neutrality (VS)","Neutrality (TS)","Positivity (TS)","Positivity (AVG)"]
    col_list_nlp = ["tb.pol","tb.sub","tb.pos","vs.pos","vs.neu","ts.neu","ts.pos","al.pos"]
    if display_col_bar :
        xticks=None
    else :
        xticks=""
    list_num = [x for x in range(0, entry_num)]
    col_list_nlp_suff = []
    col_list_nlp_suff_keyword = []
    for col in col_list_nlp :
        col_list_nlp_suff.append(col+suffix)
        col_list_nlp_suff_keyword.append(col+"_k"+suffix)
    colors_all = getColorList(entry_num)#colors = ["red","green","blue","yellow","pink","cyan","orange","purple","gray"]
    colors =colors_all[:entry_num]
    colors2 =colors_all[entry_num:entry_num*2]
    df_cat = df.set_index(agg_col)
    df_cat["xticks"]=[x.upper().replace("THE","").replace(" ","")[:4] for x in df_cat.index.str.slice(stop=10)]
    df["xticks"]=[x.upper().replace("THE","").replace(" ","")[:4] for x in df[agg_col].str.slice(stop=10)]
    size_one_graph = 5
    if disp==0 :
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(int(4*size_one_graph*(entry_num/7)), 4*size_one_graph), gridspec_kw={'wspace':0.07,"hspace":0.07},constrained_layout = True)#(fig_size, fig_size)
        fig.suptitle('Statistics about '+str(agg_col_origin)+"   (n="+str(total_initial_entry)+")  (filter: "+str(filter_col)+"="+str(filter_value)+")", fontsize=title_font_size)#,y=0.96horizontalalignment="center", verticalalignment="center",
        for i in range(len(col_list_nlp_suff)) :
            df_cat.plot.bar(y=col_list_nlp_suff[i],use_index=True,subplots=True,ax=ax[int(i/4),i%4], legend=False, title="", xlabel="", fontsize= field_font_size,stacked=False,width=0.85,color=colors,grid=True,rot=1) #, figsize=(5, 5)  ,xticks=list_num,layout=(1,1)
            df.plot.scatter(y=col_list_nlp_suff_keyword[i],x=agg_col,subplots=True,ax=ax[int(i/4),i%4], s = int(ball_size/2),c=colors2, legend=False,layout=(1, 1), fontsize= field_font_size, title="", ylabel="", xlabel="",grid=True,rot=1)#,xticks=xticks)#
            ax[int(i/4),i%4].set_title(col_list_nlp_name[i],pad=field_font_size, fontdict={'fontsize':title_font_size})
            ax[int(i/4),i%4].set_xticklabels(df_cat.xticks)#df_cat.index, 
        ### Embbeding Stats
        for i in  range(len(algo_list)):
            df.plot.scatter(x="0_"+algo_list[i]+suffix,y="1_"+algo_list[i]+suffix,subplots=True,ax=ax[2,i+1], s = ball_size,c=colors, legend=False,layout=(1, 1), xlabel=algo_list[i].upper()+" 1", ylabel=algo_list[i].upper()+" 2", title="",fontsize= field_font_size,grid=True,rot=1)
            ax[2,i+1].set_title(algo_list[i].upper()+" (1-2)",pad=field_font_size, fontdict={'fontsize':title_font_size})
        for i in  range(len(algo_list)):
            df.plot.scatter(x="0_"+algo_list[i]+suffix,y="2_"+algo_list[i]+suffix,subplots=True,ax=ax[3,i+1], s = ball_size,c=colors, legend=False,layout=(1, 1), xlabel=algo_list[i].upper()+" 1", ylabel=algo_list[i].upper()+" 3", title="",fontsize= field_font_size,grid=True,rot=1)
            ax[3,i+1].set_title(algo_list[i].upper()+" (1-3)",pad=field_font_size, fontdict={'fontsize':title_font_size})
        ### Rest Stats
        df_cat.plot.bar(y="vs.comp"+suffix,use_index=True,subplots=True,ax=ax[2,0], legend=False, title="", xlabel="", ylabel="", fontsize= field_font_size,color=colors,layout=(1,1),stacked=False,width=0.85,xticks=None)#,xticks=(0,1,2,3,4,5,6,7,8)) #, figsize=(5, 5)  
        ax[2,0].set_title("Compound",pad=field_font_size, fontdict={'fontsize':title_font_size})
        ax[2,0].set_xticklabels(df_cat.xticks)
        df_cat.plot.pie(y=count_col,use_index=True,subplots=True,ax=ax[3,0], legend=False, title="", xlabel="", ylabel="", fontsize= field_font_size,colors=colors,layout=(1,1),stacked=False,autopct="%.2f%%", explode=[0.03]*entry_num)#,xticks=(0,1,2,3,4,5,6,7,8)) #, figsize=(5, 5)  
        ax[3,0].set_title("Number of Articles",pad=field_font_size, fontdict={'fontsize':title_font_size})
    elif disp==1:

        fig, ax = plt.subplots(nrows=19, ncols=4, figsize=(int(4*size_one_graph*(entry_num/7)), 19*size_one_graph), gridspec_kw={'wspace':0.07,"hspace":0.07},constrained_layout = True)#
        fig.suptitle('Statistics about '+str(agg_col_origin)+"   (n="+str(total_initial_entry)+")  (filter: "+str(filter_col)+"="+str(filter_value)+")", fontsize=title_font_size)#,y=0.96horizontalalignment="center", verticalalignment="center",
        col_list=["tb.pol","tb.sub","tb.pos","tb.neg","vs.pos","vs.neu","vs.neg","vs.comp","ts.pos","ts.neu","ts.neg","al.pos","al.neg","tb.char_pa","tb.char_ps","tb.pol_aj","tb.sub_aj","tb.pos_aj","tb.neg_aj","vs.pos_aj","vs.neu_aj","vs.neg_aj","vs.comp_aj","ts.pos_aj","ts.neu_aj","ts.neg_aj","al.pos_aj","al.neg_aj","tb.word_pa","tb.word_ps","tb.pol_k","tb.sub_k","tb.pos_k","tb.neg_k","vs.pos_k","vs.neu_k","vs.neg_k","vs.comp_k","ts.neg_k","ts.neu_k","ts.pos_k","al.pos_k","al.neg_k","tb.noun_pa","tb.noun_ps","tb.pol_k_aj","tb.sub_k_aj","tb.pos_k_aj","tb.neg_k_aj","vs.pos_k_aj","vs.neu_k_aj","vs.neg_k_aj","vs.comp_k_aj","ts.neg_k_aj","ts.neu_k_aj","ts.pos_k_aj","al.pos_k_aj","al.neg_k_aj","tb.sent_pa","count_nlp"]#,"tb.char_pw"
        #list_name = [name[:3] for name in df_cat.index.tolist()] #xticks
        
        for x in range(4) :
            for y in range(15) :
                #print(df_cat[col_list[15*x+y]].min())
                df_cat.plot.bar(y=col_list[15*x+y],use_index=True,subplots=True,ax=ax[y,x], legend=False, title="", xlabel="", ylabel="", fontsize= field_font_size,stacked=False,width=0.85,color=colors,grid=True,rot=1)#,ylim=(0,1),color=colors) #, figsize=(5, 5)  ,xticks=list_num,xticks=xticks,xticks=tuple(list_name)
                ax[y,x].set_title(col_list[15*x+y],pad=field_font_size, fontdict={'fontsize':title_font_size})
                ax[y,x].set_xticklabels(df_cat.xticks)#df_cat.index, 
        dim_red_x=["0_tsne","0_pca","0_ipca","0_tsne_aj","0_pca_aj","0_ipca_aj","0_tsne","0_pca","0_ipca","0_tsne_aj","0_pca_aj","0_ipca_aj"]
        dim_red_y=["1_tsne","1_pca","1_ipca","1_tsne_aj","1_pca_aj","1_ipca_aj","2_tsne","2_pca","2_ipca","2_tsne_aj","2_pca_aj","2_ipca_aj"]
        for x in range(4) :
            for y in range(15,18) :
                i=y-15
                if display_col_bar :
                    df.plot.line(x=dim_red_x[3*x+i],y=dim_red_y[3*x+i],subplots=True,ax=ax[y,x],color="black", legend=False,layout=(1, 1), xlabel=dim_red_x[3*x+i], ylabel=dim_red_y[3*x+i], title="",fontsize= field_font_size,grid=True)
                df.plot.scatter(x=dim_red_x[3*x+i],y=dim_red_y[3*x+i],subplots=True,ax=ax[y,x], s = ball_size,c=colors, legend=False,layout=(1, 1), xlabel=dim_red_x[3*x+i], ylabel=dim_red_y[3*x+i], title="",fontsize= field_font_size,grid=True)
                ax[y,x].set_title(dim_red_x[3*x+i]+" to "+dim_red_y[3*x+i][0:1],pad=field_font_size, fontdict={'fontsize':title_font_size})
        text_kwargs = dict(ha='center', va='center', fontsize=10, color='C1')
        ax[18,0].text(x=0, y=0, s="hello\nhello2", **text_kwargs)
        ax[18,1].text(x=0.5, y=0.5,s="hello\nhello2", **text_kwargs)
        # fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(fig_size/4, fig_size),constrained_layout = True)
        # df_cat["vs.pos_new"]=df_cat["vs.pos"]/(df_cat["vs.pos"]+df_cat["vs.neg"])
        # df_cat["vs.neg_new"]=df_cat["vs.neg"]/(df_cat["vs.pos"]+df_cat["vs.neg"])
        # df_cat["ts.pos_new"]=df_cat["ts.pos"]/(df_cat["ts.pos"]+df_cat["ts.neg"])
        # df_cat["ts.neg_new"]=df_cat["ts.neg"]/(df_cat["ts.pos"]+df_cat["ts.neg"])
        # df_cat["vs.pos_new"]=df_cat["vs.pos_new"]-(df_cat["vs.pos_new"].sum()/df_cat.shape[0])
        # df_cat["vs.neg_new"]=df_cat["vs.neg_new"]-(df_cat["vs.neg_new"].sum()/df_cat.shape[0])
        # df_cat["ts.pos_new"]=df_cat["ts.pos_new"]-(df_cat["ts.pos_new"].sum()/df_cat.shape[0])
        # df_cat["ts.neg_new"]=df_cat["ts.neg_new"]-(df_cat["ts.neg_new"].sum()/df_cat.shape[0])
        # print(0.5-df_cat["vs.pos_new"].sum()/df_cat.shape[0])
        # df_cat.plot.bar(y=["tb.pos","tb.neg"],use_index=True,ax=ax[0], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green","red"])
        # # df_cat.plot.bar(y=["vs.pos","vs.neu","vs.neg"],use_index=True,ax=ax[1], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green",'orange',"red"])
        # # df_cat.plot.bar(y=["ts.pos","ts.neu","ts.neg"],use_index=True,ax=ax[2], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green",'orange',"red"])
        # # df_cat.plot.bar(y=["vs.pos_new","vs.neg_new"],use_index=True,ax=ax[1], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green","red"],ylim=(0,0.1))
        # # df_cat.plot.bar(y=["ts.pos_new","ts.neg_new"],use_index=True,ax=ax[2], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green","red"],ylim=(0,0.1))
        # df_cat.plot.bar(y="vs.pos_new",use_index=True,ax=ax[1], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color="black")
        # df_cat.plot.bar(y="ts.pos_new",use_index=True,ax=ax[2], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color="black")

        # # df_cat.plot.bar(y=["vs.pos_aj","vs.neg_aj"],use_index=True,ax=ax[1], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green","red"])
        # # df_cat.plot.bar(y=["ts.pos_aj","ts.neg_aj"],use_index=True,ax=ax[2], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green","red"])


        # fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(fig_size/4, fig_size),constrained_layout = True)
        # df_cat["vs.pos_new"]=df_cat["vs.pos"]/(df_cat["vs.pos"]+df_cat["vs.neg"])
        # df_cat["ts.pos_new"]=df_cat["ts.pos"]/(df_cat["ts.pos"]+df_cat["ts.neg"])
        # df_cat["tb.pos_new"]=df_cat["tb.pos"]-(df_cat["tb.pos"].sum()/df_cat.shape[0])
        # df_cat["vs.pos_new"]=df_cat["vs.pos_new"]-(df_cat["vs.pos_new"].sum()/df_cat.shape[0])
        # df_cat["ts.pos_new"]=df_cat["ts.pos_new"]-(df_cat["ts.pos_new"].sum()/df_cat.shape[0])
        # print(0.5-df_cat["vs.pos_new"].sum()/df_cat.shape[0])
        # df_cat.plot.bar(y="tb.pos_new",use_index=True,ax=ax[0], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color="gray",logy=True)
        # # df_cat.plot.bar(y=["vs.pos","vs.neu","vs.neg"],use_index=True,ax=ax[1], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green",'orange',"red"])
        # # df_cat.plot.bar(y=["ts.pos","ts.neu","ts.neg"],use_index=True,ax=ax[2], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green",'orange',"red"])
        # # df_cat.plot.bar(y=["vs.pos_new","vs.neg_new"],use_index=True,ax=ax[1], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green","red"],ylim=(0,0.1))
        # # df_cat.plot.bar(y=["ts.pos_new","ts.neg_new"],use_index=True,ax=ax[2], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green","red"],ylim=(0,0.1))
        # df_cat.plot.bar(y="vs.pos_new",use_index=True,ax=ax[1], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color="black")
        # df_cat.plot.bar(y="ts.pos_new",use_index=True,ax=ax[2], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color="black")

        # # df_cat.plot.bar(y=["vs.pos_aj","vs.neg_aj"],use_index=True,ax=ax[1], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green","red"])
        # # df_cat.plot.bar(y=["ts.pos_aj","ts.neg_aj"],use_index=True,ax=ax[2], legend=False, title="", xlabel="", fontsize= field_font_size,layout=(1,1),stacked=True,width=0.95,color=["green","red"])
        pass
    plt.show()
    return fig
    
def lineTimeSent(df) :
    col_list = ["tb.pol_aju","tb.sub_aju","tb.pos_aju","vs.pos_aju","ts.pos_aju","al.pos_aju","vs.neu_aju","ts.neu_aju"]
    fig = px.line(df, x="year", y=col_list, title="Sorted Input", text="year", orientation="h",markers=True,line_shape ="spline",color="year") #,facet_row =["2010","2011","2012","2013","2014","2015","2016"]
    fig.show(renderer="browser")

# import main_var
# mv = main_var.main_var()
# graph_folder = "graphs/"
# agg_list = ["category","year","source_title"]#,["year","category"],["year","source_title"]]
# field_list = [False,True,False]#,True,True]#,["year","category"],["year","source_title"]]
# count = 0
# for agg in agg_list :
#     df_main = openDFcsv(mv.visu_path,mv.visu_filename+"_"+str(agg))
#     if type(agg)!=type([]) :
#         fig = displayCatStats(df_main,agg,display_col_bar=field_list[count])##
#         fig.savefig(mv.visu_path+graph_folder+mv.visu_filename+"_"+agg+".png")
#     else :
#         filter_list = list(set(df_main[agg[1]].tolist()))
#         print(filter_list)
#         for filter_val in filter_list :
#             fig = displayCatStats(df_main,agg,agg[1],filter_val,display_col_bar=field_list[count])##
#             fig.savefig(mv.visu_path+graph_folder+mv.visu_filename+"_"+str(agg)+"_"+str(filter_val)+".png")
#     count = count +1

#lineTimeSent(df_main)
#figd.savefig(mv.visu_path+mv.visu_filename+"category"+".png")




# import main_var
# mv = main_var.main_var()
# from utils_art import *
# from embedding_keyword_module_lib import *
# word_cloud_folder="word_cloud/"
# px_graph_folder = "px_plots/"
# mat_graph_folder = "test/"
# #df_main = openDFcsv(mv.visu_path,"visu_file_['year', 'source_title']")


# # df_main = openDFcsv(mv.join2_path,mv.join2_filename)
# # calculateStatsNLP(df_main)

# # df_main = openDFcsv(mv.visu_path,"visu_file_['year_month', 'category']")
# # _2dList_out, _3dList_out = getRenderLists(browser=False)
# # plt_2dList = renderAllOptions(df_main,_2dList_out)
# # plt_3dList = renderAllOptions(df_main,_3dList_out)
# # savePlotList(plt_2dList,mv.visu_path+px_graph_folder,"2D_graphs")
# # savePlotList(plt_3dList,mv.visu_path+px_graph_folder,"3D_graphs")

# agg_list = ["category","year","source_title",["year","category"],["year","source_title"]]
# field_list = [False,True,False,True,True]#,["year","category"],["year","source_title"]]
# count = 0
# for agg in agg_list :
#     df_main = openDFcsv(mv.visu_path,mv.visu_filename+"_"+str(agg))
#     for disp in [0,1]:
#         fig = displayCatStats(df_main,agg,display_col_bar=field_list[count],disp=disp)##
#         fig.savefig(mv.visu_path+mat_graph_folder+mv.visu_filename+"_"+agg+"_"+str(disp)+".png")

# from embedding_keyword_module_lib import *
# df_main = openDFcsv(mv.join2_path,mv.join2_filename)
# #print(df_main["word_combined_all_sel"].mode())
# for cat in _TOPIC_LIST:
#     generateWordCloud(df_main,"word_combined_all",cat)
#par_list = parse_keywords_list(df_main,"word_combined_all_sel",False)
#print(par_list)
#generateKeywordFrequency(df_main,)
#"word_combined_all"
#word_combined_all_sel

