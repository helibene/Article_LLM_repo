# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:14:15 2024

@author: Alexandre
"""
import main_var
mv = main_var.main_var()
from pygooglenews import GoogleNews
import pandas as pd
from dateparser import parse as parse_date
import matplotlib.pyplot as plt
from utils_art import openDFcsv,openSTRtxt,openDFxlsx,saveDFcsv,saveSTRtxt,openConfFile,deleteUnnamed,display_df
import hashlib
import os
import embedding_keyword_module_lib as ekml
from datetime import timedelta
pd.set_option('expand_frame_repr', False)


_TOPIC_LIST = ["TOP","WORLD","NATION","BUSINESS","TECHNOLOGY","ENTERTAINMENT","SCIENCE","SPORTS","HEALTH"] #"CAAqJQgKIh9DQkFTRVFvSUwyMHZNR3QwTlRFU0JXVnVMVWRDS0FBUAE"
_STANDARD_SCRAPPING_FIELDS = ["title","link", "published", "source_url", "source_title", "category", "year", "year_month","pk","url_list","url_TLD", "hash_key"]

log_filename = "log"
language = "en"
country = "US"

main_path = mv.query_path  # "C:/Users/User/OneDrive/Desktop/article/files_3/1_1_query_main/"+env
filename = mv.query_filename  #  "reduction_test"

scarp_path = mv.scarp_path  #  "C:/Users/User/OneDrive/Desktop/article/files_3/1_2_scarp_main/"+env
scarp_filename = mv.scarp_filename

join1_path = mv.join1_path
join1_filename = mv.join1_filename

join2_path = mv.join2_path
join2_filename = mv.join2_filename

embdedding_path=mv.embdedding_path
embdedding_filename=mv.embdedding_filename

embdedding_path_raw = mv.embdedding_path_raw
embdedding_filename_raw = mv.embdedding_filename_raw

keyword_path = mv.keyword_path
keyword_filename = mv.keyword_filename

join2_path = mv.join2_path
join2_filename = mv.join2_filename



# path_union_stat = join1_path"C:/Users/User/OneDrive/Desktop/article/files_3/1_4_join_main/"+env
# filename_union_stat = join1_filename"join_main_test-filter"

# llm_result = "C:/Users/User/OneDrive/Desktop/article/file_2/test_llm_output/new_embedding/"

# llm_join_out = "C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/"
# llm_join_out_filename = "article_stats_embedding"


################################## QUERRY TO GET ARTICLE LIST

gn = GoogleNews(lang = language, country = country)
log_file = open(main_path+log_filename+".txt", "a", encoding="utf-8")

def getHeadlindTopic(gn, search = True, topic_sel=1, date1 = '2015-01-01', date2 = '2016-01-01') :
    if search :
        headlines_raw = gn.search(_TOPIC_LIST[topic_sel],from_=date1,to_=date2)
    else :
        if topic_sel == 0:
            headlines_raw = gn.top_news()
        else :
            headlines_raw = gn.topic_headlines(_TOPIC_LIST[topic_sel])
    return headlines_raw["entries"]

def rawHeadListToListList(rawHeadList, category="n/a") :
    new_list_d = []
    list_list_output = []
    for entry in rawHeadList :
        new_list_d.append({key: entry[key] for key in ["title", "link", "published", "source"]})
    for entry in new_list_d :
        list_for_entry = []
        link_str_list = entry["source"]["href"].split(".")
        link_str_list[0] = link_str_list[0].replace("https://","").replace("http://","").replace("www","")
        if link_str_list[0] == "" :
            link_str_list = link_str_list[1:]
        published_date = parse_date(entry["published"]).date()
        pk = entry["link"].split("articles/")[1].replace("?oc=5","")
        hash_key = hashlib.shake_256(str(pk).encode()).hexdigest(20)
        
        list_for_entry.append(entry["title"]) # "title"
        list_for_entry.append(entry["link"]) # "link"
        list_for_entry.append(published_date) # "published"
        list_for_entry.append(entry["source"]["href"]) # "source_url"
        list_for_entry.append(entry["source"]["title"]) # "source_title"
        list_for_entry.append(category) # "category"
        list_for_entry.append(published_date.year) # "year"
        list_for_entry.append(str(published_date.strftime('%Y-%m'))) # "year_month"
        list_for_entry.append(pk) # "pk"
        list_for_entry.append(link_str_list[:-1]) # "url_list"
        list_for_entry.append(link_str_list[-1]) # "url_TLD"
        list_for_entry.append(hash_key) # "hash_key"
        
        list_list_output.append(list_for_entry)
        
    return list_list_output

def fillDFwithListList(df, ListList):
    for ent in ListList:
        df.loc[len(df)] = ent
    return df

def fullWorkflow(gn, df, search = False, topic_sel=1, date1 = '2015-01-01', date2 = '2016-01-01') :
    headlines_raw = getHeadlindTopic(gn, search, topic_sel, date1, date2)
    headlinesMat = rawHeadListToListList(headlines_raw, _TOPIC_LIST[topic_sel])
    df_out = fillDFwithListList(df, headlinesMat)
    return df_out, len(headlinesMat)

def getStandardDf(col_list=[]):
    return pd.DataFrame([], columns = col_list) 

def addToDFHeadlinesParamList(gn, topic_list=[0], date_list_1 = ['2015-01-01'], date_list_2 = ['2016-01-01'], display=True,save=True,iteration=-1,use_date=True):
    if iteration != -1:
        iteration_str = str(iteration)
    else :
        iteration_str = ""
    if display :
        expected_article = logStartWorkflow(topic_list,date_list_1,date_list_2)
    df_out = getStandardDf(_STANDARD_SCRAPPING_FIELDS)
    date_list_len = len(date_list_1)
    for j in range(date_list_len) :
        ar_count = 0
        for i in topic_list :
            df_out, ar_num = fullWorkflow(gn, df_out, use_date , i, date_list_1[j], date_list_2[j])
            ar_count = ar_count + ar_num
        if display :
            log("     - Sample num "+str(j+1)+" done ("+str(ar_count)+" articles found)")
    if save :
        if display :
            log(" - Saving result table : '"+str(main_path)+str(filename)+"_"+iteration_str+".csv'")
        # df_out = deleteUnnamed(df_out,"hash_key")
        # df_out = df_out.set_index('hash_key')
        saveDFcsv(df_out, main_path, filename+"_"+iteration_str,False)
    if display :
        logEndWorkflow(df_out.shape[0], expected_article)
    return df_out

def loop_scraping(number_topics=8, startDate='2010-01-01', endDate='2024-01-01', sampling_1=14, sampling_2=6, display=True,save_steps=True,save_final=True):
    total_article_count = 0
    topic_list = list(range(0,number_topics))
    date_list_start, date_list_end = splitDateList(generateDateList(startDate=startDate, endDate=endDate,sampling=sampling_1))
    df_out = getStandardDf(_STANDARD_SCRAPPING_FIELDS)
    if display :
        expected_article = logStartWorkflow(topic_list,date_list_start,date_list_end,True)
    for i in range(len(date_list_start)) :
        date_list_entry_start, date_list_entry_end = splitDateList(generateDateList(startDate=date_list_start[i], endDate=date_list_end[i],sampling=sampling_2))
        df = addToDFHeadlinesParamList(gn, topic_list, date_list_entry_start, date_list_entry_end,display,save_steps,i)
        total_article_count = total_article_count + df.shape[0]
        df_out = pd.concat([df_out, df], ignore_index=True)
    logEndWorkflow(total_article_count, expected_article*sampling_2,True)
    if save_final :
        # saveDFcsv(df_out, main_path, filename+"_final",False)
        # df_out = df_out.set_index('hash_key')
        df_out = deleteUnnamed(df_out,"hash_key")
        saveDFcsv(df_out, main_path, filename,False)
        print("Final file saved here :",main_path)
    return df_out

################################## LOGS
def logStartWorkflow(topic_list,date_list_1,date_list_2,loop_module=False):
    min_date = parse_date(date_list_1[0])
    max_date = parse_date(date_list_2[-1])
    total_period = (max_date-min_date).days
    sampling_period = calculateRatio(total_period,len(date_list_1))
    expected_article = 100*len(topic_list)*len(date_list_1)
    articles_per_day = calculateRatio(expected_article,total_period)
    if loop_module :
        log("   -===-   -===-   Loop Scrapping module start   -===-   -===-   ")
    else :
        log("   -===-   Scrapping module start   -===-   ")
    log(" - List of topics : '"+str(topic_list)+"'  ("+str(len(topic_list))+")")
    log(" - From '"+str(min_date)+"'' to '"+str(max_date)+"' ("+str(total_period)+" days)")
    log(" - Sampling this numebr of periods : '"+str(len(date_list_1))+"'  ("+sampling_period+" days)")
    log(" - Expected articles found : '"+str(expected_article)+"''  ("+articles_per_day+"/day)")
    if loop_module :
        log("")
        log("")
    return expected_article
    
def logEndWorkflow(found_articles, expected_article,loop_module=False) :
    if loop_module :
        log("")
        log("")
    log(" - Total of '"+str(found_articles)+"' articles found ("+calculatePct(found_articles,expected_article)+"% of expected) !")
    if loop_module :
        log("   -===-   -===-   Loop Scrapping module end   -===-   -===-   ")
    else :
        log("   -===-   Scrapping module end   -===-   ")
        log("")

def displayStats(df,shape=True,schema=True,preview=False,mostCommon=1) :
    print("\n   ------   Raw dataframe   ------   \n")
    if shape :
        print("Shape : ",list(df.shape))
    if schema :
        print("Column type : ",df.dtypes)
    if preview :
        display_df(df.head(1))
    if mostCommon>0 :
        print("\n")
        print("\n   ------   Most common sources   ------   \n")
        print(df['source_title'].value_counts())
        print("\n")
    if mostCommon>1 :
        print("\n   ------   Most common topics   ------   \n")
        print(df['category'].value_counts())
        print("\n")
    if mostCommon>2 :
        print("\n   ------   Most common dates   ------   \n")
        print(df['published'].value_counts())
        print("\n")
    print("\n")
    
    
def log(string, print_str=True, log_str=True):
    if print_str :
        print(string)
    if log_str :
        log_file.write(string)
        log_file.write("\n")
        
################################## STATS CALC

def calculateStatsLength(df,groupping,display_df=True):
    rename_dict = {"text_len":"char_n","sentences":"sentence_n","noun_phrases":"noun_n","words":"words_n"}
    df = df.rename(columns=rename_dict)
    df_group = df[[groupping,"char_n","sentence_n","noun_n","words_n"]].groupby(groupping).sum(["char_n","sentence_n","noun_n","words_n"])
    df_count = df[groupping].value_counts().to_frame("count")#
    df_main = df_group.join(df_count, how="inner",on=groupping).sort_values(by=['count'],ascending=True)
    df_main[["char_per_count","sentence_per_count","noun_per_count","word_per_count"]] = df_main[["char_n","sentence_n","noun_n","words_n"]].div(df_main['count'], axis=0).astype(float)
    df_main[["char_per_sentence","noun_per_sentence","word_per_sentence"]] = df_main[["char_n","noun_n","words_n"]].div(df_main["sentence_n"], axis=0).astype(float)
    df_main[["char_per_word"]] = df_main[["char_n"]].div(df_main["words_n"], axis=0).astype(float)
    df_main = df_main.sort_values(by=["count"],ascending=True)
    if display_df :
        print("Dataframe Statistics Length Column :'"+groupping+"'")
        pd.display(df_main)
    return df_main

def calculateStatsNLP(df,groupping,display_df=True,display_stats=False,out_raw=False):
    rename_dict = {"polarity":"Polarity","subjectivity":"Subjectivity","pos1":"Positivity","neu1":"Neutrality","neg1":"Negativity","pos2":"Positivity2","neg2":"Negativity2","compound":"Compound"}
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
        df_main = df_main.sort_values(by=["count"],ascending=True)
    if display_df :
        print("Dataframe Statistics NLP Column :'"+groupping+"'")
        pd.display(df_main)
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


################################## UTILS ?

def loadFromFolder(folder_path="",force_schema=[],save=False, unicity_key="hash_key",cast_date_col="published"):
    if folder_path == "" :
        folder_path = main_path
    root_path = os.Path(folder_path)
    file_list = os.listdir(root_path)
    first_flag = True
    for filename_entry in file_list:
        if ".csv" in filename_entry :
            df = openDFcsv(folder_path,filename_entry.replace(".csv",""))
            if first_flag :
                first_flag = False
                df_out = df
            else :
                df_out = pd.concat([df_out, df], ignore_index=True)
    if "Unnamed: 0" in list(df_out.columns) :
        df_out = df_out.rename(columns={"Unnamed: 0":"index"})
    if force_schema!=[] :
        df_out = df_out[force_schema]
    if unicity_key != "" and unicity_key in list(df_out.columns) :
        df_out = df_out.drop_duplicates(subset=['hash_key'])
    if cast_date_col != "" and cast_date_col in list(df_out.columns) :
        df_out[cast_date_col+"_date_type"] = pd.to_datetime(df_out[cast_date_col])
    if save :
        df_out = df_out.set_index('hash_key')
        saveDFcsv(df_out, folder_path, filename+"_all_aggregated")
    return df_out

def generateDateList(startDate='2015-01-01', endDate='2020-01-01', sampling=10, includeEnd=True):
    date_list = []
    start = parse_date(startDate).date()
    end = parse_date(endDate).date()
    current_date = start
    interval_in_days = int((end-start).days/sampling)
    for i in range(sampling+int(includeEnd)) :
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date = current_date + timedelta(days = interval_in_days)
    return date_list

def splitDateList(date_list) :
    date_list_len = len(date_list)
    date_list1 = date_list[0:date_list_len-1]
    date_list2 = date_list[1:date_list_len]
    return date_list1, date_list2

def displayDF(df,col_name) :
    print(df[col_name].value_counts().describe([.05, .25, .5, .75, .95]))
    return df.shape[0], df[col_name].value_counts()

def getStandardDfInput(input_fields):
    return pd.DataFrame([], columns = input_fields)

################################## DISPLAY PLOTS

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
    
    
################################## MAIN JOINS

def joinQuerryAndParse(save=True,remove_invalid=True,display=True,filtered_input_df=False) :
    # rename_dict = {"title_q":"title_quer","title_p":"title_par","published_q":"published","year_q":"year","year_month_q":"year_month","source_url_q":"source_url","url_list_q":"url_list","url_TLD_q":"url_TLD","source_title_q":"source_title","category_q":"category","authors":"authors","keywords_list":"keywords_list","text_len_p":"text_len","tb.sentences":"tb.sentences","tb.noun_phrases":"tb.noun_phrases","tb.words":"tb.words","tb.polarity":"tb.polarity","tb.subjectivity":"tb.subjectivity","tb.p_pos":"tb.p_pos","tb.p_neg":"tb.p_neg","vs.neg":"vs.neg","vs.neu":"vs.neu","vs.pos":"vs.pos","vs.compound":"vs.compound","valid":"valid","link_q":"link","pk_q":"pk",}
    # rename_dict = {"title_q":"title_quer","title_p":"title_par","published_q":"published","year_q":"year","year_month_q":"year_month","source_url_q":"source_url","url_list_q":"url_list","url_TLD_q":"url_TLD","source_title_q":"source_title","category_q":"category","authors":"authors","keywords_list":"keywords_list","text_len_p":"text_len","tb.sentences":"sentences","tb.noun_phrases":"noun_phrases","tb.words":"words","tb.polarity":"polarity","tb.subjectivity":"subjectivity","tb.p_pos":"tb.pos","tb.p_neg":"tb.neg","vs.neg":"vs.neg","vs.neu":"vs.neu","vs.pos":"vs.pos","vs.compound":"vs.comp","valid":"valid","link_q":"link","pk_q":"pk"}
    rename_dict = {"pk_q":"pk"}
    df_q = openDFcsv(main_path,filename)
    df_q = deleteUnnamed(df_q,"hash_key")
    df_q_len = df_q.shape[0]
    if display :
        print("QUERRY dataset loaded from ",main_path)
        print("QUERRY dataset has entry length of :",df_q_len,"\n")
    df_p = openDFcsv(scarp_path,scarp_filename)
    df_p = deleteUnnamed(df_p,"hash_key")
    df_p_len = df_p.shape[0]
    if display :
        print("PARSSING dataset loaded from ",scarp_path)
        print("PARSSING dataset has entry length of :",df_p_len," ("+calculatePct(df_p_len,df_q_len)+"% of querry data)\n")
    df = df_q.join(df_p, how="inner",on='hash_key', lsuffix='_q', rsuffix='_p')
    df = df.rename(columns=rename_dict)
    df = df.drop_duplicates(subset=['pk'])
    # df = df.set_index('hash_key')
    # df = df[list(rename_dict.values())]
    join_df_len = df.shape[0]
    if display :
        print("JOINED dataset has entry length of :",join_df_len," ("+calculatePct(join_df_len,df_p_len)+"% of parssing data)")
    if remove_invalid :
        df = df.loc[(df['valid'] == True)]
        join_df_valid_len = df.shape[0]
        if display :
            print("JOINED dataset VALID entries :",join_df_valid_len," ("+calculatePct(join_df_valid_len,join_df_len)+"% of joined data)")
            print("JOINED dataset INVALID entries :",join_df_len-join_df_valid_len," ("+calculatePct(join_df_valid_len,join_df_len,ajust_for_denom=1)+"% of joined data)\n")
        join_df_len = join_df_valid_len
    if display :
        print("TOTAL yield : from",df_q_len," to ",join_df_len,"("+calculatePct(join_df_len,df_q_len)+"% yeald)\n")
    if save :
        # df = deleteUnnamed(df,"hash_key")
        saveDFcsv(df,join1_path,join1_filename)
        if display :
            print("JOINED dataset saved here :",join1_path+join1_filename+".csv")
    return df





################################## FILTERS AND SELECTION

def selectOnDf(df, date_start="2015-01-01", date_end="2017-06-01", categroy_list=[], source_list=[]) :
    df['published'] = df['published']
    # df = df.loc[(df['published'] >= parse_date(date_start).date()) & (df['published'] < parse_date(date_end).date())]
    if date_start != "" and date_end != "" :
        df = df.loc[(df['published'] >= date_start) & (df['published'] < date_end)]
    if categroy_list != [] :
        df = df[df['category'].isin(categroy_list)]
    if source_list != [] :
        df = df[df['source_title'].isin(source_list)]
    df = deleteUnnamed(df,"hash_key")
    return df

def filterQuerryDataset(df,thd_high=5000,thd_low=30, display_stats=True,display_end_stats=False,save=False) :
    ser_source = df['source_title'].value_counts() #.to_frame("count").sort_values(by=['count'],ascending=False)
    if thd_high > 0 and thd_high < 1 :
    #print(float(len(ser_source))*float(thd_high))
        thd_high = ser_source[int(float(len(ser_source))*float(thd_high))]
        # print(type(ser_source))
        # print(type(ser_source.values()))
        # print(list(ser_source.values())[int(float(len(ser_source))*float(thd_high))])
        
    ser_source_low = ser_source[ser_source < thd_low]
    ser_source_high = ser_source[ser_source > thd_high]
    list_source_low = list(ser_source_low.keys())
    list_source_high = list(ser_source_high.keys())
    df_wo_low = df[~df['source_title'].isin(list_source_low)]
    df_only_low = df[df['source_title'].isin(list_source_low)]
    # df_wo_high = df[~df['source_title'].isin(list_source_high)]
    df_only_high = df[df['source_title'].isin(list_source_high)]
    df_wo_low_high = df[~df['source_title'].isin(list_source_low+list_source_high)]
    # ser_source_wo_low = df_wo_low['source_title'].value_counts()
    # ser_source_wo_high = df_wo_high['source_title'].value_counts()
    # ser_source_only_high = df_only_high['source_title'].value_counts()
    df_high_start = getStandardDfInput(list(df.columns))
    for source_high in list_source_high :
        newdf = df.loc[(df['source_title'] == source_high)].sort_values(by=["hash_key"],ascending=False).head(thd_high)
        df_high_start = pd.concat([df_high_start,newdf]).reset_index(drop=True)
    final_df = pd.concat([df_wo_low_high,df_high_start]).reset_index(drop=True)
    if display_stats :
        num_entry_start, num_unique_start = displayDFsourceStats(df," - START -     ")
        displayDFsourceStats(df_only_low," - TOO LOW -   ")
        displayDFsourceStats(df_only_high," - TOO HIGH -  ")
        num_entry_wo_low, num_unique_wo_low = displayDFsourceStats(df_wo_low," - WO LOW -    ")
        displayDFsourceLoss(" - LOSS -      ",[num_entry_start, num_entry_wo_low],[num_unique_start, num_unique_wo_low])
    
    if display_end_stats :
        print(final_df['source_title'].value_counts())
    if save :
        df = deleteUnnamed(df,"hash_key")
        saveDFcsv(df,main_path,filename+"_backup_before_filter",True)
        final_df = deleteUnnamed(final_df,"hash_key")
        saveDFcsv(final_df,main_path,filename,True)
    return final_df


def randomSampleSelection(df,pct=50):
    df_sel_len = int(float(df.shape[0]*pct)/float(100))
    return df.sort_values(by=['hash_key'],ascending=False).head(df_sel_len)
    

def displayDFsourceStats(df,label=" - DEFAULT -   ") :
    num_entry = df.shape[0]
    num_unique = len(list(df['source_title'].value_counts()))
    print(label,str({"Articles sum":num_entry,
                 "Unique sources":num_unique,
                 "Unique/Sum":calculateRatio(num_entry,num_unique)}))
    return num_entry, num_unique

def displayDFsourceLoss(label=" - LOSS -      ",entry_ct_list=[],unique_ct_list=[]) :
    print(label,{"Articles sum":str(entry_ct_list[0]-entry_ct_list[1])+" (" + calculatePct(entry_ct_list[1],entry_ct_list[0],ajust_for_denom=1)+"%)",
                 "Unique sources :":str(unique_ct_list[0]-unique_ct_list[1])+" (" + calculatePct(unique_ct_list[1],unique_ct_list[0],ajust_for_denom=1)+"%)",
                 "Unique/Sum":calculateRatio(entry_ct_list[0],unique_ct_list[0]) + " -> " + calculateRatio(entry_ct_list[1],unique_ct_list[1])})
    # return num_entry, num_unique
    
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
    
    
def mainJoinOut() :
    df = openDFcsv(keyword_path,keyword_filename)
    df_out = ekml.generateNLPonKeywords(df,0,df.shape[0],True)
    # df_out = deleteUnnamed(df_out,"hash_key")
    saveDFcsv(df_out, join2_path,join2_filename)
    return df_out
    
print("IMPORT : article_scraping_lib")
