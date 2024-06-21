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
from utils_art import saveDFcsv, deleteUnnamed, openDFcsv, calculatePct, calculateRatio, display_df
import hashlib
from datetime import timedelta
pd.set_option('expand_frame_repr', False)
# import embedding_keyword_module_lib as ekml

_TOPIC_LIST = ["TOP","WORLD","NATION","BUSINESS","TECHNOLOGY","ENTERTAINMENT","SCIENCE","SPORTS","HEALTH"] #"CAAqJQgKIh9DQkFTRVFvSUwyMHZNR3QwTlRFU0JXVnVMVWRDS0FBUAE"
_STANDARD_SCRAPPING_FIELDS = ["title","link", "published", "source_url", "source_title", "category", "year", "year_month","pk","url_list","url_TLD", "hash_key"]
_MIN_SCRAPPING_FIELDS = ["hash_key","title", "category"]
_ALL_SCRAPPING_FIELDS = ["hash_key","title","category","source_title","source_url","is_https","url_list","url_tld","published_date","published_date_str","published_date_int","published_time_raw","published_time_struct","published_year","published_year_month","link","guidislink","num_sub_article","num_links","language_title","language_summary","source_in_title","id"]
_STANDARD_SCRAPPING_FIELDS_VALUES = ["title","link", "published", "source_url", "source_title", "category", "year", "year_month","pk","url_list","url_TLD", "hash_key"]
log_filename = "log"
language = "fr"
country = "FR"

################################## QUERRY TO GET ARTICLE LIST

gn = GoogleNews(lang = language, country = country)
log_file = open(mv.query_path+log_filename+".txt", "a", encoding="utf-8")

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
        
        list_for_entry.append(str(entry["title"])) # "title"
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

def rawHeadListToDictList(rawHeadList, category="n/a") :
    field_list = _STANDARD_SCRAPPING_FIELDS
    dict_list_output = []
    for a_dict_in in rawHeadList :
        a_dict_out = {}
        
        # Link
        website_https = True
        true_link = a_dict_in["source"]["href"]
        if "https" not in true_link :
            website_https = False
        link_str_list = true_link.split(".")
        link_str_list[0] = link_str_list[0].replace("https://","").replace("http://","").replace("www","")
        url_comp_list = link_str_list[:-1]
        url_tld = link_str_list[-1]
        source_in_title = False
        if str(a_dict_in["source"]["title"]) in a_dict_in["title"] :
            source_in_title = True
        
        hash_key = hashlib.shake_256(str(a_dict_in["id"]).encode()).hexdigest(20)
        
        published_date = parse_date(a_dict_in["published"]).date()
        published_year = published_date.year
        published_year_month = str(published_date.strftime('%Y-%m'))
        published_str = str(published_date.strftime('%Y-%m-%d'))
        published_int = int(published_date.strftime('%Y%m%d'))
        
        if "hash_key" in field_list :
            a_dict_out["hash_key"] = hash_key
        if "title" in field_list :
            a_dict_out["title"] = a_dict_in["title"]
        if "category" in field_list :
            a_dict_out["category"] = category
        if "source_title" in field_list :
            a_dict_out["source_title"] = a_dict_in["source"]["title"]
        if "source_url" in field_list :
            a_dict_out["source_url"] = true_link
        if "is_https" in field_list :
            a_dict_out["is_https"] = website_https
        if "url_list" in field_list :
            a_dict_out["url_list"] = url_comp_list
        if "url_tld" in field_list :
            a_dict_out["url_tld"] = url_tld
        if "published_date" in field_list :
            a_dict_out["published_date"] = published_date
        if "published_date_str" in field_list :
            a_dict_out["published_date_str"] = published_str
        if "published_date_int" in field_list :
            a_dict_out["published_date_int"] = published_int
        if "published_time_raw" in field_list :
            a_dict_out["published_time_raw"] = a_dict_in["published"]
        if "published_time_struct" in field_list :
            a_dict_out["published_time_struct"] = a_dict_in["published_parsed"]
        if "published_year" in field_list :
            a_dict_out["published_year"] = published_year
        if "published_year_month" in field_list :
            a_dict_out["published_year_month"] = published_year_month
        if "link" in field_list :
            a_dict_out["link"] = a_dict_in["link"]
        if "guidislink" in field_list :
            a_dict_out["guidislink"] = a_dict_in["guidislink"]
        if "num_sub_article" in field_list :
            a_dict_out["num_sub_article"] = len(a_dict_in["sub_articles"])
        if "num_links" in field_list :
            a_dict_out["num_links"] = len(a_dict_in["links"])
        if "language_title" in field_list :
            a_dict_out["language_title"] = a_dict_in["title_detail"]["language"]
        if "language_summary" in field_list :
            a_dict_out["language_summary"] = a_dict_in["summary_detail"]["language"]
        if "source_in_title" in field_list :
            a_dict_out["source_in_title"] = source_in_title
        if "id" in field_list :
            a_dict_out["id"] = a_dict_in["id"]
        dict_list_output.append(a_dict_out)
    return dict_list_output
    

def fillDFwithListList(df, ListList):
    for ent in ListList:
        df.loc[len(df)] = ent
    return df

def fullWorkflow(gn, df, search = False, topic_sel=1, date1 = '2015-01-01', date2 = '2016-01-01') :
    headlines_raw = getHeadlindTopic(gn, search, topic_sel, date1, date2)
    headlinesMat = rawHeadListToListList(headlines_raw, _TOPIC_LIST[topic_sel])
    # headlinesMat = rawHeadListToDictList(headlines_raw, _TOPIC_LIST[topic_sel])
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
            log(" - Saving result table : '"+str(mv.query_path)+str(mv.query_filename)+"_"+iteration_str+".csv'")
        saveDFcsv(df_out, mv.query_path, mv.query_filename+"_"+iteration_str,False)
    if display :
        logEndWorkflow(df_out.shape[0], expected_article)
    return df_out

def loop_scraping(number_topics=8, startDate='2010-01-01', endDate='2024-01-01', sampling_1=14, sampling_2=6, display=True,save_steps=True,save_final=True,feilds="s"): # feilds (s=standard) (m=min) (a=all)
    if feilds=="m" :
        _STANDARD_SCRAPPING_FIELDS=_MIN_SCRAPPING_FIELDS
    elif feilds=="a" :
        _STANDARD_SCRAPPING_FIELDS=_ALL_SCRAPPING_FIELDS
    else :
        _STANDARD_SCRAPPING_FIELDS=_STANDARD_SCRAPPING_FIELDS_VALUES
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
        df_out = deleteUnnamed(df_out,"hash_key")
        saveDFcsv(df_out, mv.query_path, mv.query_filename,False)
        print("Final file saved here :",mv.query_path)
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




################################## UTILS 


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
    print(type(df_source))
    print(df_source)
    df_date_year_month = df[time_field].value_counts().to_frame("count").sort_values(by=[time_field],ascending=True)
    axes[1].tick_params(labelcolor='black', labelright=True, labelleft=False)
    axes[1].invert_yaxis()

    plot = df_source.plot.barh(y='count',use_index=True,ax=axes[0], legend=False, ylabel="", fontsize= 8, title="Sources") #, figsize=(5, 5)
    plot = df_date_year_month.plot.barh(y='count',use_index=True,ax=axes[1], legend=False, ylabel="", fontsize= 8, title="Volume month/year") #, figsize=(5, 5)
    plot = df_category.plot.pie(y='count',use_index=True, legend=False, ylabel="", title="Distribution of categories") #, figsize=(5, 5)
    
    return plot


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

def filterQuerryDataset(thd_high=5000,thd_low=30, display_stats=True,display_end_stats=False,save=False) :
    df = openDFcsv(mv.query_path,mv.query_filename)
    ser_source = df['source_title'].value_counts() #.to_frame("count").sort_values(by=['count'],ascending=False)
    if thd_high > 0 and thd_high < 1 :
        thd_high = ser_source[int(float(len(ser_source))*float(thd_high))]
    ser_source_low = ser_source[ser_source < thd_low]
    ser_source_high = ser_source[ser_source > thd_high]
    list_source_low = list(ser_source_low.keys())
    list_source_high = list(ser_source_high.keys())
    df_wo_low = df[~df['source_title'].isin(list_source_low)]
    df_only_low = df[df['source_title'].isin(list_source_low)]
    df_only_high = df[df['source_title'].isin(list_source_high)]
    df_wo_low_high = df[~df['source_title'].isin(list_source_low+list_source_high)]
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
        saveDFcsv(df,mv.query_path,mv.query_filename+"_backup_before_filter",True)
        final_df = deleteUnnamed(final_df,"hash_key")
        saveDFcsv(final_df,mv.query_path,mv.query_filename,True)
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

print("IMPORT : scrap_lib")
