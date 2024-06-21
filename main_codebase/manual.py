# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:38:51 2024

@author: Alexandre
"""

# from utils_art import unionFiles
# main_folder = "C:/Users/User/OneDrive/Desktop/Article_LLM/main_files/1_2_scarp_main/.main/temp3_not_del/"
# file1="input1"
# file2="input2"#51562
# file3="join3"#69841   59492
# unionFiles(main_folder,file1,main_folder,file2,main_folder,"out")
#unionFiles(main_folder,"join10",main_folder,file3,main_folder,"join20")


# from utils_art import select_articles, openDFcsv
# import pandas as pd
# main_folder = "C:/Users/User/OneDrive/Desktop/Article_LLM/main_files/1_3_article_main/.main/"
# out=select_articles(main_folder)
# #out_df=pd.DataFrame(out, columns=["hash_key"])
# querry_path = "C:/Users/User/OneDrive/Desktop/Article_LLM/main_files/1_1_query_main/.main/"
# df_querry = openDFcsv(querry_path,"query_file")
# df = df_querry[df_querry['hash_key'].isin(out)]
# print(df_querry)
# print(df)
# # df = df_querry.join(out_df, how="outter",on='hash_key', rsuffix='_e')
# # print(df.shape)
# #print(out)
# # saveStats()
# # df = generateDimReducedDF(n_components=3, norm_output=True, active_sel=[True,True,True,True])
# # print(df)


# import plotly.express as px
# import pandas as pd

# import main_var
# mv = main_var.main_var()
# from dimension_reduc_lib import calculateStatsNLP2,calculateStatsLength2
# from utils_art import openDFcsv,saveDFcsv
# agg_list = ["category","year","source_title"]#,["year","category"],["year","source_title"]]
# df_main = openDFcsv(mv.join2_path,mv.join2_filename)
# for agg in agg_list :
#     df_nlp = calculateStatsNLP2(df_main,agg)
#     df_len = calculateStatsLength2(df_main,agg)
#     df_stats = df_nlp.join(df_len, how="inner",on=agg,lsuffix="_nlp",rsuffix='_len')
#     saveDFcsv(df_stats,mv.visu_path,mv.visu_filename+"_"+str(agg))

#df_stats = calculateStatsNLP2(df_main,["year","category"])#year_month
#df_stats = calculateStatsNLP2(df_main,"source_title")#year_month
# print(list(df_stats.columns))
# df_len = calculateStatsLength2(df_main,["year","category"])
#df_stats = df_stats.reset_index()
#saveDFcsv(df_stats,mv.visu_path,mv.visu_filename+"_test_new_12")



# import plotly.express as px
# df = px.data.gapminder().query("continent == 'Oceania'")
# print(df)
# import pandas as pd

# groups = [[23,135,3], [123,500,1]]
# group_labels = ['views', 'orders']

# # Convert data to pandas DataFrame.
# df = pd.DataFrame(groups, index=group_labels).T
# print(df)
# # Plot.
# # pd.concat(
# #     [
# #         df.mean().rename('average'), 
# #         df.min().rename('min'), 
# #         df.max().rename('max')
# #     ],
# #     axis=1,
# # ).plot.bar()
# df.plot.bar()

# import pandas as pd

# df = pd.DataFrame({
#     'product': ['A', 'B', 'C', 'A', 'B', 'C'],
#     'sales': [100, 200, 300, 400, 500, 600],
#     'sales2': [700, 50, 30, 0, 100, 2]
# })
# print(df)
# grouped = df.groupby('product')
# print(grouped.mean())
# print(grouped.sum("sales"))
# print(grouped.min())
# print(grouped.max())
# print(type(df.min()["product"]))
# print(df.min()["product"])
# print(df.max())

# print("lol"*(not False))