# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:14:51 2024

@author: Alexandre
"""


class main_var :
    def __init__(self,main_path="C:/Users/User/OneDrive/Desktop/Article_LLM/main_files/",env="test_new/"):
        self.main_path = main_path
        self.env = env
        self.query_path = self.main_path+"1_1_query_main/"+self.env
        self.scarp_path = self.main_path+"1_2_scarp_main/"+self.env
        self.article_path = self.main_path+"1_3_article_main/"+self.env
        self.join1_path = self.main_path+"1_4_join_main/"+self.env
        self.embdedding_path = self.main_path+"2_1_embdedding_main/"+self.env
        self.embdedding_path_raw = self.main_path+"2_1_embdedding_main/"+self.env+"raw/"
        self.keyword_path = self.main_path+"2_2_keyword_main/"+self.env
        self.completion_path = self.main_path+"2_3_completion_main/"+self.env
        self.join2_path = self.main_path+"3_1_join_main/"+self.env
        self.visu_path = self.main_path+"3_2_visu_main/"+self.env
        self.query_filename = "query_file"
        self.scarp_filename = "scarp_file"
        self.join1_filename = "join1_file"
        self.embdedding_filename = "embdedding_file"
        self.embdedding_filename_raw = "embdedding_file_raw"
        self.keyword_filename = "keyword_file"
        self.completion_filename = "completion_file"
        self.join2_filename = "join2_file"
        self.visu_filename = "visu_file"

mv = main_var()

print("IMPORT : main_var ")
