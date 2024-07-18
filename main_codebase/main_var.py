# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:14:51 2024

@author: Alexandre
"""
import os

class main_var :
    def __init__(self,main_path="C:/Users/User/OneDrive/Desktop/Article_LLM/main_files/" ,env="test/"):#"test_new/"test_new4  ".main/"
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
        self.image_path = self.main_path+"2_4_image_main/"+self.env
        self.join2_path = self.main_path+"3_1_join_main/"+self.env
        self.visu_path = self.main_path+"3_2_visu_main/"+self.env
        self.query_filename = "query_file"
        self.scarp_filename = "scarp_file"
        self.join1_filename = "join1_file"
        self.embdedding_filename = "embdedding_file"
        self.embdedding_filename_raw = "embdedding_file_raw"
        self.keyword_filename = "keyword_file"
        self.completion_filename = "completion_file"
        self.image_filename = "image_file"
        self.join2_filename = "join2_file"
        self.visu_filename = "visu_file"
        self.query_col_list = ["hash_key","title","category","source_title","published","year","year_month","source_url","url_list","url_TLD","link","pk"]




    def cleanFolders(self) :
        self.delete_files_in_directory(self.query_path)
        self.delete_files_in_directory(self.scarp_path)
        self.delete_files_in_directory(self.article_path)
        self.delete_files_in_directory(self.join1_path)
        self.delete_files_in_directory(self.embdedding_path)
        self.delete_files_in_directory(self.embdedding_path_raw)
        self.delete_files_in_directory(self.completion_path)
        self.delete_files_in_directory(self.join2_path)
        self.delete_files_in_directory(self.visu_path)
        
    def creareFolders(self) :
        self.create_folder(self.query_path)
        self.create_folder(self.scarp_path)
        self.create_folder(self.article_path)
        self.create_folder(self.join1_path)
        self.create_folder(self.embdedding_path)
        self.create_folder(self.embdedding_path_raw)
        self.create_folder(self.completion_path)
        self.create_folder(self.join2_path)
        self.create_folder(self.visu_path)

    def delete_files_in_directory(self,directory_path):
       try:
         files = os.listdir(directory_path)
         for file in files:
           file_path = os.path.join(directory_path, file)
           if os.path.isfile(file_path):
             os.remove(file_path)
         print("All files deleted successfully.")
       except OSError:
         print("Error occurred while deleting files.")
    
    def create_folder(self,directory_path) :
        try:
          os.mkdir(directory_path)
          print("Folder created successfully.")
        except OSError:
          print("Error occurred while creating the folder.")

def initFolderTree(main_path="C:/Users/User/OneDrive/Desktop/Article_LLM/test/" ,env=".main/"):
    mv = main_var(main_path,"")
    mv.create_folder(main_path)
    mv.creareFolders()
    mv2 = main_var(main_path,env)
    mv2.creareFolders()
    
print("IMPORT : main_var")

# initFolderTree()
# mv.cleanFolders()