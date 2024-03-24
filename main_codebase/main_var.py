# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:14:51 2024

@author: Alexandre
"""


class main_var :
    def __init__(self,main_path="C:/Users/User/OneDrive/Desktop/Article_LLM/main_files/",env="test/"):
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
        self.query_filename = "query_fileG_cap"
        self.scarp_filename = "scarp_file"
        self.join1_filename = "join1_file"
        self.embdedding_filename = "embdedding_file"
        self.embdedding_filename_raw = "embedding_matrix_20000"
        self.keyword_filename = "keyword_file"
        self.completion_filename = "completion_file"
        self.join2_filename = "keyword_with_nlp_more_cat"
        self.visu_filename = "visu_file"

mv = main_var()
# ####   article_scraping
# env = "arc/"
# main_path = "C:/Users/User/OneDrive/Desktop/article/files_3/1_1_query_main/"+env
# filename = "reduction_test"
# path_parse_stat = "C:/Users/User/OneDrive/Desktop/article/files_3/1_2_scarp_main/"+env
# path_union_stat = "C:/Users/User/OneDrive/Desktop/article/files_3/1_4_join_main/"+env
# filename_union_stat = "join_main_test-filter"
# llm_join_out = "C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/"
# llm_join_out_filename = "article_stats_embedding"

# ####   article_parsing
# env = "arc/"
# open_path = "C:/Users/User/OneDrive/Desktop/article/files_3/1_1_query_main/"+env
# filename_input = "main_scrap"#"query_large_2010_to_2023"#"query_main_final"
# save_path = "C:/Users/User/OneDrive/Desktop/article/files_3/1_2_scarp_main/"+env #parssing_df_main/
# filename_out = "test_vol___________XXXXXXXX"
# save_path_article = "C:/Users/User/OneDrive/Desktop/article/files_3/1_3_article_main/"+env #article_download_main_2

# ####   openai_module
# open_path = "C:/Users/User/OneDrive/Desktop/article/files_3/1_3_article_main/arc/"
# save_path = "C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embd_df/"
# filename_save = "embd_out_main_test"
# llm_result = "C:/Users/User/OneDrive/Desktop/article/file_2/test_llm_output/new_embedding/"




# ####   embedding_keyword_module

# folder_path_input = "C:/Users/User/OneDrive/Desktop/article/files_3/1_4_join_main/arc/"
# filename_input = "join_main_big"
# folder_path_embd_df="C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embd_df/"
# folder_path_embd_join="C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embd_join/"
# folder_path_embd_raw="C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/test/"
# filename_embd_df="embd_out_main_testfinal"
# filename_embd_join="embd_all_main_testfinal"
# filename_embd_raw_save="embedding_matrix_max"
# filename_embd_raw_load="embedding_matrix_XXXX"


# folder_path_keyword_out="C:/Users/User/OneDrive/Desktop/article/files_3/2_2_keyword_main/out/"
# folder_path_keyword_out_nlp="C:/Users/User/OneDrive/Desktop/article/files_3/2_2_keyword_main/out_nlp/"
# filename_keyword_out="keywords_df_main_test"
# filename_keyword_out_nlp="keywords_df_nlp_main_test"

# # # folder_path="C:/Users/User/OneDrive/Desktop/article/file_2/test_llm_output/run2/"
# # folder_path_embd="C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/embd/only_embd/"
# # folder_path_keyword="C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/keyword/"
# # folder_path_llm="C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/llm/"
# # folder_path_graph="C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/graphs/"
# # folder_path_viz="C:/Users/User/OneDrive/Desktop/article/file_2/join_2_df/viz/"
# # \files_3\1_4_join_main\arc
# # filename="article_stats_embedding"
# # filename_embd="embedding_matrix_1000"#"embedding_matrix_main"#"article_stats_embedding_smain" #""
# # filename_keyw="article_keyword_main"
# # open_path_keyw="C:/Users/User/OneDrive/Desktop/article/file_2/join_1_df/"



# ####   visualization_module

# folder_path_input_df="C:/Users/User/OneDrive/Desktop/article/files_3/3_1_join_main/arc/"
# folder_path_input_embd="C:/Users/User/OneDrive/Desktop/article/files_3/2_1_embdedding_main/embf_raw_viz/"
# filename_input_df="keyword_with_nlp_viz_col1"
# filename_input_embd="embedding_matrix_20000"

# folder_path_graph="C:/Users/User/OneDrive/Desktop/article/files_3/3_2_visu_main/"
