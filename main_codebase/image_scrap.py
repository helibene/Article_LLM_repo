# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:25:23 2024

@author: Alexandre
"""

import requests # request img from web
import shutil # save img locally

url_raw = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-JC0VmXs611nLY2FmI4JVhO5k/user-uh3YpAeeWcHuSOhPxQpZnZJW/img-5EP1F3ASFPWTjugoDASGai1o.png?st=2024-07-16T17%3A20%3A50Z&se=2024-07-16T19%3A20%3A50Z&sp=r&sv=2023-11-03&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2024-07-16T11%3A42%3A46Z&ske=2024-07-17T11%3A42%3A46Z&sks=b&skv=2023-11-03&sig=lRVzeeJUEEhGUZD1if/six5vD1UUo67cRZ%2B4aCI0we4%3D"
output_path_raw = "C:/Users/User/OneDrive/Desktop/Article_LLM/main_files/2_1_embdedding_main/.main/img/test.png"

url = url_raw
file_name = output_path_raw
res = requests.get(url, stream = True)
with open(file_name,'wb') as f:
        shutil.copyfileobj(res.raw, f)