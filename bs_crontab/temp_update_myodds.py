#%%

# 불러오기 위치 설정
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import numpy as np
import pandas as pd
import pymysql

from bs_database import base as bs

from bs_stats import preprocess as pr
from bs_stats import sample as sp
from bs_personal import personal_code as cd
from bs_stats import analytics as an

import datetime
import psutil
import time
from sqlalchemy import create_engine
#%%
import pickle


with open(cd.total_param_list_url,'rb') as f:
    total_params_list = pickle.load(f)
    #%%
    
#%%
a = an.Analytics()
s = sp.Sample()
d = bs.Database()
#%%
process = psutil.Process()
memory_info = process.memory_info()

print(f"RSS (Resident Set Size): {memory_info.rss / (1024 * 1024):.2f} MB")
print(f"VMS (Virtual Memory Size): {memory_info.vms / (1024 * 1024):.2f} MB")
#%%

engine = create_engine(cd.db_address + cd.aws_code + cd.file_aws_address)#, encoding = 'utf-8')
conn = engine.connect()
team_info_array = np.array(pd.read_sql_table('team_info',conn))
conn.close()

