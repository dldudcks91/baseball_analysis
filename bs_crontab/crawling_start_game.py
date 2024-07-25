#%%


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#%%
import requests
from bs4 import BeautifulSoup 
import json
import time
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

import pymysql

#from bs_crawling import base as cb
from bs_personal import personal_code as cd
from bs_crawling import kbo_request as kr





host = cd.aws_host
user = cd.aws_user
password = cd.aws_code
db = cd.db
today = (datetime.today() + timedelta(hours = 9)).date()


ck = kr.Crawling_kbo_request()
conn_aws = pymysql.connect(host = host, user = user, password= password, db= db, charset='utf8')
conn = conn_aws
ck.set_last_game_num_list(ck.year,conn)
conn.close()


date_str = str(today).replace("-","")
date = int(date_str)
ck.year = int(date_str[:4])
ck.craw_game_info(date)
ck.craw_lineup(date)
ck.set_date_start(date)
#%%
print(datetime.now())
#%%
conn_aws = pymysql.connect(host = host, user = user, password= password, db= db, charset='utf8')
conn = conn_aws
try:
    ck.delete_table_data(conn, 'today_lineup')
    ck.delete_table_data(conn, 'today_team_game_info')
    ck.delete_table_data(conn, 'today_game_info')
    
    conn.commit()
    print(f"Success Delete start_game_data")
except pymysql.InterfaceError as e:
    conn.rollback()
    print(f"InterfaceError 발생: {e}")
except Exception as e:
    print(f"다른예외처리: {e}")
finally:
    
    conn.close()
#%%

print(ck.game_info_array.shape)
print(ck.team_game_info_array.shape)
print(ck.lineup_array.shape)
conn_aws = pymysql.connect(host = host, user = user, password= password, db= db, charset='utf8')
conn = conn_aws
try:
    ck.array_to_db(conn, ck.game_info_array, 'today_game_info')  
    ck.array_to_db(conn, ck.team_game_info_array, 'today_team_game_info')
    ck.array_to_db(conn, ck.lineup_array, 'today_lineup')
    
    conn.commit()
    print(f"Success insert start_game_data")
except pymysql.InterfaceError as e:
    conn.rollback()
    print(f"InterfaceError 발생: {e}")
except Exception as e:
    print(f"다른예외처리: {e}")
    
finally:
    conn.close()









