#%%
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

script_dir = os.path.dirname(os.path.abspath(__file__)) 
#%%

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np 
import time
from datetime import datetime
#from bs_crawling import base as cb

from sqlalchemy import create_engine
#%%
import  json
WISE_IDX_URL = f'{script_dir}\wise_idx.json'

#%%
result_list = list()

session = requests.session()
with open(WISE_IDX_URL, 'r') as f:
    start_game_info_master_seq = json.load(f)['last_game_info_master_seq']
    print(start_game_info_master_seq)
    
    
game_round = 0
YEAR = 2024
game_info_master_seq_dic = dict()
error_dic = dict()
#%%
def find_data_by_game(game_list, year):
    
    handi_num = 0
    
    home_odds = 1.74
    away_odds = 1.74
    is_odds = False
    new_dic = dict()
    if game_list[11].string =='취소':
        return
    if game_list[3].string !='KBO':
        return
    
    for i, li in enumerate(game_list):
        # 게임번호 by round
        
        if i == 1:
            
            string = li.string
            month = string[:2]
            day = string[3:5]
            date = int(str(year) + month + day)
            
            new_dic['date'] = date
            
            game_time = string[-5:]
            new_dic['game_time'] = game_time
        
        # 종목
        elif i == 2:
            sport_type = li['class'][1]
            new_dic['sport_type'] = sport_type
        
        # 리그
        elif i == 3: 
            league_type = str(li.string)
            new_dic['league_type'] = league_type
            
        # 핸디 ('' = 노핸디 / H = 점수 핸디 / U = 언오버)
        elif i == 4: 
            
            if li['class'][0] == 'd1':
                return None
            
            
            
            if li['class'][0] =='hm':
                if li.string:
                    handi_num = 2
                    handi_score = li.string[2:]
                    
                else:
                    handi_num = 1
                    handi_score = 0
                    
            elif li['class'][0] =='hp':
                handi_num = 2
                handi_score = li.string[2:]
                
            elif li['class'][0] == 'un':
                handi_num = 3
                handi_score = li.string[2:]
                
            else:
                handi_num = -1
                handi_score = -1
                
            new_dic['handi_num'] = handi_num
            new_dic['handi_score'] = handi_score
            
        # 홈팀 이름 & 원정팀 이름
        elif i == 5:
            
            # 홈팀 이름
            new_dic['home_name'] = str(li.contents[0].string)
            
            # 원정팀 이름
            if handi_num == 3:
                
                new_dic['away_name'] = str(game_list[7].contents[0].string)
            else:
                new_dic['away_name'] = str(game_list[7].contents[2].string)
            
            
        
        # 승배당
        
        elif i == 8:
            try:
                home_odds = float(li.find(class_ = 'pt').string)
                is_odds = True
            except:
                is_odds = False
                pass
            

        # 패배당
        elif i == 10:
            try:
                away_odds = float(li.find(class_ = 'pt').string)
                is_odds = True
            except:
                is_odds = False
                pass
        
        
        odds_ratio =  1/ ((1/home_odds) + (1/away_odds))
         
        home_rate = round((1/home_odds) * odds_ratio,3)
        away_rate = round((1/away_odds) * odds_ratio,3)
        
        new_dic['home_rate'] = home_rate
        new_dic['away_rate'] = away_rate
        
        new_dic['is_odds'] = is_odds
    return new_dic
#%%
for i, game_info_master_seq in enumerate(range(start_game_info_master_seq, start_game_info_master_seq + 30)):
    
    
    
    game_round = i+1

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
               'referer': f'https://www.wisetoto.com/index.htm?tab_type=proto&game_type=pt&game_category=pt1&game_year=2024&game_round={game_round}',
               }
                
    response = requests.get(f'https://www.wisetoto.com/util/gameinfo/get_proto_list.htm?game_category=pt1&game_year=2024&game_round={game_round}&game_month=&game_day=&game_info_master_seq={game_info_master_seq}&sports=&sort=&tab_type=proto',
                    headers = headers)


    soup = BeautifulSoup(response.text,'html.parser')  
    
    
    game_info  = soup.find('div', attrs = {'class':'gameinfo'})
    game_info_ul = game_info.find_all('ul')
    
    
    old_list = list()
    for i, ul in enumerate(game_info_ul):
        game_list = ul.find_all('li')
        try:
            new_dic = find_data_by_game(game_list, YEAR)
        except:
            new_dic = None
            error_dic[game_info_master_seq] = game_info_ul
        #print(type(new_list))
        if new_dic is not None:
            new_dic['game_info_master_seq'] = game_info_master_seq
            result_list.append(new_dic)
    
    #print(game_round, len(round_dic[game_round]))
    #time.sleep(0.5)
    today = datetime.now()
    date_str = str(today.year) + str(today.month).zfill(2) + str(today.day).zfill(2)
    time_str = str(today.hour).zfill(2) + ":" + str(today.minute).zfill(2)
    for result in result_list:
        result['craw_date'] = date_str
        result['craw_time'] = time_str

    game_info_master_seq_dic[game_info_master_seq] = len(result_list)
    

print(game_info_master_seq, len(result_list), response.status_code)
print('success crawling odds data')
#%%


result_data = pd.DataFrame(result_list)
today = int(datetime.today().strftime("%Y%m%d"))
last_idx = int(result_data[result_data['date'] == today].game_info_master_seq.iloc[0])

with open(WISE_IDX_URL, 'w') as f:
        json.dump({'last_game_info_master_seq': last_idx}, f)  

print('success save last_idx')
#%%

result_data = result_data[result_data.is_odds == True]
result_data['site_name'] = 'livescore'
result_data = result_data[['date','game_time','site_name','handi_num','away_name','home_name','away_rate','home_rate','handi_score','craw_time']]

mysql_columns = ['date','time','site_name','win_type','away_name','home_name','away_odds','home_odds','handicap','craw_time']
result_data.columns = mysql_columns

print('success data preprocess')
#%%
from bs_personal import personal_code as cd
db_address = cd.db_address
code = cd.aws_code
file_address = cd.file_aws_address
engine = create_engine(db_address + code + file_address)#, encoding = 'utf-8')

result_data = result_data.drop_duplicates(subset = ['date','time','site_name','win_type','away_name','home_name','craw_time'])
#%%
result_data.to_sql('today_toto',con = engine, if_exists = 'append', index = False)

print('success save today_toto to mysql')

