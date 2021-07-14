#%%
import sys
sys.path.append('C:\\Users\\Chan\\Desktop\\BaseballProject\\python')
import datetime
from  baseball_2021 import crawling_kbo as ck
today = datetime.datetime.today()
year = str(today.year)
month = str(today.month).zfill(2)
day = str(today.day).zfill(2)
date = int(year+month+day)
c = ck.Crawling_baseball(date)
c.driver_start()
c.start_game_crawling()
c.driver.close()
