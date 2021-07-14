#%%
import sys
sys.path.append('C:\\Users\\Chan\\Desktop\\BaseballProject\\python')
import datetime
from  baseball_2021 import crawling_kbo as ck


c = ck.Crawling_baseball(20210517)
c.driver_start()
c.end_game_crawling()
c.driver.close()
