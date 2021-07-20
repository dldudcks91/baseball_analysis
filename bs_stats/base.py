#%%
import numpy as np
import pandas as pd
#%%
class Baseball:
    '''
        공통적으로 쓰이는 값들을 선언한 부모 Class
    
    '''
    
    # 팀 이름 번호에 매칭하는 dictionary
    team_dic = {'LG':1,'롯데':2,'KIA':3,'삼성':4,'두산':5,'한화':6,'SK':7,'SSG':7,'키움':8,'넥센':8,'NC':9,'KT':10,'kt':10}
    team_list = [0,'LG','롯데','KIA','삼성','두산','한화','SSG','키움','NC','KT']
    # 팀 별 Stadium dictionary
    stadium_dic = {'LG':'잠실','롯데':'사직','KIA':'광주','삼성':'대구','두산':'잠실','한화':'대전','SK':'문학','SSG':'문학','키움':'고척','넥센':'고척','NC':'창원','KT':'수원','kt':'수원'}
    
    # 분석에 사용한 year_list
    year_list = [2017,2018,2019,2020,2021]
    
    # 구장, 연도별 Park Factor dictionary
    park_factor_2016 = {'잠실': 0.954,'사직': 1.030,'광주':0.997, '대구': 1.042, '대전':1.007,'문학':1.016,'고척':1.032,'마산':0.943,'수원':1.021}
    park_factor_total = {'잠실': 0.854,'사직': 1.099,'광주':1.003, '대구': 1.153, '대전': 0.977,'문학':1.046,'고척':0.931,'창원':1.051,'수원':1.032}
    park_factor_list = [0.   , 0.863, 1.099, 1.003, 1.153, 0.863, 0.977, 1.046, 0.931, 1.051, 1.032]
    
    #타순별 기여
    pa_params = [4.756, 4.625,4.558,4.467,4.354,4.227,4.110,3.976,3.829]                   
       
    
    # csv 데이터 저장 주소
    address = 'C:\\Users\\Chan\\Desktop\\BaseballProject\\data'
    
    def __init__(self):
        return

     
    def save_csv(self,data,address,name):
        '''
        Save data_array
        
            csv 데이터 저장하기
            
        '''
        data = pd.DataFrame(data)
        data.to_csv(address + '\\' + name + '.csv',encoding = 'cp949')
        
    
    def load_csv(self,address,name):
        '''
        Load data_array
        
            csv 데이터 불러오기
            
        '''
        
        data = pd.read_csv(address + '\\' + name + '.csv',encoding = 'cp949' )
        result = np.array(data)
        return result
    


    
    ################################    DB관련코드    #############################################
    
    
