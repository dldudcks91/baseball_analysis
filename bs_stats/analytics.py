#%%
#%%
import sys
sys.path.append('D:\\BaseballProject\\python')
import numpy as np
import pandas as pd
import math
import scipy
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.model_selection import KFold
import xgboost
from sklearn.linear_model import LogisticRegression, LinearRegression, HuberRegressor, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from bs_stats import sample as sp
from bs_stats import preprocess as pr
from xgboost import XGBRegressor

#%%
s = sp.Sample()
sm.families.Gamma.safe_links.append(sm.genmod.families.links.identity)
sm.families.Gamma.safe_links.append(sm.genmod.families.links.inverse_power)
#%%
class Analytics(pr.Preprocess):
    
    def __init__(self):
        
        
        self.min_recent_sp = 1
        self.max_recent_sp = 100
        self.min_inn = 4
        
        self.max_era = 8
        self.max_fip = 8
        
        self.len_y = 2
        self.len_x = 0
        
        self.br_range_list = list()
        self.sp_range_list = list()
        self.rp_range_list = list()
        self.year_list = list()
        
        self.len_params = 0
        self.len_model = 1
        self.len_score = 3
        self.start_game_idx = 20
        
        
        self.total_score_dic = dict()
        self.total_params_list = list()
        self.total_scale_list = list()
        self.total_data_len_dic = dict()
        
        self.sr_type = 0
        self.xr_type = 0
        self.sp_type = 0
        
        self.random_state = 1
        return
    
    def get_mean(self, x, params, link = 'identity'):
        
        '''
        Get mean
        
            주어진 parameter를 통해 예측값 y_hat을 구하는 함수
            
            x(n x k): 독립변수
            params(k x 1): 예측파라미터
            link(str): 감마분포 링크함수(inverse, log, identity)
            
            return y_hat(n x 1)
        '''
        
        y_hat = np.dot(x,params).reshape(-1,1)
        if link == 'inverse':
            y_hat = 1/y_hat
            
        elif link =='log':
            y_hat = np.exp(y_hat)
            
        return y_hat
    
    def get_score(self, y, y_hat, dist = 'gamma', link = 'identity', alpha = 2):
        
        '''
        Get_score
        
             주어진 parameter를 통해 y_hat을 구하고 y와의 차이를 구하는 함수
             
             y(n x 1): 종속변수
             y_hat(n x 1): y 예측값
             dist(str): 분포 설정(cp 계산을 위한)
             link(str): 감마분포 링크함수(inverse, log, identity)
             beta(float): 감마분포 모수 beta 초기값
             
             return [rmse, msle, cp]
             
        '''
        y  = y.reshape(-1,1)
        y_hat  = y_hat.reshape(-1,1)
        
        beta = y_hat / alpha
        
        mse = np.mean((y - y_hat)**2)
        
        rmse = np.round(np.sqrt(mse),4)
        '''
        if dist == 'gamma':
            cp = np.mean(s.gamma_pdf(y,[alpha,beta]))
            
            
        else:
            cp = 0
        '''
        msle = np.round(np.mean((np.log(y+1) - np.log(y_hat+1))**2),4)
        #cee = -np.sum(np.log(y)*y_hat)
        mae = np.round(np.mean(abs(y-y_hat)),4)
        
        var = np.var(y_hat)
        bias = np.mean((y - np.mean(y_hat))**2)
        
        
        return [rmse, msle, mae]#var, bias] 
    
    
    
    def get_input(self, total_array, sr = 0, is_new = False):
        '''
        Get input
            
            preprocess에서 1차 전처리한 데이터들을 가져와 데이터분석을 위해 전처리하는 함수
            
            total_array(n x k): 분석을 위한 모든 데이터 array
            min_sp_len
        
        '''
        park_factor = {'잠실': 0, '사직': 1,'광주': 2, '대구': 3, '대전': 4,'문학': 5,'고척': 6,'마산': 7,'수원': 8}
        ground_array = total_array[:,8]
        old_array = np.zeros((1,9))
        for ground in ground_array:
            
            pf = park_factor.get(ground)
            new_array = np.zeros((1,9))
            if pf==None:
                pass
            else:
                new_array[0,pf] = 1
            
            old_array = np.vstack([old_array,new_array])
        old_array = old_array[1:]
        total_array = np.hstack([total_array,old_array])
        
        sr_type = self.sr_type
        xr_type = self.xr_type
        sp_type = self.sp_type
        # 선발투수 경기 수 구간에 맞는 데이터 추출
        recent_sp_idx = self.len_total - self.len_rp - 1
        recent_sp_array = total_array[:,recent_sp_idx].astype(np.float64)
        recent_sp_mask = (recent_sp_array >= self.min_recent_sp) & (recent_sp_array <= self.max_recent_sp)
        
        if sr_type == 1:
            recent_sp_mask = (recent_sp_array == sr)
        
        
        total_array = total_array[recent_sp_mask,:]
        
        
        
        
        #기본정보 제외한 분석용 데이터 열 추출
        record_array = total_array[:,self.len_info:]       
        Y = record_array[:,:self.len_y].astype(np.float64) # 종속변수 Y(n x 2): run, xr_run
        
        X = record_array[:,self.len_y:] # 독립변수 X(n x k)
        
        #pitcher name열 제외
        sp_name_idx = self.len_xr
        
        X = np.hstack([X[:,:sp_name_idx],X[:,sp_name_idx+1:]]).astype(np.float64)
        
        # 타자, 투수 데이터 분리
        batter_array = X[:,:self.len_xr].astype(np.float64)
        pitcher_array = X[:,self.len_xr:-9].astype(np.float64)
        ground_array = X[:,-9:]
        xr_array = batter_array[:,-1].reshape(-1,1)
        # 선발, 계투 데이터 분리
        sp_array = pitcher_array[:,:self.len_sp-1].astype(np.float64)
        rp_array = pitcher_array[:,self.len_sp-1:].astype(np.float64)
        
        # 이닝 데이터 추출 및 최소값 설정
        inn = sp_array[:,0].reshape(-1,1)
        inn = np.where(inn<=self.min_inn, self.min_inn, inn)
        
        # 선발, 계투 데이터 세부분리
        sp_era = sp_array[:,-4].reshape(-1,1) #선발투수 era
        sp_fip = sp_array[:,-3].reshape(-1,1) #선발투수 fip
        sp_var = sp_array[:,-2]
        sp_len = sp_array[:,-1].reshape(-1,1) #선발투수 등판 수
        
        sp_era = sp_era *sp_len
        rp_era = rp_array[:,-2].reshape(-1,1) #계투 era
        rp_fip = rp_array[:,-1].reshape(-1,1) #계투 fip
        
        
        
        
        
        
        
        
        # 홈원정 변수 생성        
        X_home = np.where(total_array[:,7] == 'home',1,0).reshape(-1,1)
        
        
       
        
        if xr_type == 0:
            x_batter = batter_array[:,:-6]
            
        elif xr_type ==1:
            x_batter = xr_array
        
        
        if sp_type == 0:
            x_sp = sp_array
            x_rp = rp_array
        elif sp_type ==1:
            x_sp = (sp_array *inn) / 9
            x_sp = x_sp[:,1:]
            x_rp = ((9-inn)*rp_array) / 9 
        elif sp_type == 2:
            x_sp = (inn*sp_fip) / 9 
            x_sp = x_sp[:,1:]
            x_rp = ((9-inn)*rp_array) / 9 
        elif sp_type == 3:
            x_sp = (inn*sp_array) / 9 
            x_sp = x_sp[:,1:]
            x_rp = ((9-inn)*rp_fip) / 9 
        elif sp_type == 4:
            x_sp = (inn*sp_fip) / 9 
            x_sp = x_sp[:,1:]
            x_rp = ((9-inn)*rp_fip) / 9 
        x_sp = x_sp[:,:-5]
        x_rp = x_rp[:,:-3]
        X = np.hstack([x_batter, x_sp, x_rp, X_home, ground_array, sp_len])
        #print([len(x_batter[0]), len(x_sp[0]), len(x_rp[0])])
        X = np.hstack([np.ones(len(X)).reshape(-1,1),X])
        self.len_x = len(X[0])
       
        result_array = np.hstack([Y,X])
        
        
        return result_array
    
    def set_total_params(self, range_data_dic, is_print = True):
         
        '''
        Get total params
        
            cv-fold를 통해 파라미터 평균, 스케일 평균, 스코어 평균을 구하는 함수
        '''
        total_params_list = [dict() for i in range(self.len_model)]
        total_scale_list = [dict() for i in range(self.len_model)]
        total_score_dic = dict()
        total_data_len_dic = dict()
        xgb = XGBRegressor(n_estimators = 500, learning_rate = 0.1, random_state = 1)
        #gbr = GradientBoostingRegressor()
        for br in self.br_range_list:
            for sr in self.sp_range_list:
                for rr in self.rp_range_list:
                    
                    
                    total_array = np.zeros((1,self.len_total))
                    
                    for year in self.year_list:
                        
                        for team_num in range(1,11):
                            
                            team_array = range_data_dic[(br,sr,rr)][year][team_num][self.start_game_idx:]
                            
                            total_array = np.vstack([total_array,team_array])
                    
                    
                    total_array = total_array[1:,:]
                    
                    input_array = self.get_input(total_array, sr = sr)
                    
                    
                    run = input_array[:,0]
                    run = np.where(run==0, 1, run)
                    #run = np.where(run>=13, 13, run)
                    #run_xr = input_array[:,1]
                    X = input_array[:,2:-1]
                    self.len_x = len(X[0])
                    cv = KFold(5,shuffle =True,random_state= self.random_state)


                    model_list = [list() for i in range(self.len_model)]
                    
                    
                    
                    score_list = list()
                    score_array = np.zeros((1,self.len_score))
                    
                    
                    
                    
                    scale_list = list()                
                    scale_array = np.zeros((1))
                    
                    for i in range(self.len_model):
                        score_list.append(score_array)
                        
                        scale_list.append(scale_array)
                    
                    
                    
                    score_model_list = [0 for i in range(self.len_model)]
                    len_score_model = 0
                    i = 0
                    for (train_idx, test_idx) in cv.split(X):
                        X_train = X[train_idx]
                        X_test = X[test_idx]
                        Y_train = run[train_idx]
                        Y_test = run[test_idx]
                        
                        
                        model_list[0].append(sm.GLM(Y_train,X_train,family = sm.families.Gamma(link = sm.genmod.families.links.identity())).fit())
                        
                        model_list[1].append(sm.OLS(Y_train,X_train).fit())
                        
                        #model_list[2].append(HuberRegressor().fit(X_train, Y_train))
                        model_list[2].append(Ridge().fit(X_train,Y_train))
                        #model_list[3].append((gbr.fit(X_train,Y_train)))
                        
                        
                        
                        
                        for j, model in enumerate(model_list):
                            '''
                            if j ==0:
                                
                                new_scale = model[i].scale
                                scale_list[j] = np.vstack([scale_list[j],new_scale])
                            '''
                            y_hat = model[i].predict(X_test)
                            
                            new_score = self.get_score(Y_test, y_hat, dist = 'gamma', link = 'identity')
                            
                            score_list[j] = np.vstack([score_list[j],new_score])                        
                            score_model_list[j]+=new_score[0]
                            
                            
                            
                        len_score_model+=1
                        i+=1
                    
                    
                    for i in range(self.len_model):
                        
                        
                        total_params_list[i][(br,sr,rr)] = model_list[i]
                        
                        
                        scale_mean = np.mean(scale_list[i][1:], axis = 0)
                        
                        total_scale_list[i][(br,sr,rr)] = scale_mean
                        
                    
                    
                    # 2017~2020 데이터 cv-fold 결과
                    total_score_array = np.zeros((1,self.len_score))
                    for i, score in enumerate(score_list):
                        total_score_array = np.vstack([total_score_array, np.mean(score[1:],axis = 0)])
                        #print([br,sr,rr], i, '=', np.mean(score,axis = 0))
                    
                    total_score = np.round(np.mean(total_score_array[1:], axis = 0 ),4)
                    total_score_dic[(br,sr,rr)] = total_score
                    total_data_len_dic[(br,sr,rr)] = len(X)
                    print('success' + str([br,sr,rr]))
                    if is_print:
                        print(len(input_array))
                        print([br,sr,rr], total_score)
                        
                        print(' ')
        
        self.total_score_dic = total_score_dic
        self.total_params_list = total_params_list
        self.total_scale_list = total_scale_list
        self.total_data_len_dic = total_data_len_dic