#%%
import numpy as np
import math
import scipy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#%%
class Sample():
    '''
    1. Create random sample (norm_dist & gamma_dist) 
       난수생성을 최적화하여 난수생성 라이브러리 보다 빠름
    
    2. gradient descent
       MAP,MLE,Linear Regression 모수를 추정할 때 사용 
       
    '''
    def __init__(self):
        self.data = np.zeros((1,1)) # 데이터
        self.data_sum = None # 데이터 합
        self.data_log_sum = None # 데이터 로그 합
        self.data_length = None # 데이터 길이
        self.data_mean = None # 데이터 평균
        self.data_var = None # 데이터 분산
        
        
        self.result_array = None # 결과 array
        self.isinflect = False  # 변곡점인가
        
        self.start_theta = None # 이터레이션 시작 모수
        self.mom_gamma_theta = [None,None]
        self.mom_alpha = None
        self.mom_beta = None
        
        self.alpha_theta = None # 모수 alpha의 분포
        self.beta_theta = None # 모수 beta의 분포
        
        self.size = 10000 # 샘플 사이즈
        self.max_loop = 1000 # 루프 최대 한계
        self.gradient_param = 0.0001 
        self.stop_point = 0.0001
        
    def set_data(self, data):
        
        # 데이터 set하면서 기본정보 생성하는 함수
        
        self.data = data
        self.data_sum = np.sum(self.data)
        self.data_log_sum = np.sum(np.log(self.data))
        self.data_length = len(self.data)
        self.data_mean = np.mean(self.data) 
        self.data_var = np.var(self.data)
        
        self.start_theta = [self.data_mean, self.data_var]
        
        self.mom_beta = self.data_var / self.data_mean
        self.mom_alpha = self.data_mean / self.mom_beta
        self.mom_gamma_theta = [self.mom_alpha,self.mom_beta]
        
    def set_linear_data(self,data):
        
        # 선형분석을 위한 데이터를 set하면서 기본정보 생성하는 함수
        
        self.data = data
        self.data_length = len(self.data)
        self.start_theta = [0.5,0.5]
        
        self.max_loop = 1000
        self.gradient_param = 0.01
        self.stop_point = 0.0001
    def set_basic(self, **kwargs):
        self.basic = kwargs
        
        
        '''
        
        Set basic parameters
        
        
        Parameters
        ----------
            
            max_loop : 최대 루프 개수
        
            gradient_param : 기울기 param
        
            stop_point : stop_point 이하면 변곡점
        
            start_theta : 시작모수(초기값 평균, 분산)
            
            isinflect : 변곡점에 도달했는가
            
            alpha_theta : alpha의 분포가 가지는 모수
            
            beta_theta : beta의 분포가 가지는 모수
        
        '''
        
        
        if kwargs.get('max_loop') != None:
            self.max_loop = kwargs['max_loop']
            
        if kwargs.get('gradient_param') != None:
            self.gradient_param = kwargs['gradient_param']
        
        if kwargs.get('stop_point') != None:            
            self.stop_point = kwargs['stop_point']
            
        if kwargs.get('start_theta') != None:
            self.start_theta = kwargs['start_theta']
        
        if kwargs.get('isinflect') != None:
            self.isinflect = kwargs['isinflect']
        
        if kwargs.get('alpha_theta') != None:
            self.alpha_theta = kwargs['alpha_theta']
        
        if kwargs.get('beta_theta') !=None:
            self.beta_theta = kwargs['beta_theta']
            
        if kwargs.get('size') !=None:
            self.size = kwargs['size']
        
    def norm_pdf(self,x,theta):
        '''
        정규분포 pdf 계산
        '''
        mu = theta[0]
        sigma = theta[1]
        norm_pdf = math.pow(2*math.pi,-1/2) / sigma * np.exp(-(np.power((x - mu),2) / (2 * np.power(sigma,2))))
        return norm_pdf    
    
    def dnorm_mu(self,theta):
        '''
        정규분포 log-likelihood function의 Mu에 대한 편미분값 계산
        '''
        n = self.data_length
        mu = theta[0]
        sigma = theta[1]
        dnorm_mu = (self.data_sum - n*mu) / math.pow(sigma,2)
        return dnorm_mu
    
    def dnorm_var(self,theta):
        '''
        정규분포 log-likelihood function의 Var(=sigma**2)에 대한 편미분값 계산
        '''
        n = self.data_length
        mu = theta[0]
        sigma = theta[1]
        var = math.pow(sigma,2)
        
        dnorm_var = (np.sum(np.power(self.data - mu,2)) / math.pow(var,2) ) - (n / var)
        return dnorm_var
    
    
    def gamma_pdf(self,x,theta):
        '''
        감마분포 pdf 계산
        '''
        alpha = theta[0]
        beta = theta[1]
        
        gamma_pdf = np.power(x,alpha-1) * np.exp(-x / beta) / ( np.power(beta,alpha) * scipy.special.gamma(alpha))
        return gamma_pdf 
    
    def dgamma_alpha(self,theta):
        '''
        감마분포 log-likelihood function의 Alpha에 대한 편미분값 계산
        '''
        n = self.data_length
        alpha = theta[0]
        beta = theta[1]
        
        dgamma_alpha = self.data_log_sum - (n * np.log(beta)) - (n*scipy.special.digamma(alpha))
        return dgamma_alpha
    
    def dgamma_beta(self,theta):
        '''
        감마분포 log-likelihood function의 Beta에 대한 편미분값 계산
        '''
        n = self.data_length
        alpha = theta[0]
        beta = theta[1]
        dgamma_beta = (self.data_sum / np.power(beta,2)) - (n * alpha / beta) 
        return dgamma_beta
    
    def map_pdf(self,x,theta,alpha_theta,beta_theta):
        '''
        Calculate MAP pdf
        
        f(theta|x) = f(x|theta) * f(theta)

        f(x|theta) ~ Gamma(theta)
        f(alpha) ~ Norm(alpha_theta)
        f(beta) ~ Norm(beta_theta)
        
        Parameters
        ----------
        theta = [alpha, beta]
        alpha_theta = [alpha_mu, alpha_sigma]
        beta_theta = [beta_mu,beta_sigma]
        '''
        
        alpha = theta[0]
        beta = theta[1]
        gamma_pdf = self.gamma_pdf(x,theta) # f(x | theta) ~ Gamma(theta)
        
        
        alpha_array = np.full(self.data_length,alpha)
        beta_array = np.full(self.data_length,beta)
        
        alpha_pdf = self.norm_pdf(alpha_array,alpha_theta) # f(alpha) ~ Norm(alpha_theta)
        beta_pdf = self.norm_pdf(beta_array,beta_theta) # f(beta) ~ Norm(beta_theta)
        
        map_pdf = gamma_pdf * alpha_pdf * beta_pdf
        
        return map_pdf 
    
    def dmap_alpha(self,theta,alpha_theta):
        '''
        MAP log-likelihood function(X~Gamma, theta~Norm) Alpha에 대한 편미분값 계산
        '''
        n = self.data_length
        alpha = theta[0]
        beta = theta[1]
        
        mu = alpha_theta[0]
        sigma = alpha_theta[1]
        
        dgamma_alpha = self.data_log_sum - (n * np.log(beta)) - (n*scipy.special.digamma(alpha)) - (n*(alpha-mu) / np.power(sigma,2))
        
        return dgamma_alpha
    
    def dmap_beta(self,theta,beta_theta):
        '''
        MAP log-likelihood function(X~Gamma, theta~Norm) Beta에 대한 편미분값 계산
        '''
        n = self.data_length
        alpha = theta[0]
        beta = theta[1]
        
        mu = beta_theta[0]
        sigma = beta_theta[1]
        dgamma_beta = (self.data_sum / np.power(beta,2)) - (n * alpha / beta) - (n*(beta-mu) / np.power(sigma,2))
        return dgamma_beta
    
    def linear_pdf(self,data,theta):
        y_data = self.data[:,0]
        x1_data = self.data[:,1]
        x2_data = self.data[:,2]
        new_y = y_data - x2_data
        new_x = x1_data - x2_data
        linear_pdf = np.sum(np.power((new_y - theta[0] * new_x),2)) / self.data_length
        return linear_pdf
    
    def dlinear_alpha(self,theta):
        y_data = self.data[:,0]
        x1_data = self.data[:,1]
        x2_data = self.data[:,2]
        
        new_y = y_data - x2_data
        new_x = x1_data - x2_data
        dlinear_alpha = (2 * theta[0] * np.sum(np.power(new_x,2)) - 2 * np.sum(new_x * new_y)) / self.data_length
        return dlinear_alpha
        
    def update(self,dist,theta,alpha_theta = None, beta_theta = None):
        '''
        theta값의 변화에 따른 미분값을 통해 변곡점으로 update
        
        New_theta += gradient_alpha * d(Old_theta)
        
        if abs(d(Old_theta)) <= stop_point: 변곡점(inflect) 도달 
        
        '''
        if dist == 'norm':
            
            dnorm_mu = self.dnorm_mu(theta) # Mu 기울기 계산
            theta[0] += self.gradient_param * dnorm_mu # Update Mu
            
            dnorm_var = self.dnorm_var(theta) # Var 기울기계산
            theta[1] += self.gradient_param * dnorm_var # Update Var
            
            if abs(dnorm_mu) + abs(dnorm_var) <=self.stop_point: # 변곡점확인
                self.isinflect = True
            
            
            
        elif dist == 'gamma':
            dgamma_alpha = self.dgamma_alpha(theta) # Alpha 기울기 계산
            theta[0] += self.gradient_param * dgamma_alpha # Update Alpha
            
            dgamma_beta = self.dgamma_beta(theta) # Beta 기울기 계산
            theta[1] += self.gradient_param * dgamma_beta # Update beta
            
            if abs(dgamma_alpha) + abs(dgamma_beta) <=self.stop_point: # 변곡점 확인
                self.isinflect = True
                
        elif dist == 'map':
            
            
            dmap_alpha = self.dmap_alpha(theta,alpha_theta) # Alpha 기울기 계산
            theta[0] += self.gradient_param * dmap_alpha # Update Alpha
            
            dmap_beta = self.dmap_beta(theta,beta_theta) # Beta 기울기 계산
            theta[1] += self.gradient_param * dmap_beta # Update Beta
            
            if abs(dmap_alpha) + abs(dmap_beta) <=self.stop_point: # 변곡점 확인
                self.isinflect = True
        
        elif dist == 'linear':
            dlinear_alpha = self.dlinear_alpha(theta) # Alpha 기울기 계산
            theta[0] -= self.gradient_param * dlinear_alpha # Update Alpha
            
            if abs(dlinear_alpha) * self.gradient_param <= self.stop_point: #변곡점확인
                self.isinflect = True
            elif theta[0]>=1:
                self.isinflect = True
            elif theta[0]<=0:
                self.isinflect = True
        
        return theta
    
    
    def fit(self,dist):
        old_theta = self.start_theta
        
        
        if dist == 'norm':
            old_likelihood = self.norm_pdf(self.data,old_theta)
        elif dist == 'gamma':
            old_likelihood = self.gamma_pdf(self.data,old_theta)
        elif dist == 'map':
            if self.alpha_theta == None or self.beta_theta == None:
                raise ValueError('set_alpha_beta_theta를 통해 모수를 입력해주세요')
            old_likelihood = self.map_pdf(self.data,old_theta,self.alpha_theta,self.beta_theta)
        elif dist == 'linear':
            old_likelihood = old_theta[0]
        else:
            raise ValueError('dist is not proposed')
        
        
        old_cost = np.sum(old_likelihood)
        old_array = np.array([old_cost,old_theta[0],old_theta[1]]).reshape(1,3)   
        for i in range(self.max_loop):
            
            new_theta = self.update(dist,old_theta,self.alpha_theta, self.beta_theta)
            
            if dist == 'norm':
                new_likelihood = self.norm_pdf(self.data,new_theta)
            elif dist == 'gamma':
                new_likelihood = self.gamma_pdf(self.data,new_theta)
            elif dist == 'map':
                new_likelihood = self.map_pdf(self.data,new_theta, self.alpha_theta,self.beta_theta)
            elif dist=='linear':
                new_likelihood = self.linear_pdf(self.data,new_theta)
            cost = np.sum(new_likelihood)
            
            
            new_array = np.array([cost,new_theta[0], new_theta[1]])
            old_array = np.vstack([old_array,new_array])
            
            old_theta = new_theta
            
            if self.isinflect:
                self.isinflect = False
                break
        
        return old_array
    
    def gamma_sample(self,theta):
        '''
        
        Get gamma sample array(n)
        
        theta : [alpha, beta]
        
        size : sample size
        
        
        '''

        return np.random.gamma(shape = theta[0], scale = theta[1], size = self.size)
        
    def norm_sample(self,**kwargs):
        '''
        
        Get gamma sample array(n)
        
        theta : [alpha, beta]
        
        size : sample size
        
        
        '''
        
        theta = self.start_theta
        size = self.size
        
        if kwargs.get('theta')!=None:
            theta = kwargs['theta']
            
        if kwargs.get('size')!=None:
            size = kwargs['size']
        
        
        return np.random.normal(loc = theta[0],scale = math.sqrt(theta[1]),size = size)
    
    
    def mod_array_zero_to_num(self,data,num):
        '''
        Get array(n)
            
            array의 0값을 num으로 교체
            감마분포 mle계산시 0값으로 인해 정확한 추정치를 얻을 수 없음
            
        '''
        data = np.where(data == 0, num, data)
        data = data.astype(np.float)
        return data
    def pita_model(x1,x2,a):
        y = a * x1 + (1-a) * x2
        return y