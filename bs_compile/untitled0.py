#%%
def solution(enter, leave):
    import numpy as np
    length=len(enter)
    ent = [0 for i in range(length)]
    lea = [0 for i in range(length)]
    count = 0
    lea_count = 0
    
    cap = set()
    
    while(True):
        
            
        if count<length:
            e = enter[count]
            cap.add(e)
            ent[e-1] = count
            
        l = leave[lea_count]
        
    
        if l in cap:
            lea[l-1] = count
            lea_count +=1
            
            cap.remove(l)  
        
        count+=1    
        if lea_count == length:
            break
        
        
    print(ent,lea)
    answer = []
    ent=np.array(ent).reshape(-1,1)
    lea=np.array(lea).reshape(-1,1)
    total = np.hstack([ent, lea])
    
    #total=total[:,1:]
    #print(total[:,0])
    
    for t in total:
        
        
        a=t[0]
        b=t[1]
        
        c=len(total[(total[:,0] < a)&(total[:,1] > b),:])
        d=len(total[(total[:,0] > a)&(total[:,0] < b),:])
        e =c+d
        answer.append(e)
        
        

    return answer
#%%
solution(	[1, 4, 2, 3], [2, 1, 3, 4])