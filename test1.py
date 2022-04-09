import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def save_aiformat(load_path, save_path):
    output= np.loadtxt(load_path, delimiter=',')
    indx = np.array([i for i in range(output.shape[0])])
    mask = np.equal(np.roll(output[:,:], -1, axis=0), output[:,:])
    #output = video, class ,time
    
    
    
    
    r_mask = np.logical_and(mask[:, 0], mask[:, 1])
    out = output[r_mask].astype(int)

    r_n_mask = np.logical_not(r_mask)

    res = np.hstack([(mask[r_n_mask]), output[r_n_mask]]).astype(int)
    # res = pd.DataFrame(columns=['Equal1','Equal2', index, 'video_ID', 'activityid', 'time'])
    start_time = 0
    outputs= []
    #입력비디오, Class간비교한거(Bool), 인덱스, 비디오아이디, 예측값, 프레임끝번호
    print(res[:10])
    #exit()
    for item in res:
        outputs.append([item[3],item[4], int(start_time/30), int((item[5]+15)/30)])
        if not item[0]: #입력 비디오가 달라졌는지?
            start_time = 0 #입력 시간이 0부터 시작
        elif not item[1]: #이상행동 클래스가 달라졌는지?
            start_time = item[5]+16 #입력 시간이 0부터 시작
        else:
            pass
    
    
    np.savetxt(save_path, np.array(outputs).astype(int), delimiter=' ', fmt='%d')
save_aiformat('./output/tmp.csv', './output/last.txt')

