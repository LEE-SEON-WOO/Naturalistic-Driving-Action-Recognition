import pandas as pd
from scipy import ndimage

def postprocessing(csvName,savePath):
    output = pd.read_csv(csvName, header=None)
    
    for _ in range (3):
        output[1] = ndimage.median_filter(output[1], size=25,mode='nearest')

        first=0
        for video_name in range(1,11):
            tmp = output[output[0]==video_name][1].reset_index()
            count = 0
            start = 0
            stack = []
            end = 1
            idx = 0
            for i in range(0,len(tmp[1])-1):
                if i == 0 :
                    tmp[1][i] = tmp[1][i+1]
                    count += 1
                    value = tmp[1][i+1]
                    continue
                elif i == len(tmp[1])-1 :
                    tmp[1][i] = tmp[1][i-1]
                    break
                if tmp[1][i] == tmp[1][i-1]:
                    count += 1
                    end = i
                    continue
                else :
                    stack.append([idx,value,count,[start,end]])
                    start = i
                    value = tmp[1][i]
                    end = i
                    idx += 1
                    count = 0
            
            for idx,value, count, lens in stack:
                if idx == 0 :
                    if count <= 28 :
                        tmp[1][lens[0]:lens[1]+1] = stack[idx+1][1]
                        continue
                if idx == len(stack)-1 :
                    if count <= 28 :
                        tmp[1][lens[0]:lens[1]+1] = stack[idx-1][1]      
                        continue

                if count <= 28 :
                    if stack[idx-1][2] > stack[idx+1][2]:
                        tmp[1][lens[0]:lens[1]+1] = stack[idx-1][1]
                    else :
                        tmp[1][lens[0]:lens[1]+1] = stack[idx+1][1]
            output[1][first:first+len(tmp)] = tmp[1]
            first = first + len(tmp)
            
    output.to_csv(savePath,index=False)