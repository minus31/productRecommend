import pandas as pd 
import numpy as np 

def mAP(results, k=21, AP=True):
    """
    Arguments : 
        results = related images list for sorted file list
        k = mAP@k
        AP = whether return AP list for each images 
    
    return : mAP value
    """
    aps = []
    
    for truth, result in enumerate(results):

        result = result[:k]
        print("truth", truth, "result 3: ", result[:3])
        # AP 계산
        correct = 0
        precisions = [] 
        for j, r in enumerate(result):
            print(r.split("-")[0])
            
            if int(r.split("_")[0]) == truth:
                correct += 1
                precision = correct / (j + 1)
                precisions.append(precision)
                
            else: 
                continue
                    
        if len(precisions) == 0:
            aps.append(0)
            continue
            
        aps.append(np.mean(precisions))   
    
    if AP :
        return np.mean(aps), aps
    
    return np.mean(aps)     

# def mAP(results, k=21, AP=True):
#     """
#     Arguments : 
#         results = related images list for sorted file list
#         k = mAP@k
#         AP = whether return AP list for each images 
    
#     return : mAP value
#     """

#     truths = pd.read_csv("./data/test/test.csv").related_items

#     aps = []
    
#     for i, truth in enumerate(truths):
        
#         truth = truth.split()
        
#         if len(truth) < k:
#             result = results[i][:len(truth)]
#         else: 
#             result = results[i][:k]
#             truth = [str(t) for t in truth[:k]]

#         # AP 계산
#         correct = 0
#         precisions = [] 
#         for i, r in enumerate(result):
#             if str(r) in truth:
#                 correct += 1
#                 precision = correct / (i + 1)
#                 precisions.append(precision)
                
#             else: 
#                 continue  

#         if len(precisions) == 0:
#             aps.append(0)
#             continue
            
#         aps.append(np.mean(precisions))   
    
#     if AP :
#         return np.mean(aps), aps
    
#     return np.mean(aps)     




if __name__ == '__main__':

    pass
