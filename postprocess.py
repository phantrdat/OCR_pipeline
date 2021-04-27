import json
import numpy as np
from ocr_config import Config
from ocr import OCR
import cv2
import os
from tqdm import tqdm
def distance(text1, text2):
    m_dis = np.array([text1['x1'] - text2['x1'], 
                      text1['x2'] - text2['x2'], 
                      text1['y1'] - text2['y1'], 
                      text1['y2'] - text2['y2']])
    return np.linalg.norm(m_dis)
def is_same_line(text1, text2):
    center1_y  = 0.5*(text1['y1']+ text1['y2'])
    center2_y  = 0.5*(text2['y1']+ text2['y2'])
    if center1_y >= center2_y and center1_y - center2_y > 0.5*(text1['y2'] -  text1['y1']):
        return False
    if center1_y < center2_y and center2_y - center1_y > 0.5*(text2['y2'] -  text2['y1']):
        return False
    return True

def is_overlap(text1, text2):
     
    # To check if either rectangle is actually a line
      # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}
       
    if (text1['x1'] == text1['x2'] or text1['y1'] == text1['y2'] or text2['x1'] == text2['x2'] or text2['y1'] == text2['y2']):
        # the line cannot have positive overlap
        return False
       
     
    # If one rectangle is on left side of other
    if(text1['x1'] >= text2['x2'] or text2['x1'] >= text1['x2']):
        return False
 
    # If one rectangle is above other
    if(text1['y2'] <= text2['y1'] or text2['y2'] <= text1['y1']):
        return False
 
    return True    

def merge_two_boxes(text1, text2):
    if (text1['x1']<text2['x1']):
        txt = ' '.join([text1['text'], text2['text']])
    else:
        txt = ' '.join([text2['text'], text1['text']])

    merged_box = {
            "text": txt ,
            "x1": min(text1['x1'], text2['x1']),
            "y1": min(text1['y1'], text2['y1']),
            "x2": max(text1['x2'], text2['x2']),
            "y2": max(text1['y2'], text2['y2']),
            "confdt": min(text1['confdt'] , text2['confdt'])
        }
    return merged_box


def check_merged_or_not(merged_index, idx):
    for m in merged_index:
        if idx in m:
            return True
    return False
def merge_overlap(json_list):
    merged_json_list = []
    
    merged_index = []
    single_index = []
    L = len(json_list)

    i = 0
    for i in range(L-1):
        for j in range(i+1, L):
            if check_merged_or_not(merged_index, i) == False and check_merged_or_not(merged_index, j) ==False:
                if is_overlap(json_list[i], json_list[j]) and is_same_line(json_list[i], json_list[j]):
                    merged_index.append((i,j))
    for i in range(L):
        is_merged = False
        for m in merged_index:
            if i in m:
                is_merged = True
                break
        if is_merged ==False:
            single_index.append(i)
    for m in merged_index:
        merged_json_list.append(merge_two_boxes(json_list[m[0]], json_list[m[1]]))
    for i in single_index:
        merged_json_list.append(json_list[i])

    return merged_json_list

def correct_boxes(dict_1, dict_2):
    for i, x  in enumerate(dict_1):
        if x['confdt'] < 0.99: 
            dis = 9999999999
            nearest_boxes = None
            for j, y in enumerate(dict_2):
                if distance(x,y) < dis:
                    dis = distance(x,y)
                    nearest_boxes = y
            if nearest_boxes != None:
                if x['confdt'] < nearest_boxes['confdt'] and is_same_line(x, nearest_boxes):
                    dict_1[i] = nearest_boxes       
    return dict_1

def add_padding(image, expanding_percent=0.05):
    # src = cv2.imread(f'{name}.png', cv2.IMREAD_COLOR)
    top = int(expanding_percent * src.shape[0])
    bottom = top
    left = int(expanding_percent * src.shape[1])
    right = left

    value = (255,255,255)
    borderType = cv2.BORDER_CONSTANT
    dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
    return dest

def re_regconize(image, json_list, model_obj, threshold = 0.5, max_length = 7):
    results = []
    for k, box in enumerate(json_list):
        try:
            if box['confdt'] < threshold: #or len(box['text']) > max_length:
                x1_root = box['x1']
                y1_root = box['y1']
                part = image[box['y1']:box['y2'], box['x1']:box['x2']].copy()
                sub_part, _ = ocr.split_text_vertically(part, max_length=max_length)
                reg_res = model_obj.recognize(sub_part)

                # Update recognize result
                # if sum(reg_res[1])/len(reg_res[1]) > json_list[k]['confdt']:
                json_list[k]['text'] = ''.join(reg_res[0])
                json_list[k]['confdt'] = sum(reg_res[1])/len(reg_res[1])
                # print(f"-------->{json_list[k]}\n")                                
                
        except:
            pass
    return json_list

def merge_dict(D):
    while True:
        old_L = len(D)
        D = merge_overlap(D)
        new_L = len(D)
        if old_L == new_L:
            return D


     
        
if __name__ == '__main__':
    cfg = Config()
    ocr = OCR(cfg)
    ocr.load_net()

    # idxs = ['DSC06221']


    sub_folder = 'captcha1'
    idxs = os.listdir(f'test_im/{sub_folder}')
    # idxs = [x[:-4] for x in idxs]
    idxs = ['1']
    for im_idx in tqdm(idxs):
        print(im_idx)
        dict_1 = json.load(open(f'json/normal_{im_idx}.json','r'))
        # dict_2 = json.load(open(f'json/normal_{im_idx}_ori.json','r'))
        
        
        dict_1 = merge_dict(dict_1)
        image = cv2.imread(f'test_im/{sub_folder}/{im_idx}.png')
        
        
        augmented_dict = re_regconize(image, dict_1, ocr, threshold=0.97)
        dict1 = correct_boxes(dict_1, augmented_dict)
        image = ocr.plot(image, dict_1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        prefix = 'normal'
        cv2.imwrite(f'correct/corrected_{prefix}_{im_idx}.png',image)

   

    
   
    

    