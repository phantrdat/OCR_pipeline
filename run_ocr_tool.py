from ocr_config import Config
from ocr import OCR
import cv2
import os
import json
from tqdm import tqdm
from ocr_utils import sortBBox
import math
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import numpy as np
import time
import pandas as pd

def distance(p1, p2, scale_x ,scale_y):
	return math.sqrt(((p1[0]-p2[0])/scale_x)**2 + ((p1[1]-p2[1])/scale_y)**2)
def is_merge(p1, p2, scale_x ,scale_y, threshold):
	if distance(p1, p2, scale_x, scale_y) <=threshold:
		return True
	else:
		return False
def join_line(line_parts, scale_x ,scale_y, threshold=0.03):
	merged_list = []
	if len(line_parts) <= 1:
		return line_parts
	# if len(line_parts) ==2 and is_merge(line_parts[0], line_parts[1], scale_x, scale_x ,scale_y, threshold)==False:
		# return line_parts

	merged_index = []
	i = 0
	merged_chain = []
	while i< len(line_parts)-1:
		is_reset = False
		mergeable = True
		for j in range(i+1, len(line_parts)):
			pv = line_parts[i]
			pv1 = (pv['x2'], pv['y1'])
			pv2 = (pv['x2'], pv['y2'])

			pm = line_parts[j]
			pm1 = (pm['x1'], pm['y1'])
			pm2 = (pm['x1'], pm['y2'])
			dis = distance(pv1, pm1,scale_x ,scale_y) + distance(pv2, pm2,scale_x ,scale_y)
			dis = dis/2
			if dis <= threshold:
				merged_chain+=[i,j]
				
			else:
				mergeable = False
				if i not in merged_chain:
					merged_index.append([i])
			i+=1
			break
		if len(merged_chain) !=0 and (i==len(line_parts)-1 or mergeable==False):
			merged_chain = sorted(list(set(merged_chain)))
			merged_index.append(merged_chain)
			is_reset = True
		if i==len(line_parts)-1 and i not in merged_chain:
			merged_index.append([i])
		if is_reset == True:
			merged_chain = []
		
						
	for chain in merged_index:
		part = [line_parts[i] for i in chain]
		min_x1 = min([p['x1'] for p in part])
		min_y1 = min([p['y1'] for p in part])
		min_x2 = max([p['x2'] for p in part])
		min_y2 = max([p['y2'] for p in part])
		text = ' '.join([p['text'] for p in part])
		merged_list.append({
			"text": text,
			'x1': min_x1,
			'y1': min_y1,
			'x2': min_x2,
			'y2': min_y2
		})
	return merged_list
		

def merge(all_parts, scale_x ,scale_y):
	merged_parts = []
	for lines in all_parts:
		line_parts = []
		for part in lines:
			sorted_part = []
			points = []

			if len(part)==1:
				sorted_part.extend(part)
			elif len(part)>1:
				for p in part:
					x1,y1,x2,y2 = p['x1'],p['y1'],p['x2'],p['y2']
					points.append([p['text'], [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]])

				points = sortBBox(points)
				joined_lines = []
				for each_line in points:
					joined_lines+=each_line
				points = joined_lines
				
				for pnt in points:
						# print(pnt, part)
					for p in part:
						if pnt[0] == p['text'] and pnt[1] == [p['x1'], p['y1']]:
							sorted_part.append(p)
			if sorted_part!=[]:
				min_x1 = min([p['x1'] for p in sorted_part])
				min_y1 = min([p['y1'] for p in sorted_part])
				min_x2 = max([p['x2'] for p in sorted_part])
				min_y2 = max([p['y2'] for p in sorted_part])
				text = ' '.join([p['text'] for p in sorted_part])
				line_parts.append({
					"text": text,
					'x1': min_x1,
					'y1': min_y1,
					'x2': min_x2,
					'y2': min_y2
				})
		if line_parts!=[]:
			# line_parts = join_line(line_parts, scale_x ,scale_y)
			merged_parts.append(line_parts)
	
	return merged_parts
# def re_regconize(image, model_obj, json_list):
#     incorrect_items = []
#     for x in json_list:
#         if x['confdt'] < 0.95:
#             incorrect_items.append(x)
	
# 	return incorrect_items


if __name__ == '__main__':

	cfg = Config()
	ocr = OCR(cfg)
	ocr.load_net()


	# parts ,score = ocr.split_text_vertically('./cut/011.png', s_length=7)
	
	# res = ocr.recognize(parts)
	# print(res)
	# print(''.join(res[0]))

	# x_name = [str(i) for i in range(len(score))]
	# print(argrelextrema(np.array(score), np.less))

	# plt.bar(x_name,score)
	# plt.show()



	
	sub_folder = 'resized_png'
	idxs = sorted(os.listdir(f'test_im/{sub_folder}'))

	# idxs = ['DSC05466']
	# types = ['.png']

	types = [x[-4:] for x in idxs]
	idxs = [x[:-4] for x in idxs]
	running_times = []	
	# idxs = ['009']
	for z, im_idx in enumerate(tqdm(idxs)):
			t1 = time.time()
			image = f'test_im/{sub_folder}/{im_idx}{types[z]}' # or image = imgproc.loadImage('test_im/1.png')
			h, w, _ = cv2.imread(image).shape

			# json_list, _,_,_ = ocr.ocr_with_split(image, h_slide=10, v_slide=2)
			json_list = ocr.ocr(image)
			
			json_list = ocr.re_regconize(image, json_list)
		
			running_times.append(time.time()-t1)

			if ocr.cfg.transform_type!=None:
				prefix = 'transform'
			else:
				prefix = "normal"
			image = ocr.plot(image, json_list)
			# json.dump(json_list, open(f'json/{prefix}_{im_idx}.json','w'))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			cv2.imwrite(f'result/{prefix}_{im_idx}.png',image)
	
	stat = pd.DataFrame({'Filename':idxs,'Running Time (No Split)':running_times})
	stat.to_csv()