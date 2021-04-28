from ocr_config import Config
from ocr import OCR
import cv2
import os
import json
from ocr_utils import sortBBox
import warnings
import math
import time
warnings.filterwarnings('ignore')

# PK form setting
MAX_WIDTH = 1024
V_MERGE_THRESHOLD = 0.085
H_MERGE_THRESHOLD = 0.02

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
		
def match(text, list_keys):
	if list_keys == []:
		return False
	for k in list_keys:
		if k not in text:
			return False
	return True

def find_nearest_PK_form(key, possible_values, direction):
	m = 9999999999
	item = 0
	if direction == 'horizontal':
		pk = (max(key['x1'], key['x2']) , (key['y1']+ key['y2']) * 0.5)
		for i, v in enumerate(possible_values):
				pv = (min(v['x1'], v['x2']) , (v['y1']+ v['y2']) * 0.5)
				dis = distance(pk, pv, 1, 1)
				if dis < m:
					m = dis
					item = i
		if 'is_key' not in possible_values[item]:
			return possible_values[item]
		else:
			return None
		 
# def map_PK_form(merged_parts, scale_x, scale_y):
# 	KEY_MAP = [ {"Key": 'DO No.', "With": ['DO','No'],  "Without": []},
# 	{"Key": 'AO / PO No.', "With": ['AO', 'PO'],  "Without": []},
# 	{"Key": 'ITEMCODE', "With": ['ITEMCODE'], "Without": []}, 
# 	{"Key": 'CARGO CLS DATE', "With": ['CARGO'], "Without": []},
# 	{"Key": 'LOT NO.', "With": ['LOT'], "Without": []},
# 	{"Key": 'COLOR', "With": ['COLOR'],  "Without": ['COLORS']}, 
# 	{"Key": 'PKG# CARTON QTY', "With": ['PKG#'], "Without": []},
# 	{"Key": 'CARTON QTY', "With": ['CARTON', 'QTY'],  "Without": ['PKG#']},
# 	{"Key": 'CARTON NO.', "With": ['CARTON', 'NO'], "Without": ['PKG#','QTY']}]

# 	# INV_KEY_MAP = {v:k for k,v in KEY_MAP.items()}
	
# 	for m in KEY_MAP:
# 		for i, line in enumerate(merged_parts):
# 			if match(line['text'], m['With']) == True and match(line['text'], m['Without']) == False:
# 				merged_parts[i]['text'] = m['Key']
# 				merged_parts[i]['is_key'] = True
# 				break
# 	keys = [box for box in merged_parts if 'is_key' in box]
# 	possible_values = [box for box in merged_parts if 'is_key' not in box]
# 	possible_values = merge_PK_form(possible_values, scale_x=scale_x, scale_y=scale_y, threshold=0.1, direction='vertical')
# 	for k in keys:
# 		values = [v for v in possible_values if is_same_line(k, v) == True and v['x1']>k['x1'] and len(v['text'])>1]
# 		same_line_keys = [k1 for k1 in keys if is_same_line(k, k1) == True and k!=k1 and len(k1['text'])>1]

# 		if len(values) == 0:
# 			print(f"Key: {k['text']}, Value: ")
# 		else:
# 			nearest_item = find_nearest_PK_form(k, values + same_line_keys, direction='horizontal')
# 			if nearest_item == None:
# 				print(f"Key: {k['text']}, Value: ")
# 			elif k['x1'] < nearest_item['x1']:
# 				print(f"Key: {k['text']}, Value: {nearest_item['text']}")
# 			else:
# 				print(f"Key: {k['text']}, Value: ")

def map_PK_form(merged_parts, scale_x, scale_y):

	KEY_MAP = [ {"Key": 'DO No.', "With": ['DO','No'],  "Without": [], "l_key":None, 'u_key':'AO / PO No.'},
	{"Key": 'AO / PO No.', "With": ['AO', 'PO'],  "Without": [], "l_key":"DO No.", 'u_key':'ITEMCODE'},
	{"Key": 'ITEMCODE', "With": ['ITEMCODE'], "Without": [], "l_key":'AO / PO No.', 'u_key':'COLOR'}, 
	
	{"Key": 'COLOR', "With": ['COLOR'],  "Without": ['COLORS'], "l_key":"ITEMCODE", 'u_key':'CARGO CLS DATE'}, 

	{"Key": 'PKG# CARTON QTY', "With": ['PKG#'], "Without": [], "l_key":'COLOR', 'u_key':'CARTON QTY'},
	{"Key": 'CARGO CLS DATE', "With": ['CARGO'], "Without": [], "l_key":'COLOR', 'u_key':'LOT NO.'},


	{"Key": 'CARTON QTY', "With": ['CARTON', 'QTY'],  "Without": ['PKG#'], "l_key":'PKG# CARTON QTY', 'u_key':'CARTON NO.'},
	{"Key": 'LOT NO.', "With": ['LOT', 'NO'], "Without": [], "l_key":'CARGO CLS DATE', 'u_key':None},

	
	{"Key": 'CARTON NO.', "With": ['CARTON', 'NO'], "Without": ['PKG#','QTY'], "l_key":'CARTON QTY', 'u_key':None}]

	keys = []
	for m in KEY_MAP:
		for i, line in enumerate(merged_parts):
			if match(line['text'], m['With']) == True and match(line['text'], m['Without']) == False:
				merged_parts[i]['text'] = m['Key']
				merged_parts[i]['is_key'] = True
				merged_parts[i]['l_key'] = m['l_key']
				merged_parts[i]['u_key'] = m['u_key']
				keys.append(merged_parts[i])
				break
	for i in range(len(keys)):
		pre = [lk for lk in keys if lk['text'] == keys[i]['l_key'] if keys[i]['l_key']!=None]
		pre = pre[0] if len(pre)>0 else None
		tail = [uk for uk in keys if uk['text'] == keys[i]['u_key'] if keys[i]['u_key']!=None]
		tail = tail[0] if len(tail)>0 else None
		if pre !=None:
			keys[i]['l_bound'] = pre['y2']
		else:
			keys[i]['l_bound'] = keys[i]['y1'] - int(0.3*(keys[i]['y2'] - keys[i]['y1']))
		if tail!=None:
			keys[i]['u_bound'] = tail['y1']
		else:
			keys[i]['u_bound'] = keys[i]['y2'] + int(0.65*(keys[i]['y2'] - keys[i]['y1']))

		
	
			
		
	possible_values = [box for box in merged_parts if 'is_key' not in box]
	
	result = []
	for k in keys:
		values = [v for v in possible_values if v['y1']>=k['l_bound'] and v['y2']<= k['u_bound'] and v['x1']>=k['x1'] and len(v['text'])>1]
		# values = sorted(enumerate(values), key=lambda x:x['x1'])
		values = merge_PK_form(values, scale_x=scale_x, scale_y=scale_y, threshold=V_MERGE_THRESHOLD, direction='vertical')
		# values = [v for v in possible_values if is_same_line(k, v) == True and v['x1']>k['x1'] and len(v['text'])>1]
		same_line_keys = [k1 for k1 in keys if k1['y1']>=k['l_bound'] and k1['y2']<= k['u_bound'] and k!=k1 and len(k1['text'])>1]

		if len(values) == 0:
			result.append({'Key': k['text'], 'Value':''})
			# print(f"Key: {k['text']}, Value: ")
		else:
			nearest_item = find_nearest_PK_form(k, values + same_line_keys, direction='horizontal')
			if nearest_item == None:
				result.append({'Key': k['text'], 'Value':''})
			elif k['x1'] < nearest_item['x1']:
				result.append({'Key': k['text'], 'Value':nearest_item['text']})
			else:
				result.append({'Key': k['text'], 'Value':''})
	sorted_results = []
	for m in KEY_MAP:
		for res in result:
			if res['Key'] == m['Key']:
				sorted_results.append(res)

	return sorted_results

	

def is_same_line(text1, text2):
	center1_y  = 0.5*(text1['y1']+ text1['y2'])
	center2_y  = 0.5*(text2['y1']+ text2['y2'])
	if center1_y >= center2_y and center1_y - center2_y > 0.5*(text1['y2'] -  text1['y1']):
		return False
	if center1_y < center2_y and center2_y - center1_y > 0.5*(text2['y2'] -  text2['y1']):
		return False
	return True

def check_merged_or_not(merged_index, idx):
	for m in merged_index:
		if idx in m:
			return True
	return False

def is_near(text1, text2, scale_x, scale_y, threshold, direction):
	if direction == 'horizontal':
		if text1['x1']>text2['x1']:
			temp = text1
			text1 = text2
			text2 = temp
		
		pa1 = (text1['x2'], text1['y1'])
		pa2 = (text1['x2'], text1['y2'])
		pb1 = (text2['x1'], text2['y1'])
		pb2 = (text2['x1'], text2['y2'])
		dis = distance(pa1, pb1,scale_x ,scale_y) + distance(pa2, pb2,scale_x ,scale_y)
		dis = dis/2
		if dis <= threshold:
			return True
	else:
		if text1['y1']>text2['y1']:
			temp = text1
			text1 = text2
			text2 = temp
		dis = abs(text1['y1']- text2['y2'])/scale_y 
		if dis <= threshold and  abs(text1['x1']- text2['x1'])/scale_x < threshold:
			return True 
	
	return False

def merge_two_boxes(text1, text2, direction):
	if direction == 'horizontal':
		if text1['x1']<text2['x1']:
			txt = ' '.join([text1['text'], text2['text']])
		else:
			txt = ' '.join([text2['text'], text1['text']])
	if direction == 'vertical':
		if text1['y1']<text2['y1']:
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

def merge(json_list, scale_x ,scale_y, threshold=0.03, direction='horizontal'):
	merged_json_list = []
	
	merged_index = []
	single_index = []
	L = len(json_list)

	i = 0
	for i in range(L-1):
		for j in range(i+1, L):
			if check_merged_or_not(merged_index, i) == False and check_merged_or_not(merged_index, j) == False:
				if is_near(json_list[i], json_list[j],scale_x, scale_y, threshold, direction):
					if (direction == 'horizontal'and is_same_line(json_list[i], json_list[j])) or direction=='vertical':
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
		merged_json_list.append(merge_two_boxes(json_list[m[0]], json_list[m[1]], direction))
	for i in single_index:
		merged_json_list.append(json_list[i])

	return merged_json_list
	
def merge_PK_form(json_list, scale_x, scale_y, threshold, direction):
	while True:
		old_L = len(json_list)
		json_list = merge(json_list, scale_x, scale_y, threshold=threshold, direction=direction)
		new_L = len(json_list)
		if old_L == new_L:
			return json_list



if __name__ == '__main__':

	cfg = Config()
	ocr = OCR(cfg)
	ocr.load_net()




	sub_folder = 'PK_form'
	# images = sorted(os.listdir(f'test_im/{sub_folder}'))
	images = ['DSC05472.jpg']
	# images = ['nghieng_warped_warped.png', 'nghieng.png']
	for im in images:
		for i in range(1):
			t1 = time.time()
			print(f'------------------- Image {im} ------------------------------')
			image_name = f'test_im/{sub_folder}/{im}'
			image = cv2.imread(image_name)
			h, w, _ = image.shape
			if w > MAX_WIDTH: 
				h, w = int((h*MAX_WIDTH)/w), MAX_WIDTH
				image = cv2.resize(image, (w, h))

			
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			# json_list, all_parts,_,_ = ocr.ocr_with_split(image, h_slide= 5, v_slide=2)
			json_list = ocr.ocr(image)
			# json_list = sortBBox(json_list)
			# image = cv2.imread(image_name)
			# image = ocr.plot(image, final_json_list)
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			# cv2.imwrite(f'1_{im}',image)

			# image = cv2.imread(image_name)
			json_list = merge_PK_form(json_list, scale_x=w, scale_y=h, threshold=H_MERGE_THRESHOLD, direction='horizontal')
			# image = ocr.plot(image, json_list)
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			# cv2.imwrite(f'2_{im}',image)

			# json.dump(json_list, open(f'json/2_{im}.json','w'))
			# print(json_list)
			# if len(json_list)==1 and len(json_list[0]['text'])!=3:
			json_list = ocr.re_regconize(image, json_list,min_inp_confidence=0.98, min_out_confidence=0.8, s_length=7)

			# json.dump(json_list, open(f'json/3_{im}.json','w'))

			# json_list = merge_PK_form(json_list, scale_x=w, scale_y=h, threshold=0.07, direction='vertical')


			res = map_PK_form(json_list, scale_x=w, scale_y=h)

			for  r in res:
				print (r)

			# image = cv2.imread(image_name)
			# image = ocr.plot(image, json_list)
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			# cv2.imwrite(f'captcha_result/{im}',image)

			print(time.time() - t1)

		# image = cv2.imread(image_name)
		# result = map_key(merged_parts)
		# # for x in merged_parts:
		# # 	print(x)
		# for each_res in result:
		# 	k,v = each_res
			
		# 	image = ocr.plot(image, [k], color=(255, 0, 0), line_thickness=2)
		# 	image = ocr.plot(image, [v], color=(50, 168, 82), line_thickness=2)
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# cv2.imwrite(f'2_{im}',image)


	



	
	# Use enhanced version (split + craft + scatter)
	# image = f'test_im/{im_idx}.png' # or image = imgproc.loadImage('test_im/1.png')
	# final_json_list,_,_ = ocr.ocr_with_split(image)
	# image = ocr.plot(image, final_json_list)
	# if ocr.cfg.transform_type!=None:
	# 	prefix = 'transform'
	# else:
	# 	prefix = "normal"
	# cv2.imwrite(f'result/{prefix}_{im_idx}_split.jpg',image)
