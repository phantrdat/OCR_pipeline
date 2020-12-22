from ocr_config import Config
import os
from ocr import OCR
import cv2
import json
import subprocess
import os

def preprocessing(img_path):
	pre_img_path = img_path.split('/')[-1] # pre_0.jpg
	command = './imgtxtenh/imgtxtenh ' + img_path + ' -p ' + './pre_result/'+pre_img_path # ../../imgtxtenh/imgtxtenh 0.jpg -p pre_0.jpg
	p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	scpComm = 'cp /home/haophan/quephuong/OCR_pipline/pre_result/'+pre_img_path +' /home/haophan/quephuong/OCR_pipline/pre_result/pre_result/'+pre_img_path
	q = subprocess.Popen(scpComm, shell=True, stdout=subprocess.PIPE)
	return pre_img_path

def new_check_bb(data):
	checkBB={}
	for d in data:
		key=d['text']+"_"+str(d['x1'])+"_"+str(d['y1'])+"_"+str(d['x2'])+"_"+str(d['y2'])
		checkBB[key]="True"
	return checkBB
def sorted_in_line(line):
	x1=[]
	index=0
	new_line=[]
	for i in line:
		x1.append(i['x1'])
	x1=sorted(x1)

	for x in x1:
		for i in line:
			if x==i['x1']:
				new_line.append(i)
				break
	return new_line

def merge_bb(bb1,bb2):
	new_bb={
		'text': bb1['text']+bb2['text'],
		'x1': bb1['x1'],
		'y1': min(bb1['y1'],bb2['y1']),
		'x2':bb2['x2'],
		'y2':max(bb1['y2'],bb2['y2']),
		'confdt': max(bb1['confdt'],bb2['confdt'])
	}
	return new_bb
def sorting_bb(data):
	check = new_check_bb(data)
	dem=0
	sorted_line={}
	final_list=[]

	for d in data:
		key=d['text']+"_"+str(d['x1'])+"_"+str(d['y1'])+"_"+str(d['x2'])+"_"+str(d['y2'])
		if check[key] != "False":
			line=[]
			check[key]="False"
			y2_tmp=d['y2']
			line.append(d)
			for k in data:
				keyK=k['text']+"_"+str(k['x1'])+"_"+str(k['y1'])+"_"+str(k['x2'])+"_"+str(k['y2'])
				if check[keyK] != "False":
					k_y2_tmp=k['y2']
					# print("k_y2: ",k_y2_tmp)
					if abs(y2_tmp - k_y2_tmp)<=10:
						# print(keyK)
						check[keyK]="False"
						line.append(k)
					elif abs(y2_tmp - k_y2_tmp)>10:
						# print("huhu")
						break
			dem+=1
			sorted_line[dem]=sorted_in_line(line)
			for i in range(len(sorted_line[dem])):
				if i < (len(sorted_line[dem])-1):
					if sorted_line[dem][i]['x2'] > sorted_line[dem][i+1]['x1']:
						sorted_line[dem][i]=merge_bb(sorted_line[dem][i],sorted_line[dem][i+1])
						sorted_line[dem].pop(i+1) 
	for j in sorted_line.keys():
		# print(j)
		for k in sorted_line[j]:
			final_list.append(k)
	return final_list


if __name__ == '__main__':
	img_folder='./data/'
	cfg = Config()
	ocr = OCR(cfg)
	ocr.load_net()


	
	img_list=os.listdir(img_folder)
	if not os.path.isdir('./data'):
			os.mkdir('./data')
	for img in img_list:
		# print(img_list)
		img_name=img.split('/')[-1].split('_')[0]
		print(img_name)
		if img_name=="data":
			print('hihi')
			continue
		else:
			if not os.path.isdir('./result/'+img_name):
				os.mkdir('./result/'+img_name)
		# Use original version (no_split + craft + scatter)
		# image = 'test_im/1.png' # or image = imgproc.loadImage('test_im/1.png')
			# pre_img_path=preprocessing('./data/'+img)
			# print(pre_img_path)
			# q = subprocess.Popen('cd /home/haophan/quephuong/OCR_pipline/', shell=True, stdout=subprocess.PIPE)
			json_list_original= ocr.ocr('./data/'+img)
			final_original=sorting_bb(json_list_original)

			image = ocr.plot(img, final_original)
			cv2.imwrite('result/'+img_name+'/'+'original.png',image)
			with open ('result/'+img_name+'/'+'original.json','w') as f:
				json.dump(final_original,f,ensure_ascii=True)

			# Use enhanced version (split + craft + scatter)
			# image = 'test_im/1.png' # or image = imgproc.loadImage('test_im/1.png')
			json_list_split,_,_ = ocr.ocr_with_split('./data/'+img)
			final_split=sorting_bb(json_list_split)
			image = ocr.plot(img, final_split)
			cv2.imwrite('result/'+img_name+'/'+'split.png',image)
			with open ('result/'+img_name+'/'+'split.json','w') as f:
				json.dump(final_split,f,ensure_ascii=True)
