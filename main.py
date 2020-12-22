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
			ocr.ocr('./data/'+img)

			final_original=ocr.merge_box()

			image = ocr.plot(img, final_original)
			cv2.imwrite('result/'+img_name+'/'+'original.png',image)
			with open ('result/'+img_name+'/'+'original.json','w') as f:
				json.dump(final_original,f,ensure_ascii=True)

			# Use enhanced version (split + craft + scatter)
			# image = 'test_im/1.png' # or image = imgproc.loadImage('test_im/1.png')
			ocr.ocr_with_split('./data/'+img)
			final_split=ocr.merge_box()

			image = ocr.plot(img, final_split)
			cv2.imwrite('result/'+img_name+'/'+'split.png',image)
			with open ('result/'+img_name+'/'+'split.json','w') as f:
				json.dump(final_split,f,ensure_ascii=True)
