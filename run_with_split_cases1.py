import cv2
import numpy as np
import glob
import os
import time
from shutil import copy, copy2, rmtree
import json

from test import detect_and_split
outdir = 'result_out'

def plot_one_box(img, c1, c2, label=None, score=None, color=None, line_thickness=None):
	tl = int(round(0.001 * max(img.shape[0:2])))  # line thickness
	cv2.rectangle(img, c1, c2, color, thickness=1)
	if label:
		tf = max(tl - 2, 1)  # font thickness
		text_sz = float(tl) /4
		t_size = cv2.getTextSize(label, 0, fontScale=text_sz, thickness=tf)[0]
		c2 = c1[0] + t_size[0]+15, c1[1] - t_size[1] -3
		cv2.rectangle(img, c1, c2 , color, -1)  # filled
		cv2.putText(img, '{}'.format(label), (c1[0],c1[1] - 2), 0, text_sz, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

if __name__ == '__main__':
	img_paths = 'test_im/'
	# for p in img_paths:
	# 	print(p)
	# 	name = os.path.basename(p)
	# 	name = name.split('_')[1]
	# 	print(name)
		# try:

		# if os.path.isfile(os.path.join(outdir, name+'.png')) ==False:

		# 	path_out = os.path.join(outdir, name)
		# 	if os.path.isdir(path_out)==False:
		# 		os.makedirs(path_out)
	im = cv2.imread('/home/phantrdat/Desktop/Scene_Text_Detection/CRAFT-pytorch/SI/SI_resize/CLTV11054145_BKKVK5711300_1_ocps_si@opussmtp.one-line.com_4(ocps20191105162229_4)_1.png')

	json_list , h_lines, v_lines = detect_and_split(img_paths, h_thres=1.25, v_thres=0.5)
	for b in json_list:
		x1 = b['x1'] 
		x2 = b['x2']
		y1 = b['y1']
		y2 = b['y2']
		conf = b['confdt']
		text = b['text']
		plot_one_box(im, (x1,y1), (x2,y2), label=text, score = conf, color=(0, 0, 255), line_thickness=1)
	print(json_list)
	w = im.shape[1]
	for l in h_lines:
		im = cv2.line(im, (0, l), (w, l), (0, 255, 0), 3)
	for i in range(0, len(h_lines),2):
		for v_l in v_lines[i//2]:
			im = cv2.line(im, (v_l, h_lines[i]), (v_l, h_lines[i+1]), (0, 255, 0), 2)
	cv2.imwrite('result_out/A.png', im)

