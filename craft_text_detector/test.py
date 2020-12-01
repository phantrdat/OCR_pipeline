"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
sys.path.insert(0, './cvpr20_scatter_text_recognizer')
from craft import CRAFT
from cvpr20_scatter_text_recognizer.regconization_infer import infer
from collections import OrderedDict
from tqdm import tqdm

def copyStateDict(state_dict):
	if list(state_dict.keys())[0].startswith("module"):
		start_idx = 1
	else:
		start_idx = 0
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = ".".join(k.split(".")[start_idx:])
		new_state_dict[name] = v
	return new_state_dict
def str2bool(v):
	return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.6, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, action='store_true', help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.0, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='test_im/pic1/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('--padding', default=False, action='store_true', help='Box padding or not')
parser.add_argument('--padding_ratio', type=int, default=6, help='ratio of word height and padding for detected boxes which did not cover necessary characters')
args = parser.parse_args()



""" Loading network"""
net = CRAFT()     # initialize

print('Loading weights from checkpoint (' + args.trained_model + ')')
if args.cuda:
	net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
else:
	net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
if args.cuda:
	net = net.cuda()
	net = torch.nn.DataParallel(net)
	cudnn.benchmark = False
net.eval()
# LinkRefiner
refine_net = None
if args.refine:
	from refinenet import RefineNet
	refine_net = RefineNet()
	print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
	if args.cuda:
		refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
		refine_net = refine_net.cuda()
		refine_net = torch.nn.DataParallel(refine_net)
	else:
		refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
	refine_net.eval()
	args.poly = True

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)
image_list = sorted(image_list)
result_folder = './result_out/'
if not os.path.isdir(result_folder):
	os.mkdir(result_folder)



def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None, is_rendered=False):
	t0 = time.time()

	# resize
	img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
	ratio_h = ratio_w = 1 / target_ratio

	# preprocessing
	x = imgproc.normalizeMeanVariance(img_resized)
	x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
	x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
	if cuda:
		x = x.cuda()

	# forward pass
	with torch.no_grad():
		y, feature = net(x) #CRAFT 

	# make score and link map
	score_text = y[0,:,:,0].cpu().data.numpy()
	score_link = y[0,:,:,1].cpu().data.numpy()

	# refine link
	if refine_net is not None:
		with torch.no_grad():
			y_refiner = refine_net(y, feature)
		score_link = y_refiner[0,:,:,0].cpu().data.numpy()

	t0 = time.time() - t0
	t1 = time.time()

	# Post-processing
	boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

	# coordinate adjustment
	boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
	polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
	for k in range(len(polys)):
		if polys[k] is None: polys[k] = boxes[k]

	t1 = time.time() - t1

	# render results (optional)
	ret_score_text = None
	if is_rendered !=False:
		render_img = score_text.copy()
		render_img = np.hstack((render_img, score_link))
		# ret_score_text = imgproc.cvt2HeatmapImg(render_img)
	if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
	return boxes, polys, score_text, target_ratio

def detect_and_split(images_path, h_thres=2, v_thres=0.3):
	def consec(lst):
		it = iter(lst)
		prev = next(it)
		tmp = [prev]
		for ele in it:
			if prev + 1 != ele:
				yield tmp
				tmp = [ele]
			else:
				tmp.append(ele)
			prev = ele
		yield tmp
	image_list, _, _ = file_utils.get_files(images_path)
	image_list = sorted(image_list)
	t = time.time()
	final_output = {}
	for k, image_path in enumerate(tqdm(image_list)):

		image = imgproc.loadImage(image_path)
		_, _, score_text, target_ratio = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
		
		# First split text image horizontally, then split vertically


		horizontal_line_score = np.sum(score_text, axis=1)

		horizontal_empty_line_index = np.where(horizontal_line_score<= h_thres)[0].tolist()
		horizontal_empty_line_index = list(consec(horizontal_empty_line_index))
		horizontal_empty_line_index = [e for e in horizontal_empty_line_index if len(e)>1]

		
		horizontal_cut_lines = []
		for i, line in enumerate(horizontal_empty_line_index):
			if i==0: 
				horizontal_cut_lines.append(line[-1])
			elif i==len(horizontal_empty_line_index)-1:
				horizontal_cut_lines.append(line[0])
			else:
				horizontal_cut_lines.extend([line[0],line[-1]])
		final_horizontal_cut_lines = [int(c*2*(1/target_ratio)) for c in horizontal_cut_lines]

		# Split vertically on each horizontally patches

		vertical_cut_lines = []
		for i in range(0,len(horizontal_cut_lines),2):
			patch_score = score_text[horizontal_cut_lines[i]:horizontal_cut_lines[i+1]]
			vertical_patch_score = np.sum(patch_score, axis=0)
			vertical_empty_patch_index = np.where(vertical_patch_score<= v_thres)[0].tolist()
			vertical_empty_patch_index = list(consec(vertical_empty_patch_index))
			vertical_empty_patch_index = [e for e in vertical_empty_patch_index if len(e)>1]
			vertical_patch_cut_lines = []
			for i, line in enumerate(vertical_empty_patch_index):
				if i==0: 
					vertical_patch_cut_lines.append(line[-1])
				elif i==len(vertical_empty_patch_index)-1:
					vertical_patch_cut_lines.append(line[0])
				else:
					vertical_patch_cut_lines.extend([line[0],line[-1]])
			vertical_cut_lines.append([int(c*2*(1/target_ratio)) for c in vertical_patch_cut_lines])
		final_json_list = []
		for i in range(0, len(final_horizontal_cut_lines)-1,2):
			v_l = vertical_cut_lines[i//2]
			for j in range(len(v_l)-1):
				split_im = image.copy()[final_horizontal_cut_lines[i]:final_horizontal_cut_lines[i+1], v_l[j]:v_l[j+1]]
				json_list = detect_im(split_im)
				for k in range(len(json_list)):
					json_list[k]['x1'] += v_l[j]
					json_list[k]['y1'] += final_horizontal_cut_lines[i]
					json_list[k]['x2'] += v_l[j]
					json_list[k]['y2'] += final_horizontal_cut_lines[i]
				final_json_list.extend(json_list)
		
		return final_json_list, final_horizontal_cut_lines, vertical_cut_lines
	
def detect_im(image):
	bboxes, polys, score_text, target_ratio = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
	raw_img = image[:,:,::-1]
	clone = raw_img.copy()
	all_text = {}
	coords = []
	for i  in range(len(polys)):
		try:
			pts = polys[i]
			rect = cv2.boundingRect(pts)
			x,y,w,h = rect
			
			if args.padding:
				box_padding = int(h/args.padding_ratio)
			else:
				box_padding = 0
			w += box_padding
			h += box_padding
			croped = clone[y:y+h, x:x+w].copy()
			p1 = max(0,int(pts[0][0])) 
			p2 = max(0,int(pts[0][1]))
			p3 = max(0,int(pts[2][0])) 
			p4 = max(0,int(pts[2][1])) 
			cbb = f'{p1}-{p2}_{p3}-{p4}'
			all_text[cbb] = Image.fromarray(croped)
		except Exception:
		    pass
	pred_str, pred_conf = infer(all_text)
	json_list = []
	for boxes, text, conf in zip(polys, pred_str, pred_conf):
		word_pred_dict = {}
		word_pred_dict['text'] = text
		if args.padding:
			h = max(0,int(boxes[2][1])) - max(0,int(boxes[0][1]))
			box_padding = int(h/args.padding_ratio)
		else:
			box_padding =0
		x1, y1, x2, y2  = max(0,int(boxes[0][0])), max(0,int(boxes[0][1])), max(0,int(boxes[2][0])) + box_padding, max(0,int(boxes[2][1]) + box_padding)
		word_pred_dict['x1'], word_pred_dict['y1'],word_pred_dict['x2'],word_pred_dict['y2']= x1, y1, x2, y2
		word_pred_dict['confdt'] = conf  
		json_list.append(word_pred_dict)
	

	return json_list
	

def detect(images_path):
	image_list, _, _ = file_utils.get_files(images_path)
	image_list = sorted(image_list)
	t = time.time()
	final_output = {}
	for k, image_path in enumerate(tqdm(image_list)):
		fname = os.path.basename(image_path)
		image = imgproc.loadImage(image_path)
		json_list = detect_im(image)
		final_output[fname] = json_list
	return final_output


if __name__ == '__main__':
	# load net
	# net = CRAFT()     # initialize

	# print('Loading weights from checkpoint (' + args.trained_model + ')')
	# if args.cuda:
	#     net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
	# else:
	#     net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

	# if args.cuda:
	#     net = net.cuda()
	#     net = torch.nn.DataParallel(net)
	#     cudnn.benchmark = False

	# net.eval()

	# # LinkRefiner
	# refine_net = None
	# if args.refine:
	#     from refinenet import RefineNet
	#     refine_net = RefineNet()
	#     print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
	#     if args.cuda:
	#         refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
	#         refine_net = refine_net.cuda()
	#         refine_net = torch.nn.DataParallel(refine_net)
	#     else:
	#         refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

	#     refine_net.eval()
	#     args.poly = True

	t = time.time()

	for k, image_path in enumerate(image_list):
		t1 = time.time()
		# print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
		filename, file_ext = os.path.splitext(os.path.basename(image_path))

		t_stge1 = time.time()
		image = imgproc.loadImage(image_path)
		bboxes, polys, score_text, target_ratio = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
		


		# score_text = cv2.resize(score_text, (score_text.shape[1], score_text.shape[0]*2))
		# score_text = cv2.resize(score_text, (image.shape[1], image.shape[0]))
		# cv2.imwrite(f"{filename}_heatmap.jpg", score_text)
		# print(score_text.shape)
		
		

		raw_img = image[:,:,::-1]
		fname = os.path.basename(image_path)
		clone = raw_img.copy()
		all_text = {}
		coords = []
		for i  in range(len(polys)):
			try:
				pts = polys[i]
				rect = cv2.boundingRect(pts)
				x,y,w,h = rect
				croped = clone[y:y+h, x:x+w].copy()
				p1 = max(0,int(pts[0][0])) 
				p2 = max(0,int(pts[0][1]))
				p3 = max(0,int(pts[2][0]))
				p4 = max(0,int(pts[2][1]))
				# px = max(0,int(pts[1][0]))
				# py = max(0,int(pts[1][1]))
				# pz = max(0,int(pts[3][0]))
				# pt = max(0,int(pts[3][1]))
				
				# coords.append((p1, p2, p3, p4, px, py, pz, pt))
				# pts = pts - pts.min(axis=0)
				# mask = np.zeros(croped.shape[:2], np.uint8)
				# ctr = np.array(pts).reshape((-1,1,2)).astype(np.int32)
				# cv2.drawContours(mask, [ctr], -1, (255, 255, 255), -1, cv2.LINE_AA)

				# ## (3) do bit-op
				# dst = cv2.bitwise_and(croped, croped, mask=mask)
				# ## (4) add the white background
				# bg = np.ones_like(croped, np.uint8)*255
				# cv2.bitwise_not(bg,bg, mask=mask)
				# final_crop = bg + dst

				# cropped_im = clone[p2:p4, p1:p3]
				# cropped_im = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2RGB)
				# cropped_im = Image.fromarray(cropped_im)
				# cbb = f'{c[0]}-{c[1]}_{c[2]}-{c[3]}_{c[4]}-{c[5]}_{c[6]}-{c[7]}'
				final_crop = croped
				cbb = f'{fname[:-4]}_{p1}-{p2}_{p3}-{p4}'
				cv2.imwrite(f'./test_im/text_cropped/{cbb}.jpg', final_crop)
				all_text[cbb] = Image.fromarray(final_crop)
			except Exception:
				pass
		# print("Transfer Stage", time.time()- t_stagemid)
		# t_reg = time.time()
		pred_str, pred_conf = infer(all_text)
		for ps, pc in zip(pred_str, pred_conf):
			print(ps, pc)
		raw_img = cv2.imread(image_path)
		json_list = []

		json_out_name = image_path[:-4] + '.json'
		for boxes, text, conf in zip(polys, pred_str, pred_conf):
			word_pred_dict = {}
			x1, y1, x2, y2  = max(0,int(boxes[0][0])), max(0,int(boxes[0][1])), max(0,int(boxes[2][0])), max(0,int(boxes[2][1]))
			word_pred_dict['text'] = text
			word_pred_dict['x1'], word_pred_dict['y1'],word_pred_dict['x2'],word_pred_dict['y2']= x1, y1, x2, y2
			word_pred_dict['confdt'] = conf
			
			json_list.append(word_pred_dict)
			plot_one_box(raw_img , boxes, text, score=conf, color=(0, 0, 255), line_thickness=1, poly=True)
			
			# cv2.imwrite(f'{args.test_folder}/text_cropped/{fname},{cbb}.jpg',cropped_im)
		
		filename, file_ext = os.path.splitext(os.path.basename(image_path))
		out_file = os.path.join(result_folder,filename + ".jpg")
		cv2.imwrite(out_file, raw_img)
		print('\n'+ json_out_name + '\n\n\n')
		# with open(json_out_name, 'w') as json_dump_out:
		# 	json.dump(json_list, json_dump_out)
		
	print(time.time() - t)
	print("elapsed time : {}s".format(time.time() - t))

