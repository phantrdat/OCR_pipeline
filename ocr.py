import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import cv2
from skimage import io
import numpy as np
from tqdm import tqdm
import more_itertools as mit
import string
from craft_text_detector import *
from scatter_text_recognizer import *
from ocr_utils import copyStateDict, plot_one_box, Params

class OCR:
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def load_net(self):
		""" Loading detection network"""
		self.craft = CRAFT()     # initialize
		print('Loading box detection weights from checkpoint (' + self.cfg.craft_model + ')')
		if self.cfg.cuda:
			self.craft.load_state_dict(copyStateDict(torch.load(self.cfg.craft_model)))
		else:
			self.craft.load_state_dict(copyStateDict(torch.load(self.cfg.craft_model, map_location='cpu')))
		if self.cfg.cuda:
			self.craft = self.craft.cuda()
			cudnn.benchmark = False
		
		self.craft.eval()

		# LinkRefiner
		self.refine_net = None
		if self.cfg.craft_refine:
			from refinenet import RefineNet
			self.refine_net = RefineNet()
			print('Loading weights of refiner from checkpoint (' + self.cfg.craft_refiner_model + ')')
			if self.cfg.cuda:
				self.refine_net.load_state_dict(copyStateDict(torch.load(self.cfg.craft_refiner_model)))
				self.refine_net = self.refine_net.cuda()
				# self.refine_net = torch.nn.DataParallel(self.refine_net)
			else:
				self.refine_net.load_state_dict(copyStateDict(torch.load(self.cfg.craft_refiner_model, map_location='cpu')))
			self.refine_net.eval()
			self.cfg.craft_poly = True

		""" Loading recognition network """ 
		
		if self.cfg.scatter_sensitive:
			self.cfg.scatter_character = string.printable[:-6]

		self.scatter_converter = AttnLabelConverter(self.cfg.scatter_character)
		self.cfg.scatter_num_class = len(self.scatter_converter.character)

		if self.cfg.scatter_rgb:
			self.cfg.scatter_input_channel = 3
		
		self.align_collate = AlignCollate(imgH=self.cfg.scatter_img_h, imgW=self.cfg.scatter_img_w, keep_ratio_with_pad=self.cfg.scatter_pad)
		self.scatter_params = Params(FeatureExtraction=self.cfg.scatter_feature_extraction, PAD=self.cfg.scatter_pad,
			batch_max_length=self.cfg.scatter_batch_max_length, batch_size=self.cfg.scatter_batch_size, 
			character=self.cfg.scatter_character, 
			hidden_size=self.cfg.scatter_hidden_size, imgH=self.cfg.scatter_img_h, imgW=self.cfg.scatter_img_w, 
			input_channel=self.cfg.scatter_input_channel, num_fiducial=self.cfg.scatter_num_fiducial, num_gpu=self.cfg.scatter_num_gpu, 
			output_channel=self.cfg.scatter_output_channel, 
			rgb=self.cfg.scatter_rgb, saved_model=self.cfg.scatter_model, sensitive=self.cfg.scatter_sensitive, 
			workers=self.cfg.scatter_workers, num_class=self.cfg.scatter_num_class)

		self.scatter = SCATTER(self.scatter_params)
		self.scatter = torch.nn.DataParallel(self.scatter).to(self.device)

		print('loading pretrained model from %s' % self.cfg.scatter_model)
		self.scatter.load_state_dict(torch.load(self.cfg.scatter_model, map_location=self.device))
		self.scatter.eval()

	def detection(self, image, is_rendered=False):
		if isinstance(image, str):
			image = imgproc.loadImage(image)
		t0 = time.time()

		# resize
		img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.cfg.craft_canvas_size,
			 								interpolation=cv2.INTER_LINEAR, mag_ratio=self.cfg.craft_mag_ratio)
		ratio_h = ratio_w = 1 / target_ratio

		# preprocessing
		x = imgproc.normalizeMeanVariance(img_resized)
		x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
		x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
		if self.cfg.cuda:
			x = x.to(self.device)

		# forward pass
		with torch.no_grad():
			y, feature = self.craft(x) #CRAFT 

		# make score and link map
		score_text = y[0,:,:,0].cpu().data.numpy()
		score_link = y[0,:,:,1].cpu().data.numpy()

		# refine link
		if self.refine_net is not None:
			with torch.no_grad():
				y_refiner = self.refine_net(y, feature)
			score_link = y_refiner[0,:,:,0].cpu().data.numpy()

		t0 = time.time() - t0
		t1 = time.time()

		# Post-processing
		boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.cfg.craft_text_threshold, self.cfg.craft_link_threshold,
		 self.cfg.craft_low_text, self.cfg.craft_poly)

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
		
		# print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
		return boxes, polys, score_text, target_ratio
	def recognize(self, textbb_dict):

		data = StreamDataset(self.scatter_params, textbb_dict)  # use StreamDataset
		loader = torch.utils.data.DataLoader(
			data, batch_size=self.scatter_params.batch_size,
			shuffle=False,
			num_workers=int(self.scatter_params.workers),
			collate_fn=self.align_collate, pin_memory=True)

		# predict
		
		final_preds = []
		final_conf = []
		with torch.no_grad():
			for image_tensors, image_path_list in loader:
				all_block_preds = []
				all_confidence_scores = []
				batch_size = image_tensors.size(0)
				image = image_tensors.to(self.device)
				# For max length prediction
				length_for_pred = torch.IntTensor([self.scatter_params.batch_max_length] * batch_size).to(self.device)
				text_for_pred = torch.LongTensor(batch_size, self.scatter_params.batch_max_length + 1).fill_(0).to(self.device)

				predss = self.scatter(image, text_for_pred, is_train=False)[0]
				

				for i, preds in enumerate(predss):
					confidence_score_list = []
					pred_str_list = []

					# select max probability (greedy decoding) then decode index to character
					_, preds_index = preds.max(2)
					preds_str = self.scatter_converter.decode(preds_index, length_for_pred)
					
					preds_prob = F.softmax(preds, dim=2)
					preds_max_prob, _ = preds_prob.max(dim=2)
					for pred, pred_max_prob in zip(preds_str, preds_max_prob):
						pred_EOS = pred.find('[s]')
						pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
						pred_str_list.append(pred)
						pred_max_prob = pred_max_prob[:pred_EOS]
						
						# calculate confidence score (= multiply of pred_max_prob)
						try:
							confidence_score = pred_max_prob.cumprod(dim=0)[-1].cpu().numpy()
						except:
							confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
						confidence_score_list.append(confidence_score)
				
					all_block_preds.append(pred_str_list)
					all_confidence_scores.append(confidence_score_list)
				
				all_confidence_scores =  np.array(all_confidence_scores)
				all_block_preds = np.array(all_block_preds)

				best_pred_index = np.argmax(all_confidence_scores, axis=0)
				best_pred_index = np.expand_dims(best_pred_index, axis=0)

				# Get max predition per image through blocks
				all_block_preds = np.take_along_axis(all_block_preds, best_pred_index, axis=0)[0]
				all_confidence_scores = np.take_along_axis(all_confidence_scores, best_pred_index, axis=0)[0]
				
				final_conf.extend(all_confidence_scores.tolist())
				final_preds.extend(all_block_preds.tolist())

		return final_preds, final_conf

	def ocr(self, image):
		if isinstance(image, str):
			image = imgproc.loadImage(image)
		if self.cfg.transform_type == "dilation":
			kernel = np.ones(self.cfg.transform_kernel_size,np.uint8)
			# kernel = np.random.choice([0,1], size=self.cfg.transform_kernel_size).astype(np.uint8)
			transformed_image = cv2.dilate(1-image, kernel, iterations = 1)
		else:
			transformed_image = image.copy()
		bboxes, polys, score_text, target_ratio = self.detection(transformed_image)
		raw_img = image[:,:,::-1]
		clone = raw_img.copy()
		all_text = {}
		coords = []
		for i  in range(len(polys)):
			try:
				pts = polys[i]
				rect = cv2.boundingRect(pts)
				x,y,w,h = rect
				x, y, w, h = max(x,0), max(y,0), max(w,0), max(h,0)
				
				if self.cfg.craft_padding_ratio != None:
					box_padding = int(h/self.cfg.craft_padding_ratio)
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
		pred_str, pred_conf = self.recognize(all_text)
		json_list = []
		for boxes, text, conf in zip(polys, pred_str, pred_conf):
			word_pred_dict = {}
			word_pred_dict['text'] = text
			if self.cfg.craft_padding_ratio != None:
				h = max(0,int(boxes[2][1])) - max(0,int(boxes[0][1]))
				box_padding = int(h/self.cfg.craft_padding_ratio)
			else:
				box_padding =0
			x1, y1, x2, y2  = max(0,int(boxes[0][0])), max(0,int(boxes[0][1])), max(0,int(boxes[2][0])) + box_padding, max(0,int(boxes[2][1]) + box_padding)
			word_pred_dict['x1'], word_pred_dict['y1'],word_pred_dict['x2'],word_pred_dict['y2']= x1, y1, x2, y2
			word_pred_dict['confdt'] = conf  
			json_list.append(word_pred_dict)
		self.json_list = json_list
		return json_list

	def merge_box(self):
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
		self.json_list 
		json_process = sorting_bb(self.json_list)
		return json_process

	def ocr_with_split(self, image, h_thres=2, v_thres=0.3): # Threshold for splitting line horizontally and vertically:
		def consec(lst):
			G = mit.consecutive_groups(lst)
			G = [list(g) for g in G]
			return G

		t = time.time()
		final_output = {}
		if isinstance(image, str):
			image = imgproc.loadImage(image)
			

		im_height, im_width, _ = image.shape
		_, _, score_text, target_ratio = self.detection(image)



		# First split text image horizontally, then split vertically
		horizontal_line_score = np.sum(score_text, axis=1)
		horizontal_empty_line_index = np.where(horizontal_line_score<= h_thres)[0].tolist()
		horizontal_empty_line_index = consec(horizontal_empty_line_index)
		horizontal_empty_line_index = [e for e in horizontal_empty_line_index if len(e)>1]
		horizontal_empty_line_index.insert(0,[0])

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
		if len(horizontal_cut_lines)==0:
			horizontal_cut_lines = [0, im_height]
		for i in range(0,len(horizontal_cut_lines),2):
			patch_score = score_text[horizontal_cut_lines[i]:horizontal_cut_lines[i+1]]
			vertical_patch_score = np.sum(patch_score, axis=0)
			vertical_empty_patch_index = np.where(vertical_patch_score<= v_thres)[0].tolist()
			vertical_empty_patch_index = consec(vertical_empty_patch_index)
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
			if len(v_l)==0:
				v_l = [0, im_width]
			for j in range(len(v_l)-1):
				split_im = image.copy()[final_horizontal_cut_lines[i]:final_horizontal_cut_lines[i+1], v_l[j]:v_l[j+1]]
				json_list = self.ocr(split_im)
				# cv2.imwrite(f'h_{final_horizontal_cut_lines[i]}{final_horizontal_cut_lines[i+1]}-w_{v_l[j]}{v_l[j+1]}.png', split_im)
				for k in range(len(json_list)):
					json_list[k]['x1'] += v_l[j]
					json_list[k]['y1'] += final_horizontal_cut_lines[i]
					json_list[k]['x2'] += v_l[j]
					json_list[k]['y2'] += final_horizontal_cut_lines[i]
				final_json_list.extend(json_list)
		
		return final_json_list, final_horizontal_cut_lines, vertical_cut_lines
	
	def plot(self, image, json_list):
		if isinstance(image, str):
			image = imgproc.loadImage(image)
		for b in json_list:
			# print(b)
			x1 = b['x1'] 
			x2 = b['x2']
			y1 = b['y1']
			y2 = b['y2']
			conf = b['confdt']
			text = b['text']
			plot_one_box(image, (x1,y1), (x2,y2), label=text, score = conf, color=(0, 0, 255), line_thickness=1)
		return image
	
	





