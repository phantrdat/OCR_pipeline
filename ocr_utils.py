import cv2
from collections import OrderedDict
import math
import scipy.spatial.distance as distance
import numpy as np
class Params:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

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

# def str2bool(v):
# 	return v.lower() in ("yes", "y", "true", "t", "1")

def plot_one_box(img, c1, c2, c3=None, c4=None, label=None, score=None, color=None, line_thickness=None, box_type='rectangle'):
	
	# tl = int(round(0.3 * max(img.shape[0:2])))  # line thickness
	if box_type =='rectangle':
		height = abs(c1[1] - c2[1])
		cv2.rectangle(img, c1, c2, color, thickness=1)
	else:
		height = abs(c1[1] - c3[1])
		cv2.polylines(img, [np.array([c1,c2,c3,c4])], isClosed=True, color=color, thickness=1)
	tl = int(round(0.03*height)) if int(round(0.03*height)) else 1
	if label:
		tf = max(tl - 2, 1)  # font thickness
		text_sz = float(tl)/3
		t_size = cv2.getTextSize(label, 0, fontScale=text_sz, thickness=tf)[0]
		c2 = c1[0] + t_size[0]+15, c1[1] - t_size[1] -3
		cv2.rectangle(img, c1, c2 , color, -1)  # filled
		cv2.putText(img, '{}'.format(label), (c1[0],c1[1] - 2), 0, text_sz, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)
def sortBBox(json_list):
	points = []
	for bb in json_list:
		P = [bb['text'],  [[bb['x1'], bb['y1']],
						   [bb['x2'], bb['y1']],
						   [bb['x2'], bb['y2']],
						   [bb['x1'], bb['y2']]]
						   ]
		points.append(P)
	points = list(map(lambda x:[x[0],x[1][0],x[1][2]],points))
	points_sum = list(map(lambda x: [x[0],x[1],sum(x[1]),x[2][1]],points))
	x_y_cordinate = list(map(lambda x: x[1],points_sum))
	sorted_list = []
	while True:
		try:
			new_sorted_text = []
			initial_value_A  = [i for i in sorted(enumerate(points_sum), key=lambda x:x[1][2])][0]
			threshold_value = abs(initial_value_A[1][1][1] - initial_value_A[1][3])
			threshold_value = (threshold_value/2) + 5
			del points_sum[initial_value_A[0]]
			del x_y_cordinate[initial_value_A[0]]
			A = [initial_value_A[1][1]]
			K = list(map(lambda x:[x,abs(x[1]-initial_value_A[1][1][1])],x_y_cordinate))
			K = [[count,i]for count,i in enumerate(K)]
			K = [i for i in K if i[1][1] <= threshold_value]
			sorted_K = list(map(lambda x:[x[0],x[1][0]],sorted(K,key=lambda x:x[1][1])))
			B = []
			points_index = []
			for tmp_K in sorted_K:
				points_index.append(tmp_K[0])
				B.append(tmp_K[1])
			dist = distance.cdist(A,B)[0]
			d_index = [i for i in sorted(zip(dist,points_index), key=lambda x:x[0])]
			new_sorted_text.append(initial_value_A[1])

			index = []
			for j in d_index:
				new_sorted_text.append(points_sum[j[1]])
				index.append(j[1])
			for n in sorted(index, reverse=True):
				del points_sum[n]
				del x_y_cordinate[n]
			sorted_list.append(new_sorted_text)
		except Exception as e:
			break
	
	# return list of [text, (x1,y1), (x1+y1), y2]
	sorted_list = sorted(sorted_list, key=lambda x:x[0][1][1])
	print(sorted_list)
	sorted_json_list = []
	for line in sorted_list:
		for s in line:
			for bb in json_list:
				if s[0] == bb['text'] and s[1] == [bb['x1'], bb['y1']]:
					sorted_json_list.append(bb)
	return sorted_json_list

	# final_sorted_list = []    
	# for text in sorted_list:
	# 	final_sorted_list.append([t[0] for t in text])
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

if __name__ == '__main__':

	# points = [['11/10,', 
	# [  [466.66666, 261.33334],
    #    [532.     , 261.33334],
    #    [532.     , 285.33334],
    #    [466.66666, 285.33334] ]],
    #    ['st', [[556.     , 261.33334],
    #    [582.6667 , 261.33334],
    #    [582.6667 , 285.33334],
    #    [556.     , 285.33334]]], ['Str', [[586.6667 , 261.33334],
    #    [626.6667 , 261.33334],
    #    [626.6667 , 285.33334],
    #    [586.6667 , 285.33334]]], ['R', [[377.33334, 262.66666],
    #    [400.     , 262.66666],
    #    [400.     , 285.33334],
    #    [377.33334, 285.33334]]], ['si.', [[410.66666, 264.     ],
    #    [442.66666, 264.     ],
    #    [442.66666, 285.33334],
    #    [410.66666, 285.33334]]], ['1.', [[544.     , 264.     ],
    #    [561.3333 , 264.     ],
    #    [561.3333 , 285.33334],
    #    [544.     , 285.33334]]], ['et,', [[637.3333, 264.    ],
    #    [670.6667, 264.    ],
    #    [670.6667, 288.    ],
    #    [637.3333, 288.    ]]], ['et', [[396.     , 265.33334],
    #    [414.66666, 265.33334],
    #    [414.66666, 285.33334],
    #    [396.     , 285.33334]]], ["'el", [[622.6667 , 265.33334],
    #    [641.3333 , 265.33334],
    #    [641.3333 , 285.33334],
    #    [622.6667 , 285.33334]]], ['in', [[529.3333 , 276.     ],
    #    [537.3333 , 276.     ],
    #    [537.3333 , 285.33334],
    #    [529.3333 , 285.33334]]], ['Corporati', [[378.73196, 287.75485],
    #    [482.9534 , 289.35825],
    #    [482.57034, 314.25494],
    #    [378.3489 , 312.65155]]], ['ion', [[478.66666, 288.     ],
    #    [518.6667 , 288.     ],
    #    [518.6667 , 309.33334],
    #    [478.66666, 309.33334]]], ['Colony,', [[525.82104, 285.5305 ],
    #    [614.4748 , 291.07132],
    #    [613.00653, 314.5629 ],
    #    [524.3528 , 309.02206]]], ['T.Nafgg,', [[377.85098, 309.27054],
    #    [470.8392 , 316.4235 ],
    #    [468.88623, 341.81174],
    #    [375.89804, 334.65878]]], ['Chennai', [[476.     , 313.33334],
    #    [566.6667 , 313.33334],
    #    [566.6667 , 336.     ],
    #    [476.     , 336.     ]]], ['48.', [[592.     , 313.33334],
    #    [626.6667 , 313.33334],
    #    [626.6667 , 334.66666],
    #    [592.     , 334.66666]]]]
	import json

	json_list = json.load(open('json/2_0.jpg.json'))
	# print(len(json_list))
	merged_list = sortBBox(json_list)
	for m in json_list:
		if m not in merged_list:
			print(m)
	# print(len())
