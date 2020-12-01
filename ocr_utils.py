import cv2
from collections import OrderedDict

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
