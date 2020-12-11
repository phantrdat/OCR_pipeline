from ocr_config import Config
from ocr import OCR
import cv2
if __name__ == '__main__':

	cfg = Config()
	ocr = OCR(cfg)
	ocr.load_net()

	im_idx = 3

	# # Use original version (no_split + craft + scatter)
	# image = f'test_im/{im_idx}.png' # or image = imgproc.loadImage('test_im/1.png')
	# final_json_list= ocr.ocr(image)
	# image = ocr.plot(image, final_json_list)
	# cv2.imwrite(f'result/{im_idx}_original.jpg',image)



	# Use enhanced version (split + craft + scatter)
	image = f'test_im/{im_idx}.png' # or image = imgproc.loadImage('test_im/1.png')
	final_json_list,_,_ = ocr.ocr_with_split(image)
	image = ocr.plot(image, final_json_list)
	if ocr.cfg.transform_type!=None:
		prefix = 'transform'
	else:
		prefix = "normal"
	cv2.imwrite(f'result/{prefix}_{im_idx}_split.jpg',image)