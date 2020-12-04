from ocr_config import Config
from ocr import OCR
import cv2
if __name__ == '__main__':

	cfg = Config()
	ocr = OCR(cfg)
	ocr.load_net()

	# Use original version (no_split + craft + scatter)
	
	image = 'test_im/3.png' # or image = imgproc.loadImage('test_im/1.png')
	final_json_list= ocr.ocr(image)
	image = ocr.plot(image, final_json_list)
	cv2.imwrite('result/3_original.jpg',image)



	# Use enhanced version (split + craft + scatter)
	image = 'test_im/3.png' # or image = imgproc.loadImage('test_im/1.png')
	final_json_list,_,_ = ocr.ocr_with_split(image)
	image = ocr.plot(image, final_json_list)
	cv2.imwrite('result/3_split.jpg',image)