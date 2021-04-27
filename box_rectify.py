import numpy as np
import cv2
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


if __name__== '__main__':

	pts = np.array([[129.40506, 622.15314],
 [207.91785, 619.5361 ],
 [208.75912, 644.7747 ],
 [130.24634, 647.3918 ]])
	pts= np.array([[100,75],[619,45],[711,396],[19,397]])
	image  = cv2.imread('/home/phantrdat/Desktop/Scene_Text_Detection/CRAFT-pytorch/OCR_pipline/test_im/PK_form/nghieng.png')
	# boxes = np.load('1.npy', allow_pickle=True)
	# X = []
	# Y = []
	# for box in boxes:
	# 	X+= [p[0] for p in box]
	# 	Y+= [p[1] for p in box]
	# min_x = min(X)
	# min_y = min(Y)
	# max_x = max(X) 
	# max_y = max(Y)


	# print(min_x, min_y, max_x, max_y)
	# left_p = None
	# right_p = None
	# top_p = None
	# bottom_p = None

	# for box in boxes:
	# 	for p in box:
	# 		if p[0] == min_x:
	# 			left_p = p
	# 		if p[0] == max_x:
	# 			right_p = p
	# 		if p[1] == min_y:
	# 			bottom_p = p
	# 		if p[1] == max_y:
	# 			top_p = p
	# pts = np.array([left_p, bottom_p, right_p, top_p])
	# minRect = cv2.minAreaRect(pts)
	# pts = cv2.boxPoints(minRect)

	for i, p in enumerate(pts):
		image  = cv2.imread('/home/phantrdat/Downloads/pic.jpg')
		cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
		cv2.imwrite(str(i)+'.jpg', image)
	warped  = four_point_transform(image, pts)
	
	# image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
	# cv2.imwrite('1.jpg', image)
	cv2.imwrite('warped.jpg', warped)