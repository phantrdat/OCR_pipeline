from PIL import Image
import glob
D = input('Dir:')

ims = glob.glob(D+'/*.png')
for path  in ims:
	im = Image.open(path)
	bg = Image.new("RGB", im.size, (255,255,255))
	bg.paste(im,im)
	bg.save(path.replace('.png','.jpg'))