from PIL import Image
import glob
import sys

images = []
for i in range(100):
    file_name = sys.argv[1] + '/picture-' + str(i).zfill(3) + '.png'
    im = Image.open(file_name)
    images.append(im)
images[0].save(sys.argv[1] + '/image.gif' , save_all = True , append_images = images[1:] , duration = 100 , loop = 0)