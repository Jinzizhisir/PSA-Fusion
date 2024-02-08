import math
import numpy as np
from PIL import Image
from Rotation import Rotation


def add_margin(img,top,right,bottom,left,color):
    width,height = img.size
    top = int(top)
    right = int(right)
    bottom = int(bottom)
    left = int(left)
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(img.mode,(new_width,new_height),color)
    result.paste(img,(left,top))
    return result


im = Image.open('demo.jpg')
im_data = np.asarray(im)
S1 = im_data.shape[0]
S2 = im_data.shape[1]
S3 = im_data.shape[2]
R = int(math.ceil(math.sqrt(S1 * S1 + S2 * S2)))
# R = max(S1,S2)

if (R-S1)%2==0 and (R-S2)%2==0:
    im = add_margin(im,(R-S1)/2,(R-S2)/2,(R-S1)/2,(R-S2)/2,(255,255,255,0))
elif (R-S1)%2!=0 and (R-S2)%2==0:
    im = add_margin(im,(R-S1)/2-0.5,(R-S2)/2,(R-S1)/2+0.5,(R-S2)/2,(255,255,255,0))
elif (R-S1)%2==0 and (R-S2)%2!=0:
    im = add_margin(im,(R-S1)/2,(R-S2)/2-0.5,(R-S1)/2,(R-S2)/2+0.5,(255,255,255,0))
else:
    im = add_margin(im,(R-S1)/2-0.5,(R-S2)/2-0.5,(R-S1)/2+0.5,(R-S2)/2+0.5,(255,255,255,0))
new_fig = np.zeros((R, R, S3), dtype=np.uint8)
new_fig[:,:,:3] = 255
w = 1.0
v = 1.0

for i in range(R):
    t = i/v
    angle = w*t
    rot_figure = im.rotate(angle,fillcolor='white')
    rot_figure = np.asarray(rot_figure)
    # print(rot_figure.shape)
    # input('Enter')
    new_fig[i,:,:] = rot_figure[i,:,:]
    if i%100==0:
        print(i)

img = Image.fromarray(new_fig, 'RGB')
img.save('encoded.png')
img.show()

