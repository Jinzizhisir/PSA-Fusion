import math
import numpy as np
from PIL import Image
from Rotation import Rotation
import matplotlib.pyplot as plt


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

def new_image(img,row,angle):
    img_data = np.asarray(img)
    new_img_np = np.zeros((img_data.shape[0],img_data.shape[1],4),dtype=np.uint8)
    new_img_np[:,:,:3] = 255
    new_img_np[row,:,:3] = img_data[row,:,:]
    # s = new_img_np.shape[1]
    new_img_np[row,:,3] = 100
    new_img = Image.fromarray(new_img_np,'RGBA')
    new_img = new_img.rotate(angle,fillcolor='white')
    return new_img


im = Image.open('000007.png')
im_data = np.asarray(im)
S1 = im_data.shape[0]
S2 = im_data.shape[1]
S3 = im_data.shape[2]
# R = int(math.ceil(math.sqrt(S1 * S1 + S2 * S2)))
R = max(S1,S2)

if (R-S1)%2==0 and (R-S2)%2==0:
    im = add_margin(im,(R-S1)/2,(R-S2)/2,(R-S1)/2,(R-S2)/2,(255,255,255))
elif (R-S1)%2!=0 and (R-S2)%2==0:
    im = add_margin(im,(R-S1)/2-0.5,(R-S2)/2,(R-S1)/2+0.5,(R-S2)/2,(255,255,255))
elif (R-S1)%2==0 and (R-S2)%2!=0:
    im = add_margin(im,(R-S1)/2,(R-S2)/2-0.5,(R-S1)/2,(R-S2)/2+0.5,(255,255,255))
else:
    im = add_margin(im,(R-S1)/2-0.5,(R-S2)/2-0.5,(R-S1)/2+0.5,(R-S2)/2+0.5,(255,255,255))
new_fig = np.zeros((R, R, 4), dtype=np.uint8)
new_fig[:,:,:3] = 255
# print(new_fig.shape)
# input('Enter')
new_fig = Image.fromarray(new_fig,'RGBA')
w = 10800
v = 15000

for i in range(R):
    t = i / v
    angle = w * t
    # if angle > 180:
    #     continue
    new_im = new_image(im,i,-angle)
    new_fig = Image.alpha_composite(new_im,new_fig)

    # new_fig.paste(new_im,(0,0),new_im)
    if i % 50 == 0:
        print(i)
        # new_fig.show()
        # input('Enter')

# new_fig = np.asarray(new_fig).copy()
# for j in range(new_fig.shape[0]):
#     for k in range(new_fig.shape[1]):
#         if new_fig[j,k,3]<100:
#             new_fig[j,k,3] = 0
# new_fig = Image.fromarray(new_fig,'RGBA')

# new_fig = new_fig.rotate(-(R-S1)/v)
# img = Image.fromarray(new_fig, 'RGBA')
new_fig.save('decoded.png')
new_fig.show()