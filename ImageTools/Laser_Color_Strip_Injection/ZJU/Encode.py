import math
import numpy as np
from PIL import Image,ImageFilter
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

def add_exposure(img,angle,exposure,row,n):
    img_np = np.asarray(img)
    new_img = np.zeros((img_np.shape[1],img_np.shape[2]))
    figs = []
    for i in range(n+1):
        figs = img.rotate(angle+i/n*exposure,fillcolor='white')
        f = np.asarray(figs)
        new_img += f[row,:,:]
    new_img /= (n+1)
    return new_img


im = Image.open('decoded.png')
im = im.filter(ImageFilter.GaussianBlur(radius=0.1))
im = im.filter(ImageFilter.SMOOTH_MORE)
im_data = np.asarray(im)
S1 = im_data.shape[0]
S2 = im_data.shape[1]
S3 = im_data.shape[2]
# R = int(math.ceil(math.sqrt(S1 * S1 + S2 * S2)))
R = max(S1,S2)

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
w = 10800.0 # 1/30 per round
v = 15000.0
f = 0 # 1/100 rotation rate (smaller means faster exposure time)
n = 2 # exposure times

for i in range(R):
    t = i/v
    angle = w*t
    exposure = f
    angle = angle
    # rot_figure = im.rotate(angle,fillcolor='white')
    # rot_figure = np.asarray(rot_figure)
    # new_fig[i,:,:] = rot_figure[i,:,:]
    new_row = add_exposure(im,angle,exposure,i,n)
    new_fig[i,:,:] = new_row
    if i%100==0:
        print(i)

img = Image.fromarray(new_fig, 'RGBA')
img.save('final.png')
img.show()

