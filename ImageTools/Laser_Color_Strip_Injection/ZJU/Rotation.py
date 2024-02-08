import numpy as np
import math
from PIL import Image


def Rotation(img,angle):
    S1 = img.shape[0]
    S2 = img.shape[1]
    S3 = img.shape[2]
    R = int(math.ceil(math.sqrt(S1 * S1 + S2 * S2)))
    new_fig = np.zeros((R, R, S3), dtype=np.uint8)
    new_fig[:, :, :] = 255
    R = float(R)
    mid_x = S2 / 2
    mid_y = S1 / 2
    im_data = np.rot90(img, k=3)

    for i in range(S2):
        for j in range(S1):
            r = math.sqrt(pow(i - mid_x, 2) + pow(j - mid_y, 2))
            # N = Rot_Matrix*np.array([i-mid_x,j-mid_y]).T
            theta = math.atan2(i - mid_x, j - mid_y)
            new_theta = theta+angle
            new_x = int(round(r * math.cos(new_theta)) + R / 2)
            new_y = int(round(r * math.sin(new_theta)) + R / 2)
            # print(new_x)
            # print(new_y)
            # input('Enter')
            try:
                new_fig[new_y, new_x, :] = im_data[i, j, :]
            except:
                print(new_x)
                print(new_y)
                print(r)
                input('Enter')
    new_fig = np.rot90(new_fig, k=1)
    return new_fig

if __name__=='__main__':
    im = Image.open('000007.png')
    im_data = np.asarray(im)
    new_fig = Rotation(im_data,math.pi)
    img = Image.fromarray(new_fig, 'RGB')
    # img.save('after.jpg')
    img.show()