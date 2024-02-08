import numpy as np
import math
from PIL import Image
from scipy.optimize import bisect
import matplotlib.pyplot as plt


def newton(f,Df,x0,epsilon,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.
    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            # print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            # print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    # print('Exceeded maximum iterations. No solution found.')
    return None

def Rotate(theta,R,r,w,v):
    p = lambda x: r*math.sin(w*x+theta)-R+v*x
    # Dp = lambda x: r*w*math.cos(w*x+theta)+v
    # epsilon = 0.01
    # max_iter = 10000
    # approx = newton(p, Dp, 0, epsilon, max_iter)
    approx = bisect(p,0,2*R/v,xtol=1e-3)
    # t = (R-r*math.sin(theta))/v
    # theta_new = theta + w*t
    theta_new = theta+w*approx
    theta_new = theta_new%(2*math.pi)
    return theta_new


im = Image.open('000007.png')
im_data = np.asarray(im)
S1 = im_data.shape[0]
S2 = im_data.shape[1]
S3 = im_data.shape[2]
R = int(math.ceil(math.sqrt(S1*S1+S2*S2)))
new_fig = np.zeros((R,R,S3),dtype=np.uint8)
new_fig[:,:,:] = 255
# img = Image.fromarray(new_fig,'RGB')
# img.show()
R = float(R)
w = 1.0
v = 50.0
mid_x = S2/2
mid_y = S1/2
# Rot_Matrix = np.empty((2,2))
# Rot_Matrix[0,0] = math.cos(math.pi/2)
# Rot_Matrix[0,1] = -math.sin(math.pi/2)
# Rot_Matrix[1,0] = math.sin(math.pi/2)
# Rot_Matrix[1,1] = math.cos(math.pi/2)
im_data = np.rot90(im_data,k=3)
# img = Image.fromarray(im_data,'RGB')
# img.show()


for i in range(S2):
    for j in range(S1):
        r = math.sqrt(pow(i-mid_x,2)+pow(j-mid_y,2))
        # N = Rot_Matrix*np.array([i-mid_x,j-mid_y]).T
        theta = math.atan2(i-mid_x,j-mid_y)
        new_theta = Rotate(theta,R,r,w,v)
        new_x = int(round(r*math.cos(new_theta))+R/2)
        new_y = int(round(r*math.sin(new_theta))+R/2)
        # print(new_x)
        # print(new_y)
        # input('Enter')
        try:
            new_fig[new_y,new_x,:] = im_data[i,j,:]
        except:
            print(new_x)
            print(new_y)
            print(r)
            input('Enter')
    if i%100==0:
        print('i=',i)
# print(new_fig[0:100,0:100,1])
new_fig = np.rot90(new_fig, k=1)
# for _ in range(3):
#     for i in range(1,new_fig.shape[0]-1):
#         for j in range(1,new_fig.shape[1]-1):
#             for k in range(3):
#                 if new_fig[i,j,k]==255:
#                     if new_fig[i-1,j,k]!=255 and new_fig[i+1,j,k]!=255:
#                         new_fig[i,j,k] = max(new_fig[i-1,j,k],new_fig[i+1,j,k])
#                     if new_fig[i,j-1,k]!=255 and new_fig[i,j+1,k]!=255:
#                         new_fig[i,j,k] = max(new_fig[i,j-1,k],new_fig[i,j+1,k])
# for i in range(1,new_fig.shape[0]-1):
#     for j in range(1,new_fig.shape[1]-1):
#         for k in range(3):
#             if new_fig[i,j,k]==255:
#                 if new_fig[i-1,j,k]!=255 and new_fig[i+1,j,k]!=255:
#                     new_fig[i,j,k] = max(new_fig[i-1,j,k],new_fig[i+1,j,k])
#                 if new_fig[i,j-1,k]!=255 and new_fig[i,j+1,k]!=255:
#                     new_fig[i,j,k] = max(new_fig[i,j-1,k],new_fig[i,j+1,k])
img = Image.fromarray(new_fig,'RGB')
img.save('after.jpg')
img.show()