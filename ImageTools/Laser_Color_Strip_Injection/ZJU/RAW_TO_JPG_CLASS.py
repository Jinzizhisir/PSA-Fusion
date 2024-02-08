import rawpy
import numpy as np
from PIL import Image
import pickle
import scipy
import math
import random
from scipy import optimize,stats
import csv
from sympy import symbols, Eq, solve
from datetime import datetime

class Raw_Image_Converter:
    def __init__(self,path1,path2=None):
        self.RAW_Image = rawpy.imread(path1)
        self.RAW_Image_np = None
        self.JPG_Image = None
        if path2!=None:
            self.JPG_Real = Image.open(path2)
        else:
            self.JPG_Real = None
        self.Difference = None
        self.height = self.RAW_Image.sizes[0]
        self.width = self.RAW_Image.sizes[1]

        with open('Red_Curve.pkl','rb') as f:
            self.red_curve = pickle.load(f)
        with open('Green_Curve.pkl','rb') as f:
            self.green_curve = pickle.load(f)
        with open('Blue_Curve.pkl','rb') as f:
            self.blue_curve = pickle.load(f)

    def RAW_to_JPG(self,method='original'):
        if method=='original':
            self.Original()
        elif method=='interpolation':
            self.Interpolation()
        else:
            print('Incorrect Method!')

    def Original(self,if_np = False):
        img = self.RAW_Image
        height = self.height
        width = self.width
        img_real = np.zeros((height,width,3),dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if i%2==0 and j%2==0:
                    img_real[i,j,1] = img.raw_value(i,j)
                    img_real[i,j+1,1] = img.raw_value(i,j)
                elif i%2==0 and j%2!=0:
                    img_real[i,j,2] = img.raw_value(i,j)
                    img_real[i,j-1,2] = img.raw_value(i,j)
                    img_real[i+1,j,2] = img.raw_value(i,j)
                    img_real[i+1,j-1,2] = img.raw_value(i,j)
                elif i%2!=0 and j%2==0:
                    img_real[i,j,0] = img.raw_value(i,j)
                    img_real[i,j+1,0] = img.raw_value(i,j)
                    img_real[i-1,j,0] = img.raw_value(i,j)
                    img_real[i-1,j+1,0] = img.raw_value(i,j)
                else:
                    img_real[i,j,1] = img.raw_value(i,j)
                    img_real[i,j-1,1] = img.raw_value(i,j)
        if if_np==False:
            self.RAW_Image_np = img_real
            self.JPG_Image = Image.fromarray(img_real,'RGB')
            return self.JPG_Image
        else:
            self.RAW_Image_np = img_real
            return self.RAW_Image_np

    def Interpolation(self,if_np=False):
        img = self.RAW_Image
        height = self.height
        width = self.width
        img_real = np.zeros((height,width,3),dtype=np.uint8)
        for i in range(1,height-1):
            for j in range(1,width-1):
                if i%2==0 and j%2==0:
                    img_real[i,j,0] = (img.raw_value(i-1,j)+img.raw_value(i+1,j))/2
                    img_real[i,j,1] = img.raw_value(i,j)
                    img_real[i,j,2] = (img.raw_value(i,j-1)+img.raw_value(i,j+1))/2
                elif i%2==0 and j%2!=0:
                    img_real[i,j,0] = (img.raw_value(i-1,j-1)+img.raw_value(i+1,j-1)+
                                       img.raw_value(i-1,j+1)+img.raw_value(i+1,j+1))/4
                    img_real[i,j,1] = (img.raw_value(i-1,j)+img.raw_value(i+1,j)+
                                       img.raw_value(i,j-1)+img.raw_value(i,j+1))/4
                    img_real[i,j,2] = img.raw_value(i,j)
                elif i%2!=0 and j%2==0:
                    img_real[i,j,0] = img.raw_value(i,j)
                    img_real[i,j,1] = (img.raw_value(i-1,j)+img.raw_value(i+1,j)+
                                       img.raw_value(i,j-1)+img.raw_value(i,j+1))/4
                    img_real[i,j,2] = (img.raw_value(i-1,j-1)+img.raw_value(i+1,j-1)+
                                       img.raw_value(i-1,j+1)+img.raw_value(i+1,j+1))/4
                else:
                    img_real[i,j,0] = (img.raw_value(i-1,j)+img.raw_value(i+1,j))/2
                    img_real[i,j,1] = img.raw_value(i,j)
                    img_real[i,j,2] = (img.raw_value(i,j-1)+img.raw_value(i,j+1))/2

        for j in range(width):
            if j%2==0:
                img_real[0,j,0] = img.raw_value(1,j)
                img_real[0,j,1] = img.raw_value(0,j)
                img_real[0,j,2] = img.raw_value(0,j+1)
                img_real[height-1,j,0] = img.raw_value(height-1,j)
                img_real[height-1,j,1] = (img.raw_value(height-1,j+1)+img.raw_value(height-2,j))/2
                img_real[height-1,j,2] = img.raw_value(height-2,j+1)
            else:
                img_real[0,j,0] = img.raw_value(1,j-1)
                img_real[0,j,1] = (img.raw_value(0,j-1)+img.raw_value(1,j))/2
                img_real[0,j,2] = img.raw_value(0,j)
                img_real[height-1,j,0] = img.raw_value(height-1,j-1)
                img_real[height-1,j,1] = img.raw_value(height-1,j)
                img_real[height-1,j,2] = img.raw_value(height-2,j)

        for i in range(1,height-1):
            if i%2==0:
                img_real[i,0,0] = (img.raw_value(i-1,0)+img.raw_value(i+1,0))/2
                img_real[i,0,1] = img.raw_value(i,0)
                img_real[i,0,2] = img.raw_value(i,1)
                img_real[i,width-1,0] = (img.raw_value(i-1,width-2)+img.raw_value(i+1,width-2))/2
                img_real[i,width-1,1] = (img.raw_value(i,width-2)+img.raw_value(i-1,width-1)+
                                         img.raw_value(i+1,width-1))/3
                img_real[i,width-1,2] = img.raw_value(i,width-1)
            else:
                img_real[i,0,0] = img.raw_value(i,0)
                img_real[i,0,1] = (img.raw_value(i-1,0)+img.raw_value(i+1,0)+
                                   img.raw_value(i,1))/3
                img_real[i,0,2] = (img.raw_value(i-1,1)+img.raw_value(i+1,1))/2
                img_real[i,width-1,0] = img.raw_value(i,width-2)
                img_real[i,width-1,1] = img.raw_value(i,width-1)
                img_real[i,width-1,2] = (img.raw_value(i-1,width-1)+img.raw_value(i+1,width-1))/2
        if if_np==False:
            self.RAW_Image_np = img_real
            self.JPG_Image = Image.fromarray(img_real,'RGB')
            return self.JPG_Image
        else:
            self.RAW_Image_np = img_real
            return self.RAW_Image_np

    def Comparison(self):
        img_real = self.RAW_Image_np
        height = self.height
        width = self.width
        img_r = np.asarray(self.JPG_Real)
        img_diff = np.zeros((height,width,3),dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                for k in range(3):
                    if img_real[i,j,k]>img_r[i,j,k]:
                        img_diff[i,j,k] = img_real[i,j,k]-img_r[i,j,k]
                    else:
                        img_diff[i,j,k] = img_r[i,j,k]-img_real[i,j,k]
        difference = Image.fromarray(img_diff,'RGB')
        self.Difference = difference
        return self.Difference

    def red_percentile(self,wavelength):
        return self.red_curve(wavelength)/100

    def green_percentile(self,wavelength):
        return self.green_curve(wavelength)/100

    def blue_percentile(self,wavelength):
        return self.blue_curve(wavelength)/100

    def Add_Beam(self,x=0,y=0,h=0,w=0,wavelength=632,strength=500,image_type='RAW'):
        red_percent = self.red_percentile(wavelength)
        green_percent = self.green_percentile(wavelength)
        blue_percent = self.blue_percentile(wavelength)

        add_red = strength*red_percent
        add_green = strength*green_percent
        add_blue = strength*blue_percent

        if image_type=='RAW':
            new_image = self.RAW_Image_np
        else:
            new_image = np.asarray(self.JPG_Real).copy()
        for i in range(x,(x+h)):
            for j in range(y,(y+w)):
                if new_image[i,j,0] + add_red>255:
                    new_image[i,j,0] = 255
                else:
                    new_image[i,j,0] = new_image[i,j,0]+add_red
                if new_image[i,j,1] + add_green>255:
                    new_image[i,j,1] = 255
                else:
                    new_image[i,j,1] = new_image[i,j,1]+add_green
                if new_image[i,j,2] + add_blue>255:
                    new_image[i,j,2] = 255
                else:
                    new_image[i,j,2] = new_image[i,j,2]+add_blue
        self.RAW_Image_np = new_image
        transformed = Image.fromarray(new_image,'RGB')
        return transformed

    def Add_Beam_with_Dir(self,x=0,y=0,h=0,w=0,wavelength=632,strength_min=500,strength_max=1000,mode=None,image_type='RAW'):
        red_percent = self.red_percentile(wavelength)
        green_percent = self.green_percentile(wavelength)
        blue_percent = self.blue_percentile(wavelength)

        if image_type=='RAW':
            new_image = self.RAW_Image_np
        else:
            new_image = np.asarray(self.JPG_Real).copy()

        if mode==None:
            def color_func(k):
                return (strength_min+strength_max)/2
        elif mode=='Linear_Left':
            z = np.polyfit([y,y+w],[strength_min,strength_max],deg=1)
            color_func = np.poly1d(z)
        elif mode=='Linear_Right':
            z = np.polyfit([y+w,y],[strength_min,strength_max],deg=1)
            color_func = np.poly1d(z)
        elif mode=='Sigmoid_Left':
            def color_func(k):
                p = 1/(1+np.exp(-5/w*(k-y-w/2)))*strength_max
                return p
        elif mode=='Sigmoid_Right':
            def color_func(k):
                p = 1/(1+np.exp(-5/w*(-k+y+w/2)))*strength_max
                return p
        elif mode=='Gaussian':
            def color_func(p,q):
                rho=0.0
                part_a = ((p-x-h/2)**2)/((h/2)**2)
                part_b = ((q-y-w/2)**2)/((w/4)**2)
                part_c = 2*(p-x-h/2)*(q-y-w/2)/(h*w/8)*rho
                f_value = 1/(2*math.pi*h/2*w/4*np.sqrt(1-rho**2))*np.exp(-1/(2*(1-rho**2))*(part_a+part_b+part_c))*strength_max*h*w
                return f_value
        
        for i in range(x,(x+h)):
            for j in range(y,(y+w)):
                if mode=='Gaussian':
                    add_red = color_func(i,j)*red_percent
                    add_green = color_func(i,j)*green_percent
                    add_blue = color_func(i,j)*blue_percent
                else:
                    add_red = color_func(j)*red_percent
                    add_green = color_func(j)*green_percent
                    add_blue = color_func(j)*blue_percent
                if new_image[i,j,0] + add_red>255:
                    new_image[i,j,0] = 255
                else:
                    new_image[i,j,0] = new_image[i,j,0]+add_red
                if new_image[i,j,1] + add_green>255:
                    new_image[i,j,1] = 255
                else:
                    new_image[i,j,1] = new_image[i,j,1]+add_green
                if new_image[i,j,2] + add_blue>255:
                    new_image[i,j,2] = 255
                else:
                    new_image[i,j,2] = new_image[i,j,2]+add_blue
        self.RAW_Image_np = new_image
        transformed = Image.fromarray(new_image,'RGB')
        return transformed

    def Add_Beam_with_Salt(self,x=0,y=0,h=0,w=0,wavelength=632,strength_min=500,strength_max=1000,mode=None,direction=None,image_type='RAW'):
        red_percent = self.red_percentile(wavelength)
        green_percent = self.green_percentile(wavelength)
        blue_percent = self.blue_percentile(wavelength)

        if image_type=='RAW':
            new_image = self.RAW_Image_np
        else:
            new_image = np.asarray(self.JPG_Real).copy()

        if mode==None:
            def color_func(k):
                return (strength_min+strength_max)/2
        elif mode=='Linear_Left':
            z = np.polyfit([y,y+w],[strength_min,strength_max],deg=1)
            color_func = np.poly1d(z)
        elif mode=='Linear_Right':
            z = np.polyfit([y+w,y],[strength_min,strength_max],deg=1)
            color_func = np.poly1d(z)
        elif mode=='Sigmoid_Left':
            def color_func(k):
                p = 1/(1+np.exp(-5/w*(k-y-w/2)))*strength_max
                return p
        elif mode=='Sigmoid_Right':
            def color_func(k):
                p = 1/(1+np.exp(-5/w*(-k+y+w/2)))*strength_max
                return p
        elif mode=='Gaussian':
            def color_func(p,q):
                rho=0.0
                part_a = ((p-x-h/2)**2)/((h/2)**2)
                part_b = ((q-y-w/2)**2)/((w/4)**2)
                part_c = 2*(p-x-h/2)*(q-y-w/2)/(h*w/8)*rho
                f_value = 1/(2*math.pi*h/2*w/4*np.sqrt(1-rho**2))*np.exp(-1/(2*(1-rho**2))*(part_a+part_b+part_c))*strength_max*h*w
                return f_value

        add_red = np.zeros((h,w))
        add_green = np.zeros((h,w))
        add_blue = np.zeros((h,w))
        start = datetime.now()

        vcolor_func = np.vectorize(color_func)
        if mode=='Gaussian':
            add_red = vcolor_func(np.array(range(x,x+h)).reshape(-1,1),np.array(range(y,(y+w))).reshape(1,-1))
            add_green = vcolor_func(np.array(range(x,x+h)).reshape(-1,1),np.array(range(y,(y+w))).reshape(1,-1))
            add_blue = vcolor_func(np.array(range(x,x+h)).reshape(-1,1),np.array(range(y,(y+w))).reshape(1,-1))
        else:
            for i in range(h):
                add_red[i,:] = vcolor_func(range(y,(y+w)))
                add_green[i,:] = vcolor_func(range(y,(y+w)))
                add_blue[i,:] = vcolor_func(range(y,(y+w)))
        # for i in range(x,(x+h)):
        #     for j in range(y,(y+w)):
        #         if mode=='Gaussian':
        #             add_red[i-x,j-y] = color_func(i,j)*red_percent
        #             add_green[i-x,j-y] = color_func(i,j)*green_percent
        #             add_blue[i-x,j-y] = color_func(i,j)*blue_percent
        #         else:
        #             add_red[i-x,j-y] = color_func(j)*red_percent
        #             add_green[i-x,j-y] = color_func(j)*green_percent
        #             add_blue[i-x,j-y] = color_func(j)*blue_percent
        print(datetime.now()-start)
                # if new_image[i,j,0] + add_red>255:
                #     new_image[i,j,0] = 255
                # else:
                #     new_image[i,j,0] = new_image[i,j,0]+add_red
                # if new_image[i,j,1] + add_green>255:
                #     new_image[i,j,1] = 255
                # else:
                #     new_image[i,j,1] = new_image[i,j,1]+add_green
                # if new_image[i,j,2] + add_blue>255:
                #     new_image[i,j,2] = 255
                # else:
                #     new_image[i,j,2] = new_image[i,j,2]+add_blue
        new_image = new_image.astype(np.int)
        new_image[x:(x+h),y:(y+w),0] = new_image[x:(x+h),y:(y+w),0] + add_red
        new_image[x:(x+h),y:(y+w),1] = new_image[x:(x+h),y:(y+w),1] + add_green
        new_image[x:(x+h),y:(y+w),2] = new_image[x:(x+h),y:(y+w),2] + add_blue
        new_image[new_image>255] = 255
        new_image = new_image.astype(np.uint8)

        if direction==None:
            num = random.randint(10000,20000)
            width_list = random.choices(range(1,5),k=num)
            height_list = random.choices(range(1,5),k=num)
            x_list = random.choices(range(x,x+h-np.max(height_list)-1),k=num)
            y_list = random.choices(range(y,y+w-np.max(width_list)-1),k=num)
            bright_list = random.choices(range(200,255),k=num)
        elif direction=='Left':
            num = random.randint(10000,20000)
            width_list = random.choices(range(1,5),k=num)
            height_list = random.choices(range(1,5),k=num)
            x_num = len(list(range(x,x+h-np.max(height_list)-1)))
            y_num = len(list(range(y,y+w-np.max(width_list)-1)))
            vertical_weight = np.exp(-(np.linspace(y,y+w,num=y_num)-w)/w*8)
            horizontal_weight = stats.norm.pdf((np.linspace(x,x+h,num=x_num)-x-h/2)/h*5,loc=0,scale=1)
            x_list = random.choices(range(x,x+h-np.max(height_list)-1),weights=horizontal_weight,k=num)
            y_list = random.choices(range(y,y+w-np.max(width_list)-1),weights=vertical_weight,k=num)
            bright_list = random.choices(range(240,255),k=num)
        elif direction=='Right':
            num = random.randint(10000,20000)
            width_list = random.choices(range(1,5),k=num)
            height_list = random.choices(range(1,5),k=num)
            x_num = len(list(range(x+h-np.max(height_list)-1,x,-1)))
            y_num = len(list(range(y+w-np.max(width_list)-1,y,-1)))
            vertical_weight = np.exp(-(np.linspace(y,y+w,num=y_num)-w)/w*8)
            horizontal_weight = stats.norm.pdf((np.linspace(x,x+h,num=x_num)-x-h/2)/h*5,loc=0,scale=1)
            x_list = random.choices(range(x+h-np.max(height_list)-1,x,-1),weights=horizontal_weight,k=num)
            y_list = random.choices(range(y+w-np.max(width_list)-1,y,-1),weights=vertical_weight,k=num)
            bright_list = random.choices(range(240,255),k=num)
        for i in range(num):
            for j in range(x_list[i],x_list[i]+height_list[i]):
                for k in range(y_list[i],y_list[i]+width_list[i]):
                    new_image[j,k,0] = bright_list[i]
                    new_image[j,k,1] = bright_list[i]
                    new_image[j,k,2] = bright_list[i]
            
        self.RAW_Image_np = new_image
        transformed = Image.fromarray(new_image,'RGB')
        return transformed
        
    def white_balance(self,mode='Gray_World'):
        img = self.RAW_Image_np
        img_copy = img.astype(np.float64)
        if mode=='Gray_World':
            R_average = np.mean(img_copy[:,:,0])
            G_average = np.mean(img_copy[:,:,1])
            B_average = np.mean(img_copy[:,:,2])
            K = (R_average+G_average+B_average)/3
            R_average = R_average.astype(np.uint8)
            G_average = G_average.astype(np.uint8)
            B_average = B_average.astype(np.uint8)
            K = K.astype(np.uint8)
            img_transformed = np.zeros(img.shape,dtype=np.uint8)
            for i in range(img_transformed.shape[0]):
                for j in range(img_transformed.shape[1]):
                    img_transformed[i,j,0] = (K/R_average*img[i,j,0])
                    img_transformed[i,j,1] = (K/G_average*img[i,j,1])
                    img_transformed[i,j,2] = (K/B_average*img[i,j,2])
            transformed = Image.fromarray(img_transformed,'RGB')
            return transformed
        elif mode=='Full_Reflection':
            R_max = np.max(img_copy[:,:,0].flatten())
            G_max = np.max(img_copy[:,:,1].flatten())
            B_max = np.max(img_copy[:,:,2].flatten())
            R_w = 255
            G_w = 250
            B_w = 250
            img_transformed = np.zeros(img.shape,dtype=np.uint8)
            for i in range(img_transformed.shape[0]):
                for j in range(img_transformed.shape[1]):
                    img_transformed[i,j,0] = (R_w/R_max*img[i,j,0])
                    img_transformed[i,j,1] = (G_w/G_max*img[i,j,1])
                    img_transformed[i,j,2] = (B_w/B_max*img[i,j,2])
            transformed = Image.fromarray(img_transformed,'RGB')
            return transformed
        elif mode=='QCGP':
            R_average = np.mean(img_copy[:, :, 0])
            G_average = np.mean(img_copy[:, :, 1])
            B_average = np.mean(img_copy[:, :, 2])
            R_average = R_average.astype(np.uint8)
            G_average = G_average.astype(np.uint8)
            B_average = B_average.astype(np.uint8)
            K_average = R_average/3 + G_average/3 + B_average/3
            K_average = K_average.astype(np.uint8)
            R_max = np.max(img_copy[:, :, 0].flatten())
            G_max = np.max(img_copy[:, :, 1].flatten())
            B_max = np.max(img_copy[:, :, 2].flatten())
            K_max = R_max/3 + G_max/3 + B_max/3
            K_max = K_max.astype(np.uint8)
            # For Red
            UR,VR = symbols('UR VR')
            eq1 = Eq(UR*R_average**2+VR*R_average - K_average,0)
            eq2 = Eq(UR*R_max**2 + VR*R_max - K_max,0)
            sol = solve((eq1, eq2), (UR, VR))
            UR = sol[UR]
            VR = sol[VR]
            print('R')
            # For Green
            UG,VG = symbols('UG VG')
            eq1 = Eq(UG*G_average**2+VG*G_average - K_average,0)
            eq2 = Eq(UR*G_max**2 + VG*G_max - K_max,0)
            sol = solve((eq1, eq2), (UG,VG))
            UG = sol[UG]
            VG = sol[VG]
            print('G')
            # For Blue
            UB,VB = symbols('UB VB')
            eq1 = Eq(UB*B_average**2+VB*B_average - K_average,0)
            eq2 = Eq(UB*B_max**2 + VB*B_max - K_max,0)
            sol = solve((eq1, eq2), (UB,VB))
            UB = sol[UB]
            VB = sol[VB]
            print('B')
            # Change Color
            img_transformed = np.zeros(img.shape, dtype=np.uint8)
            for i in range(img_transformed.shape[0]):
                for j in range(img_transformed.shape[1]):
                    img_transformed[i, j, 0] = UR*img[i,j,0]*img[i,j,0] + VR*img[i,j,0]
                    img_transformed[i, j, 1] = UG*img[i,j,1]*img[i,j,1] + VG*img[i,j,1]
                    img_transformed[i, j, 2] = UB*img[i,j,2]*img[i,j,2] + VB*img[i,j,2]
            transformed = Image.fromarray(img_transformed, 'RGB')
            return transformed
        elif mode=='Temp_Est':
            img_rgb = Image.fromarray(img,'RGB')
            Cr = 100
            Cb = 100
            C = np.sqrt(Cr*Cr+Cb*Cb)
            round = 0
            u=1
            v=1
            while (C>0.1 and round < 40):
                img_YCrCb = img_rgb.convert('YCbCr')
                img = np.array(img_YCrCb)
                img_transformed = img.copy()
                img_transformed = img_transformed.astype(np.float64)
                img_transformed[:, :, 1] -= 128
                img_transformed[:, :, 2] -= 128
                # Color Estimation
                count = 0
                Y = 0
                Cr = 0
                Cb = 0
                for i in range(img_transformed.shape[0]):
                    for j in range(img_transformed.shape[1]):
                        z = img_transformed[i,j,0] - np.abs(img_transformed[i,j,1]) - np.abs(img_transformed[i,j,2])
                        if z > 100:
                            Y += img_transformed[i,j,0]
                            Cr += img_transformed[i,j,1]
                            Cb += img_transformed[i,j,2]
                            count += 1
                Y /= count
                Cr /= count
                Cb /= count
                C = np.sqrt(Cr*Cr+Cb*Cb)
                print(C)
                # Color Fix
                if round > 55:
                    lam = 0.001
                elif round > 40:
                    lam = 0.005
                elif round > 25:
                    lam = 0.008
                else:
                    lam = 0.01
                if np.abs(Cr)>np.abs(Cb):
                    if Cr>0:
                        u -= lam
                    else:
                        u += lam
                else:
                    if Cb>0:
                        v -= lam
                    else:
                        v += lam
                print(u)
                print(v)
                img_c = np.array(img_rgb).copy()
                img_c[:,:,0] = img_c[:,:,0] * u
                img_c[:,:,2] = img_c[:,:,2] * v
                img_rgb = Image.fromarray(img_c,'RGB')
                round += 1
                print(round)
            # img_transformed[:,:,1] += 128
            # img_transformed[:,:,2] += 128
            # img_transformed = img_transformed.astype(np.uint8)
            transformed = Image.fromarray(img_rgb, 'RGB')
            return transformed


    def show(self,image_type=0):
        if image_type==0:
            self.JPG_Image.show()
        elif image_type==1:
            try:
                self.JPG_Real.show()
            except:
                print('Image not converted!')
        elif image_type==2:
            try:
                self.Difference.show()
            except:
                print('Image difference noe created!')


raw_file_path = 'clear.RAW'
jpg_file_path = 'BB01.jpg'

Converter1 = Raw_Image_Converter(raw_file_path,jpg_file_path)
origin = Converter1.RAW_to_JPG()

start = datetime.now()
transformed = Converter1.Add_Beam_with_Salt(x=1140,y=0,h=220,w=4000,wavelength=520,strength_min=0,strength_max=1000,mode='Sigmoid_Left',direction='Left',image_type='JPG')
# transformed = Converter1.white_balance(mode='Temp_Est')
#transformed = Converter1.Add_Beam_with_Salt(x=1700,y=0,h=130,w=4000,wavelength=450,strength_min=0,strength_max=800,mode=None,direction=None,image_type='RAW')
print(datetime.now()-start)
transformed.show()
input('Enter')
wavelength_option = [520]
strength_min_option = [0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1500,1550,1600]
strength_max_option = [200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800]
# modes=[None,'Linear_Left','Linear_Right','Sigmoid_Left','Sigmoid_Right','Gaussian']
modes = ['Linear_Right']
directions=[None,'Left','Right']

count = 0
for wavelength in wavelength_option:
    for strength_min in strength_min_option:
        for strength_max in strength_max_option:
            if strength_min>=strength_max:
                continue
            else:
                for mode in modes:
                    if mode!=None and mode[-4:]=='Left':
                        direction='Right'
                    elif mode!=None and mode[-5:]=='Right':
                        direction='Left'
                    else:
                        direction=None
                    start = datetime.now()
                    transformed = Converter1.Add_Beam_with_Salt(x=1140, y=0, h=220, w=4000, wavelength=wavelength,
                                                                strength_min=strength_min, strength_max=strength_max, mode=mode,
                                                                direction=direction, image_type='JPG')
                    print(datetime.now()-start)

                    if count<10:
                        filename = r'F:\DNG Files\Result_Search_Green_Test_Large\Result00'+str(count)+'.jpg'
                    elif count<100:
                        filename = r'F:\DNG Files\Result_Search_Green_Test_Large\Result0'+str(count)+'.jpg'
                    else:
                        filename = r'F:\DNG Files\Result_Search_Green_Test_Large\Result'+str(count)+'.jpg'
                    transformed.save(filename)
                    print('No.',count)
                    count += 1
                    try:
                        parameters = [str(wavelength),str(strength_min),str(strength_max),str(mode),str(direction)]
                    except:
                        try:
                            parameters = [str(wavelength),str(strength_min),str(strength_max),str(mode),'None']
                        except:
                            parameters = [str(wavelength),str(strength_min),str(strength_max),'None','None']
                    with open(r'F:\DNG Files\Result_Search_Green_Test_Large\Parameters.csv','a',newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(parameters)



#strengths = range(100,2100,100)
#for strength in strengths: 
#    transformed = Converter1.Add_Beam_with_Dir(x=980,y=0,h=120,w=4000,wavelength=632,strength=strength,image_type='JPG')
    #difference1 = Converter1.Comparison()
    #transformed.show()
#    if strength < 1000:
#        filename = r'D:\DNG files\Result0'+str(strength/100)+'.jpg'
#    else:
#        filename = r'D:\DNG files\Result'+str(strength/100)+'.jpg'
#    transformed.save(filename)
#    print(strength)

#interpolation = Converter1.RAW_to_JPG(method='interpolation')
#difference2 = Converter1.Comparison()
#Converter1.show(2)
