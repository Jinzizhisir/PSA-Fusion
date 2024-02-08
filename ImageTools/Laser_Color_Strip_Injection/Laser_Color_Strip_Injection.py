from PIL import Image
import math
import numpy as np
import os

# def quantum_efficiency(wavelength)
#         # 读取quantum efficiency
#         with open('Red_Curve.pkl','rb') as f:
#             self.red_curve = pickle.load(f)
#         with open('Green_Curve.pkl','rb') as f:
#             self.green_curve = pickle.load(f)
#         with open('Blue_Curve.pkl','rb') as f:
#             self.blue_curve = pickle.load(f)

#         red_percent = self.red_curve(wavelength)/100
#         green_percent = self.green_curve(wavelength)/100
#         blue_percent = self.blue_curve(wavelength)/100    
#         return red_percent,green_percent,blue_percent

def color_func(p,q,start_row,strength,h,w):
    # Gaussion
    rho=0.0
    part_a = ((p-start_row-h/2)**2)/((h/2)**2)
    part_b = ((q-w/2)**2)/((w/4)**2)
    part_c = 2*(p-start_row-h/2)*(q-w/2)/(h*w/8)*rho
    f_value = 1/(2*math.pi*h/2*w/4*np.sqrt(1-rho**2))*np.exp(-1/(2*(1-rho**2))*(part_a+part_b+part_c))*strength*h*w
    return f_value

def color_strip_injection(original_image,x=0,y=0,h=0,w=0,wavelength=632,strength=1500):
        
    # red_percent,green_percent,blue_percent = quantum_efficiency(wavelength)
    red_percent,green_percent,blue_percent = 0.6,0.1,0.1

    # add_red = strength*red_percent
    # add_green = strength*green_percent
    # add_blue = strength*blue_percent

    # original_image = Image.open(image_path)
    image_with_strip = original_image.load()
    width, height = original_image.size
    # color raw
    start_row = height//3  #
    end_row = height//3*2    # 

    for i in range(0,width):
        for j in range(start_row,end_row):
            
            strength = color_func(p=j,q=i,start_row = start_row,strength = 1500,h=height,w=width)
            # if i<10:
            #     print(strength)
            add_red = strength*red_percent
            add_green = strength*green_percent
            add_blue = strength*blue_percent

            current_pixel = list(image_with_strip[i,j])

            if current_pixel[0] + add_red>255:
                current_pixel[0] = 255
            else:
                current_pixel[0] = current_pixel[0]+add_red
            if current_pixel[1] + add_green>255:
                current_pixel[1] = 255
            else:
                current_pixel[1] = current_pixel[1]+add_green
            if current_pixel[2] + add_blue>255:
                current_pixel[2] = 255
            else:
                current_pixel[2] = current_pixel[2]+add_blue

            image_with_strip[i,j] = tuple(map(int,current_pixel))
    
    # return image_with_strip
    return original_image
    # image_with_strip.save("image_with_strip.jpg")
    # original_image.save(os.path.dirname(os.path.abspath(__file__))+"\\image_with_strip_red.jpg")

if __name__ == "__main__":
    image_path = os.path.dirname(os.path.abspath(__file__))+'\\000007.png'
    color_strip_injection(image_path)