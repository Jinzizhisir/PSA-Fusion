import PIL.Image as Image
import numpy as np
import os 

file_path= "/home/usslab/Documents2/qinhong/object/7.22/origin"

# subfile_paths = os.listdir(file_path)


def bayer_swap(img_input_path,img_output_path,line_num):
    img = Image.open(img_input_path)
    img_array = img.load()
    img_data = img.getdata()
    width, height = img.size
    img_array_new = [[0]*3 for i in range(height)]
    for x in range(0,width):
        for y in range(line_num,height):
            img_array_new[y][0] = int(img_array[x,y][1]*2.5)
            # print(img_array_new[y][0])
            img_array_new[y][2] = int(img_array[x,y][1]*2.5)
            img_array_new[y][1] = int((img_array[x,y][0]+img_array[x,y][2])/2)-50
            img_array[x,y]= tuple(img_array_new[y])
    img.save(output_img_path)

# for subfile_path in subfile_paths:
print("begin")
img_files = os.listdir(file_path)
for img_file in img_files:
    fname,ext = os.path.splitext(img_file)
    img_path = os.path.join(file_path,img_file)
    img2 = Image.open(img_path)
    # img_array = img.load()
    # img_data = img2.getdata()
    width, height = img2.size
    # img_array_new = [[0]*3 for i in range(height)]
    for line_num in range(0,height+1,5):
        # output_file_path = '/home/usslab/Documents2/qinhong/object/7.19/attack/attack5'
        output_file_path = os.path.join('/home/usslab/Documents2/qinhong/object/7.22/attack',fname)
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        output_img_path = os.path.join(output_file_path,fname+'_'+str(line_num)+'.jpg')
        # print(output_img_path)
        bayer_swap(img_path,output_img_path,line_num)
        print(line_num,'\n')
        # img.save(output_img_path)
    print(img_file + " write successfully")