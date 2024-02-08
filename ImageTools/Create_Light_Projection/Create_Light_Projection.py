# scripts for [Create]Light_Projection
# with the method of watermark

import cv2
import numpy as np
import matplotlib.pyplot as plt
# import os
from PIL import Image
# from tqdm import tqdm


def light_projection(waterMark, origin, transparency=0.6):
    # Convert both images
    waterMark = waterMark.convert('RGBA')
    origin = origin.convert('RGBA')

    # Calculate new dimensions
    new_width = int(origin.width / 1)  # This is just the original width
    new_height = int(waterMark.height * (new_width / waterMark.width))  # Scale height based on 

    # Resize
    wMark = waterMark.resize((new_width, new_height))

    # Adjust the transparency
    datas = wMark.getdata()
    newData = []
    for item in datas:
        newData.append((item[0], item[1], item[2], int(item[3] * transparency)))
    wMark.putdata(newData)

    baseImg = Image.new('RGBA', origin.size)
    baseImg.paste(origin, (0, 0), origin)

    position = (int(origin.width / 4), int((origin.height - wMark.height) / 3))
    baseImg.paste(wMark, position, wMark)

    # Convert the final image back to 'RGB' mode and return
    return baseImg.convert('RGB')


    return blended_image

def visualize_image(image):
    # Visualize the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Example usage
    # overlay_x = 0
    # overlay_y = 0
    # os.chdir('/home/usslab/SensorFusion/Dataset')
    creating_object_path = './ImageTools/Create_Light_Projection/car.png'
    waterMark = Image.open(creating_object_path)

    # # TEST ONE image
    # img_path = './ImageTools/[Create]Light_Projection/000003.png'
    # origin = Image.open(img_path)
    # blended_image = light_projection(waterMark,origin)
    # blended_image.save('test.png')


    directory_img = ''

    print('yes')
    # projected_image = projection(origin,blended_image)


    # for file_name in os.listdir(directory_img):
    #     img_path = directory_img+'\\'+file_name
    #     origin = Image.open(img_path)
    #     image_corrupted = light_projection(waterMark,origin)
    #     image.save(str(i).zfill(6)+'.png')

