from PIL import Image, ImageChops

def blooming(img, strength):

    rows, cols = img.shape[:2]

    centerX = rows / 2 - 0
    centerY = cols / 2 + 0
    radius = min(centerX, centerY)

    dst = np.zeros((rows, cols, 3), dtype="uint8")
    for i in range(rows):

        for j in range(cols):

            distance = math.pow((centerY-j), 2) + math.pow((centerX-i), 2)
            B = img[i,j][0]
            G = img[i,j][1]
            R = img[i,j][2]

            if (distance < radius * radius):

                result = (int)(strength*( 1.0 - math.sqrt(distance) / radius ))
                B = img[i,j][0] + result
                G = img[i,j][1] + result
                R = img[i,j][2] + result

                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))

                dst[i,j] = np.uint8((B, G, R))
            else:
                dst[i,j] = np.uint8((B, G, R)) 
    return dst

def resize_image(input_path, output_path, size):
    with Image.open(input_path) as img:
        resized_img = img.resize(size, Image.ANTIALIAS)
        resized_img.save(output_path)

def adjust_exposure(image, exposure_value):

    plt.imshow(image)
    if image is None:
        print("Failed to read image: {}".format(image_path))
        return None
    # Convert the image to the float type
    image = image.astype(np.float32)

    # Adjust the exposure value
    image = exposure_value * image

    # Clip the pixel values that exceed the maximum value of 255
    image = np.clip(image, 0, 255)

    # Convert the image back to the uint8 type
    image = image.astype(np.uint8)

    return image

def over_exposure(image):
    exposure_value = 20 #exposure brightness
    strength = 300 #blooming center brightness
    image = adjust_exposure(image, exposure_value)
    image = blooming(image,strength)
    return image


def hide_laser_saturation(img1,img2, alpha=0.85):

    # result=image1×(1−α)+image2×α

    if img1.size != img2.size:
        # raise ValueError("The size of two images must be identical. You can choose the over_exposure to simulate the hiding attack.")
        img2 = img2.resize(img1.size, Image.ANTIALIAS)

    # result=image1×(1−α)+image2×α
    blended = Image.blend(img1, img2, alpha)
    return blended


if __name__ == "__main__":
    image1_path = '000003.png'
    hide_laser_saturation(image1_path)