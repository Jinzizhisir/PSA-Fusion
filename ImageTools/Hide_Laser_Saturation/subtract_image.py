from PIL import Image, ImageChops

def subtract_images_cycle(image1_path, image2_path, output_path, output_size=(1242, 375)):
    # 打开两张图片
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # 确保两张图片的尺寸相同
    if img1.size != img2.size:
        raise ValueError("The size of two images must be identical")
    for i in range(1,11):
        k = i*0.1
        pixels = list(img1.getdata())
        amplified_pixels = [(int(r*k), int(g*k), int(b*k)) for r, g, b in pixels]
        amplified_img = Image.new(img1.mode, img1.size)
        amplified_img.putdata(amplified_pixels)

        # 图片做差
        diff = ImageChops.difference(amplified_img, img2)

        # 调整图片大小
        diff_resized = diff.resize(output_size)

        # 保存做差后的图片
        # diff_resized.save(output_path+str(i)+'.png', "PNG")
        
def subtract_images(image1_path, image2_path, output_path, output_size=(1242, 375)):
    # 打开两张图片
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # 确保两张图片的尺寸相同
    if img1.size != img2.size:
        raise ValueError("The size of two images must be identical")
    
    k = i*0.1
    pixels = list(img1.getdata())
    amplified_pixels = [(int(r*k), int(g*k), int(b*k)) for r, g, b in pixels]
    amplified_img = Image.new(img1.mode, img1.size)
    amplified_img.putdata(amplified_pixels)

    # 图片做差
    diff = ImageChops.difference(amplified_img, img2)

    # 调整图片大小
    diff_resized = diff.resize(output_size)

    # 保存做差后的图片
    # diff_resized.save(output_path+str(i)+'.png', "PNG")

def blend_images(image1_path, image2_path, output_path, alpha=0.85):
    # 打开两张图片
    # result=image1×(1−α)+image2×α
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # 确保两张图片的尺寸相同
    if img1.size != img2.size:
        raise ValueError("The size of two images must be identical")

    # 叠加两张图片, result=image1×(1−α)+image2×α
    blended = Image.blend(img1, img2, alpha)

    # 保存叠加后的图片
    blended.save(output_path, "PNG")

def custom_blend(image1_path, image2_path, output_path, alpha1=0.15, alpha2=1):
    # 打开两张图片
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # 确保两张图片的尺寸相同
    if img1.size != img2.size:
        raise ValueError("The size of two images must be identical")

    # 获取两张图片的像素数据
    pixels1 = list(img1.getdata())
    pixels2 = list(img2.getdata())

    # 根据权重叠加两张图片的像素值
    blended_pixels = [
        (
            int(r1 * alpha1 + r2 * alpha2),
            int(g1 * alpha1 + g2 * alpha2),
            int(b1 * alpha1 + b2 * alpha2)
        )
        for (r1, g1, b1), (r2, g2, b2) in zip(pixels1, pixels2)
    ]

    # 创建新的图片并保存
    blended_img = Image.new(img1.mode, img1.size)
    blended_img.putdata(blended_pixels)
    blended_img.save(output_path, "PNG")

image1_path = '000003.png'
image2_path = 'difference_image_1.png'
output_path = 'image_attack.png'

blend_images(image1_path, image2_path, output_path)