import math
import numpy as np
from PIL import Image,ImageFilter
import cairo

im = Image.open('stop.eps')
print(im.size)
print(im.format)
