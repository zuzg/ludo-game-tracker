import cv2
import PIL
import numpy as np

def imshow(a):
  a = a.clip(0, 255).astype('uint8')
  if a.ndim == 3:
    if a.shape[2] == 4:
      a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
    else:
      a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
  display(PIL.Image.fromarray(a))

def is_mostly_white(img:np.ndarray, threshold:float=0.7) -> bool:
    _, img_bin = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
    num_white_pixels = np.sum(img_bin == 255)
    ratio = num_white_pixels / (img_bin.shape[0] * img_bin.shape[1])
    return ratio>=threshold