import cv2
import PIL
import numpy as np
from IPython.display import display


def imshow(a):
    a = a.clip(0, 255).astype('uint8')
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(PIL.Image.fromarray(a))


def cv2_imshow(title, img):
    """
    function:
    - shows image in a separate window,
    - press any key to close the window, not the red cross!
    """
    cv2.startWindowThread()
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    # cap.set(3,640) # adjust width
    # cap.set(4,480) # adjust height

    while True:
        success, img = cap.read()
        if not success:
            cap.release()
            break

        ### ALL NECESSARY FUNCTIONS WILL BE HERE ###

        ####################################
        cv2.imshow(video_path, img)
        if cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
            cap.release()
            break
            
    cv2.destroyAllWindows()


def get_percentage_white_pixels(image):
  if len(image.shape) == 2:
    white_pixels = np.sum(image == 255)
  else:
    white_pixels = np.sum((image[:,:,0] == 255) & (image[:,:,1] == 255) & (image[:,:,2] == 255))
  percentage = white_pixels / (image.shape[0] * image.shape[1]) * 100
  return percentage
