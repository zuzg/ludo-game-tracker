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


class GameObject:
    def __init__(self, x, y, w, h, color):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
    def show(self, image):
        imshow(image[self.y:self.y+self.h, self.x:self.x+self.w])

def create_game_objects(coords:tuple[int], image:np.ndarray, col_thres:float=.0) -> dict[str:list[GameObject]]:
    colors = ['blue', 'green', 'yellow', 'red']
    game_objects = dict([(c, list()) for c in colors])
    for point in coords:
        img_cropped = image[point[0][1]:point[1][1], point[0][0]:point[1][0]]
        color = determine_color(img_cropped, col_thres)
        if color is not None:
            game_object = GameObject(point[0][0], point[0][1], point[1][0]-point[0][0], point[1][1]-point[0][1], color)
            # game_object.show(image)
            # print(game_object.color)
            game_objects[color].append(game_object)
    return game_objects

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


def get_percentage_value_pixels(image:np.ndarray, value:int=255) -> float:
  if len(image.shape) == 2:
    white_pixels = np.sum(image == value)
  else:
    white_pixels = np.sum((image[:,:,0] == value) & (image[:,:,1] == value) & (image[:,:,2] == value))
  percentage = white_pixels / (image.shape[0] * image.shape[1]) * 100
  return percentage


def determine_color(image:np.ndarray, color_percentage:float=0.25) -> str:
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_groups = list()

    # BLUE
    lower_bound = (85, 100, 100)
    upper_bound = (140, 255, 255)
    mask_blue = cv2.inRange(image_hsv, lower_bound, upper_bound)
    image_blue = cv2.bitwise_and(image, image, mask=mask_blue)
    color_groups.append(('blue', np.count_nonzero(image_blue)))

    # GREEN
    lower_bound = (40, 100, 100)
    upper_bound = (80, 255, 255)
    mask_green = cv2.inRange(image_hsv, lower_bound, upper_bound)
    image_green = cv2.bitwise_and(image, image, mask=mask_green)
    color_groups.append(('green', np.count_nonzero(image_green)))

    # YELLOW
    lower_bound = (20, 100, 100)
    upper_bound = (30, 255, 200)
    mask_yellow = cv2.inRange(image_hsv, lower_bound, upper_bound)
    image_yellow = cv2.bitwise_and(image, image, mask=mask_yellow)
    color_groups.append(('yellow', np.count_nonzero(image_yellow)))
    
    # RED
    lower_bound = (150, 100, 100)
    upper_bound = (179, 255, 255)
    mask_red = cv2.inRange(image_hsv, lower_bound, upper_bound)
    image_red = cv2.bitwise_and(image, image, mask=mask_red)
    color_groups.append(('red', np.count_nonzero(image_red)))

    # imshow(image)
    color_groups.sort(key = lambda x: x[1], reverse=True)
    color_name = color_groups[0][0]
    pix_number = color_groups[0][1]

    if  pix_number / (image.shape[0] * image.shape[1]) < color_percentage:
        return None

    return color_name
