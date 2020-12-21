import cv2
import numpy as np
from PIL import ImageGrab


def grey_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def threshold_image(image):
    threshold_image = cv2.Canny(gray_image, threshold1=50, threshold2=300)
    return threshold_image

def screen_collect(width_start, height_start, width_end, height_end_scale = 10, concat=False):
    height_end = 30 * height_end_scale + height_start

    while True:
        screen = np.array(ImageGrab.grab(bbox=(width_start, height_start, width_end, height_end)))

        if concat:
            side = int((height_end - height_start)/3 + height_start)
            sc = cv2.resize(screen, (240, 150), interpolation = cv2.INTER_AREA)

            #изображение в градациях серго
            image = grey_image(sc)
            #gray_image = cv2.cvtColor(sc, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            #черно-белое изображение
            threshold_image = cv2.Canny(image, threshold1=50, threshold2=300)
            threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR)

            #разматое изображение
            blurred = cv2.GaussianBlur(threshold_image,(5,5),0)

            #объединение изображений
            concat1 = np.concatenate((image, threshold_image, blurred), axis=0)
            blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            lines = cv2.HoughLinesP(blurred, 1, np.pi / 180, 180, 10, 5)
            lines_detect = sc
            try:
                for line in lines:
                    coords = line[0]
                    if coords[1] > 80 and coords[3] > 80:
                        if coords[1] != coords[3] and coords[0] != coords[2]:
                            cv2.line(lines_detect, (coords[0], coords[1]), (coords[2], coords[3]), [0, 255, 0], 2)
            except:
                pass

            img5 = sc
            img6 = sc
            concat2 = np.concatenate((lines_detect, img5, img6), axis=0)

            numpy_horizontal_concat = np.concatenate((screen, concat1, concat2), axis=1)
            cv2.imshow('window', cv2.cvtColor(numpy_horizontal_concat, cv2.COLOR_BGR2RGB))
        else:
            cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_collect(50, 120, 600, 15, True)