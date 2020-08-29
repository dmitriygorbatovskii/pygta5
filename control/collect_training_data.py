import keyboard
import os
import sys
import numpy as np
import cv2
import time
from PIL import ImageGrab

shot_pressed = 0
was_pressed = False

def key_check():
    output = [0, 0, 0]

    global shot_pressed
    global was_pressed

    if keyboard.is_pressed('A'):
        output[0] = 1
        was_pressed = True
    elif keyboard.is_pressed('D'):
        output[2] = 1
        was_pressed = True
    elif keyboard.is_pressed('W'):
        output[1] = 1
        was_pressed = True

    return output


file_name = '/home/da/PycharmProjects/pythonProject/pygta5_yv5/control/training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def main():

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)


    while (True):
        screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (160, 120))
        training_data.append(np.array([screen, key_check()], dtype=object))

        if len(training_data) % 1000 == 0:
            print(len(training_data))
            np.save('/home/da/PycharmProjects/pythonProject/pygta5_yv5/control/training_data.npy', training_data)

main()



