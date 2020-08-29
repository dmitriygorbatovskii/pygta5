import numpy as np
from PIL import ImageGrab
import cv2
import pynput
import pyautogui
import time
from alexnet import alexnet

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)

t_time = 0.09

def straight():
    pyautogui.keyDown('W')
    pyautogui.keyUp('D')
    pyautogui.keyUp('A')
def left():
    pyautogui.keyDown('W')
    pyautogui.keyDown('A')
    time.sleep(t_time)
    pyautogui.keyUp('A')
def right():
    pyautogui.keyDown('W')
    pyautogui.keyDown('D')
    time.sleep(t_time)
    pyautogui.keyUp('D')


def wait():
    pyautogui.keyUp('W')
    pyautogui.keyUp('D')
    pyautogui.keyUp('A')


model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(3))[::-1]:
        print(i + 1)
        time.sleep(1)

    while (True):

        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (160, 120))
        prediction = model.predict([screen.reshape(160, 120, 1)])[0]

        turn_thresh = 0.75
        fwd_thresh = 0.70
        print(prediction)
        stop = False

        '''if prediction[1] > fwd_thresh:
            straight()
        elif prediction[0] > turn_thresh:
            left()
        elif prediction[2] > turn_thresh:
            right()
        else:
            straight()
'''
main()
