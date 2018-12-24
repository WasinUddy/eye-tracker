from imutils import face_utils
import dlib
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from PIL import ImageEnhance
import pygame
import time
from moviepy.editor import VideoFileClip
import os
from tqdm import tqdm

name = 'video.mp4'
clip = VideoFileClip(name)
FPS = int(clip.fps)
duration = clip.duration
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)
p = "package/shape_predictor_68_face_landmarks.dat"
db_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
constant = 5
cam = -1
game_icon = pygame.image.load('package/ICON.png')
pygame.display.set_icon(game_icon)
white = (255, 255, 255)
gray = (220, 220, 220)
black = (0, 0 , 0)
pygame.init()
Display = pygame.display.set_mode((800, 185))
pygame.display.set_caption('Gaze_Detector')
change = False
Exit = False
click =0
Detect = False
calibrate = False


def isPointInPath(x, y, poly):
    num = len(poly)
    i = 0
    j = num - 1
    c = False
    for i in range(num):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and \
                (x < poly[i][0] + (poly[j][0] - poly[i][0]) * (y - poly[i][1]) /
                                  (poly[j][1] - poly[i][1])):
            c = not c
        j = i
    return c

def process_eye(detector, img):  # blob processing
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)

    return keypoints



def detect():
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = db_detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = face_utils.shape_to_np(predictor(gray, rect))


        left_boundary  = [(shape[36][0], shape[36][1]), (shape[37][0], shape[37][1]), (shape[38][0], shape[38][1]), (shape[39][0], shape[39][1]),
                          (shape[40][0], shape[40][1]), (shape[41][0], shape[41][1])]




        right_boundary = [(shape[42][0], shape[42][1]), (shape[43][0], shape[43][1]), (shape[44][0], shape[44][1]), (shape[45][0], shape[45][1]),
                          (shape[46][0], shape[46][1]), (shape[47][0], shape[47][1])]

        #for i in range(36, 48):
            #cv2.circle(img, (shape[i][0], shape[i][1]), 2, (255, 255, 255), 3)

    keypoints = process_eye(detector, img)

    lenth = len(keypoints)

    left_detect = False
    right_detect = False
    for i in range(0, lenth):
        x, y = (keypoints[i]).pt
        x = int(x)
        y = int(y)

        if isPointInPath(x, y, left_boundary) == True:
            cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
            left_detect = True
            L = x



        if isPointInPath(x, y, right_boundary) == True:
            cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
            right_detect = True
            R = x

        if left_detect == True and right_detect == True:
            left_detect = False
            right_detect = False
            print(L, R)

            return (L, R)





def detect_2(img):


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = db_detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = face_utils.shape_to_np(predictor(gray, rect))


        left_boundary  = [(shape[36][0], shape[36][1]), (shape[37][0], shape[37][1]), (shape[38][0], shape[38][1]), (shape[39][0], shape[39][1]),
                          (shape[40][0], shape[40][1]), (shape[41][0], shape[41][1])]




        right_boundary = [(shape[42][0], shape[42][1]), (shape[43][0], shape[43][1]), (shape[44][0], shape[44][1]), (shape[45][0], shape[45][1]),
                          (shape[46][0], shape[46][1]), (shape[47][0], shape[47][1])]

        #for i in range(36, 48):
            #cv2.circle(img, (shape[i][0], shape[i][1]), 2, (255, 255, 255), 3)

    keypoints = process_eye(detector, img)

    lenth = len(keypoints)

    left_detect = False
    right_detect = False
    for i in range(0, lenth):
        x, y = (keypoints[i]).pt
        x = int(x)
        y = int(y)

        if isPointInPath(x, y, left_boundary) == True:
            cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
            left_detect = True
            L = x



        if isPointInPath(x, y, right_boundary) == True:
            cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
            right_detect = True
            R = x

        if left_detect == True and right_detect == True:
            left_detect = False
            right_detect = False
            print(L, R)

            return (L, R)



def process(most_left_left, most_right_left, most_left_right, most_right_right, position):
    left, right = position

    eye_area_left = abs(most_left_left - most_right_left)
    eye_area_right = abs(most_left_right - most_right_right)



    position_in_left_eye = abs(most_left_left - left)

    position_in_right_eye = abs(most_left_right - right)


    left_ratio = 1920 / eye_area_left

    right_ratio = 1920 / eye_area_right

    real_left_position = left_ratio * position_in_left_eye
    real_right_positon = right_ratio * position_in_right_eye

    if real_left_position > 1920:
        real_left_position = 1920

    if real_right_positon> 1920:
        real_right_positon = 1920

    return real_left_position, real_right_positon





def message_to_screen(msg, color, position,size):
    font = pygame.font.SysFont('Agency FB', size)
    screen_text = font.render(msg, True, color)
    Display.blit(screen_text, position)

def message_to_screen_calibrate(msg, color, position,size):
    font = pygame.font.SysFont('Agency FB', size)
    screen_text = font.render(msg, True, color)
    Calibrate_Display.blit(screen_text, position)

def message_to_screen_detect(msg, color, position, size):
    font = pygame.font.SysFont('Agency FB', size)
    screen_text = font.render(msg, True, color)
    Detect_Display.blit(screen_text, position)


while not Exit:
    mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()


    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Exit = True


    Display.fill(white)
    pygame.draw.rect(Display, black, [500, 0, 500, 100])
    pygame.draw.rect(Display, gray, [505, 5, 290, 90])
    pygame.draw.rect(Display, black, [0, 0, 600, 5])
    pygame.draw.rect(Display, black, [0, 95, 450, 5])
    pygame.draw.rect(Display, black, [450,5,5,200])

    pygame.draw.rect(Display, black, [0, 0, 5, 800])
    pygame.draw.rect(Display, black, [0, 595, 800, 5])
    pygame.draw.rect(Display, black, [795, 0, 5, 600])
    pygame.draw.rect(Display, black, [500, 95, 300, 90])
    pygame.draw.rect(Display, gray, [505, 100, 290, 80])
    pygame.draw.rect(Display, black, [500,100,5,400])
    pygame.draw.rect(Display, black, [0,180,800,5])

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_1:
            cam = 1
        if event.key == pygame.K_0:
            cam = 0

    if cam == 0:
        message_to_screen("0", (0, 0, 0), [469, 70], 35)

    if cam == 1:

        message_to_screen("1", (0, 0, 0), [469, 70], 35)



    message_to_screen("G-ZAX  Gazing Concentration Analysis", (0, 0, 0), [10, 10], 30)
    message_to_screen("by Wasin Silakong", (0, 0, 0), [350, 25], 15)
    message_to_screen("Start Detect", (0, 0, 0), [600, 30], 30)
    message_to_screen("Calibrate", (0, 0, 0), [610, 125], 30)
    message_to_screen("** Note **", (255, 0, 0), [15, 100],18)
    message_to_screen("1 ). Use plain White background for highest accuracy", (255, 0, 0), [15, 115], 15)
    message_to_screen("2). Use High dynamic range camera for more accuracy", (255, 0, 0), [15, 135], 15)
    message_to_screen("3). Higher Clock_clockspeed CPU give more performance", (255, 0, 0), [15, 155], 15)
    message_to_screen("Cam",black, (465,5),20)
    if mouse_pos_x > 510 and mouse_pos_x < 795 and mouse_pos_y > 5 and mouse_pos_y < 95:
        pygame.draw.rect(Display, (170, 170, 170), [505, 5, 290, 90])
        message_to_screen("Start Detect", (0, 0, 0), [600, 30], 30)
        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(Display, (128, 128, 128), [505, 5, 290, 90])
            message_to_screen("Start Detect", (0, 0, 0), [600, 30], 30)

            Detect = True


    if mouse_pos_x >510 and mouse_pos_x < 795 and mouse_pos_y >95 and mouse_pos_y < 180:
        pygame.draw.rect(Display, (170, 170, 170), [505, 100, 290, 80])
        message_to_screen("Calibrate", (0, 0, 0), (610, 125), 30)
        message_to_screen("Cam", black, (465, 5), 20)
        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(Display, (128, 128, 128), [505, 100, 290, 80])
            message_to_screen("Calibrate", (0, 0, 0), (610, 125), 30)
            message_to_screen("Cam", black, (465, 5), 20)
            calibrate = True





    enter = 0



    if calibrate == True:
        cap = cv2.VideoCapture(cam)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

        position = []
        Calibrate_left_eye_x = []
        Calibrate_left_eye_y = []
        Calibrate_right_eye_x = []
        Calibrate_right_eye_y = []
        left_x_ratio_list = []
        left_y_ratio_list = []
        right_x_ratio_list = []
        right_y_ratio_list = []

        Calibrate_Display = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
        Calibrate_Display.fill(white)
        pygame.draw.circle(Calibrate_Display, (255,0,0), (10, 10), 10)
        pygame.draw.circle(Calibrate_Display, (255, 0, 0), (960, 10), 10)
        pygame.draw.circle(Calibrate_Display, (255, 0, 0), (1910, 10), 10)
        pygame.draw.circle(Calibrate_Display, (255, 0 ,0), (10, 540), 10)
        pygame.draw.circle(Calibrate_Display, (255, 0, 0), (10, 1070), 10)
        pygame.draw.circle(Calibrate_Display, (255, 0, 0), (960, 1070), 10)
        pygame.draw.circle(Calibrate_Display, (255, 0, 0), (1910, 1070), 10)
        pygame.draw.circle(Calibrate_Display, (255, 0, 0), (1910, 540), 10)
        pygame.draw.circle(Calibrate_Display, (255, 0, 0), (960, 540), 10)

        pygame.display.update()

        Calibrate_exit = False
        Calibrate_pos = []

        while not Calibrate_exit:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and len(Calibrate_pos) < 9:

                        pos = detect()


                        if pos == None:

                            pass

                        else:
                            Calibrate_pos.append(pos)


                            pygame.draw.rect(Calibrate_Display, (255, 255, 255), (100,100,100,100))
                            message_to_screen_calibrate(str(len(Calibrate_pos)),(0, 0, 0), (100,100), 50)

                            if len(Calibrate_pos) == 1:
                                pygame.draw.rect(Calibrate_Display, (255, 255, 255), (0, 0,50, 50))

                            if len(Calibrate_pos) == 2:
                                pygame.draw.rect(Calibrate_Display, (255, 255, 255), (940, 0, 50, 50))

                            if len(Calibrate_pos) == 3:
                                pygame.draw.rect(Calibrate_Display, (255, 255, 255), ( 1800,0,300,200))

                            if len(Calibrate_pos) == 4:
                                pygame.draw.rect(Calibrate_Display, (255,255,255), (0,500,100,100))

                            if len(Calibrate_pos) == 5:
                                pygame.draw.rect(Calibrate_Display, (255, 255,255),(940,500,100,100))

                            if len(Calibrate_pos) == 6:
                                pygame.draw.rect(Calibrate_Display, (255, 255, 255), (1800,500,300,300))

                            if len(Calibrate_pos) == 7:
                                pygame.draw.rect(Calibrate_Display, (255,255,255),(0,1000,300,300))

                            if len(Calibrate_pos) == 8:
                                pygame.draw.rect(Calibrate_Display, (255,255,255), (940,1000,300,300))

                            if len(Calibrate_pos) == 9:
                                pygame.draw.rect(Calibrate_Display, (255,255,255), (1800,1000,300,300))
                                pygame.draw.rect(Calibrate_Display, (255, 255, 255), (100, 100, 100, 100))
                                message_to_screen_calibrate("Complete", (0, 0, 0), (100, 100), 50)

                                # Start Calculation
                                Screen_Info = pygame.display.Info()
                                screen_height = Screen_Info.current_h
                                Screen_width = Screen_Info.current_w

                                most_left_left = min(Calibrate_pos[0][0], Calibrate_pos[3][0], Calibrate_pos[6][0])
                                most_right_left = max(Calibrate_pos[2][0], Calibrate_pos[5][0], Calibrate_pos[8][0])

                                most_left_right = min(Calibrate_pos[0][1], Calibrate_pos[3][1], Calibrate_pos[6][1])
                                most_right_right = max(Calibrate_pos[2][1], Calibrate_pos[5][1], Calibrate_pos[8][1])

                                for i in range(1,8):
                                    d_left_x = (Calibrate_pos[i][0] - Calibrate_pos[i - 1][0])
                                    left_x_ratio_list.append(abs(d_left_x)/960)



                                    d_right_x = (Calibrate_pos[i][1] - Calibrate_pos[i - 1][1])
                                    right_x_ratio_list.append(abs(d_right_x)/960)



                                #mean

                                list(filter((0).__ne__, left_x_ratio_list))
                                list(filter((0).__ne__, right_x_ratio_list))

                                left_x_ratio = sum(left_x_ratio_list)/len(left_x_ratio_list)

                                right_x_ratio = sum(right_x_ratio_list)/len(right_x_ratio_list)


                                print(left_x_ratio)

                                print(right_x_ratio)






                                Calibrate_exit = True

                            pygame.display.update()








        Display = pygame.display.set_mode((800, 185))
        calibrate = False






    if Detect == True:


        Detect_Display = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
        Detect_Display.fill(white)
        Detect_Image = pygame.image.load('package/initial.png.jpg').convert()
        Detect_Display.blit(Detect_Image, (0, 0))
        pygame.display.update()

        Detect_exit = False

        while not Detect_exit:
            left_init_x = []
            left_init_y = []
            right_init_x = []
            right_init_y = []

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        print("Press")
                        i =0
                        while True:

                            init = detect()

                            if init is None:
                                continue
                            else:
                                left, right = init

                                left_x = left
                                right_x= right

                                left_init_x.append(left_x)

                                right_init_x.append(right_x)


                                left_mean_init_x = int(sum(left_init_x)/len(left_init_x))

                                right_mean_init_x = int(sum(right_init_x)/len(right_init_x))


                                i += 1
                                if i > 10:
                                    break
                                else:
                                    continue

                        print("complete")


                        os.system('video.mp4')
                        vid = []
                        start = time.time()
                        while True:
                            ret, frame = cap.read()

                            if ret == True:
                                vid.append(frame)

                                if (time.time() - start >= duration):

                                    break





                        cap.release()
                        cv2.destroyAllWindows()
                        lenth = len(vid)
                        print(lenth)
                        video_frame_ratio = (FPS * duration) // lenth

                        detected_eye_pos = []

                        for i in tqdm(range(0 ,lenth - 1)):

                            detected_eye_pos.append(detect_2(vid[i]))



                        with open("save/detect_eye_position.txt", "w") as f:
                            for s in detected_eye_pos:
                                f.write(str(s) + "\n")


                        with open("save/Init.txt", "w") as f:
                            f.write(str(left_mean_init_x) + "\n")

                            f.write(str(right_mean_init_x) + "\n")


                        with open("save/ratio.txt", "w") as f:
                            f.write(str(left_x_ratio) + "\n")

                            f.write(str(right_x_ratio) + "\n")

                            f.write(str() + "\n")
                            f.write(str(video_frame_ratio))

                        print("start_Process")

                        lenth = len(detected_eye_pos) - 1

                        move_position = []
                        eye_position = []

                        for i in tqdm(range(0, lenth)):
                            record_data = detected_eye_pos[i]


                            if record_data is None:
                                move_position.append(None)
                                eye_position.append(None)
                                continue
                            else:
                                move_position.append(process(most_left_left, most_right_left, most_left_right, most_right_right, record_data))
                                print(move_position)

                        with open("save/detect_eye.txt", "w") as f:
                            for s in move_position:
                                f.write(str(s) + "\n")










                        Detect = False
                        Detect_exit = True
                        break

        Display = pygame.display.set_mode((800, 185))
        pygame.display.update()

    pygame.display.update()
