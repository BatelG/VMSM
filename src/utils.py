
import os
import time
import sys
import tkinter as tk
from tkinter import filedialog#, Tk
from tkvideo import tkvideo
import cv2
import mediapipe as mp


PATH = os.path.dirname(os.path.realpath(__file__))

# TODO check if GUI workes, then change it
# # create new popup window and set it's properties
# class Popup():
#     def __init__(self, title, width, height):
#         popup = Tk()

#         popup.title(title)
#         popup.geometry(f"{width}x{height}")

class Video():
    def __init__(self, frame):
        root = frame
        my_label = tk.Label(root)
        my_label.pack()

        self.path = self.video_loader_btn_handler()

        if self.path is not False:
            player = tkvideo(self.path, my_label, loop=1, size=(350, 250))
            player.play()

    def video_loader_btn_handler(self):
        filename_path = filedialog.askopenfilename(initialdir=PATH,
                                                title="Select a File",
                                                filetypes=(("Video files","*.mp4*")))

        if filename_path in ["", " "]:
            return False

        print(filename_path)

        return str(filename_path)

    @staticmethod
    def rescaleFrame(frame, scale=0.5):
        if frame is None:
            return

        width = int(frame.shape[1] * scale) # must be an integer
        height = int(frame.shape[0] * scale) # must be an integer
        dimensions = (width, height)

        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

    @staticmethod
    def detect_object(video_path, output):
        mp_drawing = mp.solutions.drawing_utils # set up MediaPipe
        mp_holistic = mp.solutions.holistic # set up holistic module

        # ** large photos and videos are need to be rescaling and resizing ** #

        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2) # apply styling

        cap = cv2.VideoCapture(video_path) # import video from file

        # Initialize min/max default values
        maxSize = sys.maxsize
        minSize = -sys.maxsize - 1
        minX = maxSize
        maxX = minSize
        minY = maxSize
        maxY = minSize

        # Initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            cnt = 0 # indicates the frame number

            while cap.isOpened():
                ret, frame = cap.read() # ret is unsed but nessesery due cap.read() return a tuple

                if frame is None:
                    break

                frame_resized = Video.rescaleFrame(frame, scale=1) # resize big video
                image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) # recolor feed
                results = holistic.process(image) # make detections

                print(results.pose_landmarks.landmark) # print coordinates

                # Loop on landmarks set for finding min,max of (x,y)
                for land_mark in results.pose_landmarks.landmark:
                    if minX > land_mark.x > 0:
                        minX = land_mark.x
                    if maxX < land_mark.x < 1:
                        maxX = land_mark.x
                    if minY > land_mark.y > 0:
                        minY = land_mark.y
                    if maxY < land_mark.y < 1:
                        maxY = land_mark.y

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                cv2.imshow('Detected Video', image) # show the video feed

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                cnt += 1

        print("Result:")
        print('minX = ', minX)
        print('maxX = ', maxX)
        print("##############")
        print('minY = ', minY)
        print('maxY = ', maxY)

        print("############################################")

        print("Delta calculations..")
        deltaX = maxX - minX
        deltaY = maxY - minY

        print(f'(x_0,y_0) = ({minX},{minY})')
        print('deltaX = ', deltaX)
        print('deltaY = ', deltaY)
        print(
            f'(minX+deltaX,minY+deltaY) = ({minX}+{deltaX},{minY}+{deltaY}) = ({minX + deltaX},{minY + deltaY})')

        cap.release()
        cv2.destroyAllWindows()

        print("crop video file...")

        Video.crop_video(video_path, output, minX, minY, deltaX, deltaY)
        return minX, maxX

    @staticmethod
    def crop_video(video_path, output, x_0_ratio, y_0_ratio, deltaX, deltaY):
        cap = cv2.VideoCapture(video_path) # open the video
        cnt = 0 # initialize frame counter

        # Some characteristics from the original video
        w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Calculate the original (x_0,y_0) coordinates of the frame
        x_0 = int(x_0_ratio * w_frame)
        y_0 = int(y_0_ratio * h_frame)

        # width and height if the cropped frame
        w_frame_crop = int(deltaX * w_frame)
        h_frame_crop = int(deltaY * h_frame)

        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output, 0x7634706d, fps, (w_frame_crop, h_frame_crop))

        while cap.isOpened():
            ret, frame = cap.read()
            frame = Video.rescaleFrame(frame, scale=1)

            cnt += 1  # counting frames

            # avoid problems when video finish
            if ret:
                crop_frame = frame[y_0:y_0 + h_frame_crop, x_0:x_0 + w_frame_crop]

                print(
                    f'making a cut frame #{cnt} - [x:x + w],[y:y + h] = [{x_0}-{x_0 + w_frame_crop}],[{y_0} - {y_0 + 2 * h_frame_crop}]')

                # show progress in percentage
                xx = cnt * 100 / frames
                print(int(xx), '%\n')

                out.write(crop_frame) # save the new video

                # see the video in real time
                cv2.imshow('frame', frame)
                cv2.imshow('croped', crop_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        time.sleep(2)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
