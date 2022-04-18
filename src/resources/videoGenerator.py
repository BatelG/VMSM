import os
import time
import tkinter as tk
from tkinter import filedialog

import imageio
from PIL import Image, ImageTk
#from moviepy.video.io.VideoFileClip import VideoFileClip
from tkvideo import tkvideo

global pause_video


# download video at: http://www.html5videoplayer.net/videos/toystory.mp4

# class VideoGenerator:

#     def __init__(self, frame):
#         # root = tk.Tk()
#         path = self.video_loader_btn_handler()
#         if path is not False:
#             self.video = imageio.get_reader(self.video_loader_btn_handler())
#             root = frame
#             root.title('Video in tkinter')
#             my_label = tk.Label(root)
#             my_label.pack()
#             tk.Button(root, text='start', command=self._start).pack(side=tk.LEFT)
#             tk.Button(root, text='stop', command=self._stop).pack(side=tk.LEFT)

#             pause_video = False
#             movie_frame = self.video_frame_generator()

#             while True:
#                 if not pause_video:
#                     frame_number, frame = next(movie_frame)
#                     my_label.config(image=frame)

#                 root.update()

#     def video_frame_generator(self):
#         def current_time():
#             return time.time()

#         start_time = current_time()
#         _time = 0
#         for frame, image in enumerate(self.video.iter_data()):

#             # turn video array into an image and reduce the size
#             image = Image.fromarray(image)
#             image.thumbnail((750, 750), Image.ANTIALIAS)

#             # make image in a tk Image and put in the label
#             image = ImageTk.PhotoImage(image)

#             # introduce a wait loop so movie is real time -- asuming frame rate is 24 fps
#             # if there is no wait check if time needs to be reset in the event the video was paused
#             _time += 1 / 24
#             run_time = current_time() - start_time
#             while run_time < _time:
#                 run_time = current_time() - start_time
#             else:
#                 if run_time - _time > 0.1:
#                     start_time = current_time()
#                     _time = 0

#             yield frame, image

#     def _stop(self):
#         global pause_video
#         pause_video = True

#     def _start(self):
#         global pause_video
#         pause_video = False

#     def video_loader_btn_handler(self):
#         PATH = os.path.dirname(os.path.realpath(__file__))
#         filename_path = filedialog.askopenfilename(initialdir=PATH,
#                                                    title="Select a File",
#                                                    filetypes=(("Video files",
#                                                                "*.mp4*"),
#                                                               ))

#         if filename_path == "" or filename_path == " ":
#             return False

#         print(filename_path)

#         clip = VideoFileClip(filename_path)

#         # previewing the clip at fps = 10
#         clip.without_audio().preview(fps=10)
#         return False
#         return str(filename_path)


class Video():
    def __init__(self, frame):
        root = frame
        my_label = tk.Label(root)

        my_label.pack()

        path = self.video_loader_btn_handler()
        if path is not False:
            player = tkvideo(path, my_label, loop=1, size=(350, 250))
            player.play()

    def video_loader_btn_handler(self):
        PATH = os.path.dirname(os.path.realpath(__file__))
        filename_path = filedialog.askopenfilename(initialdir=PATH,
                                                   title="Select a File",
                                                   filetypes=(("Video files",
                                                               "*.mp4*"),
                                                              ))

        if filename_path == "" or filename_path == " ":
            return False

        print(filename_path)

        # clip = VideoFileClip(filename_path)
        #
        # # previewing the clip at fps = 10
        # clip.without_audio().preview(fps=10)

        return str(filename_path)
