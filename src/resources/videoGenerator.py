import os
import time
import tkinter as tk
from tkinter import filedialog, ttk

import imageio
from PIL import Image, ImageTk
# from moviepy.video.io.VideoFileClip import VideoFileClip
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


############################################################################3


class ToasterBox(tk.Toplevel):
    """
            -----DESCRIPTION-----
    A template for new widgets.
            -----USAGE-----
    toasterbox = ToasterBox(parent)
    toasterbox.create_popup(title=[string], image=[photoimage]/[string], message=[string], life=[integer])
            -----PARAMETERS-----
    parent = The parent of the widget.
            -----CONTENTS-----
    ---VARIABLES---
    parent         = The parent of the widget.
    _width         = The width of the widget.
    _padx          = The horizontal padding of the widget.
    _pady          = The vertical padding of the widget.
    _popup_fit     = The amount of popups to fit into the widget.
    _popup_pad     = The padding of the popups.
    _popup_ipad    = The internal padding of the popups.
    _popup_height  = The height of each popup.
    ---TKINTER VARIABLES---
    None
    ---WIDGETS---
    self
    ---FUNCTIONS---
    create_popup() = Creates a new popup.
    """

    def __init__(self, parent, width=350, padx=5, pady=45, popup_fit=5, popup_pad=5, popup_ipad=3, popup_height=100,
                 *args):
        tk.Toplevel.__init__(self, parent, *args)
        self.parent = parent
        self._width = width
        self._padx = padx
        self._pady = pady
        self._popup_fit = popup_fit
        self._popup_pad = popup_pad
        self._popup_ipad = popup_ipad
        self._popup_height = popup_height

        self.attributes("-toolwindow", True, "-topmost", True)
        self.overrideredirect(True)

        self.geometry("{}x{}".format(self._width, self._popup_fit * (self._popup_height + (self._popup_pad * 2))))
        self.update()
        self.geometry("+{}+{}".format((self.winfo_screenwidth() - self.winfo_width()) - self._padx,
                                      (self.winfo_screenheight() - self.winfo_height()) - self._pady))

        ttk.Style().configure("Popup.TFrame", borderwidth=10, relief="raised")
        ttk.Style().configure("Close.Popup.TButton")
        ttk.Style().configure("Image.Popup.TLabel")
        ttk.Style().configure("Title.Popup.TLabel")
        ttk.Style().configure("Message.Popup.TLabel")

    def create_popup(self, title="", image=None, message="", life=0):
        """Creates a new popup."""
        popup = Popup(self, title=title, image=image, message=message, life=life, height=self._popup_height,
                      ipad=self._popup_ipad)
        popup.pack(side="bottom", fill="x", pady=self._popup_pad)

        return popup


class Popup(ttk.Frame):
    def __init__(self, parent, title, image, message, life, height, ipad, *args):
        ttk.Frame.__init__(self, parent, height=height, style="Popup.TFrame", *args)
        self.parent = parent
        self._title = title
        self._image = image
        self._life = life
        self._message = message
        self._ipad = ipad

        self.grid_propagate(False)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)

        image = ttk.Label(self, image=self._image, style="Image.Popup.TLabel")
        image.grid(row=0, column=0, rowspan=2, sticky="nesw", padx=self._ipad, pady=self._ipad)

        title_frame = ttk.Frame(self)
        title_frame.grid(row=0, column=1, sticky="we", padx=self._ipad, pady=self._ipad)

        label = ttk.Label(title_frame, text=self._title, style="Title.Popup.TLabel")
        label.pack(side="left", fill="both", expand=True)

        close = ttk.Button(title_frame, text="X", width=3, command=self.remove, style="Close.Popup.TButton")
        close.pack(side="right")

        message = ttk.Label(self, text=self._message, style="Message.Popup.TLabel")
        message.grid(row=1, column=1, sticky="nw", padx=self._ipad, pady=self._ipad)

        if self._life > 0:
            self.after(self._life, self.remove)

    def remove(self, event=None):
        self.pack_forget()


##################################################

# if __name__ == "__main__":
#     root = tk.Tk()
#     tbox = ToasterBox(root)
#     tbox.create_popup(title="Popup 1", message="Hello!")
#     root.mainloop()
