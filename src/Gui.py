import os
from tkinter import *
from tkinter import filedialog
from tkvideo import tkvideo


class Video():
    def __init__(self, frame):
        root = frame
        my_label = Label(root)

        my_label.pack()

        path = self.video_loader_btn_handler()
        player = tkvideo(path, my_label, loop=1, size=(450, 350))

        player.play()

    def video_loader_btn_handler(self):
        init = "C:\\Users\\batel\\Desktop\\Projects\\MediaPipe\\Videos"
        filename_path = filedialog.askopenfilename(initialdir=init,
                                                   title="Select a File",
                                                   filetypes=(("Video files",
                                                               "*.mp4*"),
                                                    ))
        if filename_path == "":
            return

        print(filename_path)

        return str(filename_path)


class Gui():
    def __init__(self):
        self.main_window = Tk()
        self.main_window.title("Final Project")
        self.main_window.geometry('650x750')
        self.main_window['bg'] = '#E2CAC5'

        self.main_frame = Frame(self.main_window, bg='#E2CAC5')
        self.main_frame.grid(row=0, column=0, padx=0, pady=0)

        self.video_label = Label(self.main_frame, text="Video Loader", bg='#E2CAC5', height=2)
        self.video_label.grid(row=0, column=0, padx=3, pady=5)
        self.video_label.config(font=("Calibri", 20))

        self.video_text = Text(self.main_frame, width=35, height=1.5)
        self.video_text.grid(row=0, column=1, padx=0, pady=5)
        self.video_text.config(state='disable')

        self.file_explore_logo = PhotoImage(file="resources\\images\\FileExploreLogo.png", width=25, height=25)

        self.video_loader_btn = Button(self.main_frame, command=self.video_handler,
                                       image=self.file_explore_logo)
        self.video_loader_btn.grid(row=0, column=2, padx=3, pady=5)

        self.border = Label(self.main_frame, width=1, height=11, bg='black')
        self.border.grid(row=3, column=1, padx=20, sticky=N)

        self.roi_label = Label(self.main_frame, text="Select ROI", bg='#E2CAC5', height=2)
        self.roi_label.config(font=("Calibri", 20))
        self.roi_label.grid(row=2, column=1, padx=5, sticky=S)

        self.right_hand_roi_choice = IntVar()
        self.right_hand_roi_choice.set(0)
        self.right_hand_btn = Checkbutton(self.border, text="Right Hand", variable=self.right_hand_roi_choice, width=10,
                                          onvalue=1, offvalue=0)
        self.right_hand_btn.config(font=("Calibri", 16))
        self.right_hand_btn.grid(row=0, column=0, sticky=SW)

        self.left_hand_roi_choice = IntVar()
        self.left_hand_roi_choice.set(0)
        self.left_hand_btn = Checkbutton(self.border, text="Left Hand  ", variable=self.left_hand_roi_choice, width=10,
                                         onvalue=1, offvalue=0)
        self.left_hand_btn.config(font=("Calibri", 16))
        self.left_hand_btn.grid(row=1, column=0, sticky=SW)

        self.pose_roi_choice = IntVar()
        self.pose_roi_choice.set(0)
        self.pose_btn = Checkbutton(self.border, text="Pose          ", variable=self.pose_roi_choice, width=10,
                                    onvalue=1, offvalue=0)
        self.pose_btn.config(font=("Calibri", 16))
        self.pose_btn.grid(row=2, column=0, sticky=SW)

    def video_handler(self):
        self.video_frame = Frame(self.main_window, bg='#E2CAC5')
        self.video_frame.grid(row=1, column=0, padx=100, pady=30, sticky=NSEW)
        Video(self.video_frame)


if __name__ == '__main__':
    _Gui = Gui()
    _Gui.main_window.mainloop()
