import os
from tkinter import *
import tkinter.messagebox
from tkinter import filedialog
from tkvideo import tkvideo
import customtkinter

from PIL import Image, ImageTk  # <- import PIL for the images


PATH = os.path.dirname(os.path.realpath(__file__))


customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class Video():
    def __init__(self, frame):
        root = frame
        my_label = Label(root)

        my_label.pack()

        path = self.video_loader_btn_handler()
        player = tkvideo(path, my_label, loop=1, size=(450, 350))

        player.play()

    def video_loader_btn_handler(self):
        init = "C:\\Users\\batel\\Desktop\\Projects\\MediaPipe\\Videos" #TODO: change path
        filename_path = filedialog.askopenfilename(initialdir=init,
                                                   title="Select a File",
                                                   filetypes=(("Video files",
                                                               "*.mp4*"),
                                                    ))
        if filename_path == "":
            return

        print(filename_path)

        return str(filename_path)


class App(customtkinter.CTk):

    WIDTH = 650
    HEIGHT = 750

    def __init__(self):
        super().__init__()

        self.title("Video Motion Synchrony Measurement")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        #self.minsize(App.WIDTH, App.HEIGHT)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============load_images=============
        # load images as PhotoImage
        image_size = 30

        file_explore_image = ImageTk.PhotoImage(Image.open(PATH + "\\resources\\images\\FileExplore.png").resize((image_size, image_size), Image.Resampling.LANCZOS))

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.video_label = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Video Loader",
                                              text_font=("Calibri Bold", -20))  # font name and size in px
        self.video_label.grid(row=1, column=0, pady=10, padx=10)

        self.file_explore_btn = customtkinter.CTkButton(master=self.frame_left, image=file_explore_image, text="", width=30, height=30,
                                   compound="right", command=self.video_handler)
        self.file_explore_btn.grid(row=2, column=0, pady=10, padx=20)

        # self.button_2 = customtkinter.CTkButton(master=self.frame_left,
        #                                         text="CTkButton 2",
        #                                         fg_color=("gray75", "gray30"),  # <- custom tuple-color
        #                                         command=self.button_event)
        # self.button_2.grid(row=3, column=0, pady=10, padx=20)

        # self.button_3 = customtkinter.CTkButton(master=self.frame_left,
        #                                         text="CTkButton 3",
        #                                         fg_color=("gray75", "gray30"),  # <- custom tuple-color
        #                                         command=self.button_event)
        # self.button_3.grid(row=4, column=0, pady=10, padx=20)

        self.switch_1 = customtkinter.CTkSwitch(master=self.frame_left)
        self.switch_1.grid(row=9, column=0, pady=10, padx=20, sticky="w")

        self.switch_2 = customtkinter.CTkSwitch(master=self.frame_left,
                                                text="Dark Mode",
                                                command=self.change_mode)
        self.switch_2.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3), weight=1)
        self.frame_right.rowconfigure(7, weight=10)
        self.frame_right.columnconfigure((0, 1), weight=1)
        self.frame_right.columnconfigure(2, weight=0)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=2, rowspan=4, pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.columnconfigure(0, weight=1)

        # self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
        #                                            text="CTkLabel: Lorem ipsum dolor sit,\n" +
        #                                                 "amet consetetur sadipscing elitr,\n" +
        #                                                 "sed diam nonumy eirmod tempor" ,
        #                                            height=100,
        #                                            fg_color=("white", "gray38"),  # <- custom tuple-color
        #                                            justify=tkinter.LEFT)
        # self.label_info_1.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)

        self.progressbar = customtkinter.CTkProgressBar(master=self.frame_info)
        self.progressbar.grid(row=1, column=0, sticky="ew", padx=15, pady=15)

        # ============ frame_right ============

        self.radio_var = tkinter.IntVar(value=0)

        self.roi_label = customtkinter.CTkLabel(master=self.frame_right,
                                                        text="Select ROI:",
                                                        text_font=("Calibri Bold", -20))
        self.roi_label.grid(row=0, column=2, columnspan=1, pady=20, padx=10, sticky="")

        self.right_hand_roi_choice = customtkinter.CTkRadioButton(master=self.frame_right,
                                                            text="Right Hand",
                                                           variable=self.radio_var,
                                                           value=0)
        self.right_hand_roi_choice.grid(row=1, column=2, pady=10, padx=20, sticky="n")

        self.left_hand_roi_choice = customtkinter.CTkRadioButton(master=self.frame_right,
                                                            text="Left Hand  ",
                                                           variable=self.radio_var,
                                                           value=1)
        self.left_hand_roi_choice.grid(row=2, column=2, pady=10, padx=20, sticky="n")

        self.pose_roi_choice = customtkinter.CTkRadioButton(master=self.frame_right,
                                                            text="Pose          ",
                                                           variable=self.radio_var,
                                                           value=2)
        self.pose_roi_choice.grid(row=3, column=2, pady=10, padx=20, sticky="n")

        self.slider_1 = customtkinter.CTkSlider(master=self.frame_right,
                                                from_=0,
                                                to=1,
                                                number_of_steps=3,
                                                command=self.progressbar.set)
        self.slider_1.grid(row=4, column=0, columnspan=2, pady=10, padx=20, sticky="we")

        self.slider_2 = customtkinter.CTkSlider(master=self.frame_right,
                                                command=self.progressbar.set)
        self.slider_2.grid(row=5, column=0, columnspan=2, pady=10, padx=20, sticky="we")

        self.slider_file_explore_btn = customtkinter.CTkButton(master=self.frame_right,
                                                       height=25,
                                                       text="CTkButton",
                                                       command=self.button_event)
        self.slider_file_explore_btn.grid(row=4, column=2, columnspan=1, pady=10, padx=20, sticky="we")

        self.slider_button_2 = customtkinter.CTkButton(master=self.frame_right,
                                                       height=25,
                                                       text="CTkButton",
                                                       command=self.button_event)
        self.slider_button_2.grid(row=5, column=2, columnspan=1, pady=10, padx=20, sticky="we")

        self.checkbox_file_explore_btn = customtkinter.CTkButton(master=self.frame_right,
                                                         height=25,
                                                         text="CTkButton",
                                                         border_width=3,   # <- custom border_width
                                                         fg_color=None,   # <- no fg_color
                                                         command=self.button_event)
        self.checkbox_file_explore_btn.grid(row=6, column=2, columnspan=1, pady=10, padx=20, sticky="we")

        self.check_box_1 = customtkinter.CTkCheckBox(master=self.frame_right,
                                                     text="CTkCheckBox")
        self.check_box_1.grid(row=6, column=0, pady=10, padx=20, sticky="w")

        self.check_box_2 = customtkinter.CTkCheckBox(master=self.frame_right,
                                                     text="CTkCheckBox")
        self.check_box_2.grid(row=6, column=1, pady=10, padx=20, sticky="w")

        # self.entry = customtkinter.CTkEntry(master=self.frame_right,
        #                                     width=120,
        #                                     placeholder_text="CTkEntry")
        # self.entry.grid(row=8, column=0, columnspan=2, pady=20, padx=20, sticky="we")

        self.button_5 = customtkinter.CTkButton(master=self.frame_right,
                                                text="CTkButton",
                                                command=self.button_event)
        self.button_5.grid(row=8, column=2, columnspan=1, pady=20, padx=20, sticky="we")

        # set default values
        self.right_hand_roi_choice.select()
        self.switch_2.select()
        self.slider_1.set(0.2)
        self.slider_2.set(0.7)
        self.progressbar.set(0.5)
        self.slider_file_explore_btn.configure(state=tkinter.DISABLED, text="Disabled Button")
        #self.pose_roi_choice.configure(state=tkinter.DISABLED)
        self.check_box_1.configure(state=tkinter.DISABLED, text="CheckBox disabled")
        self.check_box_2.select()

    def video_handler(self):
        video_frame = customtkinter.CTkFrame(self.frame_info)
        video_frame.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)
        Video(video_frame)

    def button_event(self):
        print("Button pressed")

    def change_mode(self):
        if self.switch_2.get() == 1:
            customtkinter.set_appearance_mode("dark")
        else:
            customtkinter.set_appearance_mode("light")

    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
