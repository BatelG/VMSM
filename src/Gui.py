import os
from tkinter import *
import customtkinter
from PIL import Image, ImageTk  # <- import PIL for the images
from customtkinter import CTkCheckBox

from VMSM.src.resources.videoGenerator import Video

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
PATH = os.path.dirname(os.path.realpath(__file__))


class App(customtkinter.CTk):
    WIDTH = 750
    HEIGHT = 700

    def __init__(self):
        super().__init__()

        self.title("Video Motion Synchrony Measurement")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        # self.minsize(App.WIDTH, App.HEIGHT)

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

        file_explore_image = ImageTk.PhotoImage(
            Image.open(PATH + "\\resources\\images\\FileExplore.png").resize((image_size, image_size),
                                                                             Image.Resampling.LANCZOS))

        report_image = ImageTk.PhotoImage(
            Image.open(PATH + "\\resources\\images\\report_icon.png").resize((image_size, image_size),
                                                                             Image.Resampling.LANCZOS))

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)  # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)  # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.video_label = customtkinter.CTkLabel(master=self.frame_left,
                                                  text="Video Loader",
                                                  text_font=("Calibri Bold", -20))  # font name and size in px
        self.video_label.grid(row=1, column=0, pady=10, padx=10)

        self.file_explore_btn = customtkinter.CTkButton(master=self.frame_left, image=file_explore_image, text="",
                                                        width=30, height=30,
                                                        compound="right", command=self.video_handler)
        self.file_explore_btn.grid(row=2, column=0, pady=10, padx=20)

        self.report_image_btn = customtkinter.CTkButton(master=self.frame_left, image=report_image, text="",
                                                        width=30, height=30,
                                                        compound="right", command=self.video_handler)
        self.report_image_btn.grid(row=3, column=0, pady=10, padx=20)
        self.Theme_switch = customtkinter.CTkSwitch(master=self.frame_left,
                                                    text="Dark Mode",
                                                    command=self.change_mode)
        self.Theme_switch.grid(row=10, column=0, pady=10, padx=20, sticky="w")
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
        self.progressbar = customtkinter.CTkProgressBar(master=self.frame_info)
        self.progressbar.grid(row=1, column=0, sticky="ew", padx=15, pady=15)

        # ============ frame_right ============

        self.radio_var_RHand = IntVar(value=0)
        self.radio_var_LHand = IntVar(value=0)
        self.radio_var_pose = IntVar(value=0)

        self.roi_label = customtkinter.CTkLabel(master=self.frame_right,
                                                text="Select ROIs:",
                                                text_font=("Calibri Bold", -20))
        self.roi_label.grid(row=0, column=2, columnspan=1, pady=20, padx=10, sticky="")
        self.right_hand_roi_choice = CTkCheckBox(master=self.frame_right, text="Right Hand",
                                                 command=self.toggle_state(self.radio_var_RHand),
                                                 variable=self.radio_var_RHand, onvalue="on", offvalue="off")

        self.right_hand_roi_choice.grid(row=1, column=2, pady=10, padx=20, sticky="n")

        self.left_hand_roi_choice = CTkCheckBox(master=self.frame_right, text="Right Hand",
                                                command=self.toggle_state(self.radio_var_LHand),
                                                variable=self.radio_var_LHand, onvalue="on", offvalue="off")

        self.left_hand_roi_choice.grid(row=2, column=2, pady=10, padx=20, sticky="n")
        self.pose_roi_choice = CTkCheckBox(master=self.frame_right, text="Right Hand",
                                           command=self.toggle_state(self.radio_var_pose),
                                           variable=self.radio_var_pose, onvalue="on", offvalue="off")

        self.pose_roi_choice.grid(row=3, column=2, pady=10, padx=20, sticky="n")

        self.temp_label = customtkinter.CTkLabel(master=self.frame_right,
                                                 text="Percentage Deviation: ",
                                                 text_font=("Calibri Bold", -20))  # font name and size in px
        self.temp_label.grid(row=4, column=0, columnspan=2, pady=10, sticky="NS")
        self.slider_2 = customtkinter.CTkSlider(master=self.frame_right,
                                                command=self.MyHandler, from_=0, to=1)
        self.slider_2.grid(row=5, column=0, columnspan=1, pady=10, padx=125, sticky="NS")

        self.slider_button_2 = customtkinter.CTkButton(master=self.frame_right,
                                                       height=45,
                                                       width=105,
                                                       fg_color='gray',
                                                       hover_color='green',
                                                       text="Start Analysis",
                                                       corner_radius=15,
                                                       text_font=("Calibri Bold", -18),
                                                       command=self.button_event)
        self.slider_button_2.grid(row=6, column=0, columnspan=3, pady=10, padx=135, sticky="W")
        self.Theme_switch.select()
        self.slider_2.set(0.0)
        self.progressbar.set(0.0)

    def toggle_state(self, var):
        var.set(1 - var.get())

    def MyHandler(self, val):
        #  self.progressbar.set(val)
        self.temp_label.set_text('Percentage Deviation: %.2f' % val)

    def video_handler(self):
        video_frame = customtkinter.CTkFrame(self.frame_info)
        # path = self.video_loader_btn_handler()
        video_frame.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)
        # VideoGenerator(video_frame)

        Video(video_frame)

    def button_event(self):
        print("Button pressed")

    def change_mode(self):
        if self.Theme_switch.get() == 1:
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
