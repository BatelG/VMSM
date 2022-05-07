import os
import time
from tkinter import *
import customtkinter
from PIL import Image, ImageTk  # <- import PIL for the images
from customtkinter import CTkCheckBox
from resources.videoGenerator import *


customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

PATH = os.path.dirname(os.path.realpath(__file__))


class App(customtkinter.CTk):
    WIDTH = 750
    HEIGHT = 600

    def __init__(self):
        super().__init__()

        self.title("Video Motion Synchrony Measurement")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self, width=180, corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============load_images=============

        # load images as PhotoImage
        image_size = 35

        self.file_explore_image = ImageTk.PhotoImage(
            Image.open(PATH + "\\resources\\images\\FileExplore.png").resize((image_size, image_size),
                                                                             Image.Resampling.LANCZOS))

        self.report_image = ImageTk.PhotoImage(
            Image.open(PATH + "\\resources\\images\\report_icon.png").resize((image_size, image_size),
                                                                             Image.Resampling.LANCZOS))

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)  # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)  # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.video_lbl = customtkinter.CTkLabel(master=self.frame_left, text="Video Loader",
                                                text_font=("Calibri Bold", -20))  # font name and size in px
        self.video_lbl.grid(row=1, column=0, pady=10, padx=10)

        self.file_explore_btn = customtkinter.CTkButton(master=self.frame_left, image=self.file_explore_image,
                                                        text="", width=30, height=30,
                                                        compound="right", command=self.video_handler)
        self.file_explore_btn.grid(row=2, column=0, pady=10, padx=20)

        self.Theme_switch = customtkinter.CTkSwitch(master=self.frame_left, text="Dark Mode",
                                                    command=self.change_mode)
        self.Theme_switch.grid(row=10, column=0, pady=10, padx=20, sticky="w")
        self.Theme_switch.select()

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

        # ============ frame_right ============

        self.checkbox_var_RHand = IntVar(value=0)
        self.checkbox_var_LHand = IntVar(value=0)
        self.checkbox_var_pose = IntVar(value=0)

        self.roi_lbl = customtkinter.CTkLabel(master=self.frame_right, text="Select ROIs:",
                                                text_font=("Calibri Bold", -20))
        self.roi_lbl.grid(row=0, column=2, columnspan=1, pady=20, padx=10, sticky="")

        self.right_hand_roi_choice = CTkCheckBox(master=self.frame_right, text="Right Hand",
                                                command=self.toggle_state(self.checkbox_var_RHand),
                                                variable=self.checkbox_var_RHand, onvalue="on", offvalue="off")
        self.right_hand_roi_choice.grid(row=1, column=2, pady=10, padx=20, sticky="n")

        self.left_hand_roi_choice = CTkCheckBox(master=self.frame_right, text="Left Hand  ",
                                                command=self.toggle_state(self.checkbox_var_LHand),
                                                variable=self.checkbox_var_LHand, onvalue="on", offvalue="off")
        self.left_hand_roi_choice.grid(row=2, column=2, pady=10, padx=20, sticky="n")

        self.pose_roi_choice = CTkCheckBox(master=self.frame_right, text="Pose          ",
                                           command=self.toggle_state(self.checkbox_var_pose),
                                           variable=self.checkbox_var_pose, onvalue="on", offvalue="off")
        self.pose_roi_choice.grid(row=3, column=2, pady=10, padx=20, sticky="n")

        self.perc_dev_lbl = customtkinter.CTkLabel(master=self.frame_right, text="Percentage Deviation:",
                                                    text_font=("Calibri Bold", -20))
        self.perc_dev_lbl.grid(row=4, column=0, columnspan=2, pady=10, sticky="NS")

        self.slider = customtkinter.CTkSlider(master=self.frame_right, command=self.MyHandler, from_=0, to=1)
        self.slider.grid(row=5, column=0, columnspan=1, pady=10, padx=125, sticky="NS")
        self.slider.set(0.0)

        self.start_btn = customtkinter.CTkButton(master=self.frame_right, height=45, width=105,
                                                fg_color='gray', hover_color='green', text="Start Analysis", 
                                                corner_radius=15, text_font=("Calibri Bold", -18),
                                                command=self.start_btn_handler)
        self.start_btn.grid(row=6, column=0, columnspan=3, pady=10, padx=135, sticky="W")

    def toggle_state(self, var):
        var.set(1 - var.get())

    def MyHandler(self, val):
        self.perc_dev_lbl.set_text('Percentage Deviation: %.2f' % val)

    def video_handler(self):
        video_frame = customtkinter.CTkFrame(self.frame_info)
        # path = self.video_loader_btn_handler()
        video_frame.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)
        # VideoGenerator(video_frame)

        Video(video_frame)

    def start_btn_handler(self):
        time.sleep(1)

        # at least one ROI has to be selected
        if "on" not in [self.right_hand_roi_choice.get(), self.left_hand_roi_choice.get(), self.pose_roi_choice.get()]:
            fail_popup = Tk()

            fail_popup.title("Fail")
            fail_popup.geometry(f"{440}x{130}")

            select_type_lbl = customtkinter.CTkLabel(master=fail_popup,
                                                text="At least one ROI has to be selceted.\nPlease try again.",
                                                text_color='black', text_font=("Calibri Bold", -20))
            select_type_lbl.grid(row=0, column=3, pady=10, padx=1)

            ok_btn = customtkinter.CTkButton(master=fail_popup, text="Ok", width=70, height=40,
                                            fg_color='gray', hover_color='green', compound="right", 
                                            command=fail_popup.destroy)
            ok_btn.grid(row=1, column=3, padx=190, sticky=NS)

            fail_popup.mainloop()
        else:
            success_popup = Tk()

            success_popup.title("Success")
            success_popup.geometry(f"{440}x{130}")

            report_lbl = customtkinter.CTkLabel(master=self.frame_left, text="Reports Producer",
                                                text_font=("Calibri Bold", -20))  # font name and size in px
            report_lbl.grid(row=3, column=0, pady=10, padx=10)

            report_image_btn = customtkinter.CTkButton(master=self.frame_left, image=self.report_image,
                                                        text="", width=30, height=30,
                                                        compound="right", command=self.report_btn_handler)
            report_image_btn.grid(row=4, column=0, pady=10, padx=20)

            select_type_lbl = customtkinter.CTkLabel(master=success_popup,
                                                text="The Interpersonal Synchrony Analysis\nCompleted Successfully !",
                                                text_color='black', text_font=("Calibri Bold", -20))
            select_type_lbl.grid(row=0, column=3, pady=10, padx=1)

            ok_btn = customtkinter.CTkButton(master=success_popup, text="Ok", width=70, height=40,
                                            fg_color='gray', hover_color='green', compound="right", 
                                            command=success_popup.destroy)
            ok_btn.grid(row=1, column=3, padx=190, sticky=NS)

            success_popup.mainloop()

    def report_btn_handler(self):
        checkbox_var_txt = IntVar(value=0)
        checkbox_var_pdf = IntVar(value=0)

        self.export_popup = Tk()

        self.export_popup.title("Export Results to Report")
        self.export_popup.geometry(f"{300}x{200}")

        self.txt_choice = CTkCheckBox(master=self.export_popup, text=".txt report  ", text_color='black',
                                command=self.toggle_state(checkbox_var_txt),
                                variable=checkbox_var_txt, onvalue="on", offvalue="off")
        self.txt_choice.grid(row=1, column=0, pady=10, padx=60)

        self.pdf_choice = CTkCheckBox(master=self.export_popup, text=".pdf report", text_color='black',
                                command=self.toggle_state(checkbox_var_pdf),
                                variable=checkbox_var_pdf, onvalue="on", offvalue="off")
        self.pdf_choice.grid(row=2, column=0, pady=10, padx=60)

        select_type_lbl = customtkinter.CTkLabel(master=self.export_popup, text="Select Report type:",
                                                text_color='black', text_font=("Calibri Bold", -20))
        select_type_lbl.grid(row=0, column=0, pady=10, padx=60)

        export_btn = customtkinter.CTkButton(master=self.export_popup, text="Export", width=70, height=40,
                                            fg_color='gray', hover_color='green',
                                            compound="right", command=self.export_report)
        export_btn.grid(row=3, column=0, pady=10, padx=60)

        self.export_popup.mainloop()

    def export_report(self):
        # at least one report type has to be selected
        if "on" not in [self.txt_choice.get(), self.pdf_choice.get()]:
            fail_popup = Tk()

            fail_popup.title("Fail")
            fail_popup.geometry(f"{440}x{130}")

            select_type_lbl = customtkinter.CTkLabel(master=fail_popup,
                                                text="At least one report type has to be selceted.\nPlease try again.",
                                                text_color='black', text_font=("Calibri Bold", -20))
            select_type_lbl.grid(row=0, column=3, pady=10, padx=1)

            ok_btn = customtkinter.CTkButton(master=fail_popup, text="Ok", width=70, height=40,
                                            fg_color='gray', hover_color='green', compound="right", 
                                            command=fail_popup.destroy)
            ok_btn.grid(row=1, column=3, padx=190, sticky=NS)

            fail_popup.mainloop()
        else:
            self.export_popup.destroy() # temporary
            #TODO: use file system to download the selected report files

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
