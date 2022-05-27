import os
import time
from tkinter import *
import customtkinter
from PIL import Image, ImageTk
from customtkinter import CTkCheckBox
from resources.videoGenerator import *


customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


PATH = os.path.dirname(os.path.realpath(__file__))


# TODO: think of all the cases where the data need to be reset for new data (e.g. new video)
# TODO: save empty .txt and .pdf files
# TODO: disable/unable labels/buttons
class App(customtkinter.CTk):
    WIDTH = 850
    HEIGHT = 600

    def __init__(self):
        super().__init__()

        self.title("Video Motion Synchrony Measurement")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.__on_closing)

        # ==========create_two_frames=========

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # set the left frame
        self.frame_left = customtkinter.CTkFrame(master=self, width=180, corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        #set the right frame
        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=3, sticky="nswe", padx=20, pady=20)

        # ============load_images=============

        image_size = 35
        self.video_explore_image = self.__load_image("video_explore.png", image_size, image_size)
        self.file_image = self.__load_image("file.png", image_size, image_size)

        # ============ frame_left ============

        # configure grid layout (1x11)
        for i in range(0,9):
            self.frame_left.grid_rowconfigure(i, minsize=10) # empty row with minsize as spacing

        self.frame_left.grid_rowconfigure(9, weight=10) # empty row as spacing

        # set the 'Video Loader' label and button
        self.video_lbl = customtkinter.CTkLabel(master=self.frame_left, text="Video Loader",
                                                text_font=("Calibri Bold", -20))  # font name and size in px
        self.video_lbl.grid(row=1, column=0, pady=10, padx=10, sticky="n")

        self.video_explore_btn = customtkinter.CTkButton(master=self.frame_left, image=self.video_explore_image,
                                                        text="", width=30, height=30,
                                                        compound="right", command=self.__video_btn_handler)
        self.video_explore_btn.grid(row=2, column=0, pady=10, padx=20, sticky="n")

        # set the theme switcher
        self.Theme_switch = customtkinter.CTkSwitch(master=self.frame_left, text="Dark Mode",
                                                    command=self.__change_mode)
        self.Theme_switch.grid(row=10, column=0, pady=10, padx=20, sticky="w")
        self.Theme_switch.select()

        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3), weight=1)
        self.frame_right.rowconfigure(7, weight=10)
        self.frame_right.columnconfigure((0, 1, 2), weight=1)
        self.frame_right.columnconfigure(3, weight=0)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=2, rowspan=4, pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.columnconfigure(0, weight=1)

        # ============ frame_right ============

        # set the 'Select ROIs' lables and checkboxes
        self.roi_lbl = customtkinter.CTkLabel(master=self.frame_right, text="Select ROIs:",
                                                text_font=("Calibri Bold", -20))
        self.roi_lbl.grid(row=0, column=2, columnspan=1, pady=20, padx=10, sticky="n")

        # 'Right hand' choice
        self.checkbox_var_RHand = IntVar(value=0) # init checkbox
        self.right_hand_roi_choice = CTkCheckBox(master=self.frame_right, text="Right Hand",
                                                command=self.__toggle_state(self.checkbox_var_RHand),
                                                variable=self.checkbox_var_RHand, onvalue="on", offvalue="off")
        self.right_hand_roi_choice.grid(row=1, column=2, pady=10, padx=20, sticky="n")

        # 'Left hand' choice
        self.checkbox_var_LHand = IntVar(value=0) # init checkbox
        self.left_hand_roi_choice = CTkCheckBox(master=self.frame_right, text="Left Hand  ",
                                                command=self.__toggle_state(self.checkbox_var_LHand),
                                                variable=self.checkbox_var_LHand, onvalue="on", offvalue="off")
        self.left_hand_roi_choice.grid(row=2, column=2, pady=10, padx=20, sticky="n")

        # 'Pose' choice
        self.checkbox_var_pose = IntVar(value=0) # init checkbox
        self.pose_roi_choice = CTkCheckBox(master=self.frame_right, text="Pose          ",
                                           command=self.__toggle_state(self.checkbox_var_pose),
                                           variable=self.checkbox_var_pose, onvalue="on", offvalue="off")
        self.pose_roi_choice.grid(row=3, column=2, pady=10, padx=20, sticky="n")

        # set the 'Percentage Deviation' label and values slider
        self.perc_dev_lbl = customtkinter.CTkLabel(master=self.frame_right, text="Percentage Deviation:",
                                                    text_font=("Calibri Bold", -20))
        self.perc_dev_lbl.grid(row=4, column=0, columnspan=2, pady=10, sticky="ns")

        self.slider = customtkinter.CTkSlider(master=self.frame_right, command=self.__prec_slider_handler, from_=0, to=1)
        self.slider.grid(row=5, column=0, columnspan=1, pady=10, padx=125, sticky="ns")
        self.slider.set(0.0)

        # set the 'Start Analysis' button
        self.start_btn = customtkinter.CTkButton(master=self.frame_right, height=45, width=105,
                                                fg_color='gray', hover_color='green', text="Start Analysis",
                                                corner_radius=15, text_font=("Calibri Bold", -18),
                                                command=self.__start_btn_handler)
        self.start_btn.grid(row=6, column=0, columnspan=3, pady=10, padx=135, sticky="w")

    # set the shown float values in format .2f for the 'Percentage Deviation' slider
    def __prec_slider_handler(self, val):
        self.perc_dev_lbl.set_text(f"Percentage Deviation: {round(val, 2)}")

    # upload video and set the frame for it
    def __video_btn_handler(self):
        video_frame = customtkinter.CTkFrame(self.frame_info)
        # path = self.video_loader_btn_handler()
        video_frame.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)
        # VideoGenerator(video_frame)

        self.video = Video(video_frame)

    # handler for pressing 'Start Analysis' button
    def __start_btn_handler(self):
        time.sleep(1)

        rois_choices = [self.right_hand_roi_choice, self.left_hand_roi_choice, self.pose_roi_choice]
        selected_checkboxes = []

        # only the choices that are "on"
        for choice in rois_choices:
            if choice.get() == "on":
                selected_checkboxes.append(choice)

        # at least one ROI has to be selected
        if not selected_checkboxes:
            # pop a fail message
            title = "Start Fail"
            message = "At least one ROI has to be selceted.\nPlease try again."

            self.__message(title, message, "ok")

        # video wasn't uploaded
        if not hasattr(self, 'video'):
            # pop a fail message
            title = "Start Fail"
            message = "A video must be uploaded.\nPlease try again."

            self.__message(title, message, "ok")

        else:
            # add option to export the analysis result reports (set label and button)
            self.report_lbl = customtkinter.CTkLabel(master=self.frame_left, text="Reports Producer",
                                                text_font=("Calibri Bold", -20)) # font name and size in px
            self.report_lbl.grid(row=3, column=0, pady=25, padx=10, sticky="n")

            self.file_image_btn = customtkinter.CTkButton(master=self.frame_left, image=self.file_image,
                                                        text="", width=30, height=30,
                                                        compound="right", command=self.__report_btn_handler)
            self.file_image_btn.grid(row=4, column=0, pady=0, padx=5, sticky="n")

            # show to the user the synchronization rate
            # TODO: get the actual calculated grade, instead of the 'Percentage Deviation' value !
            # TODO: in the future, send the grade as is, without 'round' method (?)
            sync_rate, padx, color = self.__get_synchronization_rate(round(self.slider.get(), 2))

            # set the 'Synchronization Rate' label
            self.sync_rate_lbl = customtkinter.CTkLabel(master=self.frame_right, text_font=("Calibri Bold", -20))
            self.sync_rate_lbl.grid(row=7, column=0, columnspan=3, pady=10, sticky="nw")

            # configure the right values, according to the grade
            self.sync_rate_lbl.configure(text=sync_rate, text_color=color)
            self.sync_rate_lbl.grid(padx=padx)

            # pop a success message
            title = "Start Success"
            message = "The Interpersonal Synchrony Analysis\nCompleted Successfully !"

            self.__message(title, message, "ok")

    # get the synchronization rate and label configuration values, according to the grade
    def __get_synchronization_rate(self, grade):
        if grade < 0 or grade > 1:
            raise ValueError("The grade isn't normalized number !")

        # the second value in 'range' method doesn't count
        if 0 <= grade <= 0.33:
            return "Weak Synchronization", 100, '#FF3200'
        if 0.34 <= grade <= 0.66:
            return "Medium Synchronization", 92, '#FF9B00'
        if 0.67 <= grade <= 0.95:
            return "Strong Synchronization", 88, '#C2C000'
        if 0.96 <= grade <= 1:
            return "Perfect Synchronization", 88, '#359C25'

    # handler for pressing the 'Reports Producer' button
    def __report_btn_handler(self):
        # set the window properties
        self.export_popup = self.__new_popup("Export Results", 300, 200)

        # export txt raw data choice
        checkbox_var_txt = IntVar(value=0) # init checkbox
        self.txt_choice = CTkCheckBox(master=self.export_popup, text="Raw Data", text_color='black',
                                command=self.__toggle_state(checkbox_var_txt),
                                variable=checkbox_var_txt, onvalue="on", offvalue="off")
        self.txt_choice.grid(row=1, column=0, pady=10, padx=60)

        # export pdf report choice
        checkbox_var_pdf = IntVar(value=0) # init checkbox
        self.pdf_choice = CTkCheckBox(master=self.export_popup, text="Report      ", text_color='black',
                                command=self.__toggle_state(checkbox_var_pdf),
                                variable=checkbox_var_pdf, onvalue="on", offvalue="off")
        self.pdf_choice.grid(row=2, column=0, pady=10, padx=60)

        # set export label and button
        select_type_lbl = customtkinter.CTkLabel(master=self.export_popup, text="Select File to Export:",
                                                text_color='black', text_font=("Calibri Bold", -20))
        select_type_lbl.grid(row=0, column=0, pady=10, padx=60)

        export_btn = customtkinter.CTkButton(master=self.export_popup, text="Export", width=70, height=40,
                                            fg_color='gray', hover_color='green',
                                            compound="right", command=self.__export_handler)
        export_btn.grid(row=3, column=0, pady=10, padx=60)

        # pop the export window
        self.export_popup.mainloop()

    # handler for pressing the 'Export' button
    def __export_handler(self):
        report_choices = [self.txt_choice, self.pdf_choice]
        selected_checkboxes = []

        # only the choices that are "on"
        for choice in report_choices:
            if choice.get() == "on":
                selected_checkboxes.append(choice)

        # at least one report type has to be selected
        if not selected_checkboxes:
            # pop a fail message
            title = "Export Fail"
            message = "At least one report type has to be selceted.\nPlease try again."

            self.__message(title, message, "ok")
        else:
            self.export_popup.destroy()
            file_cnt = 0 # determines the message of the file saved popup

            # load image
            image_size = 35
            folder_image = self.__load_image("file_explore.png", image_size, image_size)

            # set the 'Open Report' label and the 'Show in File Explorer' button.
            self.open_report_lbl = customtkinter.CTkLabel(master=self.frame_left, text="Open Report",
                                                text_font=("Calibri Bold", -20)) # font name and size in px
            self.open_report_lbl.grid(row=5, column=0, pady=40, padx=10, sticky="n")

            self.folder_btn = customtkinter.CTkButton(master=self.frame_left, image=folder_image,
                                                        text="", width=30, height=30,
                                                        compound="left", command=self.__show_in_explorer_btn_handler)
            self.folder_btn.grid(row=5, column=1, pady=80, padx=0, sticky="n")


            # TODO: save an empty txt file
            if "on" == self.txt_choice.get():
                report_image = self.__load_image("open_raw_data.jpg", image_size, image_size)

                # set the 'Raw Data' label and the 'Open Raw Data' button
                self.txt_file_image_btn = customtkinter.CTkButton(master=self.frame_left, image=report_image,
                                                            text="", width=30, height=30,
                                                            compound="left", command=self.__show_report_btn_handler)

                sticky = "n" if len(selected_checkboxes) == 1 else "nw" # appers in the middle if this report is the only one
                self.txt_file_image_btn.grid(row=5, column=0, pady=80, padx=30, sticky=sticky)

                file_cnt = file_cnt + 1

            # TODO: save an empty pdf file
            if "on" == self.pdf_choice.get():
                report_image = self.__load_image("open_report.jpg", image_size, image_size)

                # set the 'Raw Data' label and the 'Open Report' button
                self.pdf_file_image_btn = customtkinter.CTkButton(master=self.frame_left, image=report_image,
                                                            text="", width=30, height=30,
                                                            compound="left", command=self.__show_report_btn_handler)

                pad_x = 30 if selected_checkboxes.index(self.pdf_choice) == 0 else 0 # appers in the middle if this report is the only one
                self.pdf_file_image_btn.grid(row=5, column=0, pady=80, padx=pad_x, sticky="n")

                file_cnt = file_cnt + 1

            # pop a success message
            title = "Export Success"
            sub_message = "The file was" if file_cnt == 1 else "The files were"
            message = f"{sub_message} downloaded successfully !\nTake a look :)"

            self.__message(title, message, "ok")

    # handler for pressing the 'Show Report' button
    def __show_report_btn_handler(self):
        #TODO: open the file on the computer
        self.destroy()

    # handler for pressing 'Show in File Explorer' button
    def __show_in_explorer_btn_handler(self):
        #TODO: show the downloaded file in the file explorer
        self.destroy()

    # set the check box's initial state
    def __toggle_state(self, var):
        var.set(1 - var.get())

    # load image as PhotoImage
    def __load_image(self, name, width, height):
        image = ImageTk.PhotoImage(Image.open
                (PATH + f"\\resources\\images\\{name}").resize((width, height), Image.Resampling.LANCZOS))

        # PhotoImage object is garbage-collected by Python,
        # so the image is cleared even if it’s being displayed by a Tkinter widget.
        # To avoid this, the program must keep an extra reference to the image object.
        lbl = Label(image=image)
        lbl.image = image

        return lbl.image

    # create new popup window and set it's properties
    def __new_popup(self, title, width, height):
        popup = Tk()

        popup.title(title)
        popup.geometry(f"{width}x{height}")

        return popup

    # pop a message window
    def __message(self, title, message, button):
        # set the window properties
        msg_popup = self.__new_popup(title, 440, 130)

        # set the label and button
        select_type_lbl = customtkinter.CTkLabel(master=msg_popup, text=message,
                                            text_color='black', text_font=("Calibri Bold", -20))
        select_type_lbl.grid(row=0, column=3, pady=10, padx=1)

        btn = customtkinter.CTkButton(master=msg_popup, text=button, width=70, height=40,
                                        fg_color='gray', hover_color='green', compound="right",
                                        command=msg_popup.destroy)
        btn.grid(row=1, column=3, padx=190, sticky='ns')

        # pop the message
        msg_popup.mainloop()

    # handler for changing the theme mode
    def __change_mode(self):
        if self.Theme_switch.get() == 1:
            customtkinter.set_appearance_mode("dark")
            self.Theme_switch.text = "Light Mode"
        else:
            customtkinter.set_appearance_mode("light")
            self.Theme_switch.text = "Dark Mode"

    # handler for closing the main frame
    def __on_closing(self, event=0):
        self.destroy()

    # run the main frame
    def start(self):
        self.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
