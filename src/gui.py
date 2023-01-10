import os
import glob
import logging
import itertools
import subprocess
from tkinter import *
import pandas as pd
import moviepy.editor as mpy
import yaml
from fpdf import FPDF
from PIL import Image
import customtkinter
from customtkinter import CTkCheckBox
from RangeSlider.RangeSlider import RangeSliderH
from .utils import *


customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-0blue"


logger = logging.getLogger(__name__)


with open(r'src\\configuration.yaml', 'r', encoding='utf-8') as c:
    config = yaml.safe_load(c)


class App():
    # sizes of the application
    width = config['gui']['app']['width']
    height = config['gui']['app']['height']

    def __init__(self):
        self.create_vmsm()

    def create_vmsm(self):
        try:
            pre_routine()  # create results folder
        except Exception:
            pass

        self.main_window = customtkinter.CTk()
        self.my_thread = ExecThread()
        self.main_window.title("Video Motion Synchrony Measurement")
        self.main_window.geometry(f"{App.width}x{App.height}")
        self.main_window.protocol("WM_DELETE_WINDOW", self.__on_closing)

        # ==========create_two_frames=========

        # configure grid layout (2x1)
        self.main_window.grid_columnconfigure(1, weight=1)
        self.main_window.grid_rowconfigure(0, weight=1)

        # set the left frame
        self.frame_left = customtkinter.CTkFrame(master=self.main_window, width=180, corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        # set the right frame
        self.frame_right = customtkinter.CTkFrame(master=self.main_window)
        self.frame_right.grid(row=0, column=3, sticky="nswe", padx=20, pady=20)

        # ============load_images=============

        image_size = 35
        self.video_explore_image = self.__load_image("video_explore.png", image_size, image_size)
        self.file_image = self.__load_image("file.png", image_size, image_size)
        self.refresh_image = self.__load_image("refresh.png", image_size, image_size)

        # ============ frame_left ============

        # configure grid layout (1x11)
        for i in range(0, 9):
            self.frame_left.grid_rowconfigure(i, minsize=10)  # empty row with minsize as spacing

        self.frame_left.grid_rowconfigure(9, weight=10)  # empty row as spacing

        # set the 'Video Loader' label and button
        self.video_lbl = customtkinter.CTkLabel(master=self.frame_left, text="Video Loader",
                                                font=("Calibri Bold", -20))  # font name and size in px
        self.video_lbl.grid(row=1, column=0, pady=10, padx=10, sticky="n")

        # set the theme switcher
        self.video_slider_max_val = 1
        self.Theme_switch = customtkinter.CTkSwitch(master=self.frame_left, text="Dark Mode",
                                                    command=self.__change_mode)
        self.Theme_switch.grid(row=10, column=0, pady=10, padx=20, sticky="w")
        self.Theme_switch.select()

        # set cutting video slider
        self.video_slider_bg = self.Theme_switch._bg_color[1]
        self.video_slider_line_s = '#4A4D50'
        self.video_slider_line = '#AAB0B5'
        self.add_video_slider()

        self.video_explore_btn = customtkinter.CTkButton(master=self.frame_left, image=self.video_explore_image,
                                                        text="", width=30, height=30,
                                                        compound="right", command=lambda:(self.my_thread.thread_excecuter(self.__video_btn_handler)))
        self.video_explore_btn.grid(row=2, column=0, pady=10, padx=20, sticky="n")

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
                                              font=("Calibri Bold", -20))
        self.roi_lbl.grid(row=0, column=2, columnspan=1, pady=20, padx=10, sticky="n")

        # 'Right hand' choice
        self.checkbox_var_RHand = IntVar(value=0)  # init checkbox
        self.right_hand_roi_choice = CTkCheckBox(master=self.frame_right, text="Right Hand",
                                                 command=self.__toggle_state(self.checkbox_var_RHand),
                                                 variable=self.checkbox_var_RHand, onvalue="on", offvalue="off")
        self.right_hand_roi_choice.grid(row=1, column=2, pady=10, padx=20, sticky="n")

        # 'Left hand' choice
        self.checkbox_var_LHand = IntVar(value=0)  # init checkbox
        self.left_hand_roi_choice = CTkCheckBox(master=self.frame_right, text="Left Hand  ",
                                                command=self.__toggle_state(self.checkbox_var_LHand),
                                                variable=self.checkbox_var_LHand, onvalue="on", offvalue="off")
        self.left_hand_roi_choice.grid(row=2, column=2, pady=10, padx=20, sticky="n")

        # 'Pose' choice
        self.checkbox_var_pose = IntVar(value=0)  # init checkbox
        self.pose_roi_choice = CTkCheckBox(master=self.frame_right, text="Pose          ",
                                           command=self.__toggle_state(self.checkbox_var_pose),
                                           variable=self.checkbox_var_pose, onvalue="on", offvalue="off")
        self.pose_roi_choice.grid(row=3, column=2, pady=10, padx=20, sticky="n")

        # set progressbar
        self.progressbar = customtkinter.CTkProgressBar(master=self.frame_right, mode="indeterminnate",
                                                        width=100)

        # set the 'Percentage Deviation' label and values slider
        self.perc_dev_lbl = customtkinter.CTkLabel(master=self.frame_right, text="Percentage Deviation:",
                                                   font=("Calibri Bold", -20))
        self.perc_dev_lbl.grid(row=5, column=0, columnspan=2, pady=10, sticky="ns")

        self.slider = customtkinter.CTkSlider(master=self.frame_right, command=self.__prec_slider_handler,
                                              from_=0, to=1)
        self.slider.grid(row=6, column=0, columnspan=1, pady=10, padx=125, sticky="ns")
        self.slider.set(0.0)

        self.video = None

        # set the 'Start Analysis' button
        self.start_btn = customtkinter.CTkButton(master=self.frame_right, height=45, width=130,
                                                fg_color='gray', hover_color='green', text="Start Analysis",
                                                corner_radius=15, font=("Calibri Bold", -18),
                                                command=(lambda:self.my_thread.thread_excecuter(self.__start_btn_handler)))
        self.start_btn.grid(row=8, column=0, columnspan=3, pady=10, padx=150, sticky="w")

        self.main_window.mainloop()

    def strat_progressbar(self):
        self.progressbar.grid(row=5, column=2, padx=100, pady=10)
        self.progressbar.start()

    def stop_progressbar(self):
        self.progressbar.grid_forget()
        self.progressbar.stop()

    # set the shown float values in format .2f for the 'Percentage Deviation' slider
    def __prec_slider_handler(self, val):
        self.perc_dev_lbl.configure(text=f"Percentage Deviation: {round(val, 2)}")

    # upload video and set the frame for it
    def __video_btn_handler(self):
        self.strat_progressbar()
        self.video_frame = customtkinter.CTkFrame(self.frame_info)
        self.video_frame.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)
        self.video = Video(self.video_frame)
        
        if not self.video.path:
            self.stop_progressbar()
            self.video = None
            return

        self.video_slider_max_val = int(mpy.VideoFileClip(self.video.path).duration)
        self.rs1.max_val = self.video_slider_max_val
        self.add_video_slider()
        self.stop_progressbar()

    def __start_btn_handler(self):
        rois_choices = [self.right_hand_roi_choice, self.left_hand_roi_choice, self.pose_roi_choice]
        selected_checkboxes = []

        # only the choices that are "on"
        for choice in rois_choices:
            if choice.get() == "on":
                selected_checkboxes.append(choice)

        # at least one ROI has to be selected
        if not selected_checkboxes:
            self.my_thread.thread_excecuter(self.__message("Start Fail", "At least one ROI has to be selceted.\nPlease try again.", "ok"))
            return

        # video wasn't uploaded
        if not self.video:
            self.my_thread.thread_excecuter(self.__message("Start Fail", "A video must be uploaded.\nPlease try again.", "ok"))
            return

        self.disable_widgets()
        self.strat_progressbar()
        time.sleep(1)

        try:
            if (self.hVar1.get() != 0) or (self.hVar2.get() != self.video_slider_max_val):
                avg_distance, self.lst_of_dist_dict_between_objects, self.lst_of_dist_dict_objectA, self.lst_of_dist_dict_objectB = get_synchronization(self.video.cut_video(self.hVar1.get(), self.hVar2.get()), selected_checkboxes, self.right_hand_roi_choice, self.left_hand_roi_choice,
                        self.pose_roi_choice)
            else:
                avg_distance, self.lst_of_dist_dict_between_objects, self.lst_of_dist_dict_objectA, self.lst_of_dist_dict_objectB = get_synchronization(self.video.path, selected_checkboxes, self.right_hand_roi_choice, self.left_hand_roi_choice,
                            self.pose_roi_choice)
        except Exception:
            self.my_thread.thread_excecuter(self.__message("Start Fail", "The VMSM system couldn't detect one or more of the objects!\nPlease try again.", "ok"))
        else:
            sync_rate, padx, color = get_synchronization_rate(avg_distance, self.slider.get())

            # set the 'Synchronization Rate' label
            self.sync_rate_lbl = customtkinter.CTkLabel(master=self.frame_right, font=("Calibri Bold", -20))
            self.sync_rate_lbl.grid(row=7, column=0, columnspan=2, pady=10, sticky="n")

            # configure the right values, according to the grade
            self.sync_rate_lbl.configure(text=sync_rate, text_color=color)
            self.sync_rate_lbl.grid(padx=padx)

            # add option to export the analysis result reports (set label and button)
            self.report_lbl = customtkinter.CTkLabel(master=self.frame_left, text="Reports Producer",
                                                    font=("Calibri Bold", -20))  # font name and size in px
            self.report_lbl.grid(row=3, column=0, pady=25, padx=10, sticky="n")

            self.file_image_btn = customtkinter.CTkButton(master=self.frame_left, image=self.file_image,
                                                        text="", width=30, height=30,
                                                        compound="right", command=(lambda:self.my_thread.thread_excecuter(self.__report_btn_handler)))
            self.file_image_btn.grid(row=4, column=0, pady=0, padx=5, sticky="n")

            # show to the user the synchronization rate
            self.my_thread.thread_excecuter(self.__message("Start Success", "The Interpersonal Synchrony Analysis\nCompleted Successfully !", "ok"))
        finally:
            self.stop_progressbar()
            self.reload_video_handler()

    # handler for pressing the 'Reports Producer' button
    def __report_btn_handler(self):
        self.strat_progressbar()
        # set the window properties
        self.export_popup = self.__new_popup("Export Results", config['gui']['export']['width'], config['gui']['export']['height'])

        # export txt raw data choice
        checkbox_var_txt = IntVar(value=0)  # init checkbox
        self.txt_choice = CTkCheckBox(master=self.export_popup, text="TXT Raw Data", text_color='black',
                                      command=self.__toggle_state(checkbox_var_txt),
                                      variable=checkbox_var_txt, onvalue="on", offvalue="off")
        self.txt_choice.grid(row=1, column=0, pady=10, padx=60)

        # export csv raw data choice
        checkbox_var_csv = IntVar(value=0)  # init checkbox
        self.csv_choice = CTkCheckBox(master=self.export_popup, text="CSV Raw Data", text_color='black',
                                      command=self.__toggle_state(checkbox_var_csv),
                                      variable=checkbox_var_csv, onvalue="on", offvalue="off")
        self.csv_choice.grid(row=2, column=0, pady=10, padx=60)

        # export pdf report choice
        checkbox_var_pdf = IntVar(value=0)  # init checkbox
        self.pdf_choice = CTkCheckBox(master=self.export_popup, text="PDF Report", text_color='black',
                                      command=self.__toggle_state(checkbox_var_pdf),
                                      variable=checkbox_var_pdf, onvalue="on", offvalue="off")
        self.pdf_choice.grid(row=3, column=0, pady=10, padx=60)

        # set export label and button
        select_type_lbl = customtkinter.CTkLabel(master=self.export_popup, text="Select File to Export:",
                                                 text_color='black', font=("Calibri Bold", -20))
        select_type_lbl.grid(row=0, column=0, pady=10, padx=60)

        export_btn = customtkinter.CTkButton(master=self.export_popup, text="Export", width=70, height=40,
                                             fg_color='gray', hover_color='green',
                                             compound="right", command=(lambda:self.my_thread.thread_excecuter(self.__export_handler)))
        export_btn.grid(row=4, column=0, pady=10, padx=60)

        # pop the export window
        self.stop_progressbar()
        self.export_popup.mainloop()

    # handler for pressing the 'Export' button
    def __export_handler(self):
        self.strat_progressbar()
        report_choices = [self.txt_choice, self.csv_choice, self.pdf_choice]
        selected_checkboxes = []

        # only the choices that are "on"
        for choice in report_choices:
            if choice.get() == "on":
                selected_checkboxes.append(choice._text)

        # at least one report type has to be selected
        if not selected_checkboxes:
            self.my_thread.thread_excecuter(self.__message("Export Fail", "At least one report type has to be selceted.\nPlease try again.", "ok"))
        else:
            file_cnt = 0  # determines the message of the file saved popup

            # open file dialog to get the diractory to sace the report
            self.export_path = StringVar()
            folder_path = filedialog.askdirectory()
            
            if not folder_path:
                return

            self.export_path.set(folder_path)
            rois = ['right_hand', 'left_hand', 'pose']

            for choice in selected_checkboxes:
                if choice == 'TXT Raw Data':
                    file_cnt = file_cnt + 1
                    logger.info("Creating TXT raw data reports")

                    for roi in rois:
                        for roi_dictA, roi_dictB in itertools.product(self.lst_of_dist_dict_objectA, self.lst_of_dist_dict_objectB):
                            if ((roi in roi_dictA.keys()) and (roi in roi_dictB.keys())):
                                df = pd.concat([roi_dictA[roi], roi_dictB[roi]], axis=1).reset_index()
                                df = df.rename(columns={'index': 'frame'})
                                file_path = f'{self.export_path.get()}\\raw_data_{roi}.txt'

                                with open(file_path, 'w', encoding='utf-8') as f:
                                    dfAsString = df.to_string(header=True, index=False)
                                    f.write(dfAsString)
                                    logger.info(f"'{file_path}' was saved successfully!")

                if choice == 'PDF Report':
                    # check if charts were created
                    # path = f"{config['video_paths']['res']}{*.png}"
                    imagelist = glob.glob(f"{config['video_paths']['res']}*.png")
                    if not imagelist:
                        return

                    file_cnt = file_cnt + 1
                    logger.info("Creating PDF report")
                    pdf = FPDF()

                    # imagelist is the list with all image filenames
                    for image in imagelist:
                        pdf.add_page()
                        pdf.image(image, 0, 0, 210, 297)  # A4

                    file_path = f"{self.export_path.get()}\\{config['gui']['export']['pdf']['name']}" 
                    pdf.output(file_path, "F")
                    logger.info(f"'{file_path}' was saved successfully!")

                if choice == 'CSV Raw Data':
                    file_cnt = file_cnt + 1
                    logger.info("Creating CSV raw data reports")

                    for roi in rois:
                        for roi_dictA, roi_dictB in itertools.product(self.lst_of_dist_dict_objectA, self.lst_of_dist_dict_objectB):
                            if ((roi in roi_dictA.keys()) and (roi in roi_dictB.keys())):
                                df = pd.concat([roi_dictA[roi], roi_dictB[roi]], axis=1).reset_index()
                                df = df.rename(columns={'index': 'Frame'})
                                file_path = f'{self.export_path.get()}\\raw_data_{roi}.csv'

                                df.to_csv(file_path, index=False)
                                logger.info(f"'{file_path}' was saved successfully!")

            # load image for 'Show in File Explorer' button
            image_size = 35
            folder_image = self.__load_image("file_explore.png", image_size, image_size)

            # set the 'Show in File Explorer' button and label.
            self.folder_lbl = customtkinter.CTkLabel(master=self.frame_left, text="Open in File Explorer",
                                                          font=("Calibri Bold", -20))  # font name and size in px
            self.folder_lbl.grid(row=5, column=0, pady=40, padx=10, sticky="n")

            self.folder_btn = customtkinter.CTkButton(master=self.frame_left, image=folder_image,
                                                      text="", width=30, height=30,
                                                      compound="left", command=(lambda:self.my_thread.thread_excecuter(self.__show_in_explorer_btn_handler)))
            self.folder_btn.grid(row=6, column=0, pady=0, padx=0, sticky="n")

            self.stop_progressbar()

            # pop a success message
            sub_message = "The file was" if file_cnt == 1 else "The files were"
            self.my_thread.thread_excecuter(self.__message("Export Success", f"{sub_message} downloaded successfully !\nTake a look :)", "ok"))
            self.export_popup.destroy()

    # handler for pressing 'Show in File Explorer' button
    def __show_in_explorer_btn_handler(self):
        subprocess.run(['explorer', os.path.realpath(self.export_path.get())], check=False)

    # set the check box's initial state
    def __toggle_state(self, var):
        var.set(1 - var.get())

    # load image as Image
    def __load_image(self, name, width, height):
        return customtkinter.CTkImage(light_image=Image.open(PATH + f"\\resources\\images\\{name}"),
                                  dark_image=Image.open(PATH + f"\\resources\\images\\{name}"),
                                  size=(width, height))

    # create new popup window and set it's properties
    def __new_popup(self, title, width, height):
        popup = Tk()

        popup.title(title)
        popup.geometry(f"{width}x{height}")

        return popup

    # pop a message window
    def __message(self, title, message, button):
        # set the window properties
        msg_popup = self.__new_popup(title, config['gui']['popup']['width'], config['gui']['popup']['height'])

        # set the label and button
        select_type_lbl = customtkinter.CTkLabel(master=msg_popup, text=message,
                                                 text_color='black', font=("Calibri Bold", -20))
        select_type_lbl.grid(row=0, column=3, pady=10, padx=1)

        btn = customtkinter.CTkButton(master=msg_popup, text=button, width=70, height=40,
                                      fg_color='gray', hover_color='green', compound="right",
                                      command=sys.exit)
        btn.grid(row=1, column=3, padx=190, sticky='ns')

        # pop the message
        msg_popup.mainloop()

    # handler for changing the theme mode
    def __change_mode(self):
        if self.Theme_switch.get() == 1:
            customtkinter.set_appearance_mode("dark")
            self.Theme_switch.text = "Light Mode"
            self.video_slider_bg = self.Theme_switch._bg_color[1]
            self.video_slider_line_s = '#4A4D50'
            self.video_slider_line = '#AAB0B5'
        else:
            customtkinter.set_appearance_mode("light")
            self.Theme_switch.text = "Dark Mode"
            self.video_slider_bg = self.Theme_switch._bg_color[0]
            self.video_slider_line_s = '#AAB0B5'
            self.video_slider_line = '#4A4D50'

        self.add_video_slider()

    # set cutting video slider
    def add_video_slider(self):
        self.hVar1 = tk.IntVar()
        self.hVar2 = tk.IntVar()
        self.rs1 = RangeSliderH(master=self.frame_right, variables=[self.hVar1, self.hVar2], Width=400, Height=65, padX=50, min_val=0, max_val=self.video_slider_max_val, show_value=True, line_s_color=self.video_slider_line_s,
                            bgColor=self.video_slider_bg, suffix=" sec", digit_precision=".0f", font_size=9, line_color=self.video_slider_line, font_family="Calibri-bold", bar_color_inner="#1F6AA5",
                            bar_color_outer="#1F6AA5", bar_radius=8, line_width=4)
        self.rs1.grid(row=4, column=0, padx=10, pady=10)

    def disable_widgets(self):
        self.start_btn.configure(state="disabled")
        self.pose_roi_choice.configure(state="disabled")
        self.left_hand_roi_choice.configure(state="disabled")
        self.right_hand_roi_choice.configure(state="disabled")
        self.video_explore_btn.grid_forget()
        self.video_lbl.configure(text="")
        self.slider.configure(state="disabled")

    def reload_video_handler(self):
        self.video_lbl.configure(text="Reload")
        self.refresh_btn = customtkinter.CTkButton(master=self.frame_left, image=self.refresh_image, text="", width=30, height=30,
                                                    compound="right", command=self.refresh)
        self.refresh_btn.grid(row=2, column=0, pady=10, padx=20, sticky="n")
    
    # handler for closing the main frame
    def __on_closing(self):
        self.main_window.destroy()

    def refresh(self):
        self.__on_closing()

        with subprocess.Popen([r'.\venv\Scripts\python.exe', 'main.py']) as p:  # start a new process to start new analysis
            sys.exit()
            # p.wait()  # Wait for the process to complete
        # TODO: check how the interpeter path works with executable file

        # check the return code of the process
        if p.returncode == 0:
            print('Script executed successfully')
        else:
            print('Script failed')
