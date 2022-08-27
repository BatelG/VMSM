import argparse
import pathlib
import time
from tkinter import *
import customtkinter
from customtkinter import CTkCheckBox
from resources.videoGenerator import *
import cv2
import imutils
import sys
import mediapipe as mp

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

        # set the right frame
        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=3, sticky="nswe", padx=20, pady=20)

        # ============load_images=============

        image_size = 35
        self.video_explore_image = self.__load_image("video_explore.png", image_size, image_size)
        self.file_image = self.__load_image("file.png", image_size, image_size)

        # ============ frame_left ============

        # configure grid layout (1x11)
        for i in range(0, 9):
            self.frame_left.grid_rowconfigure(i, minsize=10)  # empty row with minsize as spacing

        self.frame_left.grid_rowconfigure(9, weight=10)  # empty row as spacing

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

        # set the 'Percentage Deviation' label and values slider
        self.perc_dev_lbl = customtkinter.CTkLabel(master=self.frame_right, text="Percentage Deviation:",
                                                   text_font=("Calibri Bold", -20))
        self.perc_dev_lbl.grid(row=4, column=0, columnspan=2, pady=10, sticky="ns")

        self.slider = customtkinter.CTkSlider(master=self.frame_right, command=self.__prec_slider_handler, from_=0,
                                              to=1)
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
        # self.__devide_into_frames(self.video.path)

    def frame_to_video(self):
        import cv2
        import glob

        rep_path = os.path.dirname(pathlib.Path(__file__).parent.resolve())
        img_array = []
        for filename in glob.glob(rf'{rep_path}\\frames/*.jpg'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter('rep_path\project.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def devide_into_frames(self, path):

        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()

        count = 0
        lst_of_frames = []
        rep_path = os.path.dirname(pathlib.Path(__file__).parent.resolve())
        if not os.path.exists(f"{rep_path}\\frames"):
            os.mkdir(f"{rep_path}\\frames")
        if not os.path.exists(f"{rep_path}\\detected_frames"):
            os.mkdir(f"{rep_path}\\detected_frames")
        while success:
            image_path = f"{rep_path}\\frames\\frame{count}.jpg"
            detected_image_path = f"{rep_path}\\detected_frames\\frame{count}.jpg"
            cv2.imwrite(image_path, image)  # save frame as JPEG file
            success, image = vidcap.read()
            if success:
                lst_of_frames.append(image)
            print('Read a new frame: ', success)
            count += 1

    def detectByPathImage(self, path, output_path):
        image = cv2.imread(path)
        image = imutils.resize(image, width=min(800, image.shape[1]))
        result_image = self.detect(image)
        if output_path is not None:
            cv2.imwrite(output_path, result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect(self, frame):
        HOGCV = cv2.HOGDescriptor()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        bounding_box_cordinates, weights = HOGCV.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.03)

        person = 1
        for x, y, w, h in bounding_box_cordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            person += 1

        cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f'Total Persons : {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow('output', frame)
        return frame

    def last_try(self, path):
        import cv2
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_objectron = mp.solutions.objectron

        # # For static images:
        # IMAGE_FILES = []
        # IMAGE_FILES = arr
        # with mp_objectron.Objectron(static_image_mode=True,
        #                             max_num_objects=5,
        #                             min_detection_confidence=0.5,
        #                             model_name='Shoe') as objectron:
        #     for idx, file in enumerate(IMAGE_FILES):
        #         image = cv2.imread(file)
        #         # Convert the BGR image to RGB and process it with MediaPipe Objectron.
        #         results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #
        #         # Draw box landmarks.
        #         if not results.detected_objects:
        #             print(f'No box landmarks detected on {file}')
        #             continue
        #         print(f'Box landmarks of {file}:')
        #         annotated_image = image.copy()
        #         for detected_object in results.detected_objects:
        #             mp_drawing.draw_landmarks(
        #                 annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
        #             mp_drawing.draw_axis(annotated_image, detected_object.rotation,
        #                                  detected_object.translation)
        #             cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

        # For webcam input:
        cap = cv2.VideoCapture(path)
        with mp_objectron.Objectron(static_image_mode=False,
                                    max_num_objects=5,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.99,
                                    model_name='Shoe') as objectron:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = objectron.process(image)

                # Draw the box landmarks on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        mp_drawing.draw_landmarks(
                            image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                        mp_drawing.draw_axis(image, detected_object.rotation,
                                             detected_object.translation)
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

    def __Humen_detection(self, video_path):
        pass

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
                                                     text_font=("Calibri Bold", -20))  # font name and size in px
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
        checkbox_var_txt = IntVar(value=0)  # init checkbox
        self.txt_choice = CTkCheckBox(master=self.export_popup, text="Raw Data", text_color='black',
                                      command=self.__toggle_state(checkbox_var_txt),
                                      variable=checkbox_var_txt, onvalue="on", offvalue="off")
        self.txt_choice.grid(row=1, column=0, pady=10, padx=60)

        # export pdf report choice
        checkbox_var_pdf = IntVar(value=0)  # init checkbox
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
            file_cnt = 0  # determines the message of the file saved popup

            # load image
            image_size = 35
            folder_image = self.__load_image("file_explore.png", image_size, image_size)

            # set the 'Open Report' label and the 'Show in File Explorer' button.
            self.open_report_lbl = customtkinter.CTkLabel(master=self.frame_left, text="Open Report",
                                                          text_font=("Calibri Bold", -20))  # font name and size in px
            self.open_report_lbl.grid(row=5, column=0, pady=40, padx=10, sticky="n")

            self.folder_btn = customtkinter.CTkButton(master=self.frame_left, image=folder_image,
                                                      text="", width=30, height=30,
                                                      compound="left", command=self.__show_in_explorer_btn_handler)
            self.folder_btn.grid(row=6, column=0, pady=0, padx=0, sticky="n")

            # TODO: save an empty txt file
            if "on" == self.txt_choice.get():
                report_image = self.__load_image("open_raw_data.jpg", image_size, image_size)

                # set the 'Raw Data' label and the 'Open Raw Data' button
                self.txt_file_image_btn = customtkinter.CTkButton(master=self.frame_left, image=report_image,
                                                                  text="", width=30, height=30,
                                                                  compound="left",
                                                                  command=self.__show_report_btn_handler)

                sticky = "n" if len(
                    selected_checkboxes) == 1 else "nw"  # appers in the middle if this report is the only one
                self.txt_file_image_btn.grid(row=5, column=0, pady=90, padx=30, sticky=sticky)

                file_cnt = file_cnt + 1

            # TODO: save an empty pdf file
            if "on" == self.pdf_choice.get():
                report_image = self.__load_image("open_report.jpg", image_size, image_size)

                # set the 'Raw Data' label and the 'Open Report' button
                self.pdf_file_image_btn = customtkinter.CTkButton(master=self.frame_left, image=report_image,
                                                                  text="", width=30, height=30,
                                                                  compound="left",
                                                                  command=self.__show_report_btn_handler)

                sticky = "n" if len(
                    selected_checkboxes) == 1 else "ne"  # appers in the middle if this report is the only one
                self.pdf_file_image_btn.grid(row=5, column=0, pady=90, padx=30, sticky=sticky)

                file_cnt = file_cnt + 1

            # pop a success message
            title = "Export Success"
            sub_message = "The file was" if file_cnt == 1 else "The files were"
            message = f"{sub_message} downloaded successfully !\nTake a look :)"

            self.__message(title, message, "ok")

    # handler for pressing the 'Show Report' button
    def __show_report_btn_handler(self):
        # TODO: open the file on the computer
        self.destroy()

    # handler for pressing 'Show in File Explorer' button
    def __show_in_explorer_btn_handler(self):
        # TODO: show the downloaded file in the file explorer
        self.destroy()

    # set the check box's initial state
    def __toggle_state(self, var):
        var.set(1 - var.get())

    # load image as PhotoImage
    def __load_image(self, name, width, height):
        image = ImageTk.PhotoImage(Image.open
                                   (PATH + f"\\resources\\images\\{name}").resize((width, height),
                                                                                  Image.Resampling.LANCZOS))

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

    def rescaleFrame(self, frame, scale=0.5):  # rescaling to 50% by default
        # works for images, video and live video

        if frame is None:
            return

        width = int(frame.shape[1] * scale)  # must be an integer
        height = int(frame.shape[0] * scale)  # must be an integer

        # print(f"width: {width}")
        # print(f"height: {height}")
        dimensions = (width, height)

        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

    def our_media_pipe(self, video_path, output):

        # set up MediaPipe
        mp_drawing = mp.solutions.drawing_utils

        # set up holistic module
        mp_holistic = mp.solutions.holistic

        # ** large photos and videos are need to be rescaling and resizing ** #

        # Apply Styling
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

        # Import video from file
        cap = cv2.VideoCapture(video_path)

        # Initialize min/max default values
        maxSize = sys.maxsize
        minSize = -sys.maxsize - 1
        minX = maxSize
        maxX = minSize
        minY = maxSize
        maxY = minSize

        # Initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Counter for indicate the frame number
            cnt = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break

                # resize big video
                frame_resized = self.rescaleFrame(frame, scale=1)

                # Recolor Feed
                image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # Make Detections
                results = holistic.process(image)

                # Print coordinates
                print(results.pose_landmarks.landmark)

                # Loop on landmarks set for finding min,max of (x,y)
                for land_mark in results.pose_landmarks.landmark:
                    # for land_mark in results.pose_world_landmarks.landmark:
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
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                # Show the video feed
                cv2.imshow('Detected Video', image)

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

        self.crop_video(video_path, output, minX, minY, deltaX, deltaY)
        return minX, maxX

    def crop_video(self, video_path, output, x_0_ratio, y_0_ratio, deltaX, deltaY):
        # Open the video
        cap = cv2.VideoCapture(video_path)

        # Initialize frame counter
        cnt = 0

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
        out = cv2.VideoWriter(output, 0x7634706d, fps,
                              (w_frame_crop, h_frame_crop))

        while cap.isOpened():
            ret, frame = cap.read()
            frame = self.rescaleFrame(frame, scale=1)

            cnt += 1  # Counting frames

            # Avoid problems when video finish
            if ret == True:
                # Croping the frame
                crop_frame = frame[y_0:y_0 + h_frame_crop, x_0:x_0 + w_frame_crop]
                print(
                    f'making a cut frame #{cnt} - [x:x + w],[y:y + h] = [{x_0}-{x_0 + w_frame_crop}],[{y_0} - {y_0 + 2 * h_frame_crop}]')

                # Percentage
                xx = cnt * 100 / frames
                print(int(xx), '%\n')

                # I see the answer now. Here you save all the video
                out.write(crop_frame)

                # Just to see the video in real time
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


if __name__ == "__main__":
    # app = App()

    app = App()
    video_path = r"C:\final_project\VMSM\src\resources\videos\istockphoto-1382942438-640_adpp_is.mp4"

    # crop the first object from the video
    minX, maxX = app.our_media_pipe(video_path, r'C:\final_project\VMSM\src\resources\videos\1st_result.mp4')

    # find the other object
    output = r'C:\final_project\VMSM\src\resources\videos\middle_result.mp4'

    if maxX > 0.5:
        app.crop_video(video_path, output, maxX, 0,
                       1-maxX, 1)
    if minX > 0.5:
        app.crop_video(video_path, output, 0, 0,
                       minX, 1)

    # crop the second object from remain video
    app.our_media_pipe(output, r'C:\final_project\VMSM\src\resources\videos\2nd_result.mp4')
