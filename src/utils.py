import os
import logging
import shutil
import pathlib
from threading import Thread
import time
import sys
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy
import pandas
from tkvideo import tkvideo
import cv2
import mediapipe as mp
import yaml
import moviepy.editor as mpy
import moviepy


with open(r'src\\configuration.yaml', 'r', encoding='utf-8') as c:
    config = yaml.safe_load(c)


PATH = os.path.dirname(os.path.realpath(__file__))
RES_PATH = config['video_paths']['res']
logging.basicConfig(filename=config['logger']['name'], filemode='w', level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

def pre_routine():
    logger.info("*****pre_routine*****")

    if os.path.exists(RES_PATH):
        shutil.rmtree(RES_PATH)
        os.mkdir(RES_PATH)
    else:
        os.mkdir(RES_PATH)

def post_routine():
    logger.info("*****post_routine*****")

    if os.path.exists(RES_PATH) and os.path.isdir(RES_PATH):
        shutil.rmtree(RES_PATH)

# TODO check if GUI workes, then change it
# # create new popup window and set it's properties
# class Popup():
#     def __init__(self, title, width, height):
#         popup = Tk()

#         popup.title(title)
#         popup.geometry(f"{width}x{height}")

def get_c_mass(roi, lst_res):
    logger.info("*****get_c_mass*****")

    c_mass_x, c_mass_y, x_sum, y_sum, cnt = 0, 0, 0, 0, 0
    lst_vals = []
    try:
        for result in lst_res:  # result is unused variable but necessary one!
            if eval(f'result.{roi}_landmarks') is None:
                lst_vals.append([0, 0])
            else:
                for land_mark in eval(f'result.{roi}_landmarks.landmark'):
                    if land_mark.x < 0 or land_mark.y < 0:
                        continue
                    x_sum += land_mark.x
                    y_sum += land_mark.y
                    cnt += 1
                x_sum /= cnt
                y_sum /= cnt
                c_mass_x = x_sum
                c_mass_y = y_sum
                x_sum, y_sum, cnt = 0, 0, 0
                lst_vals.append([c_mass_x, c_mass_y])
    except Exception as e:
        raise e
    return pandas.DataFrame(lst_vals, columns=['c_mass_x', 'c_mass_y'])


# calculate Euclidean distance between following pairs of frames within one object
def _create_distance_chart(lst_df, object_str):
    logger.info("*****_create_distance_chart*****")

    lst_of_dist_dict = []
    for _, dict_of_df in enumerate(lst_df):
        for key in dict_of_df.keys():
            ax_df = dict_of_df[key]['c_mass_x']
            ay_df = dict_of_df[key]['c_mass_y']

            number_of_illegal_frames = 0
            data_flag = False
            lst_vals = []

            for idx, _ in enumerate(ax_df):
                try:
                    if idx == len(ax_df) - 1:
                        break
                    # only frames where both objects were detected are counted
                    if (ax_df[idx] == 0 or ay_df[idx] == 0) and (ax_df[idx+1] == 0 or ay_df[idx+1] == 0):
                        number_of_illegal_frames += 1
                        if number_of_illegal_frames == config['video_processing']['allow']['following_frames_treshold']:
                            # TODO inform the user in case of missing information (?)
                            logger.info(f'there is not enough data at roi: {key} object - {object_str}')
                            data_flag = True
                            break
                        continue

                    ax = ax_df[idx]
                    ay = ay_df[idx]

                    bx = ax_df[idx + 1]
                    by = ay_df[idx + 1]

                    point_a = numpy.array((ax, ay))
                    point_b = numpy.array((bx, by))

                    dist = numpy.linalg.norm(point_a - point_b)
                    lst_vals.append([dist])
                except Exception:
                    continue

            if not data_flag:
                lst_of_dist_dict.append({key: pandas.DataFrame(lst_vals, columns=[object_str])})

        for dist_dict in lst_of_dist_dict:
            for key in dist_dict.keys():
                df = dist_dict[key].reset_index()
                df = df.rename(columns={'index': 'frame'})
                try:
                    df.plot(x='frame', y=object_str, kind='line')
                    plt.title(f'distance of roi {key} between following frames - {object_str}')
                    plt.savefig(f'{RES_PATH}distance chart between following frames of roi {key} - {object_str}')
                except Exception:
                    # TODO inform the user in case of missing information (?)
                    logger.info(f'there is not enough data at roi: {key} object - {object_str}')

    return lst_of_dist_dict

# calculate Euclidean distance between two objects
def create_distance_chart(lst_df, lst_df2):
    logger.info("*****create_distance_chart*****")

    if lst_df2.__class__ is str:
        return _create_distance_chart(lst_df, lst_df2)

    illegal_frames = []
    lst_of_dist_dict = []
    for index, dic_of_df in enumerate(lst_df):
        for key in dic_of_df.keys():
            ax_df = dic_of_df[key]['c_mass_x']
            ay_df = dic_of_df[key]['c_mass_y']

            bx_df = lst_df2[index][key]['c_mass_x']
            by_df = lst_df2[index][key]['c_mass_y']

            number_of_illegal_frames = 0
            lst_vals = []

            for idx, _ in enumerate(ax_df):
                try:
                    # only frames where both objects were detected are counted
                    if ax_df[idx] == 0 or bx_df[idx] == 0 or ay_df[idx] == 0 or by_df[idx] == 0:
                        number_of_illegal_frames += 1
                        continue
                    ax = ax_df[idx]
                    ay = ay_df[idx]

                    bx = bx_df[idx]
                    by = by_df[idx]

                    point_a = numpy.array((ax, ay))
                    point_b = numpy.array((bx, by))

                    dist = numpy.linalg.norm(point_a - point_b)
                    lst_vals.append([dist])
                except Exception:
                    continue

            lst_of_dist_dict.append({key: pandas.DataFrame(lst_vals, columns=['distance'])})
            illegal_frames.append({key: number_of_illegal_frames})

        for dist_dict in lst_of_dist_dict:
            for key in dist_dict.keys():
                df = dist_dict[key].reset_index()
                df = df.rename(columns={'index': 'frame'})
                try:
                    df.plot(x='frame', y='distance', kind='line')
                    plt.title(f'distance of roi {key} between the two objects')
                    plt.savefig(f'{RES_PATH}distance chart between the objects of roi {key}')
                except Exception:
                    # TODO inform the user in case of missing information (?)
                    logger.info(f'there is not enough data at roi: {key} between the objects')

    return lst_of_dist_dict


def get_df(selected_checkboxes, right_hand_roi_choice, left_hand_roi_choice, pose_roi_choice, path):
    logger.info("*****get_df*****")

    mp_holistic = mp.solutions.holistic  # set up holistic module
    cap = cv2.VideoCapture(path)  # import video from file
    # initiate holistic model
    lst_res = []
    lst_df = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            _, frame = cap.read()

            # recolor feed
            if frame is None:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # resized frame

            # make detections
            results = holistic.process(image)
            lst_res.append(results)

    if len(lst_res) == 0:
        raise Exception('MediaPipe is not working properly')

    if right_hand_roi_choice in selected_checkboxes:
        # 2. Right hand
        lst_df.append({'right_hand': get_c_mass('right_hand', lst_res)})

    if left_hand_roi_choice in selected_checkboxes:
        # 3. Left Hand
        lst_df.append({'left_hand': get_c_mass('left_hand', lst_res)})

    if pose_roi_choice in selected_checkboxes:
        # 4. Pose Detections
        lst_df.append({'pose': get_c_mass('pose', lst_res)})

    return lst_df

def get_synchronization(video_path, selected_checkboxes, right_hand_roi_choice, left_hand_roi_choice,
                pose_roi_choice):
    logger.info("*****get_synchronization*****")

    # *** The following actions are happening after user press "Start" button ***

    # crop the first object from the video
    minX, maxX = Video.detect_object(video_path, config['video_paths']['first_object'])

    # find the other object

    # if maxX > 0.5:
    #     Video.crop_video(video_path, config['video_paths']['mid_res'], maxX, 0, 1 - maxX, 1) # right
    # if minX > 0.5:
    #     Video.crop_video(video_path, config['video_paths']['mid_res'], 0, 0, minX, 1) # left

    if minX > 0.25 and minX > 0.75:
        Video.crop_video(video_path, config['video_paths']['mid_res'], 0, 0, minX, 1) # left
    elif (maxX < 0.25 and minX < 0.75) or (maxX > 0.25 and minX < 0.75):
        Video.crop_video(video_path, config['video_paths']['mid_res'], maxX, 0, 1 - maxX, 1) # right
    #####
    elif minX > 0.33 and minX > 0.77:
        Video.crop_video(video_path, config['video_paths']['mid_res'], 0, 0, minX, 1) # left
    elif (maxX < 0.33 and minX < 0.77) or (maxX > 0.33 and minX < 0.77):
        Video.crop_video(video_path, config['video_paths']['mid_res'], maxX, 0, 1 - maxX, 1) # right
    #####
    elif maxX > 0.5 and minX > 0.5:
        Video.crop_video(video_path, config['video_paths']['mid_res'], 0, 0, minX, 1) # left
    elif (maxX < 0.5 and minX < 0.5) or (maxX > 0.5 and minX < 0.5):
        Video.crop_video(video_path, config['video_paths']['mid_res'], maxX, 0, 1 - maxX, 1) # right

    # crop the second object from remain video
    Video.detect_object(config['video_paths']['mid_res'], config['video_paths']['second_object'])

    # getting the dimension of the videos
    vid = cv2.VideoCapture(config['video_paths']['first_object'])
    height1 = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width1 = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    vid = cv2.VideoCapture(config['video_paths']['second_object'])
    height2 = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width2 = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    logger.info("before transformations:")
    logger.info(f'first video - ({width1},{height1})\nsecond video - ({width2},{height2})\n')

    logger.info('start to do transformations...')

    # compare between the 2 object dimension, the bigger one pass an resize + scaling transformations
    if width1 + height1 >= width2 + height2:
        new_h = int(height2)
        new_w = int(width2)
        new_path = config['video_paths']['first_object2']
        path = config['video_paths']['first_object']

        clip = mpy.VideoFileClip(path)
        clip_resized = clip.resize((new_w, new_h))
        clip_resized_mirrored = moviepy.video.fx.all.mirror_x(clip_resized, apply_to='mask')
        clip_resized_mirrored.write_videofile(new_path)

        vid1 = cv2.VideoCapture(config['video_paths']['second_object'])
        height1 = vid1.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width1 = vid1.get(cv2.CAP_PROP_FRAME_WIDTH)

    else:
        new_h = int(height1)
        new_w = int(width1)
        path = config['video_paths']['second_object']
        new_path = config['video_paths']['second_object2']

        clip = mpy.VideoFileClip(path)
        clip_resized = clip.resize((new_w, new_h))
        clip_resized_mirrored = moviepy.video.fx.all.mirror_x(clip_resized, apply_to='mask')
        clip_resized_mirrored.write_videofile(new_path)

        vid1 = cv2.VideoCapture(config['video_paths']['first_object'])
        height1 = vid1.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width1 = vid1.get(cv2.CAP_PROP_FRAME_WIDTH)

    vid2 = cv2.VideoCapture(new_path)
    height2 = vid2.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width2 = vid2.get(cv2.CAP_PROP_FRAME_WIDTH)

    logger.info("after transformations:")
    logger.info(f'first video - ({width1},{height1})\nsecond video - ({width2},{height2})\n')

    ###############################################################

    lst_df = get_df(selected_checkboxes, right_hand_roi_choice, left_hand_roi_choice,
                    pose_roi_choice, new_path)

    lst_df2 = get_df(selected_checkboxes, right_hand_roi_choice, left_hand_roi_choice,
                        pose_roi_choice, config['video_paths']['first_object'])

    lst_of_dist_dict_between_objects = create_distance_chart(lst_df, lst_df2)
    lst_of_dist_dict_objectA = create_distance_chart(lst_df, 'Object A')
    lst_of_dist_dict_objectB = create_distance_chart(lst_df2, 'Object B')

    lst_avg = []

    for dictionary in lst_of_dist_dict_between_objects:
        # calculate the avarage distance for each roi
        for roi in dictionary.keys():
            lst_avg.append(dictionary[roi]['distance'].mean())

    rate = (sum(lst_avg)/len(lst_avg)) if len(lst_avg) > 0 else 1

    return rate, lst_of_dist_dict_between_objects, lst_of_dist_dict_objectA, lst_of_dist_dict_objectB  # avarage distance of all rois


# get the synchronization rate and label configuration values, according to the grade
def get_synchronization_rate(nd, pd):
    logger.info("*****get_synchronization_rate*****")

    if nd >= pd:
        grade = max(1-nd, 1-nd+pd)
    else:
        grade = min(1-nd, 1-nd+pd)

    # the second value in 'range' method doesn't count
    if 0 <= grade <= 0.33:
        return "Weak Synchronization", 100, '#FF3200'
    if 0.34 <= grade <= 0.66:
        return "Medium Synchronization", 92, '#FF9B00'
    if 0.67 <= grade <= 0.95:
        return "Strong Synchronization", 88, '#C2C000'
    return "Perfect Synchronization", 88, '#359C25'  # 0.96 <= grade <= 1


class ExecThread:
    def thread_excecuter(self, function, *args):
        Thread(target=function, args=args).start()


class Video:
    def __init__(self, frame):
        root = frame
        my_label = tk.Label(root)
        my_label.pack()

        self.path = self.video_loader_btn_handler()

        while not self.path:
            self.path = self.video_loader_btn_handler()

        self.player = tkvideo(self.path, my_label, loop=1, size=(350, 250))
        self.player.play()

    def video_loader_btn_handler(self):  # TODO: after debug is done, change initial dir to c folder (?)
        filename_path = filedialog.askopenfilename(initialdir=PATH,
            title="Select a File",
            filetypes=(("MP4 files", "*.mp4"), ("MOV files", "*.mov"), ("AVI files", "*.avi")))

        if filename_path in ["", " "]:
            return False

        if pathlib.Path(filename_path).suffix == ".mp4":
            mp4_video_path = filename_path
        else:  # convert to mp4
            logger.info(f"Convert '{filename_path}' into mp4 file")
            mp4_video_path = config['video_paths']['mp4_video']
            cap = cv2.VideoCapture(filename_path) # open the video

            # some characteristics from the original video
            w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps, _ = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            out = cv2.VideoWriter(mp4_video_path, 0x7634706d, fps, (w_frame, h_frame))

            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break

            # release the VideoCapture and VideoWriter objects
            cap.release()
            out.release()

        logger.info(f"Loaded video path: {mp4_video_path}")

        return str(mp4_video_path)

    def cut_video(self, start, end):
        cap = cv2.VideoCapture(self.path) # open the video

        # some characteristics from the original video
        w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, _ = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        start_frame = int(start*fps)
        end_frame = int(end*fps)
        cutted_video_path = config['video_paths']['cutted']

        # define the codec and create VideoWriter object
        out = cv2.VideoWriter(cutted_video_path, 0x7634706d, fps, (w_frame, h_frame))

        # read the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                # write the frame to the output file if it is within the desired range
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= start_frame and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
                    out.write(frame)
                # break the loop if we have reached the end of the subvideo
                if cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
                    break
            else:
                break

        # release the VideoCapture and VideoWriter objects
        cap.release()
        out.release()

        return cutted_video_path

    @staticmethod
    def rescaleFrame(frame, scale=0.5):
        if frame is None:
            return

        width = int(frame.shape[1] * scale)  # must be an integer
        height = int(frame.shape[0] * scale)  # must be an integer
        dimensions = (width, height)

        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

    @staticmethod
    def detect_object(video_path, output):
        mp_drawing = mp.solutions.drawing_utils  # set up MediaPipe
        mp_holistic = mp.solutions.holistic  # set up holistic module

        # ** large photos and videos are need to be rescaling and resizing ** #

        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)  # apply styling

        cap = cv2.VideoCapture(video_path)  # import video from file

        # initialize min/max default values
        maxSize = sys.maxsize
        minSize = -sys.maxsize - 1
        minX = maxSize
        maxX = minSize
        minY = maxSize
        maxY = minSize

        # initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            cnt = 0  # indicates the frame number

            while cap.isOpened():
                _, frame = cap.read()

                if frame is None:
                    break

                frame_resized = Video.rescaleFrame(frame, scale=1)  # resize big video
                image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # recolor feed
                results = holistic.process(image)  # make detections

                if not results.pose_landmarks:
                    break
                logger.info(results.pose_landmarks.landmark)  # write coordinates to log

                # loop on landmarks set for finding min,max of (x,y)
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

                cv2.imshow('Detected Video', image)  # show the video feed

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                cnt += 1

        logger.info("Result:")
        logger.info(f"minX = {minX}")
        logger.info(f"maxX = {maxX}")
        logger.info("##############")
        logger.info(f"minY = {minY}")
        logger.info(f"maxY = {maxY}")

        logger.info("############################################")

        logger.info("Delta calculations..")
        deltaX = maxX - minX
        deltaY = maxY - minY

        logger.info(f'(x_0,y_0) = ({minX},{minY})')
        logger.info(f"deltaX = {deltaX}")
        logger.info(f"deltaY = {deltaY}")
        logger.info(f'(minX+deltaX,minY+deltaY) = ({minX}+{deltaX},{minY}+{deltaY}) = ({minX + deltaX},{minY + deltaY})')

        cap.release()
        cv2.destroyAllWindows()

        logger.info("crop video file...")

        Video.crop_video(video_path, output, minX, minY, deltaX, deltaY)
        return minX, maxX

    @staticmethod
    def crop_video(video_path, output, x_0_ratio, y_0_ratio, deltaX, deltaY):
        cap = cv2.VideoCapture(video_path)  # open the video
        cnt = 0  # initialize frame counter

        # some characteristics from the original video
        w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # calculate the original (x_0,y_0) coordinates of the frame
        x_0 = int(x_0_ratio * w_frame)
        y_0 = int(y_0_ratio * h_frame)

        # width and height if the cropped frame
        w_frame_crop = int(deltaX * w_frame)
        h_frame_crop = int(deltaY * h_frame)
        out = cv2.VideoWriter(output, 0x7634706d, fps, (w_frame_crop, h_frame_crop))

        while cap.isOpened():
            ret, frame = cap.read()
            frame = Video.rescaleFrame(frame, scale=1)
            cnt += 1  # counting frames

            # avoid problems when video finish
            if ret:
                crop_frame = frame[y_0:y_0 + h_frame_crop, x_0:x_0 + w_frame_crop]

                logger.info(
                    f'making a cut frame #{cnt} - [x:x + w],[y:y + h] = [{x_0}-{x_0 + w_frame_crop}],[{y_0} - {y_0 + 2 * h_frame_crop}]')

                # show progress in percentage
                xx = cnt * 100 / frames
                logger.info(int(xx), '%\n')

                out.write(crop_frame)  # save the new video

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
