from src.gui import App
from src.utils import *
from moviepy.editor import *


def main():
    # clip = VideoFileClip("videos/istockphoto-1382942438-640_adpp_is.mp4").cutout(0, 14)
    # clip.write_videofile("test.mp4")
    # clip = VideoFileClip("my_video.mp4")
    # print( clip.duration )
    app = App()
    app.start()
    # double_cropping()


def double_cropping():
    video_path = r"C:\final_project\VMSM\videos\WhatsApp Video 2022-07-19 at 20.38.35.mp4"

    # *** The following actions are happening after user press "Start" button ***

    # crop the first object from the video
    minX, maxX = Video.detect_object(video_path, r'src\\resources\\videos\\results\\1st_result.mp4')

    # find the other object
    output = r'src\\resources\\videos\\results\\middle_result.mp4'

    if maxX > 0.5:
        Video.crop_video(video_path, output, maxX, 0, 1 - maxX, 1)
    if minX > 0.5:
        Video.crop_video(video_path, output, 0, 0, minX, 1)

    # crop the second object from remain video
    Video.detect_object(output, r'src\\resources\\videos\\results\\2nd_result.mp4')


if __name__ == "__main__":
    main()
