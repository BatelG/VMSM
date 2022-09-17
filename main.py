import sys
from src.utils import Video

def main():
    video_path = r'src\\resources\\videos\\istockphoto-1382942438-640_adpp_is.mp4'

    # crop the first object from the video
    minX, maxX = Video.our_media_pipe(video_path, r'src\\resources\\videos\\results\\1st_result.mp4')

    # find the other object
    output = r'src\\resources\\videos\\results\\middle_result.mp4'

    if maxX > 0.5:
        Video.crop_video(video_path, output, maxX, 0, 1-maxX, 1)
    if minX > 0.5:
        Video.crop_video(video_path, output, 0, 0, minX, 1)

    # crop the second object from remain video
    Video.our_media_pipe(output, r'src\\resources\\videos\\results\\2nd_result.mp4')

if __name__ == "__main__":
    sys.exit(main())
