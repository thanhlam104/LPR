import os
import cv2


def video_interpolate(images_path, video_path, nframe=32):
    images = [img for img in os.listdir(images_path) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(images_path, images[0]))
    print(frame.shape)
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_path, 0, nframe, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(images_path, image)))


    cv2.destroyAllWindows()
    video.release()

img_path = 'dataset/plates/train/'
video_path = 'video_1.avi'
video_interpolate(img_path, video_path)