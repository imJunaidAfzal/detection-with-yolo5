import os

import cv2
from DuplicateRemover import DuplicateRemover


class ImagesFromVideo:
    """
    Extract the Images from video and
    apply pre-processing algorithms
    on images
    """

    def __init__(self, logger, f_name=""):
        self.filename = f_name
        """
        Creating an object
        """
        self.logger = logger

    def readfile(self):
        try:
            """
            Read the video from specified path
            """
            cam = cv2.VideoCapture(str(self.filename))
            return cam
        except Exception as e:
            self.logger.error("File Not Found at this specific directory...")

    def detect_blur_images(self, image):
        """
        detect the blur image
        """
        threshold = 60
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        text = "Blurry"
        if fm >= threshold:
            text = "Not Blurry"
            data = [text, image]
            return data

        data = [text, image]
        return data

    def processing(self, cam):
        currentframe = 0
        try:
            while True:
                """
                reading from frames
                """
                cam.set(cv2.CAP_PROP_POS_MSEC, (currentframe * 1000))
                ret, frame = cam.read()
                if ret:
                    """
                    set the name of frame
                    """
                    try:
                        name = os.path.dirname(os.path.abspath(__file__)) + '/' +\
                               'frames/frame' + str(
                            currentframe) + '.jpg'

                        """
                        writing the extracted images
                        """
                        data = self.detect_blur_images(frame)
                        if data[0] == "Not Blurry":
                            cv2.imwrite(name, data[1])

                        """
                        increasing counter so that it will
                        show how many frames are created
                        """
                        currentframe += 1
                    except Exception as e:
                        self.logger.error(e)
                else:
                    break
        except Exception as e:
            self.logger.error(e)

    def detect_duplicate_images(self):
        """
        Remove Duplicates Images
        """
        dr = DuplicateRemover(os.path.dirname(os.path.abspath(__file__)) + '/' + "frames/", self.logger)
        dr.find_duplicates()
