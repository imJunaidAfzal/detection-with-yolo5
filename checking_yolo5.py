import glob
import logging
import os

import Images_from_Video
import torch


class DetectObject:
    """
    Detect Object in Images.
    """

    def __init__(self):
        self.__model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        self.__results = None
        """
                Create and configure logger
                """
        logging.basicConfig(filename="newfile.log",
                            format='%(asctime)s %(message)s',
                            filemode='w')
        """
        Creating an object
        """
        self.logger = logging.getLogger()
        """
        Setting the threshold of logger to DEBUG
        """
        self.logger.setLevel(logging.DEBUG)

    def dir_handling(self):
        """
        delete all images in these directories
        else create new directory
        """
        path = os.path.dirname(os.path.abspath(__file__)) + '/' + "detected_images/frames/"
        if os.path.isdir(path):
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
        else:
            os.makedirs(path)

        dir = os.path.dirname(os.path.abspath(__file__)) + '/' +"frames/"
        if os.path.isdir(dir):
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
        else:
            os.mkdir(dir)

    def video_to_images(self, f_name):
        self.dir_handling()
        """
        :param video filename:
        :return processed images:
        """
        img = Images_from_Video.ImagesFromVideo(self.logger,
                                                f_name)
        cam = img.readfile()
        img.processing(cam)
        img.detect_duplicate_images()
        images = glob.glob(os.path.dirname(os.path.abspath(__file__)) + '/'  + "frames/*.jpg")
        return images

    def detect_objects(self, image):
        """
        Take Image and return detected objects information.
        Parameters:
            image:
        Returns:

        """
        self.__results = self.__model(image)
        total = 0
        classes_count = {}
        for frame in self.__results.pandas().xyxy:
            detected_classes = set(frame['name'])
            list_objects = list(frame['name'])

            for obj in detected_classes:
                if obj in classes_count:
                    classes_count[obj] += list_objects.count(obj)
                else:
                    classes_count[obj] = list_objects.count(obj)
                total += list_objects.count(obj)

        classes_count['All'] = total

        return classes_count
