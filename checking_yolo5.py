import torch


class DetectObject:
    """
    Detect Object in Images.
    """
    def __init__(self):
        self.__model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        self.__results = None
        # other models to try Model
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

    def detect_objects(self, image):
        """
        Take Image and return detected objects information.
        Parameters:
            image:
        Returns:
            
        """
        self.__results = self.__model(image)
        # results.save()  # or .show()
        # print(results.imgs)
        # results.show()

        # print(results.xyxy[0])  # img1 predictions (tensor)
        # print(type(results.pandas().xyxy[0]))
        print('these are classes.')
        detected_classes = set(self.__results.pandas().xyxy[0]['name'])
        list_objects = list(self.__results.pandas().xyxy[0]['name'])
        classes_count = {}
        total = 0
        for obj in detected_classes:
            classes_count[obj] = list_objects.count(obj)
            total += list_objects.count(obj)

        classes_count['All'] = total

        print(classes_count)
        print('end classes')
        # return results.pandas().xyxy
        return classes_count
