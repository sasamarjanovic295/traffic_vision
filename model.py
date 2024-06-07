import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes


class vision_model():

    def __init__(self, signs_model_path, lights_model_path, device) -> None:
        self.signs_model= YOLO(signs_model_path)   
        self.lights_model = YOLO(lights_model_path)
        self.names, self.offset = self.__get_names(self.signs_model, self.lights_model)
        self.device = device


    def predict(self, input):
        results1 = self.signs_model.predict(input, device=self.device)
        results2 = self.lights_model.predict(input, device=self.device)

        for r1, r2 in zip(results1, results2):
            r1.names = self.names
            r1.boxes = self.__merge_boxes(r1.boxes, r2.boxes, self.offset)

        return results1
        

    def __merge_boxes(self, boxes1, boxes2, class_offset):
        boxes2_cls_offset = boxes2.data.clone()
        boxes2_cls_offset[:, -1] += class_offset
        merged_data = torch.cat((boxes1.data, boxes2_cls_offset), dim=0)
        merged_boxes = Boxes(merged_data, boxes1.orig_shape)
        return merged_boxes


    def __get_names(self, model1, model2):
        names = model1.names.copy()
        offset = len(model1.names)
        for key, value in model2.names.items():
            names[key + offset] = value
        return names, offset

