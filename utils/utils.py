from src.predict_car import predict as predict_car 
from src.predict_color import predict as predict_color
from src.vehicle_detection import predict as detect_vehicle

def load_class_names(filename):
    with open(filename, 'r', encoding='utf8') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def predict_all_feature(image, predictor, car_model, color_model, car_labels, color_labels, vehicle_classes):
    list_boxes, list_scores, list_classes = detect_vehicle(image, predictor, vehicle_classes)

    cnt = 0
    for box in list_boxes:
        vehicle_box = image[box[0]: (box[0]+box[2]), box[1]: (box[1]+box[3]), :]
        vehicle_color = predict_color(vehicle_box, color_model, color_labels)
        vehicle_name = predict_car(vehicle_box, car_model, car_labels)

        return box, list_scores[cnt], list_classes[cnt], vehicle_color, vehicle_name
