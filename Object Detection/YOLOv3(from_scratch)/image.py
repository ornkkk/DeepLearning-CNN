import tensorflow as tf
from utils import *
import numpy as np
from YOLOv3 import *

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_size = (416, 416,3)
num_classes = 80
class_name = './data/coco.names'
max_output_size = 40
max_output_size_per_class= 20
iou_threshold = 0.3
confidence_threshold = 0.3

cfgfile = './cfg/yolov3.cfg'
weightsfile = './weights/yolov3.weights'
img_path = "./data/images/test1.jpg"

def main():
    configs = parse_cfg_file(cfgfile)
    input_layer = tf.keras.layers.Input(shape=model_size)
    model = create_yolo_model(input_layer, configs, num_classes)
    #model.load_weights(weightfile)
    load_weights(model, configs, weightsfile)

    try:
        model.save_weights('./weights/yolov3_weights.tf')
        print('\nThe file \'yolov3_weights.tf\' has been saved successfully.')
    except IOError:
        print("Couldn't write the file \'yolov3_weights.tf\'.")

    
    class_names = load_class_names(class_name)

    image = cv2.imread(img_path)
    image = np.array(image)
    image = tf.expand_dims(image, 0)

    resized_frame = resize_image(image, (model_size[0],model_size[1]))

    pred = model.predict(resized_frame)

    boxes, scores, classes, nums = output_boxes( \
        pred, model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold)

    image = np.squeeze(image)
    img = draw_outputs(image, boxes, scores, classes, nums, class_names)

    win_name = 'Image detection'

    cv2.imshow(win_name, img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    #If you want to save the result, uncommnent the line below:
    cv2.imwrite('test2.jpg', img)


if __name__=="__main__":
    main()
