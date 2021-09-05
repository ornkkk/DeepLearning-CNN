import numpy as np
import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import Add, Input, LeakyReLU, UpSampling2D, Concatenate
from tensorflow.keras import Model
import cv2
import time


def parse_cfg_file(cfg_file_path):
    """
    Takes a configuration file
    
    Returns a list of configs. Each configs describes a config in the neural
    network to be built. config is represented as a dictionary in the list
    
    """
    
    cfg = open(cfg_file_path, 'r')
    lines = cfg.read().splitlines() # store the lines in a list
    # getting rid of the empty lines, comments and unnecessary spaces
    lines = [x.strip() for x in lines if len(x) > 0]
    lines = [x.strip() for x in lines if x[0] != '#']

    config = dict()
    configs = list()

    for line in lines:
        if line.startswith('['):               # This marks the start of a new config
            if len(config) != 0:          # If config is not empty, implies it is storing values of previous config.
                configs.append(config)     # add it the configs list
                config = {}               # re-init the config
            config["type"] = line[1:-1].strip()     
        else:
            key, value = line.split("=") 
            config[key.strip()] = value.strip()
    configs.append(config)

    return configs


def create_yolo_model(input_layer, configs, num_classes):
    model_info = configs[0]     #Captures the information about the input and pre-processing
    current_layer = input_layer
    
    outputs = {}
    output_filters = []
    filters = []
    scale = 0
    output_pred = list()

    current_layer = current_layer/255.0
    
    for index, layer in enumerate(configs[1:]):
        #check the type of config
        #create a new layer for the config
        #append to module_list
        #--------------------------------------------------------------------------------------
        if (layer["type"] == "convolutional"):
            #Get the info about the layer
            activation = layer["activation"]
            try:
                batch_normalize = bool(layer["batch_normalize"])
                bias = False
            except:
                batch_normalize = False
                bias = True

            filters= int(layer["filters"])
            padding = int(layer["pad"])
            kernel_size = int(layer["size"])
            stride = int(layer["stride"])


            if padding:
                padd = (kernel_size) // 2
                current_layer = ZeroPadding2D(padding=padd)(current_layer)

            #Add the convolutional layer
            current_layer = Conv2D(filters,
                                    kernel_size,
                                    strides=stride,
                                    use_bias = bias,
                                    padding="valid",
                                    activation="linear",
                                    name=f"conv_{index}")(current_layer)

            #Add the Batch Normalization Layer
            if batch_normalize:
                current_layer = BatchNormalization(name=f"batch_norm_{index}")(current_layer)

            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLOv3
            if activation == "leaky":
                current_layer = LeakyReLU(alpha=0.001, name=f"leaky_{index}")(current_layer)

        #------------------------------------------------------------------------------------------

        #If it's an upsampling layer
        #We use Bilinear2dUpsampling
        elif (layer["type"] == "upsample"):
            stride = int(layer["stride"])
            current_layer = UpSampling2D(size=stride,
                                         interpolation="bilinear",
                                         name=f"upsample_{index}")(current_layer)
        
        #------------------------------------------------------------------------------------------
        
        #If it is a route layer
        elif (layer["type"] == "route"):
            
            layer["layers"] = [int(x) for x in layer["layers"].split(',')]
            
            start = layer["layers"][0]
            #start = start - index if start > 0 else start

            if len(layer["layers"]) > 1:
                #end = layer["layers"][1] - index
                end = layer["layers"][1]
                filters = output_filters[index + start] + output_filters[end-1]
                current_layer = tf.concat([outputs[index + start], outputs[end-1]],
                                           axis=-1,
                                           name=f"route_{index}")
            else:
                filters = output_filters[index + start]
                current_layer = tf.identity(outputs[index + start],
                                            name=f"route_{index}")

        #------------------------------------------------------------------------------------------

        #shortcut corresponds to skip connection
        elif layer["type"] == "shortcut":
            
            from_ = int(layer["from"])
            current_layer = Add(name=f"shortcut_{index}")([outputs[index - 1], outputs[index + from_]])

        #------------------------------------------------------------------------------------------
        
        # Yolo detection layer
        elif layer["type"] == "yolo":
            mask = [int(x) for x in layer["mask"].split(",")]

            anchors = [int(a) for a in layer["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            out_shape = current_layer.get_shape().as_list()
            input_dim = input_layer.shape[1]
            num_anchors = len(anchors)
            #batch_size = data.shape[0]
            
            stride = input_dim // out_shape[1]
            #grid_size = input_dim // stride
            grid_size = out_shape[1]
            bounding_box_attrs = 5 + num_classes
            anchors = [(x[0]/stride, x[1]/stride) for x in anchors]

            current_layer = tf.reshape(current_layer, [-1, grid_size, grid_size, num_anchors, bounding_box_attrs])

            box_centers = current_layer[:,:,:,:,0:2]
            box_centers = tf.sigmoid(box_centers)

            box_shape = current_layer[:, :, :, :, 2:4]

            confidence = current_layer[:,:, :, :, 4:5]
            confidence = tf.sigmoid(confidence)

            classes = current_layer[:,:,:,:,5:]
            classes = tf.sigmoid(classes)

            grid = tf.range(grid_size)
            cx, cy = tf.meshgrid(grid, grid)
            cxy = tf.Variable([cx, cy])
            cxy = tf.transpose(cxy, [1,2,0])
            cxy = tf.tile(cxy, [1,1,num_anchors])
            cxy = tf.reshape(cxy, [1, grid_size, grid_size, num_anchors, 2])
            cxy = tf.cast(cxy, dtype="float32")
            box_centers = box_centers + cxy
            #box_centers *= stride

            anchors = tf.cast(anchors, dtype="float32")

            anchors = tf.tile(anchors, [grid_size*grid_size, 1])
            anchors = tf.reshape(anchors, [1, grid_size, grid_size, num_anchors, 2])

            box_shape = tf.exp(box_shape)*anchors

            box = tf.concat([box_centers, box_shape, confidence], axis=-1)
            box = box*stride

            pred = tf.concat([box, classes], axis=-1)

            '''
            #current_layer = tf.reshape(current_layer, [-1, bounding_box_attrs*num_anchors, grid_size*grid_size])
            #current_layer = tf.reshape(current_layer, [-1, grid_size*grid_size, bounding_box_attrs*num_anchors])
            #print("current: ", current_layer.get_shape().as_list())
            #current_layer = tf.transpose(current_layer, perm=[0, 2, 1])
            #current_layer = tf.reshape(current_layer, [-1, grid_size*grid_size*num_anchors, bounding_box_attrs])
            #current_layer = tf.reshape(current_layer, [-1, num_anchors * out_shape[1] * out_shape[2], 5 + num_classes])

            #current_layer = tf.reshape(current_layer, [-1, grid_size, grid_size, num_anchors, bounding_box_attrs])

            anchors = [(x[0]/stride, x[1]/stride) for x in anchors]
            
            #Sigmoid the  centre_X, centre_Y. and object confidencce
            box_centers = current_layer[:,:,:,0:2]
            box_centers = tf.sigmoid(box_centers)
            box_shape = current_layer[:, :, 2:4]
            confidence = current_layer[:,:,4:5]
            confidence = tf.sigmoid(confidence)
            classes = current_layer[:,:,5:]
            classes = tf.sigmoid(classes)


            grid = np.arange(grid_size)
            cx,cy = tf.meshgrid(grid, grid)
            print("cx: ", cx.get_shape().as_list())

            cx = tf.cast(tf.reshape(cx, [-1,1]), dtype="float32")
            print("cx: ", cx.get_shape().as_list())
            cy = tf.cast(tf.reshape(cy, [-1,1]), dtype="float32")

            

            cxy = tf.Variable([cx, cy])
            print("cxy: ", cxy.get_shape().as_list())
            cxy = tf.tile(cxy, [1, num_anchors])
            print("cxy: ", cxy.get_shape().as_list())
            cxy = tf.reshape(cxy, [1, -1, 2])

            box_centers += cxy

            anchors = tf.cast(anchors, dtype="float32")

            anchors = tf.tile(anchors, [grid_size*grid_size, 1])
            anchors = tf.reshape(anchors, [1, -1, 2])

            #confidence = tf.reshape(confidence, [-1, 1])

            box_shape = tf.exp(box_shape)*anchors

            box = tf.concat([box_centers, box_shape, confidence], axis=-1)

            box *= stride

            pred = tf.concat([box, classes], axis=-1)
            
            '''
            '''
            current_layer = tf.reshape(current_layer, [-1, num_anchors * grid_size * grid_size, 5 + num_classes])

            box_centers = current_layer[:, :, 0:2]
            box_shapes = current_layer[:, :, 2:4]
            confidence = current_layer[:, :, 4:5]
            classes = current_layer[:, :, 5:num_classes + 5]

            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)

            #anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)
            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, num_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])
            strides = (input_dim // out_shape[1], \
                        input_dim // out_shape[2])
            box_centers = (box_centers + cxy) * strides

            pred = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
            '''
            pred = tf.reshape(pred, [-1, grid_size*grid_size*num_anchors, 5+num_classes])
            
            if scale:
                output_pred = tf.concat([output_pred, pred], axis=1)
            else:
                output_pred = pred
                scale = 1

        outputs[index] = current_layer
        output_filters.append(filters)
    
    #return (net_info, layer)
    model = Model(input_layer, output_pred)
    #model.summary()
    return model
    #return output_pred

def load_weights(model, configs, weightfile):
    # Open the weights file
    weights = open(weightfile, "rb")
    # The first 5 values are header information
    np.fromfile(weights, dtype=np.int32, count=5)
    
    for index, config in enumerate(configs[1:]):
        if (config["type"] == "convolutional"):
            conv_layer = model.get_layer(f"conv_{index}")
            print("layer: ",index+1,conv_layer)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]
            if "batch_normalize" in config:
                norm_layer = model.get_layer(f"batch_norm_{index}")
                print("layer: ",index+1,norm_layer)
                size = np.prod(norm_layer.get_weights()[0].shape)
                bn_weights = np.fromfile(weights, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            else:
                conv_bias = np.fromfile(weights, dtype=np.float32, count=filters)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(
                weights, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])
            if "batch_normalize" in config:
                norm_layer.set_weights(bn_weights)
                conv_layer.set_weights([conv_weights])
            else:
                conv_layer.set_weights([conv_weights, conv_bias])
                
    assert len(weights.read()) == 0, 'failed to read all data'
    weights.close()
    return


if __name__=="__main__":
    weightfile = "./weights/yolov3.weights"
    cfg_file_path = "./cfg/yolov3.cfg"
    test_file = "./dog-cycle-car.png"
    configs = parse_cfg_file(cfg_file_path)
    input_shape=(416, 416, 3)
    num_classes=80
    input_layer=Input(shape=input_shape)
    model = create_yolo_model(input_layer, configs, num_classes)
    load_weights(model, configs, weightfile)

    try:
        model.save_weights('./weights/yolov3_weights.tf')
        print('\nThe file \'yolov3_weights.tf\' has been saved successfully.')
    except IOError:
        print("Couldn't write the file \'yolov3_weights.tf\'.")

    from numba import cuda 
    device = cuda.get_current_device()
    device.reset()


