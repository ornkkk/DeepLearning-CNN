import numpy as np
import tensorflow as tf
from tensorflow import Variable
import cv2

def write_results(prediction, confidence, num_classes, nms_conf=0.4):

    confidence_mask = tf.cast((prediction[:,:,4] > confidence),"float32")
    confidence_mask = tf.expand_dims(confidence_mask, axis=2)
    prediction = prediction*confidence_mask

    prediction = tf.Variable(prediction)
    box_corner = tf.Variable(prediction[:,:,:])
    box_corner[:,:,0].assign(prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1].assign(prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2].assign(prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3].assign(prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4].assign(box_corner[:,:,:4])

    batch_size = prediction.shape[0]
    write = False
    for index in range(batch_size):
        image_pred = prediction[index]

        #print(image_pred[:, 5:])
        max_confidence = tf.math.argmax(image_pred[:, 5:5+num_classes], axis=1)
        max_confidence_score = tf.math.reduce_max(image_pred[:, 5:5+num_classes], axis=1)

        max_confidence = tf.cast(max_confidence, dtype="float32")
        max_confidence = tf.expand_dims(max_confidence, axis=1)

        max_confidence_score = tf.cast(max_confidence_score, dtype="float32")
        max_confidence_score = tf.expand_dims(max_confidence_score, axis=1)

        sequence = (image_pred[:,:5], max_confidence, max_confidence_score)
        image_pred = tf.concat(sequence, axis=1)

        zero = tf.constant(0, dtype=tf.float32)
        non_zero_ind =  tf.where(tf.not_equal(image_pred[:,4], zero))

        print("OK!!!")
        sq = tf.make_ndarray(tf.squeeze(non_zero_ind))
        print("OK1!!!")
        print(image_pred[sq,:])
        image_pred_ = tf.reshape(image_pred[sq,:], [-1,7])

        #try:
            #image_pred_ = tf.reshape(image_pred[tf.squeeze(non_zero_ind),:], [-1,7])
        #except:
            #print("OK1!!!")
            #continue
        
        if image_pred_.shape[0] == 0:
            print("OK2!!!")
            continue 

        img_classes = unique(image_pred_[:, -1])

        for cls in img_classes:
            #print(cls)
            #get the detections with one particular class
            cls_mask = tf.cast(image_pred_*(image_pred_[:,-1] == cls), "float32")
            cls_mask = tf.expand_dims(cls_mask, axis=1)

            class_mask_ind = tf.where(tf.not_equal(cls_mask[:,-2], zero))
            class_mask_ind = tf.squeeze(class_mask_ind)

            image_pred_class = tf.reshape(image_pred_[class_mask_ind], [-1,7])

            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = tf.sort(image_pred_class[:,4], direction="DESCENDING")[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.shape[0]   #Number of detections

            for i in range(idx):
                try:
                    ious = bbox_iou(tf.expand_dims(image_pred_class[i], axis=0), image_pred_class[i+1:])
                except ValueError:
                    break
                
                except IndexError:
                    break
                
                #Zero out all the detections that have IoU > treshhold
                iou_mask = tf.cast(ious < nms_conf, "float32")
                iou_mask = tf.expand_dims(iou_mask, axis=1)
                image_pred_class[i+1:] *= iou_mask

                #Remove the non-zero entries
                non_zero_ind = tf.where(tf.not_equal(image_pred_class[:,4], zero))
                non_zero_ind = tf.squeeze(non_zero_ind)
                image_pred_class = tf.reshape(image_pred_class[non_zero_ind], [-1,7])
            
            batch_ind = tf.fill([image_pred_class.shape[0], 1], value=index)   
            #Repeat the batch_id for as many detections of the class cls in the image
            seq = [batch_ind, image_pred_class]

            if not write:
                output = tf.concat(seq,1)
                write = True
            else:
                out = tf.concat(seq,1)
                output = tf.concat([output,out])

    #print("output: ", output)

    try:
        return output
    except:
        return 0




def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  tf.max(b1_x1, b2_x1)
    inter_rect_y1 =  tf.max(b1_y1, b2_y1)
    inter_rect_x2 =  tf.min(b1_x2, b2_x2)
    inter_rect_y2 =  tf.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = tf.clip_by_value(inter_rect_x2 - inter_rect_x1 + 1, clip_value_min=0) * tf.clip_by_value(inter_rect_y2 - inter_rect_y1 + 1, clip_value_min=0)
 
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou



def unique(tensor):
    tensor_np = tensor.numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = tf.convert_to_tensor(unique_np)
    
    tensor_res = tf.identity(unique_tensor)
    return tensor_res


def load_classes(file_path):
    fp = open(file_path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1]
    img = tf.convert_to_tensor(img, dtype="float32") / 255.
    img = tf.expand_dims(img, axis=0)
    return img

def tf_index_select(input_, dim, indices):
    """
    input_(tensor): input tensor
    dim(int): dimension
    indices(list): selected indices list
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape)-1
    shape[dim] = 1

    tmp = []
    for idx in indices:
        begin = [0]*len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)

    return res

