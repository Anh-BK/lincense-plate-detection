import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time 


import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/license_plate.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'class_names.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "text_size" : 3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],#[image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // 600
        fontScale=1
        ObjectsList = []
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            #label = '{} {:.2f}'.format(predicted_class, score)
            label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom-top)/2+top
            mid_v = (right-left)/2+left

            # put object rectangle
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

            # get text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

            # add everything to list
            ObjectsList.append([top, left, bottom, right,label])

        return ObjectsList

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        r_image  = self.detect_image(original_image_color)
        return r_image

    
if __name__=="__main__":
    font = cv2.FONT_HERSHEY_DUPLEX
    faceCascade = cv2.CascadeClassifier('./cascade.xml')
    yolo = YOLO()
    fps = 0
    display_time = 2
    Start_time = time.time()
    cap = cv2.VideoCapture(0)
    #cap.set(3,640)
    #cap.set(4,480)
    if not cap.isOpened():
    	raise IOError("W cannot open webcam")
	
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        plate = faceCascade.detectMultiScale(
            gray,
            scaleFactor =1.1,
            minNeighbors = 7,
            minSize = (40,30)
            )
        for (x,y,w,h) in plate: #left=x;top=y; right = x + w; bottom = y + h
            #if x!=0 & y!=0 & w!=0 & h!=0:
            roi_color = frame[y:(y+h),x:(x+w)]
            #roi_color = cv2.resize(roi_color, (0,0), fx=1.0, fy=1.0)
            roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            objectlist = yolo.detect_image(roi_color) #[top, left, bottom, right, label]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            #print(objectlist)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            for (top, left, bottom, right, label) in objectlist:
                top += y
                left += x
                bottom += y
                right += x
                cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
                #cv2.rectangle(frame,(left,bottom - 35),(right,bottom),(0,0,255),cv2.FILLED)
                cv2.putText(frame,label,(left + 5, top - 5),font,1,(255,255,0),1)
        
        fps += 1
        TIME = time.time() -  Start_time
        if TIME > display_time:
            print("FPS", fps/TIME)
            fps = 0
            Start_time = time.time()
        frame=cv2.resize(frame,(1024,1024))
        cv2.imshow("video", frame)

        if cv2.waitKey(1) == 27 & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
    cap.release()
    cap.destroyAllWindows()
    yolo.close_session()
    


