import cv2
import numpy as np
from styx_msgs.msg import TrafficLight
from keras.models import load_model
import h5py
from keras import __version__ as keras_version


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        '''

        f = h5py.File('model_test.h5', mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras_version).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version ', keras_version,
                ', but the model was built using ', model_version)
        '''

        model = load_model('./light_classification/model_test.h5')

        self.dictClass ={0:TrafficLight.RED,1:TrafficLight.YELLOW,2:TrafficLight.GREEN,3:TrafficLight.UNKNOWN}

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction

        state = TrafficLight.UNKNOWN

        
        image = np.expand_dims(image, axis=0)

        images = np.vstack([image])

        class_arr = model.predict(images, batch_size=1)

        argmax = np.argmax(class_arr)

        state = dictClass[argmax]

        
        # import ipdb; ipdb.set_trace()

        return TrafficLight.UNKNOWN