import cv2
import numpy as np
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass


    def hsv_select(self,img,Min,Max):

            # 1) Convert to HLS color space
            #hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            H = img[:,:,0]
            S = img[:,:,1]
            V = img[:,:,2]
    
            binary_output = np.zeros_like(S)
            binary_output[(H > Min[0]) & (H <= Max[0]) & (S > Min[1]) & (S <= Max[1]) & (V > Min[2]) & (V <= Max[2])] = 255
    
            #print('There is an error in the channel value')
            # 3) Return a binary image of threshold result
            #binary_output = np.copy(img) # placeholder line
            return binary_output


    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction


        hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        #red
        RED_MIN = np.array([0, 120, 120],np.uint8)
        RED_MAX = np.array([10, 255, 255],np.uint8)
        #frame_threshed = cv2.inRange(hsv_img, RED_MIN, RED_MAX)
        frame_threshed = self.hsv_select(hsv_img,RED_MIN, RED_MAX)
        r = cv2.countNonZero(frame_threshed)
        if r > 50:
            return TrafficLight.RED

        YELLOW_MIN = np.array([40.0/360*255, 120, 120],np.uint8)
        YELLOW_MAX = np.array([66.0/360*255, 255, 255],np.uint8)
        #frame_threshed = cv2.inRange(hsv_img, YELLOW_MIN, YELLOW_MAX)
        frame_threshed = self.hsv_select(hsv_img, YELLOW_MIN, YELLOW_MAX)
        y = cv2.countNonZero(frame_threshed)
        if y > 50:
            return TrafficLight.YELLOW

        GREEN_MIN = np.array([90.0/360*255, 120, 120],np.uint8)
        GREEN_MAX = np.array([140.0/360*255, 255, 255],np.uint8)
        #frame_threshed = cv2.inRange(hsv_img, GREEN_MIN, GREEN_MAX)
        frame_threshed = self.hsv_select(hsv_img, GREEN_MIN, GREEN_MAX)
        g = cv2.countNonZero(frame_threshed)
        if g > 50:
            return TrafficLight.GREEN

        # import ipdb; ipdb.set_trace()

        return TrafficLight.UNKNOWN