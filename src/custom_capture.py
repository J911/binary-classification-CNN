# Jaemin Lee (aka, J911)
# 2019

import cv2
import numpy as np 

class CustomCapture:
    
    def _cam_initialize(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)
        
    def _cam_release(self):
        self.cam.release()

    def capture(self, label=''):
        capture_data = []
        capture_labels = []

        self._cam_initialize()
        while True:
            f, img = self.cam.read()
            img = cv2.cvtColor(img, f)
            
            display_img = cv2.putText(cv2.flip(img, 1),  label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
            cv2.imshow("Custom Capture", display_img)

            img = np.moveaxis(img, -1, 0)

            key = cv2.waitKey(1)
            if key == 97: # left
                capture_data.append(img)
                capture_labels.append(0)
                print("save left photos!")

            if key == 100:  #right
                capture_data.append(img)
                capture_labels.append(1)
                print("save right photos!")

            if key == 27: # esc
                break

        self._cam_release()
        return (capture_data, capture_labels)
        
if __name__ == '__main__':
    cap = CustomCapture()
    x, y = cap.capture()
    print(x, y)