import numpy as np
import cv2 as cv

def main():

    img = cv.VideoCapture(0)
    img.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    img.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        
        ret, frame0 = img.read()
        frame = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        Z = frame.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        gauss = cv.GaussianBlur(frame, (5,5), 0)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((gauss.shape))
        
        canny = cv.Canny(res2, 10, 50)
        (contours,_) = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        contour_image = cv.cvtColor(res2, cv.COLOR_GRAY2BGR)    
        cv.drawContours(contour_image, contours, -1, (0,0,255),3)
        for contour in contours:
            cx = sum(i[0] for i in contour[0])//len(contour[0])
            cy = sum(i[1] for i in contour[0])//len(contour[0])
            cv.circle(contour_image , (cx, cy), 1, (0, 255, 0), 5)
        # print(len(contours))
        

        # for i in contours:
        #     M = cv.moments(i)
        #     if M['m00'] != 0:
        #         cx = int(M['m10']/M['m00'])
        #         cy = int(M['m01']/M['m00'])
        #         cv.circle(contour_image , (cx, cy), 1, (0, 255, 0), 5)
       
 

          


        cv.imshow('res2',contour_image)
        if cv.waitKey(1) == 27:
            break





if __name__ == "__main__":
    main()