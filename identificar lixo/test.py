import numpy as np
import cv2 as cv

def main():

    img = cv.VideoCapture(0)
    img.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    img.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        
        ret, frame0 = img.read()
        frame = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gauss = cv.GaussianBlur(frame, (5,5), 0)
        Z = gauss.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)        
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((gauss.shape))
        
        ret, thresh = cv.threshold(res2, 150, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_image = cv.cvtColor(res2, cv.COLOR_GRAY2BGR)    
                
        min_area = 30000
        max_area = 180000

        for contour in contours:
            area = cv.contourArea(contour)
            M = cv.moments(contour)
            if min_area < area < max_area:
                cv.drawContours(contour_image, [contour], -1, (0, 0, 255), 3)
                
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv.circle(contour_image , (cx, cy), 1, (0, 255, 0), 5)
                    
                    (x,y),radius = cv.minEnclosingCircle(contour)
                    radius = int(radius)
                    if radius <= 250:
                        cv.circle(contour_image, (cx,cy), radius, (0, 255, 0), 2)
                        print(cx,cy)

       
 

          


        cv.imshow('res2',contour_image)
        if cv.waitKey(1) == 27:
            break





if __name__ == "__main__":
    main()