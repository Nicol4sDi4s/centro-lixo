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
        
        canny = cv.Canny(gauss, 50, 150)
        (contours,_) = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        contour_image = cv.cvtColor(res2, cv.COLOR_GRAY2BGR)

        min_area = 150
        max_area = 400

        for contour in contours:
            area = cv.contourArea(contour)
            if min_area < area < max_area:
                cv.drawContours(contour_image, [contour], -1, (0, 0, 255), 3)
                
        #cv.drawContours(res2, contours, -1, (255,0,0),3)

        # circles = cv.HoughCircles(frame, cv.HOUGH_GRADIENT_ALT, 1.5, 50,
        #                           param1=100,param2=0.1, minRadius=0, maxRadius=0)
        
        # circles = np.uint16(np.around(circles))
        # for i in circles[0,:]:
        #     # draw the outer circle
        #     cv.circle(res2,(i[0],i[1]),i[2],(0,255,0),2)
        #     # draw the center of the circle
        #     cv.circle(res2,(i[0],i[1]),2,(0,0,255),3)




        cv.imshow('res2',contour_image)
        if cv.waitKey(1) == 27:
            break





if __name__ == "__main__":
    main()