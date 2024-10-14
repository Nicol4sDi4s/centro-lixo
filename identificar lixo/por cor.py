import numpy as np
import cv2 as cv

def main():

    img = cv.VideoCapture(0)
    img.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    img.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    def nothing(x):
        pass
    cv.namedWindow("Track")
    cv.createTrackbar('L-H', 'Track', 0, 180, nothing)
    cv.createTrackbar('L-S', 'Track', 0, 255, nothing)
    cv.createTrackbar('L-V', 'Track', 0, 255, nothing)
    cv.createTrackbar('U-H', 'Track', 180, 180, nothing)
    cv.createTrackbar('U-S', 'Track', 255, 255, nothing)
    cv.createTrackbar('U-V', 'Track', 255, 255, nothing)
   


    while True:
        
        ret, frame = img.read()
        hsvframe = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        l_h = cv.getTrackbarPos("L-H", "Track")
        l_s = cv.getTrackbarPos("L-S", "Track")
        l_v = cv.getTrackbarPos("L-V", "Track")
        u_h = cv.getTrackbarPos("U-H", "Track")
        u_s = cv.getTrackbarPos("U-S", "Track")
        u_v = cv.getTrackbarPos("U-V", "Track")
        

        low_white = np.array([l_h, l_s, l_v])
        upper_white = np.array([u_h, u_s, u_v])

        mask = cv.inRange(hsvframe, low_white, upper_white)
 

















        cv.imshow('frame', frame)
        cv.imshow('Mask', mask)
        if cv.waitKey(1) == 27:
            break





if __name__ == "__main__":
    main()