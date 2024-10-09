
from matplotlib import pyplot as plt
import cv2 as cv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def main():

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _ , mask = cv.threshold(gray, 60, 255, cv.THRESH_BINARY)
        mask = cv.erode(mask, np.ones((7, 7), np.uint8))

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        df_mean_color = pd.DataFrame()
        for idx, contour in enumerate(contours):
            area = int(cv.contourArea(contour))

        # if area is higher than 3000:
        if area < 3000:
            filtered_contours.append(contour)
        # get mean color of contour:
            masked = np.zeros_like(frame[:, :, 0])  # This mask is used to get the mean color of the specific bead (contour), for kmeans
            cv.drawContours(masked, [contour], 0, 255, -1)

            B_mean, G_mean, R_mean, _ = cv.mean(frame, mask=masked)
            df = pd.DataFrame({'B_mean': B_mean, 'G_mean': G_mean, 'R_mean': R_mean}, index=[idx])
            df_mean_color = pd.concat([df_mean_color, df])






        cv.imshow('gray', gray)
        if cv.waitKey(1) == 27:
            break





if __name__ == "__main__":
    main()