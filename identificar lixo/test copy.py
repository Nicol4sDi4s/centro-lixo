# import numpy as np
# import cv2 as cv

# def main():

#     img = cv.VideoCapture(0)
#     img.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
#     img.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

#     while True:
        
#         ret, frame0 = img.read()
#         frame = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
#         Z = frame.reshape((-1,3))
#         # convert to np.float32
#         Z = np.float32(Z)
#         # define criteria, number of clusters(K) and apply kmeans()
#         criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#         K = 2
#         ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
#         # Now convert back into uint8, and make original image
#         gauss = cv.GaussianBlur(frame, (5,5), 0)
#         center = np.uint8(center)
#         res = center[label.flatten()]
#         res2 = res.reshape((gauss.shape))
        
#         canny = cv.Canny(res2, 10, 50)
#         (contours,_) = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#         print(len(contours))
#         contour_image = cv.cvtColor(res2, cv.COLOR_GRAY2BGR)    
#         cv.drawContours(contour_image, contours, -1, (0,0,255),3)
#         for contour in contours:
#             cx = sum(i[0] for i in contour[0])//len(contour[0])
#             cy = sum(i[1] for i in contour[0])//len(contour[0])
#             cv.circle(contour_image , (cx, cy), 1, (0, 255, 0), 5)
#         # print(len(contours))
        

#         # for i in contours:
#         #     M = cv.moments(i)
#         #     if M['m00'] != 0:
#         #         cx = int(M['m10']/M['m00'])
#         #         cy = int(M['m01']/M['m00'])
#         #         cv.circle(contour_image , (cx, cy), 1, (0, 255, 0), 5)
       
 

          


#         cv.imshow('res2',contour_image)
#         if cv.waitKey(1) == 27:
#             break





# if __name__ == "__main__":
#     main()

import cv2
import numpy as np

# Inicializa os parâmetros
Brightness = 0.0
Contrast = 0.0
Saturation = 0.0
Gain = 0.0

winName = "Live"
cap = cv2.VideoCapture(0)

# Função chamada quando os trackbars são alterados
def onTrackbar_changed(x):
    global Brightness, Contrast, Saturation, Gain

    # Atualiza os valores das variáveis globais a partir dos sliders
    Brightness = float(cv2.getTrackbarPos("Brightness", winName)) / 100
    Contrast = float(cv2.getTrackbarPos("Contrast", winName)) / 100
    Saturation = float(cv2.getTrackbarPos("Saturation", winName)) / 100
    Gain = float(cv2.getTrackbarPos("Gain", winName)) / 100

    print(f"Brightness: {Brightness}, Contrast: {Contrast}, Saturation: {Saturation}, Gain: {Gain}")

# Verifica se a captura de vídeo foi iniciada com sucesso
if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

print("Pressione 's' para salvar a imagem")

cv2.namedWindow(winName)

# Define valores iniciais para os trackbars
cv2.createTrackbar("Brightness", winName, 50, 100, onTrackbar_changed)  # 50 para um valor inicial neutro
cv2.createTrackbar("Contrast", winName, 50, 100, onTrackbar_changed)
cv2.createTrackbar("Saturation", winName, 50, 100, onTrackbar_changed)
cv2.createTrackbar("Gain", winName, 50, 100, onTrackbar_changed)

def apply_post_processing(frame, brightness, contrast, saturation, gain):
    # Converte a imagem de BGR para HSV para manipular a saturação
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Ajuste de saturação
    hsv[:, :, 1] *= (1.0 + saturation)  # Saturação multiplicada
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)  # Limita os valores

    # Converte de volta para BGR após modificar a saturação
    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Ajuste de brilho e contraste (aumento de ganho em pós-processamento)
    frame = cv2.convertScaleAbs(frame, alpha=(1.0 + contrast), beta=(brightness * 100))

    # Aplica o ganho como uma multiplicação adicional de brilho
    frame = np.clip(frame * (1.0 + gain), 0, 255).astype(np.uint8)

    return frame

i = 0
while True:
    ret, frame = cap.read()  # Captura um novo frame da câmera
    if not ret:
        print("Falha ao capturar frame.")
        break

    # Aplica os ajustes de pós-processamento
    processed_frame = apply_post_processing(frame, Brightness - 0.5, Contrast - 0.5, Saturation - 0.5, Gain)

    # Mostra o frame processado
    cv2.imshow(winName, processed_frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('s'):
        filename = f"{i}.jpg"
        cv2.imwrite(filename, processed_frame)
        i += 1
        print(f"Imagem salva como {filename}")
    
    # Fecha o programa ao pressionar a tecla 'ESC'
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


