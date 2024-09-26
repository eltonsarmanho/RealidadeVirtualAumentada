import cv2
import numpy as np
import logging as log
import datetime as dt
from time import sleep

from triton.language import dtype

cascPathHand = "haarcascade_hand.xml"

# Se quiserem me solicitem este arquivo pelo Whatsapp que envio ele pra vocês: 31.9.8587.2081 (pois ele é muito grande e complexo)
# É este arquivo que vai identificar a mão de vocês possinilitando assim a IMERSÃO

handCascade = cv2.CascadeClassifier(cascPathHand)

log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

# Dimensões da tela
width, height = int(video_capture.get(3)), int(video_capture.get(4))

# Inicializa a posição da caixa preta
box_size_black = 100
box_position_black = [10, 10]
hand_in_box_black = False

# AO CLICAR NA CAIXA VERMELHA ELA VAI DESAPARECER POR ALGUNS MILÉSIMOS DE SEGUNDO

# Inicializa a posição da caixa vermelha
box_size_red = 100
box_position_red = [width-box_size_red-10, 10]
hand_in_box_red = False

# Configurações do texto "CELI ROAD"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_thickness = 4
text_color = (0, 0, 0)  # Cor preta

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of skin color in HSV
    lower_skin = np.array([0,20,70],dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale for hand detection
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Detect hands
    hands = handCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around hands
    for (x,y,w,h) in hands:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Verifica se a mão toca a caixa preta
        if box_position_black[0] < x < box_position_black[0] + box_size_black and \
           box_position_black[1] < y < box_position_black[1] + box_size_black:
            hand_in_box_black = True
            hand_in_box_red = False
            # Adiciona o texto "CELI ROAD" em baixo da tela, no centro, em preto e negrito
            text_size = cv2.getTextSize('CELI ROAD', font, font_scale, font_thickness)[0]
            text_position = ((width - text_size[0]) // 2, height - 20)
            cv2.putText(frame, 'CELI ROAD', text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Verifica se a mão toca a caixa vermelha
        elif box_position_red[0] < x < box_position_red[0] + box_size_red and \
             box_position_red[1] < y < box_position_red[1] + box_size_red:
            hand_in_box_red = True
            hand_in_box_black = False
        else:
            hand_in_box_black = False
            hand_in_box_red = False

    # Adiciona a caixa preta na posição atualizada
    frame[box_position_black[1]:box_position_black[1] + box_size_black,
          box_position_black[0]:box_position_black[0] + box_size_black] = (0, 0, 0)

    # Adiciona a caixa vermelha no lado direito
    if not hand_in_box_red:
        frame[box_position_red[1]:box_position_red[1] + box_size_red,
              box_position_red[0]:box_position_red[0] + box_size_red] = (0, 0, 255)

    # Exibe o resultado
    cv2.imshow('Video',frame);

    # Aguarda pressionar 'q' para encerrar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Quando terminar, libera os recursos
video_capture.release()
cv2.destroyAllWindows()