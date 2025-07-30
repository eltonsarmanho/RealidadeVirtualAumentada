import cv2
import mediapipe as mp
import numpy as np
import time  # Para calcular o framerate
from typing import Union

# Definindo Tipagem para melhor legibilidade
webcam_image = np.ndarray
confidence = float
coords_vector = Union[int, list[int]]
rgb_tuple = tuple[int, int, int]

class VanzeDetector:
    def __init__(self,
                 mode: bool = False,
                 number_hands: int = 2,
                 model_complexity: int = 1,
                 min_detec_confidence: confidence = 0.5,
                 min_tracking_confidence: confidence = 0.5):
        """
        Inicializa o detector de mãos do Mediapipe.

        :param mode: Define se o modelo estará sempre em modo estático.
        :param number_hands: Número máximo de mãos que serão detectadas.
        :param model_complexity: Complexidade do modelo (0 ou 1).
        :param min_detec_confidence: Confiança mínima para detecção de mãos.
        :param min_tracking_confidence: Confiança mínima para rastreamento de mãos.
        """
        self.mode = mode
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detec_confidence
        self.tracking_con = min_tracking_confidence

        # Inicializa os módulos do Mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                         self.max_num_hands,
                                         self.complexity,
                                         self.detection_con,
                                         self.tracking_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]  # IDs das pontas dos dedos

    def find_hands(self, img: webcam_image, draw_hands: bool = True):
        """
        Detecta mãos na imagem e desenha as conexões, se solicitado.

        :param img: Imagem capturada da webcam.
        :param draw_hands: Desenha as conexões das mãos, se True.
        :return: Imagem com ou sem as conexões desenhadas.
        """
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para RGB
        self.results = self.hands.process(img_RGB)  # Processa a imagem

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw_hands:
                    self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img: webcam_image, hand_number: int = 0):
        """
        Encontra as coordenadas dos marcos (landmarks) da mão.

        :param img: Imagem capturada.
        :param hand_number: Qual mão será analisada (0 = primeira mão detectada).
        :return: Lista de coordenadas dos landmarks (x, y).
        """
        self.required_landmark_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark):
                height, width, _ = img.shape
                center_x, center_y = int(lm.x * width), int(lm.y * height)
                self.required_landmark_list.append([id, center_x, center_y])

        return self.required_landmark_list

    def fingers_up(self):
        """
        Determina quais dedos estão levantados com base nos landmarks.

        :return: Lista indicando os dedos levantados (1 = levantado, 0 = abaixado).
        """
        fingers = []

        try:
            # Dedão (comparação no eixo X)
            if self.required_landmark_list[self.tip_ids[0]][1] < self.required_landmark_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Outros dedos (comparação no eixo Y)
            for id in range(1, 5):
                if self.required_landmark_list[self.tip_ids[id]][2] < self.required_landmark_list[self.tip_ids[id] - 2][
                    2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        except:
              fingers = []
        return fingers

    def draw_in_position(self, img: webcam_image, x_vector: coords_vector, y_vector: coords_vector,
                         rgb_selection: rgb_tuple = (255, 0, 0), thickness: int = 10):
        """
        Desenha círculos nas posições especificadas.

        :param img: Imagem capturada.
        :param x_vector: Coordenadas x para desenhar os círculos.
        :param y_vector: Coordenadas y para desenhar os círculos.
        :param rgb_selection: Cor dos círculos (padrão: vermelho).
        :param thickness: Espessura dos círculos.
        :return: Imagem com os círculos desenhados.
        """
        x_vector = x_vector if isinstance(x_vector, list) else [x_vector]
        y_vector = y_vector if isinstance(y_vector, list) else [y_vector]

        for x, y in zip(x_vector, y_vector):
            cv2.circle(img, (x, y), thickness, rgb_selection, cv2.FILLED)

        return img

# Função Principal (Testando a classe)
def main():
    previous_time = 0  # Para calcular o FPS
    capture = cv2.VideoCapture(0)  # Captura o vídeo da webcam

    detector = VanzeDetector()

    while True:
        ret, img = capture.read()
        if not ret:
            break

        img = detector.find_hands(img)  # Detecta mãos
        landmark_list = detector.find_position(img)  # Encontra landmarks
        fingers = detector.fingers_up()
        if fingers:
           print(f"fingers: {fingers}")

        #if landmark_list:
        #   print(f"Posição do Landmark 8 (Indicador): {landmark_list[8]}")

        # Calcula FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Exibe o FPS na imagem
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255), 3)

        # Mostra a imagem
        cv2.imshow("Deteccao de Hands", img)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
