import cv2
import numpy as np


class FaceDetection:
    def __init__(self, image_path, cascade_file='haarcascade_frontalface_default.xml'):
        """
        Inicializa a classe FaceDetection.

        :param image_path: Caminho para a imagem que será processada.
        :param cascade_file: Nome do arquivo Haar Cascade XML para detecção facial.
        """
        self.image_path = image_path
        self.cascade_file = cv2.data.haarcascades + cascade_file

    def load_image(self):
        """
        Carrega a imagem e converte para tons de cinza.

        :return: Imagem colorida e imagem em tons de cinza.
        """
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError("Imagem não encontrada. Verifique o caminho da imagem.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    def detect_faces(self, gray_image, scaleFactor=1.1, minNeighbors=12):
        """
        Detecta rostos em uma imagem usando Haar Cascades.

        :param gray_image: Imagem em tons de cinza.
        :param scaleFactor: Redução do tamanho da imagem em cada escala.
        :param minNeighbors: Número de vizinhos para validar uma detecção.
        :return: Lista de retângulos representando os rostos detectados.
        """
        cascade = cv2.CascadeClassifier(self.cascade_file)
        rectangles = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        return rectangles

    def draw_rectangles(self, image, rectangles):
        """
        Desenha retângulos ao redor dos rostos detectados.

        :param image: Imagem colorida original.
        :param rectangles: Lista de coordenadas dos retângulos detectados.
        """
        for x, y, w, h in rectangles:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    def run(self):
        """
        Executa o pipeline completo de detecção facial.
        """
        print("Carregando imagem...")
        img, gray = self.load_image()

        print("Detectando rostos...")
        faces = self.detect_faces(gray)

        print(f"Contagem de rostos detectados: {len(faces)}")
        print("Coordenadas dos rostos detectados:", faces)

        print("Desenhando retângulos...")
        self.draw_rectangles(img, faces)

        cv2.imshow('Deteccao Facial', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Exemplo de uso
if __name__ == "__main__":
    # Substitua o caminho abaixo pela localização da imagem desejada
    image_path = '../assets/fotos/lady.jpg'
    # image_path = '../assets/fotos/group 1.jpg'
    # image_path = '../assets/fotos/group 2.jpg'
    face_detection = FaceDetection(image_path)
    face_detection.run()
