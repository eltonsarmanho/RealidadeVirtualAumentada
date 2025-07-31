import cv2;
import numpy as np
import os;
import cv2.aruco as aruco
import pyautogui

def loadAugImages(path):
    """
    Carrega todas as imagens de aumento (augmentação) da pasta especificada.

    Args:
        path (str): Caminho para a pasta contendo as imagens.

    Returns:
        dict: Um dicionário onde a chave é o nome do arquivo (sem extensão, convertido para inteiro)
              e o valor é a imagem carregada como um array do OpenCV.
    """
    myList = os.listdir(path)
    augDicts = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDicts[key] = imgAug
    return augDicts
def findArucoMarkers(img, markerSize=4, totalMarkers=250, draw=True):
    """
    Detecta marcadores ArUco em uma imagem.

    Args:
        img (numpy.ndarray): Imagem de entrada (BGR).
        markerSize (int, opcional): Tamanho do marcador ArUco (default=4).
        totalMarkers (int, opcional): Número total de marcadores no dicionário (default=250).
        draw (bool, opcional): Se True, desenha os marcadores detectados na imagem (default=True).

    Returns:
        list: Uma lista contendo os bounding boxes dos marcadores detectados e seus IDs.
    """
    # Converte a imagem para escala de cinza
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define o dicionário com o tamanho e a quantidade de marcadores desejada
    arucoDict = aruco.getPredefinedDictionary(getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}'))

    # Define os parâmetros do detector
    arucoParam = aruco.DetectorParameters()

    # Detecta os marcadores na imagem em escala de cinza
    detector = aruco.ArucoDetector(arucoDict, arucoParam)
    bboxs, ids, rejected = detector.detectMarkers(imgGray)

    # Imprime os IDs dos marcadores detectados
    #print(ids)

    if draw:
        aruco.drawDetectedMarkers(img,bboxs)
    return [bboxs,ids]
def augmentAruco(bbox, id, img, imgAug, drawId=True):
    """
    Sobrepõe uma imagem de augmentação sobre o marcador ArUco detectado.

    Args:
        bbox (numpy.ndarray): Coordenadas dos vértices do marcador detectado.
        id (int): ID do marcador ArUco.
        img (numpy.ndarray): Imagem original onde o marcador foi detectado.
        imgAug (numpy.ndarray): Imagem de augmentação a ser sobreposta.
        drawId (bool, opcional): Se True, desenha o ID do marcador na imagem (default=True).

    Returns:
        numpy.ndarray: Imagem resultante com a augmentação aplicada.
    """
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img,pts1.astype(int),(0,0,0))
    imgOut = img+imgOut
    if drawId:
        org = (int(tl[0]), int(tl[1]))
        thickness = 2
        lineType = cv2.LINE_AA
        cv2.putText( imgOut,str(id),org,cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness, lineType)
    return imgOut
def main():
    """
    Função principal que captura vídeo da webcam, detecta marcadores ArUco e aplica augmentação.

    Pressione 'q' para sair do programa.
    """
    cap = cv2.VideoCapture(0)
    # Obtém o caminho absoluto da pasta Markers
    markers_path = os.path.join(os.path.dirname(__file__), "..", "Markers")
    augDics = loadAugImages(markers_path)

    while True:
        sccuess , img = cap.read()
        arucoFound = findArucoMarkers(img)
        if len(arucoFound[0]) != 0:
            for bbox,id in zip(arucoFound[0],arucoFound[1]):
                if int(id[0]) in augDics:
                    img = augmentAruco(bbox, id, img, augDics[int(id[0])])
        cv2.imshow("Image",img)
        # Sai do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main();