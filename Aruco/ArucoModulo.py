import cv2;
import numpy as np
import os;
import cv2.aruco as aruco
import pyautogui

def loadAugImages(path):
    myList = os.listdir(path)
    augDicts = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDicts[key] = imgAug
    return augDicts
def findArucoMarkers(img, markerSize=4, totalMarkers=250, draw=True):
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
    cap = cv2.VideoCapture(0)
    augDics = loadAugImages("../Markers")

    while True:
        sccuess , img = cap.read()
        arucoFound = findArucoMarkers(img)
        if len(arucoFound[0]) != 0:
            for bbox,id in zip(arucoFound[0],arucoFound[1]):
                if int(id[0]) in augDics:
                    img = augmentAruco(bbox, id, img, augDics[int(id[0])])
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main();