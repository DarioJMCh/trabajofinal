import cv2
import numpy as np

# ---------- Función para ordenar los puntos ----------
def ordenar_puntos(puntos):
    puntos = np.array(puntos, dtype=np.float32)
    suma = puntos.sum(axis=1)
    resta = np.diff(puntos, axis=1).flatten()
    top_left = puntos[np.argmin(suma)]
    bottom_right = puntos[np.argmax(suma)]
    top_right = puntos[np.argmin(resta)]
    bottom_left = puntos[np.argmax(resta)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

# ---------- Cargar imagen de fondo ----------
background = cv2.imread("timesquare.jpg")  # Reemplaza con tu imagen
if background is None:
    print("No se pudo cargar la imagen de fondo.")
    exit()

h_bg, w_bg = background.shape[:2]

# ---------- Selección de puntos para cámara ----------
print("Selecciona 4 puntos en la imagen de fondo donde se proyectará la cámara.")
points_dst = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_dst) < 4:
        points_dst.append([x, y])
        print(f"Punto {len(points_dst)}: ({x}, {y})")

clone = background.copy()
cv2.namedWindow("Selecciona puntos", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Selecciona puntos", select_points)

while True:
    img_display = clone.copy()
    for idx, point in enumerate(points_dst):
        cv2.circle(img_display, tuple(point), 5, (0, 255, 0), -1)
        cv2.putText(img_display, str(idx + 1), tuple(point), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Selecciona puntos", img_display)

    if len(points_dst) == 4:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        print("Selección cancelada.")
        exit()

cv2.destroyWindow("Selecciona puntos")
points_dst = ordenar_puntos(points_dst)

# ---------- Definir puntos destino para el segundo video (estáticos) ----------
# Modifica aquí los puntos donde quieres que aparezca el segundo video
puntos_video2 = np.array([
    [1119, 302],
    [1466, 234],
    [1475, 556],
    [1126, 565]
], dtype=np.float32)
puntos_video2 = ordenar_puntos(puntos_video2)

# ---------- Captura de video principal (cámara) ----------
cap1 = cv2.VideoCapture(0)
if not cap1.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

# ---------- Segundo video desde archivo ----------
cap2 = cv2.VideoCapture("video.mp4")
if not cap2.isOpened():
    print("No se pudo abrir el segundo video.")
    exit()

print("Iniciando proyección...")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # ---------- Proyección del video 1 (cámara) ----------
    h1, w1 = frame1.shape[:2]
    pts_src1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    matrix1 = cv2.getPerspectiveTransform(pts_src1, points_dst)
    warped1 = cv2.warpPerspective(frame1, matrix1, (w_bg, h_bg))

    mask1 = np.zeros((h_bg, w_bg), dtype=np.uint8)
    cv2.fillConvexPoly(mask1, points_dst.astype(int), 255)
    mask1_inv = cv2.bitwise_not(mask1)

    fondo_visible = cv2.bitwise_and(background, background, mask=mask1_inv)
    video_visible = cv2.bitwise_and(warped1, warped1, mask=mask1)
    resultado = cv2.add(fondo_visible, video_visible)

    # ---------- Proyección del video 2 (archivo) ----------
    h2, w2 = frame2.shape[:2]
    pts_src2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
    matrix2 = cv2.getPerspectiveTransform(pts_src2, puntos_video2)
    warped2 = cv2.warpPerspective(frame2, matrix2, (w_bg, h_bg))

    mask2 = np.zeros((h_bg, w_bg), dtype=np.uint8)
    cv2.fillConvexPoly(mask2, puntos_video2.astype(int), 255)
    mask2_inv = cv2.bitwise_not(mask2)

    resultado = cv2.bitwise_and(resultado, resultado, mask=mask2_inv)
    video2_visible = cv2.bitwise_and(warped2, warped2, mask=mask2)
    resultado = cv2.add(resultado, video2_visible)

    # ---------- Mostrar resultado y debug ----------
    cv2.namedWindow("Video_Proyectado", cv2.WINDOW_NORMAL)
    cv2.imshow("Video_Proyectado", resultado)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
