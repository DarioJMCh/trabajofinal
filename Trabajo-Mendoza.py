import cv2
import numpy as np

cv2.namedWindow('Video Proyectado',cv2.WINDOW_NORMAL)
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
background = cv2.imread("timessquare.jpg")  # Reemplaza con tu imagen
if background is None:
    print("No se pudo cargar la imagen de fondo.")
    exit()

h_bg, w_bg = background.shape[:2]

# ---------- Selección de puntos ----------
print("Selecciona 4 puntos en la imagen de fondo donde se proyectará el video.")
points_dst = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_dst) < 4:
        points_dst.append([x, y])
        print(f"Punto {len(points_dst)}: ({x}, {y})")

clone = background.copy()
cv2.namedWindow("Selecciona puntos",cv2.WINDOW_NORMAL)
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
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        print("Selección cancelada.")
        exit()

cv2.destroyWindow("Selecciona puntos")

# Ordenar los puntos automáticamente
points_dst = ordenar_puntos(points_dst)

# ---------- Captura de video ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()


print("Iniciando proyección...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        continue

    frame_w = frame.shape[1]
    frame_h = frame.shape[0]
    points_src = np.array([[0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]], dtype=np.float32)

    # Matriz de transformación
    matrix = cv2.getPerspectiveTransform(points_src, points_dst)
    warped = cv2.warpPerspective(frame, matrix, (w_bg, h_bg))

    # Crear máscara para combinar
    mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
    cv2.fillConvexPoly(mask, points_dst.astype(int), 255)
    cv2.imshow("asi esta la mascara", mask)
    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow("asi esta la mascara invertida", mask_inv)

    # Combinar fondo con el video proyectado
    fondo_visible = cv2.bitwise_and(background, background, mask=mask_inv)
    cv2.imshow("fondo visible",fondo_visible)

    video_visible = cv2.bitwise_and(warped, warped, mask=mask)
    cv2.imshow("video visible", video_visible)

    resultado = cv2.add(fondo_visible, video_visible)

    cv2.imshow("Video Proyectado", resultado)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
