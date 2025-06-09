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

# ---------- Selección de puntos ----------
print("Selecciona 4 puntos en la imagen de fondo donde se proyectará el video.")
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
    mask_inv = cv2.bitwise_not(mask)

    # Combinar fondo con el video proyectado
    fondo_visible = cv2.bitwise_and(background, background, mask=mask_inv)
    video_visible = cv2.bitwise_and(warped, warped, mask=mask)
    resultado = cv2.add(fondo_visible, video_visible)

    # ---------- Mostrar vistas combinadas ----------
    # Convertir máscaras a BGR
    mascara_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mascara_inv_bgr = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

    # Redimensionar imágenes para debug
#    debug_size = (320, 240)
#    def resize(img):
#        return cv2.resize(img, debug_size)

#    fila1 = np.hstack((resize(mascara_bgr), resize(mascara_inv_bgr)))
#    fila2 = np.hstack((resize(fondo_visible), resize(video_visible)))
#    debug_view = np.vstack((fila1, fila2))

    # Tamaño deseado de ventana debug
    debug_window_width = 1280
    debug_window_height = 960
    single_width = debug_window_width // 2
    single_height = debug_window_height // 2


    def resize(img):
        return cv2.resize(img, (single_width, single_height), interpolation=cv2.INTER_LINEAR)


    fila1 = np.hstack((resize(mascara_bgr), resize(mascara_inv_bgr)))
    fila2 = np.hstack((resize(fondo_visible), resize(video_visible)))
    debug_view = np.vstack((fila1, fila2))

    cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Debug View", debug_window_width, debug_window_height)
    cv2.imshow("Debug View", debug_view)

    # Mostrar proyección final
    cv2.namedWindow("Video_Proyectado", cv2.WINDOW_NORMAL)
    cv2.imshow("Video_Proyectado", resultado)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
