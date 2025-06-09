import cv2
import numpy as np

# Cargar imagen de fondo
background = cv2.imread("timesquare.jpg")
if background is None:
    print("No se pudo cargar la imagen de fondo.")
    exit()

h_bg, w_bg = background.shape[:2]

print("Selecciona 8 puntos:")
print("1-4: pantalla izquierda (arriba izq, arriba der, abajo der, abajo izq)")
print("5-8: pantalla derecha (arriba izq, arriba der, abajo der, abajo izq)")

points_dst = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_dst) < 8:
        points_dst.append([x, y])
        print(f"Punto {len(points_dst)}: ({x}, {y})")

clone = background.copy()
cv2.namedWindow("Selecciona puntos")
cv2.setMouseCallback("Selecciona puntos", select_points)

while True:
    img_display = clone.copy()
    for idx, point in enumerate(points_dst):
        cv2.circle(img_display, tuple(point), 5, (0, 255, 0), -1)
        cv2.putText(img_display, str(idx + 1), tuple(point), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Selecciona puntos", img_display)

    if len(points_dst) == 8:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        print("Selección cancelada.")
        exit()

cv2.destroyWindow("Selecciona puntos")

# Puntos por superficie
dst_left = np.array(points_dst[0:4], dtype=np.float32)
dst_right = np.array(points_dst[4:8], dtype=np.float32)

# Captura de cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

frame_w, frame_h = 320, 240
half_w = frame_w // 2

src_quad = np.array([[0, 0], [half_w, 0], [half_w, frame_h], [0, frame_h]], dtype=np.float32)

print("Proyectando video sobre dos superficies...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    try:
        frame = cv2.resize(frame, (frame_w, frame_h))
    except:
        continue

    # Dividir video en dos mitades
    frame_left = frame[:, :half_w]
    frame_right = frame[:, half_w:]

    # Homografías
    H_left = cv2.getPerspectiveTransform(src_quad, dst_left)
    H_right = cv2.getPerspectiveTransform(src_quad, dst_right)

    warped_left = cv2.warpPerspective(frame_left, H_left, (w_bg, h_bg))
    warped_right = cv2.warpPerspective(frame_right, H_right, (w_bg, h_bg))

    # Máscaras
    mask_left = np.zeros((h_bg, w_bg), dtype=np.uint8)
    mask_right = np.zeros((h_bg, w_bg), dtype=np.uint8)
    cv2.fillConvexPoly(mask_left, dst_left.astype(int), 255)
    cv2.fillConvexPoly(mask_right, dst_right.astype(int), 255)

    mask_total = cv2.bitwise_or(mask_left, mask_right)
    mask_inv = cv2.bitwise_not(mask_total)

    fondo_visible = cv2.bitwise_and(background, background, mask=mask_inv)
    proj_left = cv2.bitwise_and(warped_left, warped_left, mask=mask_left)
    proj_right = cv2.bitwise_and(warped_right, warped_right, mask=mask_right)

    resultado = cv2.add(fondo_visible, cv2.add(proj_left, proj_right))

    cv2.imshow("Video Proyectado con Homografía", resultado)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
