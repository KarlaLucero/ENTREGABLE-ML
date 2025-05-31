import cv2

# Abrir la cámara (intenta con 0, cambia a 1 si no funciona)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ ERROR: No se pudo acceder a la cámara.")
    exit()

# Cargar clasificadores Haar
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smileClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

print("✅ Cámara abierta correctamente. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo capturar imagen de la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detectar ojos
        eyes = eyeClassif.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        # Detectar sonrisa
        smiles = smileClassif.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            cv2.putText(frame, "Sonriendo", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Mostrar imagen
    cv2.imshow("Detector Facial", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
