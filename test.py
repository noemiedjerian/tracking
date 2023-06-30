import cv2

# charge video
cap = cv2.VideoCapture(0)

# lis la silhouette
silhouette = cv2.imread('poisson.png', cv2.IMREAD_GRAYSCALE)

# dimensions de la vidéo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# retaille la silhouette
silhouette = cv2.resize(silhouette, (width, height))

# Créer un objet soustracteur d'arrière-plan
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=100, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # applique la soustraction
    fgmask = fgbg.apply(frame)

    # trouve les contours des poissons en mouvements
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrez les petits contours et sélectionnez uniquement les contours des poissons
    fish_contours = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 150:  # réajuster le seuil si besoin
            fish_contours.append(cnt)

    all_inside = True  # variable pour savoir si tout les poissons sont à l'intérieur ou non

    for cnt in fish_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        center_x = x + w // 2
        center_y = y + h // 2

        # Vérifiez si le centre de gravité du poisson est à l'intérieur de la silhouette
        if not (0 <= center_y < silhouette.shape[0] and 0 <= center_x < silhouette.shape[1] and silhouette[
            center_y, center_x] == 0):
            all_inside = False  # boolen faux si un seul poisson à l'exterieur

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Détecter les contours dans l'image de la silhouette
    canny = cv2.Canny(silhouette, 30, 100)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    # Afficher le statut basé sur all_inside
    if all_inside:
        cv2.putText(frame, 'Tous Interieur', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Au moins un dehors', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Tracez des lignes reliant les centres des rectangles ( visuel )
    for i in range(len(fish_contours) - 1):
        cnt1 = fish_contours[i]
        cnt2 = fish_contours[i + 1]
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        center_x1 = x1 + w1 // 2
        center_y1 = y1 + h1 // 2
        center_x2 = x2 + w2 // 2
        center_y2 = y2 + h2 // 2
        cv2.line(frame, (center_x1, center_y1), (center_x2, center_y2), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
