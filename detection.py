import cv2

cap = cv2.VideoCapture(0)

# use haar cascade for detection
face_cascade = cv2.CascadeClassifier("cascade.xml")
while True:
    # get every frame and convert to grayscale
    _, image = cap.read()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=3)
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

