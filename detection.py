import cv2
import face_recognition


cap = cv2.VideoCapture(0)

subash = face_recognition.load_image_file("./images/subash.png")
esther = face_recognition.load_image_file("./images/esther.jpg")
subash_encoding = face_recognition.face_encodings(subash)[0]
esther_encoding = face_recognition.face_encodings(esther)[0]

known_encodings = [subash_encoding, esther_encoding]


# use haar cascade for detection
face_cascade = cv2.CascadeClassifier("cascade.xml")
while True:
    subash_found, esther_found = False, False
    # get every frame and convert to grayscale
    _, image = cap.read()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        face_encoding = face_recognition.face_encodings(image)
        # face_recognised = face_recognition.compare_faces(known_encodings, face_encoding)

        for unknown_encoding in face_encoding:
            results = face_recognition.compare_faces(known_encodings, unknown_encoding)
            if results[0] : subash_found = True
            if results[1] : esther_found = True
    except Exception as e:
        print("No face found")
        
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=3)
        if subash_found:
            cv2.putText(image, "Subash", (x,y), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), thickness=3)
        if esther_found:
            cv2.putText(image, "Esther", (x,y), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), thickness=3)
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

