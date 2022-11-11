import cv2
import numpy as np
import face_recognition
import os
import glob

images_encoding=[]
images_names=[]

print("\n🔃 Loading Image Database 🔃")
for i in glob.iglob('imgfolder/*'):
    img = cv2.imread(i)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images_encoding.append(face_recognition.face_encodings(rgb_img)[0])
    person_name = os.path.basename(i)
    person_name= os.path.splitext(person_name)
    images_names.append(person_name[0])
print("\n📜 DataBase Loaded Successfully 📜")
print("\n⏳ Recognizing Faces in Realtime, Press 'X' to Exit ⏳\n")

result = face_recognition.compare_faces(images_encoding, images_encoding[1])

vidcap = cv2.VideoCapture(0)

while True:
    ret, frameflip = vidcap.read()
    frame = cv2.flip(frameflip,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    curr_face_locations = face_recognition.face_locations(rgb_frame)
    detected_face_locations = np.array(curr_face_locations)
    curr_face_encodings = face_recognition.face_encodings(rgb_frame,curr_face_locations)

    detected_face_names = []

    for i in curr_face_encodings:
        curr_name = "Kon hai ye?"
        matches = face_recognition.compare_faces(images_encoding, i)
        face_distances = face_recognition.face_distance(images_encoding, i)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            curr_name = images_names[best_match_index]
        detected_face_names.append(curr_name)

    for floc, fname in zip(detected_face_locations, detected_face_names):
        y1, x2, y2, x1 = floc[0], floc[1], floc[2], floc[3]
        if fname == "Kon hai ye?":
            cv2.putText(frame,fname,(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200),2)
        else:
            cv2.putText(frame, "Ye toh "+fname+" hai",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200),2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

    cv2.imshow('Realtime Face Detector', frame)
    if cv2.waitKey(1) == ord('x'):
        print("✅ Detection Completed ✅\n")
        break

vidcap.release()
cv2.destroyAllWindows()
