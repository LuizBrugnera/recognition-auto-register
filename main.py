import face_recognition
import cv2
import numpy as np
import os
import threading
import time
import uuid

def face_confidence(face_distance, threshold=0.6):
    """
    Converts face distance into a confidence percentage.
    """
    if face_distance > threshold:
        return f"0%"
    else:
        confidence = (1 - face_distance / threshold) * 100
        return f"{round(confidence, 2)}%"

class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        self.new_data_event = threading.Event()

    def load_known_faces(self):
        """
        Loads and encodes known faces from the 'faces' folder.
        """
        faces_path = "faces"
        if not os.path.exists(faces_path):
            print(f"Folder '{faces_path}' not found. Please ensure it exists.")
            return

        for image_name in os.listdir(faces_path):
            image_path = os.path.join(faces_path, image_name)
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if not face_encodings:
                    print(f"No face detected in {image_name}. Skipping...")
                    continue

                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(os.path.splitext(image_name)[0])
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

        print(f"Known faces loaded: {self.known_face_names}")

    def save_new_face(self, frame, location):
        """
        Saves the image of an unknown face to the 'faces' folder.
        """
        # Map face coordinates to original size
        top, right, bottom, left = location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Generate a unique filename
        filename = f"new_face_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join("faces", filename)

        # Extract face region in original size
        face_image = frame[top:bottom, left:right]

        # Save the face to the 'faces' folder
        cv2.imwrite(filepath, face_image)
        print(f"Unknown face saved as {filename}.")

    def trainer_faces(self, stop_event):
        """
        Detects and recognizes faces in real-time using camera 0.
        """
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print(f"Error accessing camera {0}.")
            return

        try:
            while not stop_event.is_set():
                ret, frame = video_capture.read()
                if not ret:
                    print(f"Error capturing frame from camera {0}.")
                    break

                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                rgb_small_frame = np.array(rgb_small_frame, dtype=np.uint8)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                try:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                except Exception as e:
                    print(f"Error calculating encodings: {e}")
                    continue

                for face_encoding, location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            print(f"Recognized face: {self.known_face_names[best_match_index]}")
                        else:
                            print("Unknown face detected.")
                            # Save the unknown face
                            self.save_new_face(frame, location)
                            # Set the stop event and exit immediately
                            stop_event.set()
                            return

        finally:
            # Release camera resources
            video_capture.release()
            print(f"Camera {0} released.")

    def recognize_faces(self, stop_event):
        """
        Detects and recognizes faces in real-time using the specified camera.
        """
        video_capture = cv2.VideoCapture(1)

        if not video_capture.isOpened():
            print(f"Error accessing camera {1}.")
            return

        try:
            while not stop_event.is_set():
                ret, frame = video_capture.read()
                if not ret:
                    print(f"Error capturing frame from camera {1}.")
                    break

                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                rgb_small_frame = np.array(rgb_small_frame, dtype=np.uint8)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                try:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                except Exception as e:
                    print(f"Error calculating encodings: {e}")
                    continue

                face_names = []

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])
                            face_names.append(f"{name} ({confidence})")
                        else:
                            face_names.append("Unknown (0%)")
                    else:
                        face_names.append("Unknown (0%)")

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                cv2.imshow(f"Camera {1} - Face Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
        finally:
            video_capture.release()
            print(f"Camera {1} released.")

def start_recognition(fr, stop_event):
    fr.recognize_faces(stop_event)

def start_trainer(fr, stop_event):
    fr.trainer_faces(stop_event)

threads = []
stop_event = threading.Event()

def start_threads(fr):
    global threads, stop_event
    stop_event.clear()  # Reset the stop event

    thread1 = threading.Thread(target=start_recognition, args=(fr, stop_event))
    thread2 = threading.Thread(target=start_trainer, args=(fr, stop_event))

    threads = [thread1, thread2]
    for thread in threads:
        thread.start()

def stop_threads():
    global threads, stop_event
    stop_event.set()  # Signal all threads to stop
    for thread in threads:
        if thread.is_alive():
            thread.join()  # Wait until the thread terminates
    threads = []  # Clear the thread list

    # Call cv2.destroyAllWindows() once after all threads have stopped
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("Starting threads...")
        fr = FaceRecognition()
        start_threads(fr)

        # Wait until the stop event is set
        stop_event.wait()

        print("Stopping threads...")
        stop_threads()

        # Reset the stop event for the next cycle
        stop_event.clear()
        print("Threads successfully restarted.")
