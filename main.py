import face_recognition
import cv2
import numpy as np
import os
import threading
import time
import uuid

def face_confidence(face_distance, threshold=0.6):
    """
    Converte a distância facial em uma porcentagem de confiança.
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
        Carrega e codifica os rostos conhecidos a partir da pasta `faces`.
        """
        faces_path = "faces"
        if not os.path.exists(faces_path):
            print(f"Pasta '{faces_path}' não encontrada. Certifique-se de que ela exista.")
            return

        for image_name in os.listdir(faces_path):
            image_path = os.path.join(faces_path, image_name)
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if not face_encodings:
                    print(f"Nenhum rosto detectado em {image_name}. Pulando...")
                    continue

                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(os.path.splitext(image_name)[0])
            except Exception as e:
                print(f"Erro ao processar a imagem {image_name}: {e}")

        print(f"Rostos conhecidos carregados: {self.known_face_names}")

    def save_new_face(self, frame, location):
        """
        Salva a imagem do rosto desconhecido na pasta `faces`.
        """
        # Mapeia as coordenadas do rosto para o tamanho original
        top, right, bottom, left = location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Gera um nome único para o arquivo
        filename = f"new_face_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join("faces", filename)

        # Extrai a região do rosto no tamanho original
        face_image = frame[top:bottom, left:right]

        # Salva o rosto na pasta `faces`
        cv2.imwrite(filepath, face_image)
        print(f"Rosto desconhecido salvo como {filename}.")

    def train_faces(self):
        """
        Realiza o treinamento sempre que novos dados são adicionados.
        """

        while True:
            print("Aguardando novos dados para treinar...")
            self.new_data_event.wait()  # Espera até que novos dados sejam sinalizados
            self.new_data_event.clear()

            print("Treinando com novos dados...")
            self.load_known_faces()  # Recarrega os rostos conhecidos
            time.sleep(1)  # Simula tempo de processamento
  
        
    def trainer_faces(self):
        """
        Detecta e reconhece rostos em tempo real usando a câmera 0.
        """
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print(f"Erro ao acessar a câmera {0}.")
            return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print(f"Erro ao capturar o frame da câmera {0}.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            rgb_small_frame = np.array(rgb_small_frame, dtype=np.uint8)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            try:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            except Exception as e:
                print(f"Erro ao calcular encodings: {e}")
                continue

            for face_encoding, location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        print(f"Rosto reconhecido: {self.known_face_names[best_match_index]}")
                    else:
                        print("Rosto desconhecido detectado.")
                        # Salvar o rosto desconhecido
                        self.save_new_face(frame, location)

                        # Sinalizar que há novos dados
                        #self.new_data_event.set()

            cv2.imshow(f"Camera {0} - Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        
    def recognize_faces(self):
        """
        Detecta e reconhece rostos em tempo real usando a câmera especificada.
        """
        video_capture = cv2.VideoCapture(1)

        if not video_capture.isOpened():
            print(f"Erro ao acessar a câmera {1}.")
            return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print(f"Erro ao capturar o frame da câmera {1}.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            rgb_small_frame = np.array(rgb_small_frame, dtype=np.uint8)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            try:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            except Exception as e:
                print(f"Erro ao calcular encodings: {e}")
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
                        face_names.append("Desconhecido (0%)")
                else:
                    face_names.append("Desconhecido (0%)")

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
                break

        video_capture.release()
        cv2.destroyAllWindows()

def start_recognition(fr):
    fr.recognize_faces()
    
def start_trainer(fr):
    fr.trainer_faces()
    
def train_faces(fr):    
    fr.train_faces()

if __name__ == "__main__":
    # Criar threads para duas câmeras
    fr = FaceRecognition()
    
    thread1 = threading.Thread(target=start_recognition, args=(fr,))
    thread2 = threading.Thread(target=start_trainer, args=(fr,))
    thread3 = threading.Thread(target=train_faces, args=(fr,))

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()
