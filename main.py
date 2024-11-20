import face_recognition
import cv2
import numpy as np
import os

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
                # Carregar imagem
                image = face_recognition.load_image_file(image_path)
                
                # Detectar encodings
                face_encodings = face_recognition.face_encodings(image)
                if not face_encodings:
                    print(f"Nenhum rosto detectado em {image_name}. Pulando...")
                    continue

                # Adicionar o primeiro encoding encontrado
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(os.path.splitext(image_name)[0])
            except Exception as e:
                print(f"Erro ao processar a imagem {image_name}: {e}")

        print(f"Rostos conhecidos carregados: {self.known_face_names}")

    def recognize_faces(self):
        """
        Detecta e reconhece rostos em tempo real usando a webcam.
        """
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("Erro ao acessar a webcam.")
            return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Erro ao capturar o frame.")
                break

            # Reduzir o tamanho do frame para acelerar o processamento
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Garantir que o frame está no formato correto
            rgb_small_frame = np.array(rgb_small_frame, dtype=np.uint8)

            # Localizar rostos e calcular encodings
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if not face_locations:
                print("Nenhum rosto detectado no frame atual.")
                continue

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

            # Exibir os resultados
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Escalar as coordenadas de volta para o tamanho original
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Desenhar um retângulo ao redor do rosto
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # Exibir o frame com os rostos identificados
            cv2.imshow("Video - Face Recognition", frame)

            # Pressione 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    fr = FaceRecognition()
    fr.recognize_faces()
