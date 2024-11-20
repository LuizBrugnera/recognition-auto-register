import cv2
import os

if not os.path.exists("faces"):
    os.makedirs("faces")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível acessar a câmera.")
else:
    while True:
        ret, frame = cap.read()
        
        cv2.imshow('Pressione ESPAÇO para capturar ou ESC para sair', frame)
        key = cv2.waitKey(1) & 0xFF
        

        if key == ord(' '):  # Espaço
            file_name = input("Digite o nome do arquivo para salvar a foto: ") + ".jpg"
            file_path = os.path.join("faces", file_name)
            
            cv2.imwrite(file_path, frame)
            print(f"Foto salva como {file_path}")
        
        elif key == 27:  # Esc
            break

cap.release()
cv2.destroyAllWindows()
