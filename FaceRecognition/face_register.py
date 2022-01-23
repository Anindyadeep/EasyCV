import os
import cv2 
import pyttsx3
import numpy as np
from pathlib import Path 
from face_encoding_save import FaceEncoding

class FaceRegister(object):
    __BASEDIR = Path(__file__).resolve(strict=True).parent.parent

    def __init__(self, name_of_face_to_register):
        '''
        This class will help to register 11 different positions of the face
        into a single folder and those faces, will be used in generating the 
        encodings for more robust face recognition
        '''
        self.face_name = name_of_face_to_register
        self.__fe = FaceEncoding()
    
    def register_face(self, take_positions = 11, interval = 5, use_font = True):
        '''
        Parameters:
        ----------
        take_positions : How many different types of positions are to be taken by the camera
                        *NOTE* By default, only 11 positions are set and 11 times just the TTS
                        engine will speak, if tweaked this parameter, then after 11th time, its
                        just silent face position intake by the user.

        interval       : For how many seconds, each position is taken, default is 5 seconds
                         This interval is for the user to get time for transitioning from each
                         positions.

        use_font       : Along with the voice, font will also be written on the camera screen
        '''
        folder_path = os.path.join(FaceRegister.__BASEDIR, f'registered_faces/{self.face_name}')
        if os.path.isdir(folder_path) and len(os.listdir(folder_path)) > 10:
            raise Exception ("Folder Already exists")
        
        if os.path.isdir(folder_path) and len(os.listdir(folder_path)) < 10:
            os.rmdir(folder_path)

        os.mkdir(folder_path)
        
        engine = pyttsx3.init()
        engine.setProperty('rate', 120)

        video_capture  = cv2.VideoCapture(0)
        print('Camera starting ....')

        fps_tracker = 0
        start_counter = 0

        finished = take_positions
        positions = [
            'Face at the center.',
            'Tilt over right',
            'Tilt over left.',
            'Gently move the face towards right',
            'Now come to center.',
            'Now move the face towards left ',
            'Center and expresss something.',
            'Now move upwards.',
            'Now move downwards.',
            'Come to center',
            'You are all set'
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        done_asking_for_position = False 
        while True:
            ret, frame = video_capture.read()
            if not done_asking_for_position:
                engine.say(positions[start_counter])
                engine.runAndWait()
                done_asking_for_position = True 
            
            if ret == True:
                if use_font:
                    cv2.putText(
                        frame,
                        positions[start_counter],
                        (30, 50),
                        font, 1,
                        (0, 255, 255),
                        2,
                        cv2.LINE_4
                    )
                
                cv2.imshow('Video', frame)
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                if fps_tracker % (fps * interval) == 0:
                    cv2.imwrite(f"{folder_path}/movement_{fps_tracker}.jpeg", frame)
                    start_counter += 1
                    done_asking_for_position = False 
                fps_tracker += 1
            else:
                break 

            if start_counter == finished:
                break 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
        video_capture.release()
        cv2.destroyAllWindows()
        print("Face registration completed ...")
    
    def register_face_and_encode(self):
        '''
        This will register the face as well as its corresponding encoding in the Database
        '''
        face_folder_path = os.path.join(FaceRegister.__BASEDIR, f'registered_faces/{self.face_name}')
        if not os.path.isdir(face_folder_path):
            self.register_face()
        self.__fe.get_encodings_per_face(self.face_name)



    def remove_registered_face(self):
        """
        This will remove the registered face from the database
        """
        # do not forget to remove the encoding at the same time
        face_folder_path = os.path.join(FaceRegister.__BASEDIR, f'registered_faces/{self.face_name}')
        os.rmdir(face_folder_path)
        self.__fe.remove_face_encoding(self.face_name)
        
if __name__ == '__main__':
    fr = FaceRegister('Anindya')
    fr.register_face_and_encode()
