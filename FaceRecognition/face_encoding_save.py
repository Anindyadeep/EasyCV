import os
import cv2 
import pickle
import numpy as np 
from numpy import save 
from tqdm import tqdm 
from pathlib import Path
import face_recognition as fr 

class FaceEncoding(object):
    __BASEDIR = Path(__file__).resolve(strict=True).parent.parent

    def __init__(self):
        '''
        This will make the encoding of the corresponding faces present in the 
        registered face folder
        '''
    
    def get_encodings_per_face(self, face_name, save_encodings = True):
        '''
        Parameters:
        -----------
        save_encodings : This will save the face encodings on the specified file
        '''
        registered_face_path = os.path.join(FaceEncoding.__BASEDIR, f'registered_faces/{face_name}')
        registered_encoding_path = os.path.join(FaceEncoding.__BASEDIR, f'registered_face_encodings')

        if not os.path.isdir(registered_face_path):
            raise Exception (f"There is no such registered face named {face_name}")

        face_encodings = []
        for img_file in tqdm(os.listdir(registered_face_path), total=len(os.listdir(registered_face_path)), desc=f"Starting to encode the registered face images of {face_name}"):
            image = fr.load_image_file(os.path.join(registered_face_path, img_file))
            encodings = fr.face_encodings(image)
            if len(encodings) == 1:
                face_encodings.append(encodings[0])
        
        face_encodings = np.array(face_encodings)
        if save_encodings:
            file_name = os.path.join(registered_encoding_path, f'{face_name}.npy')
            save(file_name, face_encodings)
            print('Encoding saved successfully ...')
        return face_encodings
    
    def get_encodings_for_all_faces(self):
        registered_face_folders = os.path.join(FaceEncoding.__BASEDIR, f'registered_faces')
        for face in os.listdir(registered_face_folders):
            face_encodings = self.get_encodings_per_face(face)
        print("Registered all the faces ...")
    
    def remove_face_encoding(self, face_name):
        registered_face_encodings_folder = os.path.join(FaceEncoding.__BASEDIR, f'registered_face_encodings')
        os.remove(os.path.join(registered_face_encodings_folder, f'{face_name}.pkl'))
        print("Registered fce and its encoding removed successfully ...")



if __name__ == '__main__':
    fe = FaceEncoding()
    #fe.get_encodings_per_face('Anindya')
    fe.get_encodings_for_all_faces()