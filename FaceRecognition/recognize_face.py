import os 
import cv2
import pickle
import numpy as np
import face_recognition as fr 

# TODO: More optimization of this code part

class FaceRecognition(object):
    def __init__(self, face_encoding_path):
        self.face_encoding_path = face_encoding_path
        self.registered_faces = os.listdir(self.face_encoding_path)
        self.known_face_encodings_for_match = []
        self.known_face_encodings_one_each = []

        for encoding_npy in os.listdir(self.face_encoding_path):
            encoding_path = os.path.join(self.face_encoding_path, encoding_npy)
            self.known_face_encodings_for_match.append(np.load(encoding_path))
        
        for encodings in self.known_face_encodings_for_match:
            self.known_face_encodings_one_each.append(encodings[0])
    
    def __get_encoding_from_npy(self):
        encodings = []
        for file in os.listdir(self.face_encoding_path):
            face_encoded_path = os.path.join(self.face_encoding_path, file)
            array = list(np.load(face_encoded_path))
            encodings.append(array)
        return encodings
    
    def score_match(self, known_encoding : list, encoding_to_match : np.array, tolerance : float = 0.8) -> bool:
        '''
        This will give a score for a specific set of same face encodings to the target encoding
        Parameters:
        ----------
        known_encoding : list of the same face encodings
        encoding_to_match : the encoding we get from the face extracted from the image
        tolerance : the % threshold after which we can say that both person are alike
        '''
        total = len(known_encoding)
        results = fr.compare_faces(known_encoding, encoding_to_match)
        score = sum(results) / total
        if score < tolerance:
            return False 
        else:
            return True 

    def match_faces_through_score(self, known_encodings : list, encoding_to_match : np.array):
        '''
        It will match will all the registered encodings of all the registered faces to target face encoding
        '''
        results = []
        for known_encoding in known_encodings:
            results.append(self.score_match(known_encoding, encoding_to_match))
        return results

    def recognise_faces(self, cam_num = 0):
        cap = cv2.VideoCapture(cam_num)
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True 

        while True:
            ret, frame = cap.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            if process_this_frame:
                face_locations = fr.face_locations(rgb_small_frame)
                face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
                face_names = []

                for face_encoding in face_encodings:
                    matches = self.match_faces_through_score(self.known_face_encodings_for_match, face_encoding)
                    name = "Unknown"
                    face_distances = fr.face_distance(self.known_face_encodings_one_each, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.registered_faces[best_match_index]
                    face_names.append(name)

            process_this_frame = not process_this_frame
            print(face_names)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4 
                right *= 4 
                bottom *= 4 
                left *= 4 

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    
    def recognize_face_per_frame(self, frame, face_locations, face_encodings, face_names, process_this_frame = True):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = fr.face_locations(rgb_small_frame)
            face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
            #face_names = []

            for face_encoding in face_encodings:
                matches = self.match_faces_through_score(self.known_face_encodings_for_match, face_encoding)
                name = "Unknown"
                face_distances = fr.face_distance(self.known_face_encodings_one_each, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.registered_faces[best_match_index]
                face_names.append(name)

        process_this_frame = not process_this_frame
        print(face_locations, face_names)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4 
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        return frame, face_locations, face_encodings, face_names


    def recognize_face_on_cam(self, cam_num = 0):
        cap = cv2.VideoCapture(cam_num) 
        face_locations = []
        face_encodings = []
        face_names = []

        while True:
            ret, frame = cap.read()
            frame, face_locations, face_encodings, face_names = self.recognize_face_per_frame(frame, face_locations, face_encodings, face_names)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
        cv2.destroyAllWindows()


if __name__ == '__main__':
    base_path = os.getcwd()
    face_encoding_path = os.path.join(base_path, 'registered_face_encodings')
    FR = FaceRecognition(face_encoding_path=face_encoding_path)
    #FR.recognise_faces()
    FR.recognize_face_on_cam()
