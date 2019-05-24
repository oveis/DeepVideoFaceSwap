import os
import cv2 as cv
import numpy as np
from google_images_download import google_images_download as gid
from os.path import exists, isfile, join


FRONT_FACE_CASCADE = cv.CascadeClassifier('scripts/haarcascades/haarcascade_frontalface_default.xml')
PROFILE_FACE_CASCADE = cv.CascadeClassifier('scripts/haarcascades/haarcascade_profileface.xml')

# TODO: Need a function to put images in S3 bucket.
# TODO: Retrieve face images from a given video file.

class FaceImageCollector():
    """ Face photo collector """        
    
    def collect(self, keywords, output_dir, limit=50, face_img_shape=(64, 64)):
        self.face_img_shape = face_img_shape
        images_dir = join(output_dir, 'images')
        faces_dir = join(output_dir, 'faces')

        # Images are downloaded in 'images_dir/<keywords>'.
        self._download_images_from_google(keywords, images_dir, limit)
        
        # Extract faces from images.
        self._detect_and_save_faces(join(images_dir, keywords), join(faces_dir, keywords))
        
        
    def _download_images_from_google(self, keywords, output_dir, limit):
        self._check_dir_path(output_dir)
            
        downloader = gid.googleimagesdownload()
        downloader.download({
            'keywords': keywords,
            "limit": limit,
            'output_directory': output_dir
        })


    def _save_faces(self, img, faces, output_dir, file_id):
        self._check_dir_path(output_dir)

        for i in range(len(faces)):
            x, y, w, h = faces[i]
            face_img = img[y:y+h, x:x+w]
            output_file_path = join(output_dir, '{}_{}.jpeg'.format(file_id, i))
            
            print(output_file_path)
            face_img = cv.resize(face_img, self.face_img_shape)
            
            cv.imwrite(output_file_path, face_img)
            

    def _detect_and_save_faces(self, images_dir, faces_dir):
        self._check_dir_path(images_dir)
        self._check_dir_path(faces_dir)
            
        file_names = [f for f in os.listdir(images_dir) if isfile(join(images_dir, f))]

        for file_name in file_names:
            file_id = file_name.split('.')[0]
            img = cv.imread(join(images_dir, file_name))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            frontal_faces = FRONT_FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
            self._save_faces(img, frontal_faces, join(faces_dir, 'frontal'), file_id)

            profile_faces = PROFILE_FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
            self._save_faces(img, profile_faces, join(faces_dir, 'profile'), file_id)
        
        
    def _check_dir_path(self, dir_path):
        if not exists(dir_path):
            os.makedirs(dir_path)