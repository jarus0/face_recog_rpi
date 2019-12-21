print ('libraries importing starting')

from PIL import Image
import os
import cv2
import numpy
import face_recognition
print ('imported all libraries')

destination = '/home/pi/face_recog_rpi/unknowImages'
find_dest = '/home/pi/face_recog_rpi/findedFaces/'
images = []
arr = os.listdir(destination)
for jImage_str in arr:
	image1 = destination + '/' + jImage_str
	images.append(face_recognition.load_image_file(image1))

counter = 0
for image in images:
	face_locations = face_recognition.face_locations(image)
	for face_location in face_locations:
		top, right, bottom, left = face_location
		face_image = image[top:bottom, left:right]

		pil_image = face_image
		opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
		cv2.imwrite( find_dest + str(counter) + ".jpg", opencvImage );
		counter = counter + 1
