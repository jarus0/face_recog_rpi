import numpy as np
import cv2
import os
import os.path


cap = cv2.VideoCapture(0)

fileNameCounter = 0
while os.path.exists("unknowImages/%s.jpg" % fileNameCounter):
    fileNameCounter += 1


while(True):
	ret, src = cap.read()
	img90 = np.rot90(src)
	img90 = np.rot90(img90)

	cv2.imshow('frame',img90)
	key = cv2.waitKey(1)
	if key & 0xFF == ord('q'):
		break
	if key == ord(' '):
		fileName = 'unknowImages/'+str(fileNameCounter)+'.jpg'
		fileNameCounter +=1 
		cv2.imwrite(fileName, img90)
		print (fileName + ' saved')
cap.release()
cv2.destroyAllWindows()