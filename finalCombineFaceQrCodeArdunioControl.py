import pyttsx3;
import face_recognition
import cv2
import math
import os
import os.path
import pickle
import argparse
import serial
import time

from PIL import Image, ImageDraw
from sklearn import neighbors
from face_recognition.face_recognition_cli import image_files_in_folder
from pyzbar import pyzbar
from sklearn import neighbors


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    #if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #    raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)


    X_img = X_img_path
    X_face_locations = face_recognition.face_locations(X_img)

    if len(X_face_locations) == 0:
        return []
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()
	
fileNameCounter = 0;
while os.path.exists("unknowImages/%s.jpg" % fileNameCounter):
    fileNameCounter += 1
engine = pyttsx3.init();
arduino = serial.Serial('COM11', 9600, timeout=.1)
currentTime = time.time()
oldTime = currentTime

authenticatedPerson = {'suraj1', 'Aakanksha', 'ankita'}
authenticatedCar = {'myCar1', 'myCar2', 'myCar3'}
myTick = False
if __name__ == "__main__":
	print("Training KNN classifier...")
	classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=6)
	print("Training complete!")

	video_capture = cv2.VideoCapture(0)
	while True:        
		ret, frame = video_capture.read()
		image = frame
		frame = cv2.flip( frame, 1)
		rgb_frame = frame[:, :, ::-1]
		predictions = predict(rgb_frame, model_path="trained_knn_model.clf")
		currentTime = time.time()
		print(currentTime - oldTime)
		barcodes = pyzbar.decode(image)
		for barcode in barcodes:
			(x, y, w, h) = barcode.rect
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
			barcodeData = barcode.data.decode("utf-8")
			barcodeType = barcode.type
			print(barcodeData)
			if currentTime - oldTime > 5 :
				myTick = True;
				for carFromList in authenticatedCar:
					if barcodeData == carFromList:
						time.sleep(0.4)
						arduino.write(b'fo\n') 
						oldTime = time.time()
						myTick = False
				if myTick :
					engine.say(barcodeData + " is not permited ");
					engine.runAndWait();
				myTick = True;
			text = "{} ({})".format(barcodeData, barcodeType)
			cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		frame = image
		for name, (top, right, bottom, left) in predictions:
			left 	= left-10
			top 	= top-50
			right 	= right+10
			bottom 	= bottom+10
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
			if name =='unknown':
				fileName = 'unknowImages/'+str(fileNameCounter)+'.jpg'
				fileNameCounter +=1 
				roi = frame[top:bottom, left:right]
				cv2.imwrite(fileName, roi)
				print('Unknown images Found,saved {}'.format(fileName) )
			else :
				print("- Found {} at ({}, {})".format(name, left, top))
				if currentTime - oldTime > 5:
					myTick = True;
					for personFromList in authenticatedPerson:
						if name == personFromList:
							time.sleep(0.4)
							arduino.write(b'ho\n') 
							oldTime = time.time()
							myTick = False
					if myTick:		
						engine.say(name + " is not permited ");
						engine.runAndWait();
					myTick = True;
		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()