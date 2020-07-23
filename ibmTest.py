import json
from watson_developer_cloud import VisualRecognitionV3
import os
import sys

class ibmRecognition:
	def __init__(self, imagePath):
		#initialize parameters
		self.visual_recognition = VisualRecognitionV3(
		    '2018-03-19',
		    iam_apikey='Say7-Zq2Z-cv7FEhl8DI7C-0dhUcJ1d0AV3X-s9HJfbc')
		self.classifier_ids = ["food"]
		self.imagePath = imagePath

	def getResult(self):
		with open(self.imagePath, "rb") as images_file:
			return self.visual_recognition.classify(images_file=images_file, classifier_ids=self.classifier_ids).get_result()
		#print(json.dumps(classes_result, indent=2))

ibr = ibmRecognition(sys.argv[1])
result = ibr.getResult()
print(json.dumps(result, indent=2))

