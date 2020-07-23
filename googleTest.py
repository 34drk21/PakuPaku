import io
import os
import sys

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types


class labelImage:
	def __init__(self, imagePath):
		# Instantiates a client
		self.client = vision.ImageAnnotatorClient()

		# The name of the image file to annotate
		self.imagePath = imagePath

		# Loads the image into memory
		with io.open(imagePath, 'rb') as image_file:
		    content = image_file.read()

		self.image = types.Image(content=content)

	def getLabel(self, client, image):
		# Performs label detection on the image file
		response = self.client.label_detection(image=image)
		return response.label_annotations


#run proglam
li = labelImage(sys.argv[1])
print('Labels:')
labels = li.getLabel(li.client, li.image)
for label in labels:
	print(label.description)
