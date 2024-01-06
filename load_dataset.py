# Βιβλιοθήκες
from tensorflow.keras.datasets import mnist
import numpy as np

def load_az_dataset(datasetPath):
	# λίστα με data και labels
	data = []
	labels = []
 
	# περνά σε κάθε στήλη του αρχείου
	for row in open(datasetPath):
		#διαχωρίζει labels από images
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")
  
		# μετατρέπει την εικόνα από λίστα με 784 στοιχεία(αναπαριστούν το grascale)
		# σε 28*28 πίνακα
		 
		image = image.reshape((28, 28))
  
		# προσθέτει images και labels στις αντίστοιχες λίστες
		data.append(image)
		labels.append(label)
  
  # μεττρέπει τα data kai τα labels σε NumPy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels, dtype="int")
 
	# επιστρέφει tuple με data και labels
	return (data, labels)

def load_mnist_dataset():
	# φπρτώνει το MNIST dataset και ενώνει τα test data και train data
	#και τα test labels με τα train labels
	
	((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
	data = np.vstack([trainData, testData])
	labels = np.hstack([trainLabels, testLabels])
 
	# επιστρέφει tuplei με data και labels
	return (data, labels)

