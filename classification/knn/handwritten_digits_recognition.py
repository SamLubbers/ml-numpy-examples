import numpy as np
from os import path, getcwd, listdir

digits_dir = 'digits_data'
train_digits = path.join(getcwd(), digits_dir, 'trainingDigits')
test_digits = path.join(getcwd(), digits_dir, 'testDigits')

def file_to_vector(filename):
    """
    reads all the characters in a file and stores them in a vector of type numpy.ndarray
    :param filename: path to the file from which we want to extract the contents
    :return: numpy.ndarray of one column and n lines, n corresponding to the number of characters in the file
    """
    with open(filename, 'r') as f:
        text = f.read()
    num_characters = sum(len(line.strip()) for line in text)
    return_vector = np.zeros((1,num_characters)) # initialize vector
    # populate vector
    i = 0
    for line in text: 
        line = line.strip()
        for character in line:
            return_vector[0, i] = int(character)
            i += 1
    
    return return_vector

def label_extractor(filename):
    """
    extracts the label associated with a file by reading the filename
    :param filename: name of file from which we want to extract the label
    :return: label associated with the parsed file
    """
    return int(filename.split('_')[0])

def handwritten_digits_test():
    """
    measures the performance of knn when classifying handwritten digits
    """
    labels = []
    training_files = listdir(train_digits)
    num_instances = len(training_files)
    characters_per_instance = 1024
    X_train = np.zeros((num_instances, characters_per_instance))
    for i in range(num_instances):
        labels.append(label_extractor(training_files[i]))
        file_path = path.join(train_digits, training_files[i])
        X_train[i, :] = file_to_vector(file_path)