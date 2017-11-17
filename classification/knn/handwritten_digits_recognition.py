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
    with open(file_path, 'r') as f:
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