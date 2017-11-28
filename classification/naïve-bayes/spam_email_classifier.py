import re
from os import path, listdir
import random
from math import floor

def parse_document(document):
    """converts document in string format onto a bag of words in list format with only relevant words

    :param document: string of words and other characters
    :return: bag of words (list) with all the relevant words contained in the document
    """
    stopwords = ['for', 'and', 'you', 'your']

    document = document.lower()
    document = re.sub(r'[^a-zA-Z]', ' ', document) # remove special characters and numbers
    document = re.sub(r' +', ' ', document).strip() # remove extra spaces
    bag_of_words = document.split()

    bag_of_words = [word for word in bag_of_words if word not in stopwords and len(word) > 2] # remove stopwords
    return bag_of_words

# load email data
email_dataset = 'email_data'
spam_directory = path.join(email_dataset, 'spam')
non_spam_directory = path.join(email_dataset, 'ham')

spam_files = [file for file in listdir(spam_directory)]
non_spam_files = [file for file in listdir(non_spam_directory)]

emails = []
labels = []

for file in spam_files:
    parsed_email = parse_document(open(path.join(spam_directory, file)).read())
    emails.append(parsed_email)
    labels.append(1)


for file in non_spam_files:
    parsed_email = parse_document(open(path.join(non_spam_directory, file)).read())
    emails.append(parsed_email)
    labels.append(0)

# convert bag of words to vectors
from word_embedding import create_vocabulary, word2vector

vocabulary = create_vocabulary(emails)
X = [word2vector(vocabulary, email) for email in emails]

# TODO train test split

# TODO classify test sets and calcualte error rate
