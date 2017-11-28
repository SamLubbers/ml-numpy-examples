import re
from os import path


def parse_document(document):
    """converts document in string format onto a bag of words in list format with only relevant words

    :param document: string of words and other characters
    :return: bag of words (list) with all the relevant words contained in the document
    """
    stopwords = ['for', 'a', 'is', 'and', 'we']

    document = document.lower()
    document = re.sub(r'[^a-zA-Z]', ' ', document) # remove special characters and numbers
    document = re.sub(r' +', ' ', document).strip() # remove extra spaces
    bag_of_words = document.split()

    bag_of_words = [word for word in bag_of_words if word not in stopwords] # remove stopwords
    return bag_of_words

email_dataset = 'email_data'
email_file = path.join(email_dataset, 'spam','1.txt')

mail = open(email_file).read()
parsed_email = parse_document(mail)
print(parsed_email)