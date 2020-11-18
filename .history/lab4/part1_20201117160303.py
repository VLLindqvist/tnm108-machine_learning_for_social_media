d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1,d2,d3,d4)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

print(vectorizer)
