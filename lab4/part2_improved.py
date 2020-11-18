from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
import numpy as np

# loading all files.
moviedir = os.getcwd() + r'\movie_reviews'
movie = load_files(moviedir, shuffle=True)

# Split data into training and test sets
docs_train_data, docs_test_data, docs_train_target, docs_test_target = train_test_split(movie.data, movie.target,
                                                          test_size=0.10, random_state=84)

# # use all 25K words. Higher accuracy
# movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)

# # Training a Multimoda Naive Bayes classifier.
# review_clf = Pipeline([
#     ('vect', CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)),
#     ('tfidf', TfidfTransformer()),
#     ('clf', MultinomialNB()),
# ])
# review_clf.fit(docs_train_data, docs_train_target)
# predicted = review_clf.predict(docs_test_data)
# print("multinomialBC accuracy ", np.mean(predicted == docs_test_target))

# Training SVM classifier
review_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=1e-5, random_state=84, max_iter=1000, tol=None)),
])
review_clf.fit(docs_train_data, docs_train_target)
predicted = review_clf.predict(docs_test_data)
print("SVM accuracy ", np.mean(predicted == docs_test_target))

# # Training Logistic Regression classifier - NO GOOD
# review_clf = Pipeline([
#  ('vect', CountVectorizer()),
#  ('tfidf', TfidfTransformer()),
#  ('clf', LogisticRegression()),
# ])
# review_clf.fit(docs_train_data, docs_train_target)
# predicted = review_clf.predict(docs_test_data)
# print("Logistic Regression accuracy ", np.mean(predicted == docs_test_target))

print(metrics.classification_report(docs_test_target, predicted, target_names=movie.target_names))

# print(metrics.confusion_matrix(docs_test_target, predicted))

# =================== Grid Search ===================
parameters = {
 'vect__ngram_range': [(1, 1), (1, 2)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(review_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(docs_train_data[:400], docs_train_target[:400])

# print(gs_clf.best_score_)

print(movie.target_names[gs_clf.predict(['Victor was excellent!'])[0]])

# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride',
               'Steven Seagal was terrible', 'Steven Seagal shone through.',
               'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
               "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
               'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

# have classifier make a prediction
pred = gs_clf.predict(reviews_new)
# print out results
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))
