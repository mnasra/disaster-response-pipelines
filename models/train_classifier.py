import sys
import pandas as pd
import sqlalchemy
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """Loads data from SQLite database and splits to X and Y dataframes

    Keyword arguments:
    database_filepath - string containing file path for SQLite databse
    
    Returns:
    X - pandas dataframe containing the message posted by users
    Y - pandas dataframe containing binary values that indicate whether the text
        is part of that category
    category_names - array of strings that contain the names of the categories
    """
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Message', engine)
    X = df['message']
    Y = df.iloc[:,5:] #Skip genre and add later after one hot encoding
    Y_dummies = pd.get_dummies(df[['genre']].astype(str))
    Y = Y.merge(Y_dummies, how='inner',left_index=True,right_index=True, validate='1:1')
    Y['related'] = Y['related'].apply(lambda x: 1 if x==2 else x) #Change 2s into 1s so this is binary
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """Takes a piece of text and does the following:
        - Removes stop words
        - Lemmatizes words
        - Removes punctuation
        - Lower cases all words
        - Tokenizes words

    Keyword arguments:
    text - string containing piece of text
    
    Returns:
    tokens - array containing tokens associated with that text
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """Builds a machine learning pipeline and sets up grid search ready to be trained

    Returns:
    model - GridSearchCV object ready for training on training data
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range':[(1,1),(1,2)],
        'tfidf__use_idf': (True, False), 
        'clf__estimator__p':[1,2]
    }

    model = GridSearchCV(pipeline, parameters)
    return model

def evaluate_model(model, X_test, y_test, category_names):
    """ Evaluates the model by printing f1 score, precision, and recall

    Keyword arguments:
    model - model object that was used to make predictions
    X_test - Pandas dataframe containing testing data from the X dataframe
    y_test - Pandas dataframe containing testing data from the Y dataframe
    category_names - category_names - array of strings that contain the names of the categories 
    """
    y_pred = model.predict(X_test)
    for i in range(y_pred.shape[1]):
        col_name = y_test.columns[i]
        if col_name in category_names:
            y_test_i = y_test.iloc[:,i]
            y_pred_i = [item[i] for item in y_pred]
            print(col_name + ":")
            print(classification_report(y_test_i, y_pred_i))

def save_model(model, model_filepath):
    """ Saves the model in the specified file path using pickle

    Keyword arguments:
    model - model object that was used to make predictions
    model_filepath - String containing location you wish to save model
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()