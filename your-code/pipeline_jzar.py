import pandas as pd
import regex as re


# 1. PREPARE TEXT ------------------------------------------------

def vowel(c):
    if c.lower() in 'aeiou':
        return True
    else:
        return False

def repeated(s):
    '''
    This fuction removes repeated more than 2 times vowels and repeated more than 1 time cons
    '''
    result = ''
    rep2 = ''
    rep = ''
    for c in s:
        if not vowel(c) and (rep != c):
            result += c
        if vowel(c) and ((rep2 != c) or (rep2 == c and not rep == c)):
            result += c
            
        rep2 = rep
        rep = c
    return result


def clean_up(s, min_word_size, max_word_size):
    """
    Cleans up numbers, URLs, and special characters from a string.

    Args:
        s: The string to be cleaned up.

    Returns:
        A string that has been cleaned up.
    """
    #lower
    s1 = s.lower()
    
    # remove ironhack and http
    s2 = re.sub(r'http\S+', ' ',s1)
    
    # remove punctuations
    s3 = re.sub(r'[^A-Za-z]+', ' ',s2)
    
    # remove repetitive chars
    s4 = repeated(s3)

    # remove small words
    s5 = ' '.join(word for word in s4.split() if (len(word)>min_word_size and len(word)<max_word_size))
   
    return s5

def tokenize(sentence):
    """
    Tokenize a string.

    Args:
        s: String to be tokenized.

    Returns:
        A list of words as the result of tokenization.
    """
    wordfreq = {}
    return sentence.split()


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
nltk.download('wordnet')

def stem_and_lemmatize(sentence):
    """
    Perform stemming and lemmatization on a list of words.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after being stemmed and lemmatized.
    """
    return [lemmatizer.lemmatize(word) for word in sentence]

from stop_words import get_stop_words
def remove_stopwords(l, extra_stop_words):
    """
    Remove English stopwords from a list of strings.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after stop words are removed.
    """
    stop_words = get_stop_words('en')
    stop_words += extra_stop_words
    return [word for word in l if word not in stop_words]


def prepare_text(sentence, min_word_size, max_word_size, extra_stop_words):
    sentence_clean = clean_up(sentence, min_word_size, max_word_size)
    tokens = tokenize(sentence_clean)
    stems = stem_and_lemmatize(tokens)
    return remove_stopwords(stems, extra_stop_words)

def prepare_df(df, min_word_size, max_word_size, extra_stop_words):
    df['text_processed'] = df.text.apply(lambda x: prepare_text(x, min_word_size, max_word_size, extra_stop_words))
    return df
  

# 2. GET TOP KEYS ------------------------------------------------------------

from nltk.probability import FreqDist
def get_top_x(simple_sen, list_size):
    flat_list = [item for sublist in simple_sen.text_processed.values for item in sublist]   
    all_words = nltk.FreqDist(flat_list)
    return list(all_words.keys())[:list_size]

# 3. BUILD FEATURES ---------------------------------------------------------------

def find_features(document, word_features):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
              
def build_features(data, top_x):
    return [(find_features(row.text, top_x), row.is_positive) for index, row in data.iterrows()]    
    
# 4. NAIVE-BAYES MODELING -------------------------------------------------------

def naive_bayes_model(featuresets,train_test_rel):
    # Train-test split
    size_f = int(round(len(featuresets)*train_test_rel,0))
    
    testing_set = featuresets[:size_f]
    training_set = featuresets[size_f:]
    
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    
    print("Classifier accuracy:",(nltk.classify.accuracy(classifier, testing_set)))
    
    classifier.show_most_informative_features(20)
    
    return nltk
    
    
# 5. All together ---------------------------------------------------------------

def sentiment_pipeline(df, sample_size, key_words_num, extra_stop_words = [], min_word_size = 3, max_word_size = 10, train_test_rel = 0.3, random_state=42):
    '''
    This function receives text data and developes a sentiment model
    input:
    df -> the dataframe to be analyzed, it requires 2 columns: text and is_positive
    sample_size -> the sample size of the df you want to model
    key_words_num -> number of key words
    extra_stop_words -> Add a list of stopwords to remove
    min_word_size -> min size of the key words 
    max_word_size -> max size of the key words
    train_test_rel -> relation between train and test split
    random_state -> random state
    
    output:
    the model
    
    '''    
    print("1. Sampling...")
    sampled = df.sample(sample_size, random_state=42)
          
    print("2. Preparing...")    
    prepared = prepare_df(sampled, min_word_size, max_word_size, extra_stop_words)
    
    print("4. Getting top",key_words_num)              
    top = get_top_x(prepared, key_words_num)
    
    print("5. Creating featuresets...")    
    featuresets = build_features(sampled, top) 
    
    print("6. Naive Bayes modeling...")
    return naive_bayes_model(featuresets, train_test_rel)
    
    