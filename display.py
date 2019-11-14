import pandas as pd
import ast, math, operator, os, pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


def create_inverted_index(x_data, x_cols):
    for row in x_data.itertuples():
        index = getattr(row, 'Index')
        data = []
        for col in x_cols.keys():
            if col != "id":
                col_values = getattr(row, col)
                parameters = x_cols[col]
                if parameters is None:
                    data.append(col_values if isinstance(col_values, str) else "")
                else:
                    col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
                    if type(col_values)==bool:
                        continue
                    else:
                        for col_value in col_values:
                            for param in parameters:
                                data.append(col_value[param])
            insert(index, pre_processing(' '.join(data)))


def lemetization(stemmed_data):
    for sentence in range(stemmed_data):
        lemetized_data=+sentence

def pre_processing(data_string):
    tokens = tokenizer.tokenize(data_string)
    processed_data = []
    for t in tokens:
        if t not in stopword:
            processed_data.append(lemmatizer.lemmatize(t).lower())
    return processed_data

def insert(index, tokens):
    for token in tokens:
        if token in inverted_index:
            value = inverted_index[token]
            if index in value.keys():
                value[index] += 1
            else:
                value[index] = 1
                value["df"] += 1
        else:
            inverted_index[token] = {index: 1, "df": 1}
def build_doc_vector():
    for token_key in inverted_index:
        token_values = inverted_index[token_key]
        idf = math.log10(N / token_values["df"])
        for doc_key in token_values:
            if doc_key != "df":
                tf_idf = (1 + math.log10(token_values[doc_key])) * idf
                if doc_key not in document_vector:
                    document_vector[doc_key] = {token_key: tf_idf, "_sum_": math.pow(tf_idf, 2)}
                else:
                    document_vector[doc_key][token_key] = tf_idf
                    document_vector[doc_key]["_sum_"] += math.pow(tf_idf, 2)

    for doc in document_vector:
        tf_idf_vector = document_vector[doc]
        normalize = math.sqrt(tf_idf_vector["_sum_"])
        for tf_idf_key in tf_idf_vector:
            tf_idf_vector[tf_idf_key] /= normalize

def get_relevant_docs(query_list):
    relevant_docs = set()
    for query in query_list:
        if query in inverted_index:
            keys = inverted_index[query].keys()
            for key in keys:
                relevant_docs.add(key)
    if "df" in relevant_docs:
        relevant_docs.remove("df")
    print(relevant_docs)
    return relevant_docs

def build_query_vector(processed_query):
    query_vector = {}
    tf_vector = {}
    idf_vector = {}
    sum = 0
    for token in processed_query:
        if token in inverted_index:
            # tf_idf = (1 + math.log10(processed_query.count(token))) * math.log10(N/inverted_index[token]["df"])
            tf = (1 + math.log10(processed_query.count(token)))
            tf_vector[token] = tf
            idf = (math.log10(N / inverted_index[token]["df"]))
            idf_vector[token] = idf
            tf_idf = tf * idf
            query_vector[token] = tf_idf
            sum += math.pow(tf_idf, 2)
    sum = math.sqrt(sum)
    for token in query_vector:
        query_vector[token] /= sum
    return query_vector, idf_vector, tf_vector

def cosine_similarity(relevant_docs, query_vector, idf_vector, tf_vector, processed_query):
    score_map_final = {}
    score_map_idf = {}
    score_map_tf = {}
    score_idf_term = {}
    idf_term_new = {}
    score_tf_term = {}
    tf_term_new = {}
    score_tf_idf_term = {}
    tf_idf_term_new = {}
    for doc in relevant_docs:
        score_final = 0
        score_idf = 0
        score_tf = 0
        score_tf_idf = 0
        for token in query_vector:
            score_final += query_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)

        for token in query_vector:
            score_tf_idf = query_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
            score_tf_idf_term[token] = score_tf_idf
            score_tf_idf_term_keys = list(score_tf_idf_term.keys())
            score_tf_idf_term_values = list(score_tf_idf_term.values())

            final_score_tf_idf_term = list(zip(score_tf_idf_term_keys, score_tf_idf_term_values))

        for token in query_vector:
            score_idf = idf_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
            score_idf_term[token] = score_idf
            score_idf_term_keys = list(score_idf_term.keys())
            score_idf_term_values = list(score_idf_term.values())

            final_score_idf_term = list(zip(score_idf_term_keys, score_idf_term_values))

        for token in tf_vector:
            score_tf = tf_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
            score_tf_term[token] = score_tf
            score_tf_term_keys = list(score_tf_term.keys())
            score_tf_term_values = list(score_tf_term.values())

            final_score_tf_term = list(zip(score_tf_term_keys, score_tf_term_values))

        score_map_final[doc] = score_final
        score_map_idf[doc] = score_idf
        score_map_tf[doc] = score_tf

        idf_term_new[doc] = final_score_idf_term
        tf_term_new[doc] = final_score_tf_term
        tf_idf_term_new[doc] = final_score_tf_idf_term
    sorted_score_map_final = sorted(score_map_final.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_score_map_final[:50], tf_term_new, idf_term_new, tf_idf_term_new

def get_results(query):
    global inverted_index, document_vector
    initialize()
    if os.path.isfile("invertedIndexPickle.pkl"):
        inverted_index = pickle.load(open('invertedIndexPickle.pkl', 'rb'))
        document_vector = pickle.load(open('documentVectorPickle.pkl', 'rb'))
    else:
        print("In else of get_scores:")
        build()
        save()
    return eval_score(query)

def initialize():
    global data_folder, credits_cols, meta_cols, noise_list, credits_data, meta_data, N, tokenizer, stopword, stemmer, inverted_index, document_vector, lemmatizer

    # Data Fetch
    # data_folder = 'C:/Users/yashd/PycharmProjects/txt_search/'
    meta_cols = {"id": None,"original_title": None, "overview": None, "release_date":None}
    noise_list = ['(voice)', '(uncredited)']

    meta_data = pd.read_csv('movies_metadata.csv', usecols=meta_cols.keys(), index_col="id")
    meta_data = meta_data.dropna(subset = ["overview"])
    N = meta_data.shape[0]

    # Pre-processing
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    stopword = stopwords.words('english')
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed_data = 10
    lemetized_data=stemmed_data = 10
    lemetized_data=lemetization(stemmed_data)

    inverted_index = {}
    document_vector = {}

def build():
    print("Creating inverted index for meta data...")
    create_inverted_index(meta_data, meta_cols)
    print("Building doc vector...")
    build_doc_vector()
    lemetized_data=stemmed_data = 10
    lemetized_data=lemetization(stemmed_data)
    print("Built index and doc vector")

def save():
    pickle.dump(inverted_index, open('invertedIndexPickle.pkl', 'wb+'))
    pickle.dump(document_vector, open('documentVectorPickle.pkl', 'wb+'))
    print("Saved both")

def eval_score(query):
    processed_query = pre_processing(query)
    relevant_docs = get_relevant_docs(processed_query)
    query_vector, idf_vector, tf_vector = build_query_vector(processed_query)
    sorted_score_list, tf_new, idf_new, tf_idf_new = cosine_similarity(relevant_docs, query_vector, idf_vector, tf_vector, processed_query)
    search_result = get_movie_info(sorted_score_list, tf_new, idf_new, tf_idf_new)
    lemetized_data=stemmed_data = 10
    lemetized_data=lemetization(stemmed_data)

    return search_result, processed_query

def get_movie_info(sorted_score_list, tf_new, idf_new, tf_idf_new):
    result = []
    for entry in sorted_score_list:
        doc_id = entry[0]
        row = meta_data.loc[doc_id]
        info = (row["original_title"],
                row["overview"] if isinstance(row["overview"], str) else "", entry[1], idf_new[doc_id], tf_new[doc_id], tf_idf_new[doc_id], row["release_date"])

        result.append(info)

    new_score = None
    print(result[0:5])
    return result