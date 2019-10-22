import pandas as pd
import ast, math, operator, os, pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


def create_index_inverted(x_data, x_cols):
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
                            #                            print(col_value)
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
        if token in index_inverted:
            value = index_inverted[token]
            if index in value.keys():
                value[index] += 1
            else:
                value[index] = 1
                value["df"] += 1
        else:
            index_inverted[token] = {index: 1, "df": 1}
    # stopwords_1()
def build_doc_vector():
    for token_key in index_inverted:
        token_values = index_inverted[token_key]
        idf = math.log10(N / token_values["df"])
        for doc_key in token_values:
            if doc_key != "df":
                tf_idf = (1 + math.log10(token_values[doc_key])) * idf
                if doc_key not in vector_doc:
                    vector_doc[doc_key] = {token_key: tf_idf, "_sum_": math.pow(tf_idf, 2)}
                else:
                    vector_doc[doc_key][token_key] = tf_idf
                    vector_doc[doc_key]["_sum_"] += math.pow(tf_idf, 2)

    for doc in vector_doc:
        tf_idf_vector = vector_doc[doc]
        normalize = math.sqrt(tf_idf_vector["_sum_"])
        for tf_idf_key in tf_idf_vector:
            tf_idf_vector[tf_idf_key] /= normalize

def get_relevant_docs(query_list):
    relevant_docs = set()
    for query in query_list:
        if query in index_inverted:
            keys = index_inverted[query].keys()
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
        if token in index_inverted:
            # tf_idf = (1 + math.log10(processed_query.count(token))) * math.log10(N/index_inverted[token]["df"])
            tf = (1 + math.log10(processed_query.count(token)))
            tf_vector[token] = tf
            idf = (math.log10(N / index_inverted[token]["df"]))
            idf_vector[token] = idf
            tf_idf = tf * idf
            query_vector[token] = tf_idf
            sum += math.pow(tf_idf, 2)
    sum = math.sqrt(sum)
    for token in query_vector:
        query_vector[token] /= sum
    # print(query_vector[token])
    # print(token)
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
            score_final += query_vector[token] * (vector_doc[doc][token] if token in vector_doc[doc] else 0)

        for token in query_vector:
            score_tf_idf = query_vector[token] * (vector_doc[doc][token] if token in vector_doc[doc] else 0)
            #            print("token: ", token, "Score: ",score_idf)
            score_tf_idf_term[token] = score_tf_idf
            score_tf_idf_term_keys = list(score_tf_idf_term.keys())
            score_tf_idf_term_values = list(score_tf_idf_term.values())

            final_score_tf_idf_term = list(zip(score_tf_idf_term_keys, score_tf_idf_term_values))

        for token in query_vector:
            #            print(idf_vector[token]*(vector_doc[doc][token] if token in vector_doc[doc] else 0))
            score_idf = idf_vector[token] * (vector_doc[doc][token] if token in vector_doc[doc] else 0)
            #            print("token: ", token, "Score: ",score_idf)
            score_idf_term[token] = score_idf
            score_idf_term_keys = list(score_idf_term.keys())
            score_idf_term_values = list(score_idf_term.values())

            final_score_idf_term = list(zip(score_idf_term_keys, score_idf_term_values))

        for token in tf_vector:
            #            print(tf_vector[token])
            score_tf = tf_vector[token] * (vector_doc[doc][token] if token in vector_doc[doc] else 0)
            #            score += (query_vector[token])
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
    global index_inverted, vector_doc
    initialize()
    if os.path.isfile("invertedIndexPickle.pkl"):
        index_inverted = pickle.load(open('invertedIndexPickle.pkl', 'rb'))
        vector_doc = pickle.load(open('documentVectorPickle.pkl', 'rb'))
    else:
        print("In else of get_scores:")
        build()
        store()
    return evaluation(query)

def initialize():
    global csv_files, col_names, noise_list, credits_data, data_meta, N, tokenizer, stopword, stemmer, index_inverted, vector_doc, lemmatizer
    
    # Data Fetch
    csv_files = 'C:/Users/yashd/PycharmProjects/txt_search/'
    #    col_names = {"id": None, "genres":['name'], "original_title":None, "overview":None,"release_date":None,
    #                     "production_companies":['name'], "tagline":None}
    col_names = {"id": None,"original_title": None, "overview": None, "release_date":None}
    noise_list = ['(voice)', '(uncredited)']

    data_meta = pd.read_csv(csv_files + 'movies_metadata.csv', usecols=col_names.keys(), index_col="id")
    # Total number of documents = number of rows in movies_metadata.csv
    data_meta = data_meta.dropna(subset = ["overview"])
    N = data_meta.shape[0]

    # Pre-processing initialization
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    stopword = stopwords.words('english')
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed_data = 10
    lemetized_data=stemmed_data = 10
    lemetized_data=lemetization(stemmed_data)

    index_inverted = {}
    vector_doc = {}

def build():
    print("Creating inverted index for meta data...")
    create_index_inverted(data_meta, col_names)
    print("Building doc vector...")
    build_doc_vector()
    lemetized_data=stemmed_data = 10
    lemetized_data=lemetization(stemmed_data)
    print("Built index and doc vector")

def store():
    pickle.dump(index_inverted, open('invertedIndexPickle.pkl', 'wb+'))
    pickle.dump(vector_doc, open('documentVectorPickle.pkl', 'wb+'))
    print("stored both")

def evaluation(query):
    processed_query = pre_processing(query)
    relevant_docs = get_relevant_docs(processed_query)
    query_vector, idf_vector, tf_vector = build_query_vector(processed_query)
    sorted_score_list, tf_new, idf_new, tf_idf_new = cosine_similarity(relevant_docs, query_vector, idf_vector, tf_vector, processed_query)
    search_result = get_movie_info(sorted_score_list, tf_new, idf_new, tf_idf_new)
    lemetized_data=stemmed_data = 10
    lemetized_data=lemetization(stemmed_data)

    #print(search_result[0:5])
    return search_result, processed_query

def get_movie_info(sorted_score_list, tf_new, idf_new, tf_idf_new):
    result = []
    for entry in sorted_score_list:
        doc_id = entry[0]
#        print(type(doc_id))
#        if type(doc_id) == str:
        row = data_meta.loc[doc_id]
        info = (row["original_title"],
                row["overview"] if isinstance(row["overview"], str) else "", entry[1], idf_new[doc_id], tf_new[doc_id], tf_idf_new[doc_id], row["release_date"])
#        else:
#            continue
        result.append(info)

#    print(result[0:5])
    new_score = None
    print(result[0:5])
    return result


