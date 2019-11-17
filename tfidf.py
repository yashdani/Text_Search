import pandas as pd
import ast, math, operator, os, pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

class tfidf:
    def __init__(self):
        # Data Fetch
        # data_folder = 'C:/Users/yashd/PycharmProjects/txt_search/'
        self.meta_cols = {"id": None, "original_title": None, "overview": None, "release_date": None}
        meta_data = pd.read_csv('movies_metadata.csv', usecols=self.meta_cols.keys(), index_col="id")
        self.meta_data = meta_data.dropna(subset=["overview"])
        self.N = self.meta_data.shape[0]

        # Pre-processing
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
        self.stopword = stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        self.inverted_index = {}
        self.document_vector = {}

        if os.path.isfile("invertedIndexPickle.pkl"):
            self.inverted_index = pickle.load(open('invertedIndexPickle.pkl', 'rb'))
            self.document_vector = pickle.load(open('documentVectorPickle.pkl', 'rb'))
        else:
            print("In else of get_scores:")
            self.build()
            self.save()
    
    def build(self):
        print("Creating inverted index for meta data...")
        self.create_inverted_index()
        print("Building document vector")
        self.build_doc_vector()
        print("Built inverted index and document vector")

    def save(self):
        pickle.dump(self.inverted_index, open('invertedIndexPickle.pkl', 'wb+'))
        pickle.dump(self.document_vector, open('documentVectorPickle.pkl', 'wb+'))
        print("Saved both")

    def create_inverted_index(self):
        for row in self.meta_data.itertuples():
            index = getattr(row, 'Index')
            data = []
            for col in self.meta_cols.keys():
                if col != "id":
                    col_values = getattr(row, col)
                    parameters = self.meta_cols[col]
                    if parameters is None:
                        data.append(col_values if isinstance(col_values, str) else "")
                    else:
                        col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
                        if type(col_values) == bool:
                            continue
                        else:
                            for col_value in col_values:
                                for param in parameters:
                                    data.append(col_value[param])
                self.insert(index, self.pre_processing(' '.join(data)))

    def build_doc_vector(self):
        for token_key in self.inverted_index:
            token_values = self.inverted_index[token_key]
            idf = math.log10(self.N / token_values["df"])
            for doc_key in token_values:
                if doc_key != "df":
                    tf_idf = (1 + math.log10(token_values[doc_key])) * idf
                    if doc_key not in self.document_vector:
                        self.document_vector[doc_key] = {token_key: tf_idf, "_sum_": math.pow(tf_idf, 2)}
                    else:
                        self.document_vector[doc_key][token_key] = tf_idf
                        self.document_vector[doc_key]["_sum_"] += math.pow(tf_idf, 2)

        for doc in self.document_vector:
            tf_idf_vector = self.document_vector[doc]
            normalize = math.sqrt(tf_idf_vector["_sum_"])
            for tf_idf_key in tf_idf_vector:
                tf_idf_vector[tf_idf_key] /= normalize

    def insert(self, index, tokens):
        for token in tokens:
            if token in self.inverted_index:
                value = self.inverted_index[token]
                if index in value.keys():
                    value[index] += 1
                else:
                    value[index] = 1
                    value["df"] += 1
            else:
                self.inverted_index[token] = {index: 1, "df": 1}

    def pre_processing(self, data_string):
        tokens = self.tokenizer.tokenize(data_string)
        processed_data = []
        for t in tokens:
            if t not in self.stopword:
                processed_data.append(self.lemmatizer.lemmatize(t).lower())
        return processed_data

    def get_relevant_docs(self, query_list):
        relevant_docs = set()
        for query in query_list:
            if query in self.inverted_index:
                keys = self.inverted_index[query].keys()
                for key in keys:
                    relevant_docs.add(key)
        if "df" in relevant_docs:
            relevant_docs.remove("df")
        print(relevant_docs)
        return relevant_docs

    def build_query_vector(self, processed_query):
        query_vector = {}
        tf_vector = {}
        idf_vector = {}
        sum = 0
        for token in processed_query:
            if token in self.inverted_index:
                # tf_idf = (1 + math.log10(processed_query.count(token))) * math.log10(N/inverted_index[token]["df"])
                tf = (1 + math.log10(processed_query.count(token)))
                tf_vector[token] = tf
                idf = (math.log10(self.N / self.inverted_index[token]["df"]))
                idf_vector[token] = idf
                tf_idf = tf * idf
                query_vector[token] = tf_idf
                sum += math.pow(tf_idf, 2)
        sum = math.sqrt(sum)
        for token in query_vector:
            query_vector[token] /= sum
        return query_vector, idf_vector, tf_vector

    def similarity(self, relevant_docs, query_vector, idf_vector, tf_vector):
        FinalScore = {}
        IdfScore = {}
        TfScore = {}
        TermIdf = {}
        idf_term_new = {}
        TermTf = {}
        tf_term_new = {}
        score_tf_idf_term = {}
        tf_idf_term_new = {}
        for doc in relevant_docs:
            score_final = 0
            score_idf = 0
            score_tf = 0
            score_tf_idf = 0

            for token in query_vector:
                score_final += query_vector[token] * (
                    self.document_vector[doc][token] if token in self.document_vector[doc] else 0)

            for token in query_vector:
                score_tf_idf = query_vector[token] * (
                    self.document_vector[doc][token] if token in self.document_vector[doc] else 0)
                score_tf_idf_term[token] = score_tf_idf
                score_tf_idf_term_keys = list(score_tf_idf_term.keys())
                score_tf_idf_term_values = list(score_tf_idf_term.values())

                final_score_tf_idf_term = list(zip(score_tf_idf_term_keys, score_tf_idf_term_values))

            for token in query_vector:
                score_idf = idf_vector[token] * (self.document_vector[doc][token] if token in self.document_vector[doc] else 0)
                TermIdf[token] = score_idf
                TermIdf_keys = list(TermIdf.keys())
                TermIdf_values = list(TermIdf.values())

                final_TermIdf = list(zip(TermIdf_keys, TermIdf_values))

            for token in tf_vector:
                score_tf = tf_vector[token] * (self.document_vector[doc][token] if token in self.document_vector[doc] else 0)
                TermTf[token] = score_tf
                TermTf_keys = list(TermTf.keys())
                TermTf_values = list(TermTf.values())

                final_TermTf = list(zip(TermTf_keys, TermTf_values))

            FinalScore[doc] = score_final
            IdfScore[doc] = score_idf
            TfScore[doc] = score_tf

            idf_term_new[doc] = final_TermIdf
            tf_term_new[doc] = final_TermTf
            tf_idf_term_new[doc] = final_score_tf_idf_term
        sorted_FinalScore = sorted(FinalScore.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_FinalScore[:50], tf_term_new, idf_term_new, tf_idf_term_new

    def get_movie_info(self, sorted_score_list, tf_new, idf_new, tf_idf_new):
        result = []
        for entry in sorted_score_list:
            doc_id = entry[0]
            row = self.meta_data.loc[doc_id]
            info = (row["original_title"],
                    row["overview"] if isinstance(row["overview"], str) else "", entry[1], idf_new[doc_id],
                    tf_new[doc_id], tf_idf_new[doc_id], row["release_date"])
            result.append(info)
        new_score = None
        print(result[0:5])
        return result
        