from tfidf import tfidf

def get_results(query):
    tf = tfidf()
    processed_query = tf.pre_processing(query)
    relevant_docs = tf.get_relevant_docs(processed_query)
    query_vector, idf_vector, tf_vector = tf.build_query_vector(processed_query)
    sorted_score_list, tf_new, idf_new, tf_idf_new = tf.similarity(relevant_docs, query_vector, idf_vector, tf_vector)
    search_result = tf.get_movie_info(sorted_score_list, tf_new, idf_new, tf_idf_new)
    return search_result, processed_query
