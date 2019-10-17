# Movie-Recommendation
Dataset: [Movie Lens](https://www.kaggle.com/rounakbanik/the-movies-dataset/home)
- To develop the search feature for this web application,we used 'The Movies Dataset'.These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages,production companies, countries, TMDB vote counts and vote averages.This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.

## Web Application URL:
Web App:[Web App](http://ec2-3-14-250-128.us-east-2.compute.amazonaws.com:5000/)

## Project Video
[Watch the video](https://youtu.be/0GpSSf-ZNrw)

## [Movie Search](https://nandanpandya.netlify.com/post/movie-search/)

Blog: [https://nandanpandya.netlify.com/post/movie-search/](https://nandanpandya.netlify.com/post/movie-search/)


Search feature calculates cosine similarity between vectors space of search query and movies and top movies are returned.
### Loading the Data
 - First, we need to load the data from the dataset. We need to pick the movie credit data such as the cast and crew. The other data we  upload metadata for the movie which has columns such as genre,original title,poster path for the poster image, production companies.

 - Along with the data in code I have also initialized the initial conditions which are essential in next steps such as stemming and tokenizing. Only the seperate parts of the code are called in the initlize block of the code. This ensures easy debugging od parts of code and keeps each process atomic.
```
def initialize():
    global data_folder, credits_cols, meta_cols, noise_list, credits_data, meta_data, N, tokenizer, stopword, stemmer, inverted_index, document_vector
    # Data configurations
    #data_folder = '/home/npandya/mysite/data/'
    data_folder = 'data/'
    credits_cols = {"id": None, "cast":['character', 'name'], "crew":['name']}
    meta_cols = {"id": None, "genres":['name'], "original_title":None, "overview":None,"poster_path":None,
                     "production_companies":['name'], "tagline":None}
    noise_list = ['(voice)', '(uncredited)']

    # Read data
    credits_data = pd.read_csv(data_folder +'credits.csv', usecols=credits_cols.keys(), index_col="id")
    meta_data = pd.read_csv(data_folder + 'movies_metadata.csv', usecols=meta_cols.keys(), index_col="id")
    # Total number of documents = number of rows in movies_metadata.csv
    N = meta_data.shape[0]

    # Pre-processing initialization
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    stopword = stopwords.words('english')
    stemmer = PorterStemmer()
 ```
### Pre-Processing:Stemming
In this step we will stem the data where we have noise values and replace it with nulls. We will then tokenize the data and remove the stop words using porter stemmer. We will also enable lowercase for the token.
```
def pre_processing(data_string):
    for noise in noise_list:
        data_string = data_string.replace(noise, "")
    tokens = tokenizer.tokenize(data_string)
    processed_data = []
    for t in tokens:
        if t not in stopword:
            processed_data.append(stemmer.stem(t).lower())
    return processed_data
```
### Creation Of Inverted Index:
Inverted index is very essential for fast search results retrievel. This index is created on the id column of the metadata. It is thus our primary key which joins it with credits file.
```
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
                    for col_value in col_values:
                        for param in parameters:
                            data.append(col_value[param])
        insert(index, pre_processing(' '.join(data)))
```
### Insert the token created in the index
We take the token and then search it with the token created.
```
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
```

### Calculation for TF IDF:

Calculate TF and IDF of each document:
```
def build_doc_vector():
    for token_key in inverted_index:
        token_values = inverted_index[token_key]
        idf = math.log10(N / token_values["df"])
        for doc_key in token_values:
            if doc_key != "df":
                log_tf = 1 + math.log10(token_values[doc_key])
                tf_idf = log_tf * idf
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
```
The equation for TF IDF is given as below.

![TF IDF Equation:](https://github.com/napandya/Movie-Recommender/blob/master/tf_idf_equation.jpeg)

The TF-IDF score is given by:

![TF IDF Score Equation:](https://github.com/napandya/Movie-Recommender/blob/master/tf_idf_score.jpeg)

### Build the Query Vector

```
def build_query_vector(processed_query):
    query_vector = {}
    sum = 0
    for token in processed_query:
        if token in inverted_index:
            tf_idf = (1 + math.log10(processed_query.count(token))) * math.log10(N/inverted_index[token]["df"])
            query_vector[token] = tf_idf
            sum += math.pow(tf_idf, 2)
    sum = math.sqrt(sum)
    for token in query_vector:
        query_vector[token] /= sum
    return query_vector
```
### Calculate the cosine similarity
Once our query vector is created we will create a score map and assign it to the search result we get.
```
def cosine_similarity(relevant_docs, query_vector):
    score_map = {}
    for doc in relevant_docs:
        score = 0
        for token in query_vector:
            score += query_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
        score_map[doc] = score
    sorted_score_map = sorted(score_map.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_score_map[:50]
```
The Cosine similarity is given by the equation is given below:

![Cosine Similarity:](https://github.com/napandya/Movie-Recommender/blob/master/Cosine%20Similarity.png)


## [Movie Classifier](https://nandanpandya.netlify.com/post/movie-classifier/)

Blog: [https://nandanpandya.netlify.com/post/movie-classifier/](https://nandanpandya.netlify.com/post/movie-classifier/)

**Naive Bayes Classifier:**
- It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
- Each movie can be thus classified into multiple genres.
- The below code snippet implements the logic for multinomial Naive Bayes Classifier.

Get the probability of each genre.
```
def get_results(query):
    global prior_probability, post_probability
    initialize()
    if os.path.isfile("classifierPickle.pkl"):
        prior_probability = pickle.load(open('classifierPicklePrior.pkl', 'rb'))
        post_probability = pickle.load(open('classifierPicklePost.pkl', 'rb'))
    else:
        (prior_probability, post_probability) = build_and_save()
    return eval_result(query)

def eval_result(query):
    processed_query = pre_processing(query)
    genre_score = {}
    for genre in prior_probability.keys():
        score = prior_probability[genre]
        # print("For genre: ", genre, ", prior score: ", score)
        for token in processed_query:
            if (genre, token) in post_probability.keys():
                score = score * post_probability[(genre, token)]
                # print("token: ", token, ", score: ", score)
        genre_score[genre] = score
    sorted_score_map = sorted(genre_score.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_score_map
 ```
From the keywords iterate over the genres and get the highest probability for the match
```
def build_and_save():
    row_count = 0
    token_count = 0
    post_probability = {}
    token_genre_count_map = {}
    genre_count_map = {}
    for row in meta_data.itertuples():
        keywords = []
        genres = []
        for col in meta_cols.keys():
            col_values = getattr(row, col)
            parameters = meta_cols[col]
            # Paramter is None for tagline and overview columns, so appending data in keywords[]
            if parameters is None:
                keywords.append(col_values if isinstance(col_values, str) else "")
            # Else it is genres as it has a parameter "Name". So append in genres[]
            else:
                col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
                for col_value in col_values:
                    for param in parameters:
                        genres.append(col_value[param])

        tokens = pre_processing(' '.join(keywords))
        for genre in genres:
            if genre in genre_count_map:
                genre_count_map[genre] += 1
            else:
                genre_count_map[genre] = 1
            for token in tokens:
                token_count += 1
                if (genre, token) in token_genre_count_map:
                    token_genre_count_map[(genre, token)] += 1
                else:
                    token_genre_count_map[(genre, token)] = 1

        row_count += 1
        # Uncomment below lines for reading specific number of rows from excel instead of the whole
        # if (row_count == 2):
        #     print(genre_count_map)
        #     break
    for (genre, token) in token_genre_count_map:
        post_probability[(genre, token)] = token_genre_count_map[(genre, token)] / token_count

    prior_probability = {x: genre_count_map[x]/row_count for x in genre_count_map}
    save(prior_probability, post_probability)
    return (prior_probability, post_probability)
```
 
## [Movie Recommender](https://nandanpandya.netlify.com/post/movie-recommender/)

Blog: [https://nandanpandya.netlify.com/post/movie-recommender/](https://nandanpandya.netlify.com/post/movie-recommender/)

**Metadata Based Recommender**

- We will be using the information such as Credits,Keywords, Ratings and Movie details to recommend movies to a user.
- For any Recommender the more metadata it has the more it is accurate.
- To build a recommender,the following are the steps involved:
	- Decide on the metric or score to rate movies on.
	- Calculate the score for every movie.
	- Sort the movies based on the score and output the top results.

First lets clean the data
```
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
```
Now, Lets Apply the clean data function to our feature extractor:
```
Features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)
```
Lets create a Metadata which combines all the features. and apply it to our count vectorizer

```
    metadata['soup'] = metadata.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['soup'])
```
 
## References
* [Kaggle Kernels](https://www.kaggle.com/rounakbanik/the-movies-dataset/kernels)
* [https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf](https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf)
* [https://docs.python.org/2/library/collections.html](https://docs.python.org/2/library/collections.html)
* [https://www.numpy.org/devdocs/](https://www.numpy.org/devdocs/)
* [https://www.ics.uci.edu/~welling/teaching/CS77Bwinter12/presentations/course_Ricci/13-Item-to-Item-Matrix-CF.pdf](https://www.ics.uci.edu/~welling/teaching/CS77Bwinter12/presentations/course_Ricci/13-Item-to-Item-Matrix-CF.pdf)
* [https://www.kaggle.com/rounakbanik/the-movies-dataset/kernels](https://www.kaggle.com/rounakbanik/the-movies-dataset/kernels)
* [https://nlp.stanford.edu/IR-book/pdf/06vect.pdf](https://nlp.stanford.edu/IR-book/pdf/06vect.pd)
* [http://flask.pocoo.org/docs/](http://flask.pocoo.org/docs/)
* [http://pandas.pydata.org/pandas-docs/stable/](http://pandas.pydata.org/pandas-docs/stable/)
* [https://docs.aws.amazon.com/efs/latest/ug/gs-step-one-create-ec2-resources.html](https://docs.aws.amazon.com/efs/latest/ug/gs-step-one-create-ec2-resources.html)
