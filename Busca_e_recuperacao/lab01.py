"""
Source: https://www.datacamp.com/community/tutorials/recommender-systems-python

The following are the steps involved:

Decide on the metric or score to rate movies on.
Calculate the score for every movie.
Sort the movies based on the score and output the top results.


"""

# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('datasets/data/movies_metadata.csv', low_memory=False)

# Print the first three rows
# print(metadata.head(10))

# Weighted Rating (WR) = ((v/v+m) * R)+((m/v+m)*C)

# v is the number of votes for the movie;
# m is the minimum votes required to be listed in the chart;
# R is the average rating of the movie; And
# C is the mean vote across the whole report


# Calculate C
C = metadata['vote_average'].mean()
# print(C)

# Next, let's calculate the number of votes, m, received by a movie in the 90th percentile.
# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
# print(m)

# Next, you can filter the movies that qualify for the chart, based on their vote counts:
# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
# print(q_movies.shape)

# saída: (505, 28)

# You use the .copy() method to ensure that the new q_movies DataFrame created is independent of your
# original metadata DataFrame. In other words, any changes made to the q_movies DataFrame does not
# affect the metadata.

# You see that there are 4555 movies which qualify to be in this list. Now, you need to calculate 
# your 
# metric for each qualified movie. To do this, you will define a function, weighted_rating() and
# define a new feature score, of which you'll calculate the value by applying this function to your
# DataFrame of qualified movies:

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

# Finally, let's sort the DataFrame based on the score feature and output the title, vote count, 
# vote average and weighted rating or score of the top 15 movies.

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

"""
Print the top 15 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))

You see that the chart has a lot of movies in common with the IMDB Top 250 chart: for example, 
your top two movies, "Shawshank Redemption" and "The Godfather", are the same as IMDB.


Content-Based Recommender in Python
Plot Description Based Recommender

In this section, you will try to build a system that recommends movies that are similar to a 
particular movie. More specifically, you will compute pairwise similarity scores for all movies 
based on their plot descriptions and recommend movies based on that similarity score.

The plot description is available to you as the overview feature in your metadata dataset. Let's 
inspect the plots of a few movies:


Print plot overviews of the first 5 movies.
print(metadata['overview'].head())


In its current form, it is not possible to compute the similarity between any two overviews. To do 
this, you need to compute the word vectors of each overview or document, as it will be called from 
now on.

You will compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each document. This 
will give you a matrix where each column represents a word in the overview vocabulary (all the words 
that appear in at least one document) and each column represents a movie, as before.

In its essence, the TF-IDF score is the frequency of a word occurring in a document, down-weighted 
by the number of documents in which it occurs. This is done to reduce the importance of words that 
occur frequently in plot overviews and therefore, their significance in computing the final 
similarity score.

Fortunately, scikit-learn gives you a built-in TfIdfVectorizer class that produces the TF-IDF matrix 
in a couple of lines.

"""

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Output the shape of tfidf_matrix
#print(tfidf_matrix.shape)

# saída: (45466, 75827)

"""

You see that over 75,000 different words were used to describe the 45,000 movies in your dataset.

With this matrix in hand, you can now compute a similarity score. There are several candidates for 
this; such as the euclidean, the Pearson and the cosine similarity scores. Again, there is no right 
answer to which score is the best. Different scores work well in different scenarios and it is often 
a good idea to experiment with different metrics.

You will be using the cosine similarity to calculate a numeric quantity that denotes the similarity 
between two movies. You use the cosine similarity score since it is independent of magnitude and is 
relatively easy and fast to calculate (especially when used in conjunction with TF-IDF scores, which 
will be explained later). Mathematically, it is defined as follows: 


Since you have used the TF-IDF vectorizer, calculating the dot product will directly give you the 
cosine similarity score. Therefore, you will use sklearn's linear_kernel() instead of 
cosine_similarities() since it is faster.

"""

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# You're going to define a function that takes in a movie title as an input and outputs a list of the 
# 10 most similar movies. Firstly, for this, you need a reverse mapping of movie titles and DataFrame 
# indices. In other words, you need a mechanism to identify the index of a movie in your metadata 
# DataFrame, given its title.


#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


# You are now in a good position to define your recommendation function. These are the following steps 
# you'll follow:

# Get the index of the movie given its title.
# Get the list of cosine similarity scores for that particular movie with all movies. Convert it into 
# a list of tuples where the first element is its position and the second is the similarity score.
# Sort the aforementioned list of tuples based on the similarity scores; that is, the second element.
# Get the top 10 elements of this list. Ignore the first element as it refers to self (the movie most 
# similar to a particular movie is the movie itself).
# Return the titles corresponding to the indices of the top elements.

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

# print(get_recommendations('The Dark Knight Rises'))
# get_recommendations('The Godfather')

"""
You see that, while your system has done a decent job of finding movies with similar plot descriptions, the quality 
of recommendations is not that great. "The Dark Knight Rises" returns all Batman movies while it more likely that the 
people who liked that movie are more inclined to enjoy other Christopher Nolan movies. This is something that cannot 
be captured by your present system.

Credits, Genres and Keywords Based Recommender

It goes without saying that the quality of your recommender would be increased with the usage of better metadata. 
That is exactly what you are going to do in this section. You are going to build a recommender based on the following 
metadata: the 3 top actors, the director, related genres and the movie plot keywords.

The keywords, cast and crew data is not available in your current dataset so the first step would be to load and 
merge them into your main DataFrame.


Nesse ponto não baixei os outros datasets usados para fazer recomendações baseadas em gêneros
palavras-chave ou créditos.

"""