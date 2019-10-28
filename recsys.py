# https://towardsdatascience.com/how-to-build-a-simple-recommender-system-in-python-375093c3fb7d
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# 1. Read csv files
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# 2. Calculate the Mean Rating of all ratings for every user
meanUserRating = ratings[["userId", "rating"]].groupby("userId").agg(np.average)

# 3. Calculation of a Normalized Rating for every movie of every user by
# subtracting the mean rating of every user from every rating of user's movies
ratings["normRating"] = ratings.apply(lambda row: (row["rating"] - meanUserRating.at[int(row["userId"]), "rating"]), axis=1)

# 4. Building the basic dataframe by merging ratings and movies on the common column movieId
df = pd.merge(ratings, movies, on="movieId").drop(columns='timestamp')

# 5. Extract genres
genres = movies.genres.str.split("|") # extract every genre of every movie

# 6. Add the genres features as a column for every genre and set value to 1 for 
# the presence of the feature else set value to 0
allgenres = set(genres.sum()) # find all unique genres
for genre in allgenres:
    df[genre] = df["genres"].apply(lambda movie_genres: 1 if genre in movie_genres else 0)

# 7. Prepare the df_movies for calculating cosine similarity
df_movie_1 = df.groupby("movieId").agg({"normRating":np.average}) # the average of normalized ratings
df_movie_2 = df[["movieId"] + [col for col in df.columns if col in allgenres]].groupby("movieId").agg(np.average) # the genres
df_movies = pd.merge(df_movie_1, df_movie_2, on="movieId") # merge the two dataframes

# 8. Standardize the data of the dataframe
df_movies_s = pd.DataFrame(StandardScaler().fit_transform(df_movies), index=df_movies.index, columns=df_movies.columns)

# 9. Build a dataframe with cosine similarity of every movie to all others
df_theta = pd.DataFrame(cosine_similarity(df_movies_s), index=df_movies_s.index, columns=df_movies_s.index)

# 10. Sort users by rating of every movie after choosing the positive normalized
# ratings for every movie and keep only user-movie IDs
sortedUserMovies = ratings[ratings.normRating>0].sort_values(["userId", "normRating"], ascending=[True, False])[["userId", "movieId"]]

# 11. The most similar movies according to cosine similarity for every movie
def chooseMostSimilar(movieId, n=10, df_theta=df_theta):
    return list(df_theta[movieId].sort_values(ascending=False).head(n).index)

sortedUserMovies["recommend"] = sortedUserMovies.movieId.apply(chooseMostSimilar) 

# 12. Recommendations of every user
userRecoms = sortedUserMovies.groupby("userId").agg({"recommend": lambda col:col.sum()})


# Functions
def retrieveTitle(movieId, movies=movies): # Returns title of a movieId from movies
    return movies[movies["movieId"] == movieId]["title"].iloc[0]

def getTitles(Ids, movies=movies): # Returns movie titles from a list of movie IDs
    return [retrieveTitle(Id) for Id in Ids]

def getUserMovieRecom(userId, movieId): # Returns recommendations for a user for a movie that has been rated over his average rating
    return sortedUserMovies[(sortedUserMovies.userId==userId) & (sortedUserMovies.movieId==movieId)]["recommend"].iloc[0]

def getUserRecomId(userId): # Returns the recommended movie Ids for a userID
    return userRecoms.recommend.loc[userId]

# The Recommender Function
def getUserRecomTitles(userId, movieId=None, top=10): # Returns the recommended movie titles for a userID or for a user and movie Id
    if not movieId: Ids = getUserRecomId(userId)
    else: Ids = getUserMovieRecom(userId, movieId)
    return getTitles(Ids)[:top]

print('User 1:', getUserRecomTitles(1)) # The top 10 recommendations of user with ID '1' for all seen movies
print('User 1 for Movie 157:', getUserRecomTitles(1, 157, 5)) # The top 5 recommendations of user with ID '1' for  movie id '157'
