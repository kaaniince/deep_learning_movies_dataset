
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
ratings_df = pd.read_csv("ratings.csv")
credits_df = pd.read_csv("credits.csv")
keywords_df = pd.read_csv("keywords.csv")
links_df = pd.read_csv("links.csv")
links_small_df = pd.read_csv("links_small.csv")
ratings_small_df = pd.read_csv("ratings_small.csv")
movies_metadata_df = pd.read_csv("movies_metadata.csv")

class DeepLearningMovies():
    def __init__(self,ratings_df,credits_df,keywords_df,links_df,links_small_df,ratings_small_df,movies_metadata_df):
        self.ratings_df=ratings_df
        self.credits_df=credits_df
        self.keywords_df=keywords_df
        self.links_df=links_df
        self.links_small_df=links_small_df
        self.ratings_small_df=ratings_small_df
        self.movies_metadata_df=movies_metadata_df
    def analyzing_dataset(self,data):
        #Analyze each movie genres based on ratings.
        data['id'] = pd.to_numeric(data['id'], errors='coerce')
        merged_df = pd.merge(self.ratings_df, data, left_on='movieId', right_on='id')
        genre_ratings = merged_df.groupby('genres')['rating'].mean()
        genre_list = merged_df['genres'].str.split('|')
        genre_ratings_df = pd.DataFrame({'genre': [genre for sublist in genre_list for genre in sublist], 
                                 'title': merged_df['title'], 
                                 'rating': merged_df['rating']})

        genre_ratings = genre_ratings_df.groupby(['title', 'genre'])['rating'].mean().reset_index()
        print(genre_ratings)

        #Analyze user movie preference based on their past ratings. Then determine user types. Give a type to each person 
        self.user_ratings = ratings_df.groupby('userId')['rating'].mean().reset_index()
        print(self.user_ratings)
        def determine_user_type(rating):
            if rating < 2.5:
                return "low"
            elif rating < 4:
                return "medium"
            else:
                return "high"
        self.user_ratings['type'] = self.user_ratings['rating'].apply(determine_user_type)
        print(self.user_ratings)
    def classification_dataset(self):
        #Construct a NN algorithm. Train your algorithm for user types which is determined in last section. Show your test accuracy of the classification algorithm.
        ratings = self.user_ratings['rating'].values
        types = self.user_ratings['type'].values
        types_one_hot = pd.get_dummies(types).values
        scaler = StandardScaler()
        ratings_scaled = scaler.fit_transform(ratings.reshape(-1, 1))
        ratings_train, ratings_test, types_train, types_test = train_test_split(ratings_scaled, types_one_hot, test_size=0.2)
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=[1]),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(ratings_train, types_train, epochs=10)
        loss, accuracy = model.evaluate(ratings_test, types_test)
        print("Test accuracy:", accuracy)



def main():
    dl_movies=DeepLearningMovies(ratings_df,credits_df,keywords_df,links_df,links_small_df,ratings_small_df,movies_metadata_df)
    dl_movies.analyzing_dataset(movies_metadata_df)
    dl_movies.classification_dataset()

main()