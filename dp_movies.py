
import pandas as pd
ratings_df = pd.read_csv("ratings.csv")
credits_df = pd.read_csv("credits.csv")
keywords_df = pd.read_csv("keywords.csv")
links_df = pd.read_csv("links.csv")
links_small_df = pd.read_csv("links_small.csv")
ratings_small_df = pd.read_csv("ratings_small.csv")
movies_metadata_df = pd.read_csv("movies_metadata.csv")

class DeepLearning():
    def __init__(self,ratings_df,credits_df,keywords_df,links_df,links_small_df,ratings_small_df,movies_metadata_df):
        self.ratings_df=ratings_df
        self.credits_df=credits_df
        self.keywords_df=keywords_df
        self.links_df=links_df
        self.links_small_df=links_small_df
        self.ratings_small_df=ratings_small_df
        self.movies_metadata_df=movies_metadata_df
    def analyzing_dataset(self,data):
        data['id'] = pd.to_numeric(data['id'], errors='coerce')
        merged_df = pd.merge(self.ratings_df, data, left_on='movieId', right_on='id')
        genre_ratings = merged_df.groupby('genres')['rating'].mean()
        genre_list = merged_df['genres'].str.split('|')
        genre_ratings_df = pd.DataFrame({'genre': [genre for sublist in genre_list for genre in sublist], 
                                 'title': merged_df['title'], 
                                 'rating': merged_df['rating']})

        genre_ratings = genre_ratings_df.groupby(['title', 'genre'])['rating'].mean().reset_index()
        print(genre_ratings)

        #step 2
        user_ratings = ratings_df.groupby('userId')['rating'].mean().reset_index()
        print(user_ratings)
        def determine_user_type(rating):
            if rating < 2.5:
                return "low"
            elif rating < 4:
                return "medium"
            else:
                return "high"
        user_ratings['type'] = user_ratings['rating'].apply(determine_user_type)
        print(user_ratings)




def main():

    dp_movies=DeepLearning(ratings_df,credits_df,keywords_df,links_df,links_small_df,ratings_small_df,movies_metadata_df)
    dp_movies.analyzing_dataset(movies_metadata_df)

main()