#Merge Data
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from itertools import islice
import sys


class dataLoader(object):
    #-----------------------------------------------------------------------------------------------#
    '''
    Data Loading and Cleaning Tool for MovieLens 100k Dataset

    PARMS:
        path: Specify path top data directory
        init contains all column naming conventions to be used in df generation


    RETURNS:
        A single dataframe object merging source data from user_ratings / user_meta / item_meta
            - Drop Unnecessary temporal columns
                - Timestamp
                - Release Date (Extract Year for Year-Diff in similarity comp)
            - Get Dummies for Features:
                - Gender / Occupation
                    -Used for user similarity calc?
        REVISION: Leave as separate DFs for easier column indexing. May merge later
    '''

    #-----------------------------------------------------------------------------------------------#
    def __init__(self, path):
        self.path = os.getcwd() + '\\movieLens_data'
        self.encoding = "iso-8859-1"
        self.user_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        self.item_cols = [
            "movie_id", "movie_title", "release_date", "video_release_date",
            "IMDb_URL", "genre_unknown", "genre_Action", "genre_Adventure",
            "genre_Animation", "genre_Childrens", "genre_Comedy",
            "genre_Crime", "genre_Documentary", "genre_Drama", "genre_Fantasy",
            "genre_FilmNoir", "genre_Horror", "genre_Musical", "genre_Mystery",
            "genre_Romance", "genre_SciFi", "genre_Thriller", "genre_War",
            "genre_Western"
        ]
        self.user_meta_cols = [
            'user_id', 'age', 'gender', 'occupation', 'zipcode'
        ]


    def load_user(self):
        user = self.path + '\\u.data'
        return pd.read_csv(user, sep='\t', header=None, names=self.user_cols)

    def load_item(self):
        item = self.path + '\\u.item'
        item_cols = [i.lower() for i in self.item_cols]
        item_df = pd.read_csv(item,
                              sep="|",
                              header=None,
                              encoding=self.encoding,
                              names=item_cols)
        item_df['movie_title'] = item_df['movie_title'].str.replace(
            r"\(.*\)", "")  #Remove Year From Movie Titles

        item_df['release_year'] = pd.DatetimeIndex(
            item_df['release_date']).year #Extract year from release date and delete. Don't need granular temporal components

        item_cols_to_drop = ['video_release_date', 'imdb_url', 'release_date']
        item_df.drop(columns=item_cols_to_drop,
                     inplace=True)  #Drop unnecessary Columns

        return item_df

    def load_data(self):
        user_df = self.load_user()
        item_df = self.load_item()
        return user_df,item_df

class CollabFiltering(dataLoader):
    #-----------------------------------------------------------------------------------------------#
    '''
    Collaborative Filtering Class
    #-----------------------------#
    1. Weigh all users/items with respect to their similarity with the current user/item
    2. Select a subset of the users/items (neighbors) as recommenders
    3. Predict the rating of the user for specific items using neighbors’ ratings for the same (or similar) items
    4. Recommend items with the highest predicted rank

    Following Slide 14: We only incorporate user ratings, so we don't really need any item metadata
    We also compute average ratings as nanmeans to account for the sparsity (ignore blank/0 cells so as to not saturate)


    Item-Based
    #--------#
    - Compute Average Ratings  --> Calculate item-item Similarity    --> user_i predicted rating for item_i

    User-Based
    #--------#
    - Compute Average Ratings  --> Calculate User-User Similarity    --> user_i predicted rating for item_i

    RETURNS:
        Generator Object
    '''

    #-----------------------------------------------------------------------------------------------#
    def __init__(self):
        self.path = os.getcwd() + '\\movieLens_data'
        self.user_df, self.item_df = dataLoader(
            path=self.path).load_data()
        #We fill missing data with NA in LEC14 slides
        #Convert to sparse item-product matrix here
        self.data = self.user_df.pivot(index='user_id',
                                       columns='movie_id',
                                       values='rating').values
        #Compute means without zero elements
        self.avg_user_ratings = np.nanmean(self.data, axis=1)
        self.avg_item_ratings = np.nanmean(self.data, axis=0)
        #fill zero elements for spacial agreements
        self.data = self.user_df.pivot(index='user_id',
                                       columns='movie_id',
                                       values='rating').fillna(0).values

    #-----------------------------------------------------------------------------------------------#
    def cf_update_step(self,
                       data,
                       user_id,
                       item_id,
                       neighborhood,
                       cf_type='user'):
        if cf_type != 'user':
            data = data.T #transpose if user-item matrix instead
            avg_rating = self.avg_item_ratings[item_id] #storage for mean item rating
        else:
            avg_rating = self.avg_user_ratings[user_id] #storage for mean user ratings

        pred_rating = avg_rating #temp for y^hat

        vector = data[user_id]  #get vector representation for i (user or item).

        distances = {}
        for i, j in enumerate(data):
            if i != user_id:
                distances[i] = cosine(vector, j)

        distances = {
            k: v
            for k, v in sorted(distances.items(),
                               key=lambda item: item[1],
                               reverse=False)[:neighborhood]
        }

        neighbor_rankings = []
        for k, v in distances.items():  # Most similar items of item 𝑖
            sim_i_j = v  #get sim

            if cf_type == 'item':
                num1 = self.avg_item_ratings[k]  # item j’s mean rating
                num2 = data[k,
                              user_id]  # Observed rating (item, user) matrix
            elif cf_type =='user':
                num1 = self.avg_user_ratings[k]  # User 𝑣’s mean rating
                num2 = self.data[
                    k, item_id]  # Observed rating of user 𝑣 for item 𝑖
            neighbor_rankings.append(num2)

            #Only include second half of numerator if it isn't an empty element
            #Otherwise we drag down scores with negative numerators
            if num2 != 0:
                num = sim_i_j * (num2 - num1)
            else:
                num = 0


            denom = sum(distances.values())
            pred_rating += (num / sum(distances.values()))

        return avg_rating, distances, neighbor_rankings, pred_rating

    #-----------------------------------------------------------------------------------------------#
    def print_logs(self,
                   user_id,
                   item_id,
                   avg_rating,
                   distances,
                   neighbor_rankings,
                   pred_rating,
                   neighborhood_size,
                   cf_type='user'):
        movie = self.item_df[self.item_df['movie_id'] == item_id +
                             1]['movie_title'].values[0] #movie index starts at 1
        if cf_type != 'user':
            cf_type = 'item'
        print(
            'Performing {}-Based Rating Prediction for User {} on Movie: {} with neighborhood size {}'
            .format(cf_type, user_id + 1, movie, neighborhood_size))
        print('-' * 75)
        print(f'most similar neighbors {[i for i in distances.keys()]}')
        print(f'neighbor ratings for {movie}: {neighbor_rankings}')
        print(f'Average Rating: {avg_rating:.2f}')
        print(f'Predicted Rating: {pred_rating:.2f}')
        print(
            f'Actual Rating (if available): {self.data[user_id, item_id]:.2f}')
        print('-' * 75)

    def item_based(self, user_id, item_id, neighborhood_size, logs=True):
        #-----------------------------------------------------------------------------------------------#
        '''
        Item-Based CF Method
        ---------------------
        PARMS:
            user_id: int; userID for inference (user of interest)
            item_id: int; itemID for inference (item of interest)
            neighborhood_size: int; decide on the size of the lookup

        RETURNS:
            user_i predicted rating for item_i
        '''
        #-----------------------------------------------------------------------------------------------#
        assert all(
            isinstance(i, int) for i in [user_id, item_id, neighborhood_size]
        ), 'invalid input dtypes. user, item and neighborhood size must be ints'
        #-----------------------------------------------------#
        #ids start at 1 while matrix index starts at 0
        #-----------------------------------------------------
        user_id -= 1
        item_id -= 1
        #-----------------------------------------------------
        #Compute all user-user similarities + step through collab filtering
        #-----------------------------------------------------
        avg_rating, distances, neighbor_rankings, pred_rating = self.cf_update_step(
            data=self.data,
            user_id=user_id,
            item_id=item_id,
            neighborhood=neighborhood_size,
            cf_type='item')

        #-----------------------------------------------------
        #Logs
        #-----------------------------------------------------#
        self.print_logs(user_id,
                        item_id,
                        avg_rating,
                        distances,
                        neighbor_rankings,
                        pred_rating,
                        neighborhood_size,
                        cf_type='item')

        return [np.round(pred_rating, 2),
                self.data[user_id, item_id]]  #pred_rating, actual rating

    def user_based(self, user_id, item_id, neighborhood_size, logs=True):
        #-----------------------------------------------------------------------------------------------#
        '''
        User-Based CF Method
        ---------------------
        PARMS:
            user_id: int; userID for inference (user of interest)
            item_id: int; itemID for inference (item of interest)
            neighborhood_size: int; decide on the size of the lookup

        RETURNS:
            user_i predicted rating for item_i
        '''
        #-----------------------------------------------------------------------------------------------#
        assert all(
            isinstance(i, int) for i in [user_id, item_id, neighborhood_size]
        ), 'invalid input dtypes. user, item and neighborhood size must be ints'
        #-----------------------------------------------------#
        #ids start at 1 while matrix index starts at 0
        #-----------------------------------------------------#
        user_id -= 1
        item_id -= 1
        neighbor_rankings = []

        #-----------------------------------------------------#
        #Compute all user-user similarities + step through collab filtering
        #-----------------------------------------------------#
        avg_rating, distances, neighbor_rankings, pred_rating = self.cf_update_step(
            data=self.data,
            user_id=user_id,
            item_id=item_id,
            neighborhood=neighborhood_size,
            cf_type='user')

        #-----------------------------------------------------
        #Logs
        #-----------------------------------------------------#
        self.print_logs(user_id,
                        item_id,
                        avg_rating,
                        distances,
                        neighbor_rankings,
                        pred_rating,
                        neighborhood_size,
                        cf_type='user')

        return [np.round(pred_rating, 2),
                self.data[user_id, item_id]]  #pred_rating, actual rating

    def recommend(self, user_id, item_id, neighborhood_size, logs=True):
        self.user_based(user_id, item_id, neighborhood_size, logs=True)
        self.item_based(user_id, item_id, neighborhood_size, logs=True)


if __name__ == '__main__':
    user_id = int(sys.argv[1])
    item_id = int(sys.argv[2])
    neighborhood_size = int(sys.argv[3])
    assert user_id < 942, 'invalid user id entry. Enter a lower number'
    assert item_id < 1681, 'invalid item id entry. Enter a lower number'
    recsys = CollabFiltering()
    recsys.recommend(user_id=user_id, item_id=item_id, neighborhood_size=neighborhood_size)
