import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
# Get the base path of the script
base_path = os.getcwd()
helper_functions_path = base_path + "/modules"
sys.path.insert(0, helper_functions_path)
from DataPreparer import DataPreparer
import traceback as tb
from datetime import datetime as dt
from joblib import dump, load

#Models
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score

#MAIN FUNCTION
if __name__ == '__main__':

    try:

        start_time = dt.now()
        print("Genre_Prediction_Model_Generator_Script Start Time : {}".format(start_time))

        ################################################################################
        ## Data Loading and Cleaning
        ################################################################################

        #Read the initial excel file
        dim_tracks_df = pd.read_excel('./datasets/spotify_datasets.xlsx', sheet_name='dim_tracks')

        #Sample a random dataset from the original dataset and store in a file
        random_test_df = dim_tracks_df.sample(1000)
        random_test_df.to_csv('./datasets/sample_music_dataset.csv',index=False)

        # Drop the sampled rows from the original DataFrame
        dim_tracks_df = dim_tracks_df.drop(random_test_df.index)

        #Read the genre mapping file
        genre_mapping_df = pd.read_csv('./datasets/genre_groups.csv')

        # Apply the mapping to create a new column
        dim_tracks_df = pd.merge(dim_tracks_df,genre_mapping_df,how='inner',on='primary_genre')

        #Select the feature list
        feature_list = ['duration_ms',
                        'danceability',
                        'energy', 
                        'loudness', 
                        'speechiness', 
                        'acousticness',
                        'instrumentalness', 
                        'liveness', 
                        'valence', 
                        'tempo',
                        'explicit']

        # Perform preprocessing steps
        preparer = DataPreparer(dim_tracks_df,target='genre_group',features=feature_list)

        #Encoding the categorical features
        dp = preparer.encode_categorical()

        
        #Split the data frame into the train and test set
        X_train, X_test, y_train, y_test, nf_train, nf_test, label_encoder, scaler = dp.split_data(test_size=0.2, scale_method='standard')

        print("Training Set : {}".format(X_train.shape))
        print("Test Set : {}".format(X_test.shape))

        #Performing Class-Imbalance Correction using ADASYN technique
        #X_train, y_train = dp.handle_class_imbalance(X_train, y_train, mode='adasyn')
        
        ################################################################################
        ## Model Training and Fit
        ################################################################################

        print("Model Training in progress...")

        #Random Forest with best parameters
        rf_model = RandomForestClassifier(n_estimators=1000, max_depth=30, random_state=42)

        #XGBoost with best parameters
        xgb_model = XGBClassifier(objective='multi:softmax', max_depth=8, learning_rate=0.1, n_estimators=1000, random_state=42)

        #CatBoost with best parameters
        cat_model = CatBoostClassifier(loss_function='MultiClass',learning_rate=0.1, depth=10, bagging_temperature=1.5, 
                                       n_estimators=1000,random_state=42, verbose=0)

        #LGBMClassifier with best parameters
        lgbm_model = LGBMClassifier(objective='multiclass', learning_rate=0.1, n_estimators=1000, max_depth=30, random_state=42, verbose=0)

        #Estimator List
        estimator_list = [('rf', rf_model),
                          ('xgb', xgb_model),
                          ('cb', cat_model),
                          ('lgb', lgbm_model)]

        #Setup the model pipeline
        ensemble_model = VotingClassifier(estimators=estimator_list, voting='soft')
        
        #Train the ensemble model
        ensemble_model.fit(X_train, y_train)

        #Save the model in a file
        dump(ensemble_model, './model_files/genre_prediction_model.joblib')

        #Save the scaler
        dump(scaler, './model_files/scaler.pkl')

        #Save the label encoder
        dump(label_encoder, './model_files/label_encoder.pkl')        
        
        print("Model Training Completed and saved.")

        ################################################################################
        ## Model Testing
        ################################################################################  

        print("Model Testing in progress...")

        #Predict on Test Data
        y_pred_encoded = ensemble_model.predict(X_test)

        #Final test dataset with the original and the predicted labels
        final_dataset_df = dp.revert_to_original(X_test,y_test,y_pred_encoded,nf_test,scaler,label_encoder)

        print("Classification Report:")
        print(classification_report(final_dataset_df['original_genre_group'], final_dataset_df['predicted_genre_group']))
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(final_dataset_df['original_genre_group'], final_dataset_df['predicted_genre_group'])
        print(f"Cohen's Kappa: {kappa:.2f}")
        
        # F1-Score
        f1 = f1_score(final_dataset_df['original_genre_group'], final_dataset_df['predicted_genre_group'], average='weighted')
        print(f"Weighted F1-Score: {f1:.2f}")

        #Save the final dataset
        final_dataset_df.to_csv('./datasets/predicted_music_test_dataset.csv',index=False)

        print("Model Testing Completed and test dataset saved successfully.")
        
        end_time = dt.now()
        print("Genre_Prediction_Model_Generator_Script Completed:{}".format(end_time))
        duration = end_time - start_time
        td_mins = int(round(duration.total_seconds() / 60))
        print('The difference is approx. %s minutes' % td_mins)
     
    except Exception as e:
        
        error = "Genre_Prediction_Model_Generator_Script Failure :: Error Message {} with Traceback Details: {}".format(e,tb.format_exc())        
        print(error)
      