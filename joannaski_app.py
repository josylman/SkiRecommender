from flask import Flask, render_template, request, url_for, flash, redirect
import requests
from pandas import DataFrame
import pandas as pd
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn import base
from sklearn.feature_extraction import DictVectorizer
import json
from bokeh.plotting import figure
from bokeh.embed import components
import numpy as np
import os
from forms import RegistrationForm, LoginForm


app = Flask(__name__)
# obtain the Data
# Run the model

df = pd.read_csv('/Users/joannasylman//Desktop/DataIncubatorPrep/MilestoneProject/cleanskidata.txt', error_bad_lines=False)


class CenteringTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    Centers the features about 0
    """

    def fit(self, X, y=None):
        self.means = X.mean(axis=0)  # means of each column
        return self

    def transform(self, X):
        return X - self.means


# Deal with the text data
df_with_dummies = pd.get_dummies(df, columns=['Continent', 'Country'])

# Remove columns that are not needed
df_removal = df_with_dummies.drop(['Currency', 'Unnamed: 0', 'ResortName'], axis=1)

# Convert to an array
df_array = df_removal.values

# Deal with numeric data
ct = CenteringTransformer()
df_norm = ct.fit_transform(df_array)

# compute cosine similarity
cosine_sim = cosine_similarity(df_norm)

# Construct reverse map of indices and ski resort names
indices = pd.Series(df.index, index=df['ResortName']).drop_duplicates()

# Make a function to get recommendations


def get_recommendations(resort, cosine_sim=cosine_sim):
    # Convert resort to be in format
    resort = resort.lower()
    resort = resort.replace(" ", "-")

    # Get the index of the movie that matches the title
    idx = indices[resort]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    ski_indices = [i[0] for i in sim_scores]

    # the top 10 most similar movies
    names = df.iloc[ski_indices]
#     names = names.str.title()
#     names = names.replace('-', ' ', regex=True)

    # Return the top 10 most similar movies
    return names


smalldata = (get_recommendations('Breckenridge'))

feature_names = ['Altitude', 'Ski_resort_size', 'Child', 'Youth', 'Easy', 'Intermediate', 'Difficult', 'Snowreliability', 'Variety', 'Food', 'Accommodations']


def create_figure(current_feature_name):
    smallresorts = smalldata['ResortName'].tolist()
    values = smalldata[current_feature_name].tolist()
    p = figure(x_range=smallresorts, title=current_feature_name)
    p.vbar(x=smallresorts, top=values, width=0.9, fill_color="skyblue")
    p.xaxis.major_label_orientation = "vertical"
    return p


@app.route('/', methods=['GET', 'POST'])
def indexplotter():

    # Determine selected feature
    current_feature_name = request.args.get("feature_name")
    if current_feature_name == None:
        current_feature_name = 'Adult'

    # Create the plot
    plot = create_figure(current_feature_name)

    # Embed plot into HTML via Flask Render
    script, div = components(plot)
    return render_template("plotter.html", script=script, div=div, current_feature_name=current_feature_name, feature_names=feature_names)


if __name__ == '__main__':
    app.run(debug=True, port=5432)
