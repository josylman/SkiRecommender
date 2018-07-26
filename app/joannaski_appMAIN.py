from flask import Flask, render_template, request, url_for, flash, redirect
import requests
import pandas as pd
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn import base
from sklearn.feature_extraction import DictVectorizer
import json
#from bokeh.plotting import figure
#from bokeh.embed import components
import numpy as np
import os



app = Flask(__name__)
# obtain the Data
# Run the model

df = pd.read_csv('static/cleanskidata.txt', error_bad_lines=False)


class CenteringTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    Centers the features about 0
    """

    def fit(self, X, y=None):
        self.means = X.mean(axis=0)  # means of each column
        return self

    def transform(self, X):
        X = X.copy()
        return X - self.means


class ColumnWeightTransformer(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self

    def transform(self, X):
        X = X.copy()
        for name in self.col_names:
            X.loc[:, name] *= 500

        return X


# Deal with the text data
df_with_dummies = pd.get_dummies(df, columns=['Continent', 'Country'])

# Remove columns that are not needed
df_removal = df_with_dummies.drop(['Currency', 'Unnamed: 0', 'ResortName'], axis=1)
df_withnames = df_with_dummies.drop(['Currency', 'Unnamed: 0'], axis=1)

# Deal with numeric data
ct = CenteringTransformer()
df_norm = ct.fit_transform(df_removal)


# Construct reverse map of indices and ski resort names
indices = pd.Series(df.index, index=df['ResortName']).drop_duplicates()
allresorts = df["ResortName"]
middlenames = allresorts.str.title()
allresorts1 = middlenames.replace('-', ' ', regex=True)
allresorts1 = allresorts1.tolist()

# Make a function to get cosine similarities
def get_cosinesim(df_norm, price, snow, apres, location):
    # All possible variables to be weighted
    country_var = [col for col in df_norm if col.startswith('Country_')]
    apres_var = ['Food', 'Accommodations']
    price_var = ['Adult']
    snow_var = ['Variety', 'Snowreliability', 'Slopepreparation']
    priorities = []
    if location == "True":
        priorities.extend(country_var)
    if apres == "True":
        priorities.extend(apres_var)
    if price == "True":
        priorities.extend(price_var)
    if snow == "True":
        priorities.extend(snow_var)
    print(priorities)
    cwt = ColumnWeightTransformer(priorities)
    df_spec = cwt.fit_transform(df_norm)

    # Convert to an array
    df_array = df_spec.values

    # compute cosine similarity
    cosine_sim = cosine_similarity(df_array)

    return cosine_sim

    # Make a function to get recommendations


def get_recommendations(resort, cosine_sim):
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
    new_df = df.iloc[ski_indices]
    names = df['ResortName']
    middlenames = names.str.title()
    new_df['ResortName'] = middlenames.replace('-', ' ', regex=True)
    new_df['Adult'] = round(new_df['Adult'], 2)
    newdic = new_df.to_dict(orient='records')

#     names = names.str.title()
#     names = names.replace('-', ' ', regex=True)

    # Return the top 10 most similar movies
    return newdic


def get_recommendations_list(resort, cosine_sim):
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
    new_df = df.iloc[ski_indices]
    names = df['ResortName']
    middlenames = names.str.title()
    new_df['ResortName'] = middlenames.replace('-', ' ', regex=True)
    new_df['Adult'] = new_df['Adult'].round(2)
    newdic = new_df.T.to_dict().values()
#     names = names.str.title()
#     names = names.replace('-', ' ', regex=True)

    # Return the top 10 most similar movies
    return newdic


feature_names = ['Altitude', 'Ski_resort_size', 'Child', 'Youth', 'Easy', 'Intermediate', 'Difficult', 'Snowreliability', 'Variety', 'Food', 'Accommodations']


def create_figure(current_feature_name, smalldata):
    smallresorts = smalldata['ResortName'].tolist()
    values = smalldata[current_feature_name].tolist()
    p = figure(x_range=smallresorts, title=current_feature_name)
    p.vbar(x=smallresorts, top=values, width=0.9, fill_color="skyblue")
    p.xaxis.major_label_orientation = "vertical"
    p.xaxis.major_label_text_font_size = "10pt"
    return p


app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():

    return render_template('home.html')


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/insights")
def insights():
    return render_template('insights.html', title='About')


@app.route('/skiform', methods=['GET', 'POST'])
def form_1():
    if request.method == 'GET':
        return render_template('skiform1.html')
    else:
        # request was a POST

        return redirect(url_for('indexplotter'))


@app.route('/recommendations', methods=['GET', 'POST'])
def indexplotter():

    resort = request.args['name_skiresort']

    if resort == '' or resort not in allresorts1:
        resort = 'Vail'
    price = request.args.get('price', False)
    snow = request.args.get('snow', False)
    apres = request.args.get('apres', False)
    location = request.args.get('location', False)

    cosinesim = get_cosinesim(df_norm, price, snow, apres, location)
    # Get the resort data
    smalldata = get_recommendations(resort, cosinesim)
    listdata = get_recommendations_list(resort, cosinesim)
    # Determine selected feature
    #current_feature_name = request.args.get("feature_name")
    # if current_feature_name == None:
    #    current_feature_name = 'Adult'
    # Create the plot
    #plot = create_figure(current_feature_name, smalldata)

    # Embed plot into HTML via Flask Render
    #script, div = components(plot)
    return render_template("googlechart.html", price=price, snow=snow, apres=apres, location=location, resort=resort, cosinesim=cosinesim, df_norm=df_norm, feature_names=feature_names, smalldata=smalldata, listdata=listdata)


if __name__ == '__main__':
    app.run(debug=True, port=5427)
