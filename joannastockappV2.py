from flask import Flask, render_template, request
import requests
from pandas import DataFrame
import json
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.charts import Bar, output_file, show
import numpy as np
import os

app = Flask(__name__)
# obtain the Data
# Set the stock we are interested in, AAPL is Apple stock code
stock = 'AAPL'
# Your code
api_url = 'https://www.quandl.com/api/v1/datasets/WIKI/%s.json' % stock
session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
raw_data = session.get(api_url)

# Probably want to check that requests.Response is 200 - OK here
# to make sure we got the content successfully.

# requests.Response has a function to return json file as python dict
aapl_stock = raw_data.json()
# We can then look at the keys to see what we have access to
aapl_stock.keys()
# column_names Seems to be describing the individual data points
aapl_stock['column_names']
# A big list of data, lets just look at the first ten points...
data = (aapl_stock['data'][0:20])

# convert to pandas dataframe
df = DataFrame(data, columns=['Date', 'open', 'high', 'low', 'close', 'volume', 'ex-dividend', 'split-ratio', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volumne'])

feature_names = ['open', 'high', 'low', 'close', 'volume', 'ex-dividend', 'split-ratio', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volumne']


def create_figure(current_feature_name):

    p = figure(title=stock, x_axis_label='Date', x_axis_type="datetime", y_axis_label=current_feature_name)

    Dates = np.array(df['Date'], dtype=np.datetime64)
    p.line(Dates, df[current_feature_name], line_width=2)
    return p


@app.route('/', methods=['GET', 'POST'])
def indexplotter():

    # Determine selected feature
    if request.method == 'GET':
        current_feature_name = 'adj_close'
    else:
        current_feature_name = request.args.get("feature_name")

    # Create the plot
    plot = create_figure(current_feature_name)

    # Embed plot into HTML via Flask Render
    script, div = components(plot)
    return render_template("plotter.html", script=script, div=div, current_feature_name=current_feature_name, feature_names=feature_names)



# With debug=True, Flask server will auto-reload
# when there are code changes
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33407))
    app.run(host='0.0.0.0', port=port, debug=True)
