import numpy as np
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import datetime
import time


username = 'asrafulashiq'
api_key = 'KNdCHZP6NsxUZpxpi8iq'
stream_token = 'a3da4ceagj'

py.sign_in(username, api_key)

stream_id = stream_token

stream_1 = go.Stream(
    token=stream_id,  # link stream id to 'token' key
     maxpoints=80      # keep a max of 80 pts on screen
)

trace1 = go.Scatter(
     x=[],
     y=[],
     mode='lines+markers',
     stream=stream_1         # (!) embed stream id, 1 per trace
   )


data = go.Data([trace1])

layout = go.Layout(title='Time Series')
fig = go.Figure(data=data, layout=layout)

s = py.Stream(stream_id)

s.open()
count = 0

i = 0    # a counter
k = 5    # some shape parameter

while True:
    if s != []:
       x = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
       y = (np.cos(k*i/50.)*np.cos(i/50.)+np.random.randn(1))[0]
       count += 1
       print count
       s.write({'x': x, 'y' : x})
       print "stream write\n-------\n"
s.close()
