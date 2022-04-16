import pandas as pd
import tweepy as tw
import re

tweets = pd.read_pickle('twitter_privacy1000.pkl')
print(len(tweets))
tweets['text_processed'] = tweets['text'].map(lambda x: re.sub('[,\.!?@]', '', x))

tweets['text_processed'] = tweets['text_processed'].map(lambda x: x.lower())

print(tweets['text_processed'].head())

# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(papers['paper_text_processed'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()