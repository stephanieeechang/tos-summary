import os
import tweepy as tw
import pandas as pd
import json
import os

cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

with open('./config.json') as f:
    config = json.load(f)
        
API_KEY = config['API_KEY']
API_KEY_SECRET = config['API_KEY_SECRET']
ACCESS_TOKEN = config['ACCESS_TOKEN']
ACCESS_TOKEN_SECRET = config['ACCESS_TOKEN_SECRET']
BEARER_TOKEN = config['BEARER_TOKEN']
SERVER = config['SERVER']
TEST = eval(config['TEST'])

auth = tw.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tw.API(auth, wait_on_rate_limit=True)

search_words = "#TOS"
search_hash = "TOS"
new_search = search_words + " -filter:retweets"
date_since = "2000-01-01"
number_tweets = 1000

def scrape(words, date_since, numtweet):
 
    # Creating DataFrame using pandas
    db = pd.DataFrame(columns=[ 'text',
                                'retweetcount',
                                'favoritecount',
                                'hashtags'])

    # We are using .Cursor() to search
    # using .items(number of tweets) to restricted number of tweet
    tweets = tw.Cursor(api.search_tweets,
                        words, lang="en",
                        since_id=date_since,
                        tweet_mode='extended').items(numtweet)

    list_tweets = [tweet for tweet in tweets]

    # Counter for Tweet Count
    # i = 0

    # we will iterate over each tweet to extract info
    for tweet in list_tweets:
        retweetcount = tweet.retweet_count
        favoritecount = tweet.favorite_count
        hashtags = tweet.entities['hashtags']

        # In case the tweet is an invalid reference,
        # except block will be executed
        try:
            text = tweet.retweeted_status.full_text
        except AttributeError:
            text = tweet.full_text
        hashtext = list()
        has_hash = False
        for j in range(0, len(hashtags)):
            hashtag_text = hashtags[j]['text']
            if hashtag_text.lower() == search_hash.lower():
                has_hash = True
            hashtext.append(hashtag_text)
        if not has_hash:
            continue

        # Appending all the information in the DataFrame
        ith_tweet = [text, retweetcount, favoritecount, hashtext]
        db.loc[len(db)] = ith_tweet

        # Function call to print tweet data on screen
        # printtweetdata(i, ith_tweet)
        # i = i+1
    return db

def printtweetdata(n, ith_tweet):
    print()
    print(f"Tweet {n}:")
    print(f"Tweet Text:{ith_tweet[0]}")
    print(f"Retweet Count:{ith_tweet[1]}")
    print(f"Favorite Count:{ith_tweet[2]}")
    print(f"Hashtags Used:{ith_tweet[3]}")

db = scrape(search_words, date_since, number_tweets)
print(len(db))
# print(db)
sorted_db = db.sort_values(["retweetcount", "favoritecount"], ascending = (False, False))