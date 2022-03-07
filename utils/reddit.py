import os
import sys
from pathlib import Path
from os import PathLike

import scrapy

import urllib
from urllib import request, parse

PUSHSHIFT_API_REDDIT_BASE_URL = r'https://api.pushshift.io/reddit'


class PushShiftAndRedditAPICrawler(scrapy.Spider):
    """
    This crawler uses the pushshift reddit cache to search for terms from a large time window and uses the Reddit API to get the most up-to-date information from Reddit.
    This is necessary because pushshift stores reddit object at the time it was posted so no elaborate information is contained in the pushshift cache. We can use the Submission
    ID and comment ID to reconstruct what we need.
    """

    def _search_submission(self, keyword: str) -> scrapy.Request:
        query_string = parse.urlencode([('q', keyword)])
        return scrapy.Request(url=f'{PUSHSHIFT_API_REDDIT_BASE_URL}/submission/?{query_string}')

    # def searc