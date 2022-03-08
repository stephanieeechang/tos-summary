import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Iterable, List, Dict, Any, Tuple

from urllib import parse

import scrapy
import scrapy.crawler

PUSHSHIFT_API_REDDIT_BASE_URL = r"https://api.pushshift.io/reddit"
REDDIT_API_BASE_URL = r'https://www.reddit.com'


class PushShiftAndRedditAPICrawler(scrapy.Spider):
    """
    This crawler uses the pushshift reddit cache to search for terms from a large time window and uses the Reddit API to get the most up-to-date information from Reddit.
    This is necessary because pushshift stores reddit object at the time it was posted so no elaborate information is contained in the pushshift cache. We can use the Submission
    ID and comment ID to reconstruct what we need.

    The flow can be described as follows:
        -> search pushshift cache for relevant submissions
        -> use the submission link_ids to fetch the most recent information from the Reddit API
        -> use the submission link_ids to fetch the most recent comments from the reddit API
    """

    name = 'PushShiftAndRedditAPICrawler'

    def __init__(self, keywords: List[str], start_time: Union[datetime, int, float],
                 end_time: Union[datetime, int, float],
                 interval: Union[timedelta, int, float], submissions_per_interval: int = 50,
                 pushshift_fields: List[str] = ['link_id'], output_dir: Path = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.keywords: List[str] = keywords
        self.output_dir: Path = output_dir
        self.start_time: datetime
        self.end_time: datetime
        self.interval: timedelta
        self.submissions_per_interval = min(submissions_per_interval, 500)
        self.pushshift_fields = pushshift_fields

        if not self.output_dir.exists():
            self.output_dir.mkdir()

        if isinstance(start_time, datetime):
            self.start_time = start_time
        elif isinstance(start_time, int) or isinstance(start_time, float):
            self.start_time = datetime.fromtimestamp(start_time)

        if isinstance(end_time, datetime):
            self.end_time = end_time
        elif isinstance(end_time, int) or isinstance(end_time, float):
            self.end_time = datetime.fromtimestamp(end_time)

        if isinstance(interval, timedelta):
            self.interval = interval
        elif isinstance(interval, int) or isinstance(interval, float):
            self.interval = timedelta(seconds=interval)

    def _intervals(self) -> Iterable[datetime]:
        current_datetime = self.start_time
        while current_datetime < self.end_time:
            next_datetime = current_datetime + self.interval
            if next_datetime < self.end_time:
                yield next_datetime
            else:
                break
            current_datetime = next_datetime
        yield self.end_time

    def _search_pushshift_submissions(self, keyword: str, before: int, after: int, size: int = 50,
                                      fields: List[str] = None) -> scrapy.Request:
        query_parameters = [
            ("q", keyword),
            ('before', before),
            ('after', after),
            ('size', size)
        ]

        if fields is not None:
            query_parameters.append((
                'fields', ','.join(fields)
            ))

        query_string = parse.urlencode(query_parameters)
        return scrapy.Request(
            url=f"{PUSHSHIFT_API_REDDIT_BASE_URL}/search/submission/?{query_string}",
            cb_kwargs={"identifier": query_string}  # to identify output file
        )

    def _fetch_submission(self, submission_link_id: str, raw_json: int = 1) -> scrapy.Request:
        query_string = parse.urlencode([('raw_json', raw_json)])
        return scrapy.Request(
            url=f'{REDDIT_API_BASE_URL}/by_id/{submission_link_id}.json?{query_string}',
            cb_kwargs={'identifier': query_string}
        )

    def _fetch_comments(self, submission_link_id: str, raw_json: int = 1, identifier: str = '') -> scrapy.Request:
        link_id_without_type = submission_link_id[3:]  # strips the type prefixes: https://www.reddit.com/dev/api
        query_string = parse.urlencode([('raw_json', raw_json)])
        return scrapy.Request(
            url=f'{REDDIT_API_BASE_URL}/comments/{link_id_without_type}.json?{query_string}',
            cb_kwargs={'identifier': identifier}
        )

    def start_requests(self):
        for keyword in self.keywords:
            time_range_start = self.start_time
            for time_range_end in self._intervals():
                req = self._search_pushshift_submissions(
                    keyword=keyword,
                    after=int(time_range_start.timestamp()),
                    before=int(time_range_end.timestamp()),
                    size=self.submissions_per_interval,
                    fields=self.pushshift_fields
                )
                req.callback = self.parse
                yield req
                time_range_start = time_range_end

    def parse(self, response: scrapy.http.TextResponse, **kwargs):
        # parse the searched submissions returned by pushshift api
        identifier = kwargs['identifier']
        data = response.json()['data']
        for submission in data:
            link_id = submission['id']
            req = self._fetch_comments(submission_link_id=f't3_{link_id}', identifier=identifier)
            req.callback = self.parse_comments
            yield req

    def parse_comments(self, response: scrapy.http.TextResponse, **kwargs):
        identifier = kwargs['identifier']
        reddit_response_object = response.json()  # this is a JSON array
        submission: Dict[str, Any]
        comments: List[Dict[str, Any]]
        submission_reddit_response_object, comments_reddit_response_object = reddit_response_object
        submission = submission_reddit_response_object['data']['children'][0]
        comments = comments_reddit_response_object['data']['children']

        yield {
            'submission': submission,
            'comments': comments,
            'identifier': identifier
        }


class RawSubmissionCommentsItemPipeline:

    def __init__(self):
        self.logger: logging.Logger
        self.output_dir: Path

    def open_spider(self, spider: PushShiftAndRedditAPICrawler):
        self.logger = spider.logger
        self.output_dir = spider.output_dir

    def process_item(self, item: Dict[str, Dict[str, Any]], spider: scrapy.Spider):
        submission, comments, identifier = item['submission'], item['comments'], item['identifier']
        submission_name = submission['data'][
            'name']  # fullname to locate this submission on Reddit. This looks like t3_<base36 id>
        raw_combined_json = {
            'submission': submission,
            'comments': comments
        }
        output_file = self.output_dir / f'{identifier}-{submission_name}.json'
        json.dump(raw_combined_json, output_file.open('w'))
        self.logger.info(f'Combined JSON wrote to {str(output_file.absolute())}')

        return item


if __name__ == '__main__':
    crawling_settings = {
        'ITEM_PIPELINES': {
            'crawlers.RawSubmissionCommentsItemPipeline': 800
        },
        'LOG_LEVEL': 'INFO'
    }

    crawler_process = scrapy.crawler.CrawlerProcess(
        settings=crawling_settings
    )

    crawler_process.crawl(
        PushShiftAndRedditAPICrawler,
        keywords=['terms of service', 'privacy policy'],
        start_time=datetime(year=2019, month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
        end_time=datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
        interval=timedelta(days=7),
        submissions_per_interval=100,
        pushshift_fields=['id'],
        output_dir=Path.cwd() / f'data/{PushShiftAndRedditAPICrawler.name}-output/'
    )
    crawler_process.start()
