import dataclasses
from dataclasses import dataclass

from typing import Union, List


@dataclass
class RedditResponseObject:
    kind: str
    data: 'RedditObject'


@dataclass
class RedditObject:
    before: float
    after: float
    dist: int
    modhash: str
    geo_filter: ""
    children: List[RedditResponseObject]


@dataclass
class Submission:
    approved_at_utc: float
    subreddit: str
    selftext: str
    author_fullname: str
    saved: bool
    mod_reason_title: str
    gilded: int
    clicked: bool
    title: str
    link_flair_richtext: []
    subreddit_name_prefixed: str
    hidden: bool
    pwls: int
    link_flair_css_class: str
    downs: int
    thumbnail_height: int
    top_awarded_type: str
    hide_score: bool
    name: str
    quarantine: bool
    link_flair_text_color: str
    upvote_ratio: float
    author_flair_background_color: str
    subreddit_type: str
    ups: int
    total_awards_received: int
    media_embed: {}
    thumbnail_width: int
    author_flair_template_id: str
    is_original_content: bool
    user_reports: []
    secure_media: str
    is_reddit_media_domain: bool
    is_meta: bool
    category: str
    secure_media_embed: {}
    link_flair_text: str
    can_mod_post: bool
    score: int
    approved_by: str
    is_created_from_ads_ui: bool
    author_premium: bool
    thumbnail: str
    edited: bool
    author_flair_css_class: str
    author_flair_richtext: []
    gildings: {}
    content_categories: str
    is_self: bool
    mod_note: str
    created: float
    link_flair_type: str
    wls: int
    removed_by_category: str
    banned_by: str
    author_flair_type: str
    domain: str
    allow_live_comments: bool
    selftext_html: str
    likes: str
    suggested_sort: str
    banned_at_utc: str
    view_count: str
    archived: bool
    no_follow: bool
    is_crosspostable: bool
    pinned: bool
    over_int: bool
    all_awardings: []
    awarders: []
    media_only: bool
    link_flair_template_id: str
    can_gild: bool
    spoiler: bool
    locked: bool
    call_to_action: str
    author_flair_text: str
    treatment_tags: []
    visited: bool
    removed_by: str
    num_reports: str
    distinguished: str
    subreddit_id: str
    author_is_blocked: bool
    mod_reason_by: str
    removal_reason: str
    link_flair_background_color: str
    id: str
    is_robot_indexable: bool
    report_reasons: str
    author: str
    discussion_type: str
    num_comments: int
    send_replies: bool
    whitelist_status: str
    contest_mode: bool
    mod_reports: []
    author_patreon_flair: bool
    author_flair_text_color: str
    permalink: str
    parent_whitelist_status: str
    stickied: bool
    url: str
    subreddit_subscribers: int
    created_utc: float
    num_crossposts: int
    media: str
    is_video: bool


# @dataclass
# class Comment:
#     subreddit_id: str
#     approved_at_utc: float
#     author_is_blocked: bool
#     comment_type: null
#     awarders: []
#     mod_reason_by: null
#     banned_by: null
#     author_flair_type: str
#     total_awards_received: 0
#     subreddit: str
#     author_flair_template_id: null
#     likes: null
#     replies: ""
#     user_reports: []
#     saved: bool
#     id: str
#     banned_at_utc: null
#     mod_reason_title: null
#     gilded: 0
#     archived: bool
#     collapsed_reason_code: null
#     no_follow: bool
#     author: str
#     can_mod_post: bool
#     created_utc: 1646671513.0
#     send_replies: bool
#     parent_id: str
#     score: 1
#     author_fullname: str
#     approved_by: null
#     mod_note: null
#     all_awardings: []
#     collapsed: bool
#     body: str
#     edited: bool
#     top_awarded_type: null
#     author_flair_css_class: null
#     name: str
#     is_submitter: bool
#     downs: 0
#     author_flair_richtext: []
#     author_patreon_flair: bool
#     body_html: str
#     removal_reason: null
#     collapsed_reason: null
#     distinguished: null
#     associated_award: null
#     stickied: bool
#     author_premium: bool
#     can_gild: bool
#     gildings: {}
#     unrepliable_reason: null
#     author_flair_text_color: null
#     score_hidden: bool
#     permalink: str
#     subreddit_type: str
#     locked: bool
#     report_reasons: null
#     created: 1646671513.0
#     author_flair_text: null
#     treatment_tags: []
#     link_id: str
#     subreddit_name_prefixed: str
#     controversiality: 0
#     depth: 0
#     author_flair_background_color: null
#     collapsed_because_crowd_control: null
#     mod_reports: []
#     num_reports: null
#     ups: 1
