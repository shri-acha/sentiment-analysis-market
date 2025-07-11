from collections import defaultdict
import praw

# Fetch logic


def fetch_reddit_data():

    # # Using old.reddit.com testing old.reddit.com for better data
    # urls = ['https://old.reddit.com/r/wallstreetbets/hot.json',
    #         'https://old.reddit.com/r/stocks/hot.json']
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
    # }
    # for url in urls:
    #     page = requests.get(url, headers=headers)
    #     resulting_json = json.loads(page.text)
    #     children_arr = resulting_json['data']['children']
    #     print(f'No of posts: {len(children_arr)}')
    #     print('Post titles:')
    #     for child in children_arr:
    #         print(child['data']['title'])

    result = list(defaultdict())

    reddit = praw.Reddit(
        client_id='EzucZsNMCoEBqiJ8mTxuPw',
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        client_secret='SpVszVM1EZdbCyu8Epl5-6YNi2KiNg',
    )

    subreddit = reddit.subreddit('wallstreetbets')

    for (inx, submission) in enumerate(subreddit.hot(limit=5)):
        # print(f"Post Title: {submission.title}")
        # print(f"Post Content: {submission.selftext}")
        result_dict = defaultdict()

        result_dict['title'] = submission.title
        result_dict['content'] = submission.content
        result_dict['comments'] = list()

        submission.comments.replace_more(limit=0)
        top_comments = submission.comments[:10]

        for i, comment in enumerate(top_comments, 1):
            result_dict['content'].append(comment.body[:150])
            # print(f'Top comment [{i}]: {comment.body[:150]}')
        result.append(result_dict)

    return result

