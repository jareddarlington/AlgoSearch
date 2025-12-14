import json
import urllib.request
import feedparser
import time

'''
See arxiv.org/help/api/user-manual.html
'''

DATA_PATH = "data/data.jsonl"
ALGO_PATH = "data/algorithms/"

base_url = 'http://export.arxiv.org/api/'
method_name = 'query'
search_query = 'cat:cs.DS'
id_list = ''
start = 0  # number of result to start from
total_results = 10  # total number of results to fetch
results_per_iteration = 5  # number of results to fetch per api call
wait_time = 1  # number of seconds to wait between calls

with open(DATA_PATH, "w") as f:
    for i in range(start, total_results, results_per_iteration):  # make requests in batches

        print(f'--- Results {i} - {i + results_per_iteration - 1} ---\n')

        url = f'{base_url}{method_name}?search_query={search_query}&id_list={id_list}&start={i}&max_results={results_per_iteration}'

        data = urllib.request.urlopen(url).read()
        feed = feedparser.parse(data)

        for entry in feed.entries:

            title = entry.title
            authors = [author.name for author in entry.authors]
            published = entry.published
            updated = entry.updated
            abstract = entry.summary
            categories = [tag['term'] for tag in entry.tags]
            arxiv_id = entry.id.split("/abs/")[-1]
            pdf_link = f'https://arxiv.org/pdf/{arxiv_id}'

            paper = {
                "title": title,
                "authors": authors,
                "published": published,
                "updated": updated,
                "abstract": abstract,
                "categories": categories,
                "arxiv_id": arxiv_id,
                "pdf_link": pdf_link,
            }

            # TODO: query LLM to extract algorithms from paper and iterate through them to create json objects

            algo_id = "TODO"
            text_path = "TODO"
            algo_name = "TODO"
            description = "TODO"
            variables = "TODO"
            embeddings = "TODO"

            algorithm = {
                "algo_id": algo_id,
                "arxiv_id": arxiv_id,
                "title": title,
                "published": published,
            }

            f.write(json.dumps(algorithm) + "\n")

            print(f'Title: {title}')
            print(f'{entry.id}')
            print()

        time.sleep(wait_time)
