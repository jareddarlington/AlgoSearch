from enum import member
import json
import urllib.request
import feedparser
import time
import tarfile
from pathlib import Path

'''
See arxiv.org/help/api/user-manual.html
'''

DATA_PATH = 'data/data.jsonl'  # algorithm metadata storage
ALGO_PATH = 'data/algorithms/'  # for algorithm text files

base_url = 'http://export.arxiv.org/api/'
method_name = 'query'
search_query = 'cat:cs.DS'
id_list = ''
start = 11  # number of result to start from
total_results = 1  # total number of results to fetch
results_per_iteration = 1  # number of results to fetch per api call
wait_time = 1  # number of seconds to wait between calls

total_papers_fetched = 0
total_papers_no_tex = 0


def extract_tex(url: str, max_size: int = 5000000) -> str:
    '''
    url: url of tex source archive in string format.
    max_size: max size of tex files in bytes.
    '''

    tex = ''

    with urllib.request.urlopen(url) as response:  # get source archive
        with tarfile.open(fileobj=response, mode='r:gz') as tar:  # extact files from archive
            for member in tar:
                # Skip irregular files and non tex files
                if not member.isfile() or not member.name.lower().endswith('.tex'):
                    continue

                # Grab file
                file = tar.extractfile(member)
                if file is None:
                    continue

                # TODO: instead of basing max_size on bytes, do it on token count and make sure (file token count + prompt token count <= max model input length) so i can put prompts and file into summarization model together without weird chunking

                # Skip papers whose tex files are too large
                data = file.read(max_size + 1)
                if len(data) > max_size:
                    raise Exception('One or more TeX files are too large')

                tex += f'FILE NAME: {member.name}\n\n' + data.decode('utf-8', errors='replace') + '\n\n'

    if not tex:
        raise Exception('No valid TeX files')

    return tex


# TODO: abstract into functions

with open(DATA_PATH, 'w') as f:
    for i in range(start, start + total_results, results_per_iteration):  # make requests in batches
        print(f'--- Results {i} - {i + results_per_iteration - 1} ---\n')

        # Build query URL and fetch data
        url = f'{base_url}{method_name}?search_query={search_query}&id_list={id_list}&start={i}&max_results={results_per_iteration}'
        data = urllib.request.urlopen(url).read()
        feed = feedparser.parse(data)

        for entry in feed.entries:  # iterate through papers in batch

            print(f'Title: {entry.title}')
            print()

            # Grab paper metadata
            title = entry.title
            authors = [author.name for author in entry.authors]
            published = entry.published
            updated = entry.updated
            abstract = entry.summary
            categories = [tag['term'] for tag in entry.tags]
            arxiv_id = entry.id.split('/abs/')[-1]

            # Download and extract tex source
            tex_source_url = f'https://arxiv.org/src/{arxiv_id}'
            total_papers_fetched += 1

            try:
                tex = extract_tex(tex_source_url)
                with open('test.txt', 'w') as thing:
                    thing.write(tex)

            except Exception as e:
                total_papers_no_tex += 1
                break  # for now, skip papers without tex source

            paper = {
                'title': title,
                'authors': authors,
                'published': published,
                'updated': updated,
                'abstract': abstract,
                'categories': categories,
                'arxiv_id': arxiv_id,
            }

            # TODO: query LLM to extract algorithms from paper and iterate through them to create json objects

            algo_id = 'TODO'
            text_path = 'TODO'
            algo_name = 'TODO'
            description = 'TODO'
            variables = 'TODO'
            embeddings = 'TODO'

            algorithm = {
                'algo_id': algo_id,
                'text_path': text_path,
                'algo_name': algo_name,
                'paper': paper,
                'description': description,
                'variables': variables,
                'embeddings': embeddings,
            }

            f.write(json.dumps(algorithm) + '\n')

        time.sleep(wait_time)

print(total_papers_no_tex / total_papers_fetched)


def __main__():
    print("test")


if __name__ == "__main__":
    __main__()
