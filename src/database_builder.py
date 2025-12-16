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

BASE_URL = 'http://export.arxiv.org/'
METHOD_NAME = 'query'
SEARCH_QUERY = 'cat:cs.DS'
ID_LIST = ''

START = 11
TOTAL_RESULTS = 1
BATCH_SIZE = 1
WAIT_TIME = 1


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


def get_feed(
    method_name: str = METHOD_NAME,
    search_query: str = SEARCH_QUERY,
    id_list=ID_LIST,
    start: int = 0,
    num_results: int = 1,
):
    url = f'{BASE_URL}api/{method_name}?search_query={search_query}&id_list={id_list}&start={start}&max_results={num_results}'
    data = urllib.request.urlopen(url).read()
    feed = feedparser.parse(data)

    return feed


def extract_paper_metadata(entry):
    return {
        'title': entry.title,
        'authors': [author.name for author in entry.authors],
        'published': entry.published,
        'updated': entry.updated,
        'abstract': entry.summary,
        'categories': [tag['term'] for tag in entry.tags],
        'arxiv_id': entry.id.split('/abs/')[-1],
    }


def extract_algorithms():
    # TODO: use llm to do this

    pass


def build_algorithm_json():
    pass
    # return {
    #     'algo_id': algo_id,
    #     'text_path': text_path,
    #     'algo_name': algo_name,
    #     'paper_metadata': paper_metadata,
    #     'description': description,
    #     'variables': variables,
    #     'embeddings': embeddings,
    # }


def build_database(
    data_path: str,
    start: int,
    total_results: int,
    batch_size: int,
    wait_time: int,
    verbose: bool = True,
):
    '''
    data_path: path to .jsonl file for algorithm data.
    start: starting index for request.
    total_results: total number of papers to request.
    batch_size: number of results to fetch per api call
    wait_time: number of seconds between query batches
    verbose: if paper details should be printed
    '''

    total_papers_fetched = 0
    total_papers_no_tex = 0

    with open(data_path, 'w') as f:
        for i in range(start, start + total_results, batch_size):  # make requests in batches
            if verbose:
                print(f'--- Results {i} - {i + batch_size - 1} ---\n')

            feed = get_feed(start=i, num_results=batch_size)

            for entry in feed.entries:  # iterate through papers in batch
                total_papers_fetched += 1

                if verbose:
                    print(f'Paper {total_papers_fetched}')
                    print(f'Title: {entry.title}')
                    print()

                paper_metadata = extract_paper_metadata(entry)

                try:  # download and extract tex source
                    tex_source_url = f'{BASE_URL}src/{paper_metadata['arxiv_id']}'
                    tex = extract_tex(tex_source_url)
                    # with open('test.txt', 'w') as thing:
                    #     thing.write(tex)
                except Exception:
                    total_papers_no_tex += 1
                    break  # skip papers with no tex source

                algorithms = extract_algorithms()

                for i, algo in enumerate(algorithms):
                    # TODO: create json for algos
                    algo_json = ''
                    f.write(json.dumps(algo_json) + '\n')

            f.flush()
            time.sleep(wait_time)

    return total_papers_no_tex / total_papers_fetched  # return ratio of papers with no tex


def __main__():
    ratio = build_database(DATA_PATH, START, TOTAL_RESULTS, BATCH_SIZE, WAIT_TIME)
    print(ratio)


if __name__ == "__main__":
    __main__()
