from enum import member
import json
import urllib.request
import feedparser
import time
import tarfile
from pathlib import Path
from clean_tex import clean_tex

'''
See arxiv.org/help/api/user-manual.html
'''

DATA_PATH = 'data/papers.jsonl'  # algorithm metadata storage
ALGO_PATH = 'data/algorithms/'  # for algorithm tex txt files

BASE_URL = 'http://export.arxiv.org/'
METHOD_NAME = 'query'
SEARCH_QUERY = 'cat:cs.DS'
ID_LIST = ''

START = 0
TOTAL_RESULTS = 20
BATCH_SIZE = 5
WAIT_TIME = 1  # time to wait between batches (in seconds)


def build_tex_file(id: str, max_size: int = 5000000):
    '''
    id: arxiv id of paper
    max_size: max size of tex files in bytes.
    '''

    url = f'{BASE_URL}src/{id}'
    tex = ''

    with urllib.request.urlopen(url) as response:  # get source archive
        # Check if there's actually a gzipped file
        content_type = response.headers.get('Content-Type', '')
        if 'gzip' not in content_type and 'application/x-tar' not in content_type:
            raise Exception(f'Source not available (received {content_type})')

        try:
            with tarfile.open(fileobj=response, mode='r:gz') as tar:  # extact files from archive
                for member in tar:
                    # Skip irregular files and non tex files
                    if not member.isfile() or not member.name.lower().endswith('.tex'):
                        continue

                    # Grab file
                    file = tar.extractfile(member)
                    if file is None:
                        continue

                    # Skip papers whose tex files are too large
                    data = file.read(max_size + 1)
                    if len(data) > max_size:
                        raise Exception('One or more .tex files are too large')

                    tex += f'FILE NAME: {member.name}\n\n' + data.decode('utf-8', errors='replace') + '\n\n'
        except tarfile.ReadError:
            raise Exception('Invalid or corrupted archive - source may not be available')

    if not tex:
        raise Exception(f'No valid .tex files found')

    tex = clean_tex(tex)

    # write tex to txt for processing later
    with open(f'{ALGO_PATH}/{id}.txt', 'w', encoding='utf-8') as f:
        f.write(tex)


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


def build_paper_metadata(entry):
    return {
        'title': entry.title,
        'authors': [author.name for author in entry.authors],
        'published': entry.published,
        'updated': entry.updated,
        'abstract': entry.summary,
        'categories': [tag['term'] for tag in entry.tags],
        'id': entry.id.split('/abs/')[-1],
    }


def build(
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

    total_fetched = 0
    total_successful = 0

    # Load existing paper IDs to avoid reprocessing
    existing_ids = set()
    if Path(data_path).exists():
        with open(data_path, 'r') as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    continue

    with open(data_path, 'a') as f:
        for i in range(start, start + total_results, batch_size):  # make requests in batches

            if verbose:
                print(f'--- Results {i} - {i + batch_size - 1} ---\n')

            feed = get_feed(start=i, num_results=batch_size)

            for entry in feed.entries:  # iterate through papers in batch
                id = entry.id.split('/abs/')[-1]

                if verbose:
                    print(f'Paper {total_fetched}')
                    print(f'Title: {entry.title} ({id})')

                total_fetched += 1

                # Skip if already attempted
                if id in existing_ids:
                    if verbose:
                        print('Skipping: attempted previously\n')
                    continue

                metadata = build_paper_metadata(entry)

                try:  # process paper
                    build_tex_file(id)
                    metadata['status'] = 'success'
                    metadata['error'] = None
                    f.write(json.dumps(metadata) + "\n")

                    total_successful += 1

                    if verbose:
                        print('Processed successfully\n')
                except Exception as e:  # store metadata for failed papers
                    metadata['status'] = 'failed'
                    metadata['error'] = str(e)
                    f.write(json.dumps(metadata) + "\n")

                    if verbose:
                        print(f'Error: {e}\n')

                existing_ids.add(id)

            f.flush()
            time.sleep(wait_time)

    return total_fetched, total_successful


def __main__():
    total_fetched, total_successful = build(DATA_PATH, START, TOTAL_RESULTS, BATCH_SIZE, WAIT_TIME)
    print(total_successful, total_fetched)


if __name__ == '__main__':
    __main__()
