from enum import member
import json
import sqlite3
import urllib.request
import feedparser
import time
import tarfile
import subprocess
import os
import signal
from clean_tex import clean_tex

'''
See arxiv.org/help/api/user-manual.html
'''

DATA_PATH = 'data/papers.db'  # algorithm metadata storage
ALGO_PATH = 'data/temp/'  # for algorithm tex txt files

BASE_URL = 'http://export.arxiv.org/'
METHOD_NAME = 'query'
SEARCH_QUERY = 'cat:cs.DS'
ID_LIST = ''

START = 1000
TOTAL_RESULTS = 6000
BATCH_SIZE = 100
WAIT_TIME = 1  # time to wait between batches (in seconds)

# TODO: clean up, optimize, and abstract further
# TODO: store more paper metadata
# TODO: change TOTAL_RESULTS to END?


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
    with open(f'{ALGO_PATH}{id}.txt', 'w', encoding='utf-8') as f:
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
        'status': 'processing',
        'scrape_error': None,
        'extraction_error': None,
        'processed': False,
        'algo_count': 0,
        'model': None,
    }


def init_database(data_path: str):
    '''
    Initialize the SQLite database with the papers table if it doesn't exist.
    '''
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT NOT NULL,
            published TEXT NOT NULL,
            updated TEXT NOT NULL,
            abstract TEXT NOT NULL,
            categories TEXT NOT NULL,
            status TEXT NOT NULL,
            scrape_error TEXT,
            extraction_error TEXT,
            processed INTEGER NOT NULL DEFAULT 0,
            algo_count INTEGER NOT NULL DEFAULT 0,
            model TEXT
        )
    '''
    )

    conn.commit()
    conn.close()


def build(
    data_path: str,
    start: int,
    total_results: int,
    batch_size: int,
    wait_time: int,
    verbose: bool = True,
):
    '''
    data_path: path to .db file for algorithm data.
    start: starting index for request.
    total_results: total number of papers to request.
    batch_size: number of results to fetch per api call
    wait_time: number of seconds between query batches
    verbose: if paper details should be printed
    '''

    total_fetched = 0
    total_successful = 0

    # Initialize database
    init_database(data_path)

    # Connect to database
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()

    # Load existing paper IDs to avoid reprocessing
    cursor.execute('SELECT id FROM papers')
    existing_ids = {row[0] for row in cursor.fetchall()}

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

                # Insert into database
                cursor.execute(
                    '''
                    INSERT INTO papers (id, title, authors, published, updated, abstract, categories, status, scrape_error, extraction_error, processed, algo_count, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                    (
                        metadata['id'],
                        metadata['title'],
                        json.dumps(metadata['authors']),
                        metadata['published'],
                        metadata['updated'],
                        metadata['abstract'],
                        json.dumps(metadata['categories']),
                        metadata['status'],
                        metadata['scrape_error'],
                        metadata['extraction_error'],
                        1 if metadata['processed'] else 0,
                        metadata['algo_count'],
                        metadata['model'],
                    ),
                )

                total_successful += 1

                if verbose:
                    print('Processed successfully\n')

            except Exception as e:  # store metadata for failed papers
                metadata['status'] = 'failed'
                metadata['scrape_error'] = str(e)

                # Insert into database
                cursor.execute(
                    '''
                    INSERT INTO papers (id, title, authors, published, updated, abstract, categories, status, scrape_error, extraction_error, processed, algo_count, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                    (
                        metadata['id'],
                        metadata['title'],
                        json.dumps(metadata['authors']),
                        metadata['published'],
                        metadata['updated'],
                        metadata['abstract'],
                        json.dumps(metadata['categories']),
                        metadata['status'],
                        metadata['scrape_error'],
                        metadata['extraction_error'],
                        1 if metadata['processed'] else 0,
                        metadata['algo_count'],
                        metadata['model'],
                    ),
                )

                if verbose:
                    print(f'Error: {e}\n')

            existing_ids.add(id)

        conn.commit()
        time.sleep(wait_time)

    conn.close()

    return total_fetched, total_successful


def __main__():
    # Prevent Mac from sleeping during execution
    caffeinate_process = None
    try:
        # Start caffeinate to keep system awake
        caffeinate_process = subprocess.Popen(
            ['caffeinate', '-d', '-i', '-m', '-s'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print('System sleep prevention enabled (caffeinate running)')

        total_fetched, total_successful = build(DATA_PATH, START, TOTAL_RESULTS, BATCH_SIZE, WAIT_TIME)
        print(total_successful, total_fetched)

    finally:
        # Ensure caffeinate is terminated when script ends
        if caffeinate_process:
            caffeinate_process.terminate()
            caffeinate_process.wait()
            print('System sleep prevention disabled')


if __name__ == '__main__':
    __main__()
