import urllib, urllib.request

method_name = 'query'
parameters = {
    'search_query': 'all:electron',
    'id_list': '',
    'start': '0',
    'max_results': '1',
}
url = f'http://export.arxiv.org/api/{method_name}?{"&".join([k + "=" + v for k, v in parameters.items()])}'
data = urllib.request.urlopen(url)
print(data.read().decode('utf-8'))
