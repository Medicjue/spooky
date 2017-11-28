#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:00:21 2017

Google Way


### Need to understand Google Custom Search API
"""
import urllib
import json

from googleapiclient.discovery import build

service = build("customsearch", "v1",
            developerKey="AIzaSyA1REQLMfkVqCodNwdtarSoFGRRJP_ICGU")

res = service.cse().list(
      q='Still, as I urged our leaving Ireland with such inquietude and impatience, my father thought it best to yield.',
      cx='015610619087729652789:h_5xksiuwpa',
    ).execute()

raw_output = str(res)
'Mary Shelley' in raw_output
'Allan Poe' in raw_output
'HP Lovecraft' in raw_output

def search(search_string):
  query = urllib.parse.urlencode({'q': search_string})
  url = 'http://ajax.googleapis.com/ajax/services/search/web?v=1.0&{}&rsz=large'.format(query)
  search_response = urllib.request.urlopen(url)
  search_results = search_response.read().decode("utf8")
  results = json.loads(search_results)
  print(results)
  data = results['responseData']
  
  print('Total results: %s' % data['cursor']['estimatedResultCount'])
  hits = data['results']
  print('Top %d hits:' % len(hits))
  for h in hits: print(' ', h['url'])
  print('For more results, see %s' % data['cursor']['moreResultsUrl'])
  return hits

search('QQ')
