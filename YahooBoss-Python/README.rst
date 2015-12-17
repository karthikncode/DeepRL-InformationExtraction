================
YahooBoss-Python
================

Simple Wrapper Around the Yahoo Boss Search API

See https://developer.yahoo.com/boss/search/

---------
YahooBoss
---------

Base class. Should not be called directly.

-----------
PlaceFinder
-----------

Use PlaceFinder to locate an address or place

Usage
=====

Find the coordinates of a place o address

::

	from yahooboss import PlaceFinder

	pf = Placefinder(your_key,your_secret)
	results = pf.lookup( 'Address or Place Name' )
	latitude = results[0].get('latitude')
	longitude = results[0].get('longitude')


Find an address by latitude and longitude

::

	from yahooboss import PlaceFinder
	pf = Placefinder(your_key, your_secret)
	results = pf.reverse( lat, lon )
	address = results[0]

	street = address.get('line1')
	city = address.get('city')
	state = address.get('state') # or address.get('statecode') for the abbreviation
	zip = address.get('uzip')


Methods
=======

`__init__( key, secret )`
-------------------------

Extends ``YahooBoss``, returns a ``PlaceFinder`` object

Arguments:

- `key`- YahooBoss oAuth key.
- `secret` - YahooBoss oAuth secret

`lookup( place )`
----------------------

Accepts an address or place name and returns a validated address and geocode

Arguments:

- `place` - Address or place name

Returns ``dict``

::

	{
		"count": "1",
		"start": "0",
		"request": "count=1&flags=J&q=2435%2BN.%2BCentral%2BExpressway%252C%2BRichardson%2BTX%2B75080",
		"results": [
			{
				"neighborhood": "",
				"house": "2435",
				"county": "Dallas County",
				"street": "N Central Expy",
				"radius": "400",
				"quality": "87",
				"unit": "",
				"city": "Richardson",
				"countrycode": "US",
				"woeid": "12790359",
				"xstreet": "",
				"line4": "United States",
				"line3": "",
				"line2": "Richardson, TX 75080-2717",
				"line1": "2435 N Central Expy",
				"state": "Texas",
				"latitude": "32.98529",
				"hash": "4289EC860C737D3F",
				"unittype": "",
				"offsetlat": "32.985249",
				"statecode": "TX",
				"postal": "75080-2717",
				"name": "",
				"uzip": "75080",
				"country": "United States",
				"longitude": "-96.711983",
				"countycode": "",
				"offsetlon": "-96.713814",
				"addressmatchtype": "POINT_ADDRESS",
				"woetype": "11"
			}
		]
	}

`reverse( lat, lon )`
---------------------

Finds an address based on the given coordinates.

Arguments:

- `lat` - Latitude of the location
- `lon` - Longitude of the location

Returns ``dict``

::

	{
		"count": "1", 
		"start": "0", 
		"request": "count=1&flags=J&gflags=R&location=32.98529%2B-96.711983&q=2435%2BN.%2BCentral%2BExpressway%252C%2BRichardson%252C%2BTX%2B75080", 
		"results": [
			{
				"neighborhood": "", 
				"house": "2435", 
				"county": "Dallas County", 
				"street": "N Central Expy", 
				"radius": "400", 
				"quality": "87", "unit": "", 
				"city": "Richardson", 
				"countrycode": "US", 
				"woeid": "12790359", 
				"xstreet": "", 
				"line4": "United States", 
				"line3": "", 
				"line2": "Richardson, TX 75080-2717", 
				"line1": "2435 N Central Expy", 
				"state": "Texas", 
				"latitude": "32.98529", 
				"hash": "4289EC860C737D3F", 
				"unittype": "", 
				"offsetlat": "32.98529", 
				"statecode": "TX", 
				"postal": "75080-2717", 
				"name": "32.98529 -96.711983", 
				"uzip": "75080", 
				"country": "United States", 
				"longitude": "-96.711983", 
				"countycode": "", 
				"offsetlon": "-96.711983", 
				"addressmatchtype": "POINT_ADDRESS", 
				"woetype": "11"
			}
		]
	}

`placefinder( place )`
----------------------

Alias for `lookup()` to remain compatible with 0.1.x

----------
BossSearch
----------

Use BossSearch to search the web or news articles

Usage
=====

::

	from yahooboss import BossSearch
	bs = BossSearch(your_key, your_secret)

	web_results = bs.search_web( 'search parameters' )

	for wr in web_results:
		print 'Title: %s' % wr.get('title')
		print 'Summary: %s' % wr.get('abstract')
		print 'URL: %s' % wr.get('url')

	news_results = bs.search_news( 'search parameters' )

	for r in news_results:
		print 'Title: %s' % r.get('title')
		print 'Summary: %s' % r.get('abstract')
		print 'Source: %s' % r.get('source')
		print 'Date: %s' % r.get('date')
		print 'URL: %s' % r.get('url')


Methods
=======

`__init__(key, secret, **kwargs)`
---------------------------------

Extends YahooBoss, returns a ``BossSearch`` object

Arguments

- `key` - YahooBoss oAuth key
- `secret` - YahooBoss oAuth secret

Keyword arguments:

- `age` - Max age of the results (e.g. "7d". See the Search BOSS documentation for a complete list)
- `urls` - list of urls to search. (Can be partial urls, see the BOSS documentation)
- `results_per_page` - number of results per page (or per request)
- `sites` - list or comma separated string of sites. BOSS seems to be using this in favor of urls.

`search_web(query, page_num=1,**kwargs)`
----------------------------------------

Searches the web for the specified query and returns a list of results

Arguments:

- `query` - the query string to search for
- `page_num` - start at page (default: 1)

Keyword arguments:

- Same arguments as passed to `__init__()`

Returns a list of ``dict`` results:

::

	[{
		"dispurl": "starwars.wikia.com/wiki/<b>Yoda</b>",
		"title": "<b>Yoda</b> - Wookieepedia, the Star Wars Wiki",
		"url": "http://starwars.wikia.com/wiki/Yoda",
		"abstract": "<b>Yoda</b> was one of the most renowned and powerful Jedi Masters in galactic history. He was known for his legendary wisdom, mastery of the Force and skills in lightsaber ...",
		"clickurl": "http://starwars.wikia.com/wiki/Yoda",
		"date": ""
	}]

`search_news(query,page_num=1,**kwargs)`
----------------------------------------

Search Yahoo News for the specified query

Arguments:

- `query` - string to search for
- `page_num` - start at page (default: 1)

Keyword arguments:

- same arguments as passed to `__init__()`

Returns a list of ``dict`` results:

::

	[{
		"sourceurl": "http://abcnews.go.com/",
		"language": "en english",
		"title": "Ferguson Library Becomes Oasis of Calm Amid Strife",
		"url": "http://abcnews.go.com/US/ferguson-library-refuge-adults-children-amid-strife/story?id=25050930",
		"abstract": "Ferguson library has become an oasis of calm and activities for children while school postpone during the street protests over the police shooting of Michael Brown.",
		"clickurl": "",
		"source": "ABC News",
		"date": "1408554014"
	}]


`make_request(bucket, query, page_num)`
---------------------------------------

Used internally to make a raw request to YahooBoss services. Can be used to make a request
to one of the services not currently covered by a wrapper function.

Arguments:

- `bucket` - yahoo service bucket (e.g. "news")
- `query` - query string to search
- `page_num` - page to start at

Returns a list of dicts. The structure of the dict depends on the return value from Yahoo

----
TODO
----

The module currently does not support Yahoo's "LimitedWeb", "Images", "Spelling", "Blogs" or "Related Search".

These can be requested via the ``BossSearch.make_request`` method

------
Author
------

ConstituentVoice - opensource@constituentvoice.com

-----------------
Copyright / Legal
-----------------

Use of this module requires YahooBoss credentials and agreement to Yahoo Inc.'s Terms of Service

Yahoo, YahooBoss, BossSearch, and PlaceFinder are trademarks and property of Yahoo Inc.

ConstituentVoice is not affiliated in any way with Yahoo Inc.

This software is licensed under the terms of the BSD License. It is provided to you free to modify
or redistribute but with NO WARRANTY. See LICENSE.txt for details.

Copyright (c) 2015 ConstituentVoice

