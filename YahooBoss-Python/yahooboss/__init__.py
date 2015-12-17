# for accessing yahoo news search
import oauth2 as oauth
import time
import urllib
import json
from traceback import format_exc
import requests

class GeoCodeError(Exception):
	pass


class YahooBoss(object):

	def __init__(self, key, secret,**kwargs):
		self.key = key
		self.secret = secret

class BossSearch(YahooBoss):

	def __init__(self, key, secret, **kwargs):
		super(BossSearch, self).__init__(key,secret,**kwargs)

		self.params = {
			'oauth_version': "1.0",
			'format': 'json'
		}

		self._set_params(**kwargs)

	
	def _set_params( self, **kwargs ):
		self.params['count'] = kwargs.get('results_per_page',50)
		self.age = kwargs.get('age', '7d')
		self.age = kwargs.get('age','7d')
		self.urls = kwargs.get('urls',[])

		if self.urls:
			param_str = ")OR(".join(self.urls)
			self.params['url'] = "(" + param_str + ")"
		
		# looks like Yahoo is going away from urls and uses "sites" now
		if kwargs.get('sites'):
			sites = kwargs.get('sites')

			if isinstance(sites, list ):
				self.params['sites'] = ','.join(sites)
			else:
				self.params['sites'] = sites

	def search_news(self,q,page_num=1,**kwargs):
		
		self._set_params(**kwargs)

		start = (page_num - 1) * self.params.get('count')

		return self.make_request('news',q, start)


	def search_web(self, q, page_num=1,**kwargs ):
		self._set_params(**kwargs)

		start = (page_num - 1) * self.params.get('count')
		return self.make_request('web',q,start)


	def make_request(self,bucket, q,start_at):
		url = "http://yboss.yahooapis.com/ysearch/"+bucket
		consumer = oauth.Consumer(key=self.key,secret=self.secret)
	
		try:
			new_params = {
				'oauth_nonce': oauth.generate_nonce(),
				'oauth_timestamp': int(time.time()),
				'q' : urllib.quote_plus(q),
				'start': start_at
			}
		except:
			# weird unicode problems sometimes
			return []

		r_params = dict( self.params.items() + new_params.items() )
		if bucket == 'news':
			r_params['age'] = self.age

		req = oauth.Request(method="GET",url=url,parameters=r_params)
		signature_method = oauth.SignatureMethod_HMAC_SHA1()

		req.sign_request(signature_method, consumer, None)
		

		#f = urllib.urlopen(req.to_url())
		f = requests.get(req.to_url())

		retval = f.json()
		self.total_results = retval.get('bossresponse').get(bucket).get('totalresults')

		if self.total_results:
			results = retval.get('bossresponse').get(bucket).get('results')
		else:
			results = []

		return results


class PlaceFinder(YahooBoss):
	# placefinder geocoding

	def __init__(self, key,secret, **kwargs):
		super(PlaceFinder,self).__init__(key,secret,**kwargs)
		self.params = {}

	def placefinder(self, q):
		self.params['q'] = urllib.quote_plus(q)
		return self.make_request()

	def lookup(self,q):
		return self.placefinder(q)

	def reverse(self, lat, lon ):
		self.params['gflags'] = 'R'
		self.params['location'] = urllib.quote_plus(str(lat) + ' ' + str(lon))
		return self.make_request()
	
	def make_request(self):
		params = {	
			'oauth_version': "1.0",
			'oauth_nonce': oauth.generate_nonce(),
			'oauth_timestamp': int(time.time()),
			'count': '1',
			'flags': 'J'
		}

		params.update( self.params )

		bucket = 'placefinder'
		url =  "http://yboss.yahooapis.com/geo/" + bucket

		consumer = oauth.Consumer(key=self.key,secret=self.secret)

		req = oauth.Request(method="GET", url=url, parameters=params)
	
		signature_method = oauth.SignatureMethod_HMAC_SHA1()

		req.sign_request(signature_method, consumer, None)
		
		retval = requests.get(req.to_url())
		
		response = retval.json()

		if response and response.get('bossresponse'):
			return response.get('bossresponse').get('placefinder')
		else:
			raise GeoCodeError( retval )


