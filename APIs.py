import os
from enum import Enum


class APIs(Enum):
    COINCAP_MARKETS = {
        'title': 'Coincap Markets',
        'url': 'https://api.coincap.io/v2/markets?exchangeId=kraken'
    }
    COINCAP_RATES = {
        'title': 'Coincap Rates',
        'url': 'https://api.coincap.io/v2/rates'
    }
    CRYPTO_COMPARE = {
        'title': 'Crypto Compare Top Coins',
        'url': 'https://min-api.cryptocompare.com/data/top/mktcapfull?limit=10&tsym=USD'
    }
    BONBAST = {
        'title': 'Bonbast Currency',
        'url': 'bonbast export'
    }
    #
    # news_api_key = os.getenv('NEWS_API_KEY')
    # NEWS_API = {
    #     'title': 'Blockchain News',
    #     'url': 'https://newsapi.org/v2/everything?q=blockchain&apiKey={news_api_key}'
    # }

    @property
    def title(self):
        return self.value['title']

    @property
    def url(self):
        return self.value['url']
