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
    CRYPTO_Rates = {
        'title': 'CoinGecko Crypto Rates',
        'url': 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100'
               '&page=1&sparkline=false'
    }
    BONBAST = {
        'title': 'Bonbast Currency',
        'url': 'bonbast export'
    }

    @property
    def title(self):
        return self.value['title']

    @property
    def url(self):
        return self.value['url']
