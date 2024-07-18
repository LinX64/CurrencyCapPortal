import asyncio
import json
import os
import subprocess

import aiohttp
from aiohttp import ClientSession
from dotenv import load_dotenv

from APIs import APIs

load_dotenv()
last_successful_data = None


# Asynchronous function to fetch a single URL
async def fetch(url, session):
    async with session.get(url.url) as response:
        if response.status == 200:
            return await response.json()
        else:
            return {
                "error": f'{url.title} API request failed',
                "status_code": response.status
            }


async def run_bonbast():
    try:
        # Run the 'bonbast export' command asynchronously
        process = await asyncio.create_subprocess_exec(
            'bonbast', 'export',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Wait for the command to complete and capture the output
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"Error occurred in running bonbast command: {stderr.decode()}")
            return None

        return stdout.decode()
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in running bonbast command: {e}")
        return None


def combine_data(results):
    data = {}
    markets = results[0]['data']
    rates = results[1]['data']
    cryptos = results[2]
    bonbast = results[3]

    data['markets'] = markets
    data['rates'] = rates
    data['crypto'] = [crypto.copy() for crypto in cryptos]
    data['bonbast'] = [{k: v} for k, v in bonbast.items()]

    return data


async def aggregator():
    global last_successful_data

    async with ClientSession() as session:
        urls = [APIs.COINCAP_MARKETS, APIs.COINCAP_RATES, APIs.CRYPTO_Rates]
        tasks = [fetch(url, session) for url in urls]
        results = await asyncio.gather(*tasks)

        for result in results:
            if isinstance(result, dict) and 'error' in result:
                print(f"Encountered error: {result['error']}")
                if last_successful_data:
                    print("Using last successful data")
                    return last_successful_data
                else:
                    return result, result['status_code']

        bonbast_result_str = await run_bonbast()
        bonbast_result_json = json.loads(bonbast_result_str)
        results.append(bonbast_result_json)

        combined_data = combine_data(results)
        last_successful_data = combined_data
        return combined_data


async def getBlockchainNews():
    news_api_key = os.getenv('NEWS_API_KEY')
    if not news_api_key:
        raise ValueError("NEWS_API_KEY environment variable is not set")

    url = f'https://newsapi.org/v2/everything?q=blockchain&apiKey={news_api_key}'

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                try:
                    news = await response.json()
                    return news
                except Exception as e:
                    print(f"Error parsing JSON: {e}")
                    return None
            else:
                return {
                    "error": "Blockchain news API request failed",
                    "status_code": response.status
                }
