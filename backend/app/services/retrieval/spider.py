import sys
import scrapy
from scrapy.crawler import CrawlerProcess

class ArticleSpider(scrapy.Spider):
    name = "article_spider"
    
    def __init__(self, url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [url]

    def parse(self, response):
        yield {
            'html': response.text,
            'url': response.url
        }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
        
    target_url = sys.argv[1]
    output_file = sys.argv[2]
    
    # Run the spider and output to the temp file
    process = CrawlerProcess(settings={
        'USER_AGENT': 'LokLensBot/1.0 (https://loklens.ai; contact@loklens.ai)',
        'LOG_LEVEL': 'ERROR',
        'ROBOTSTXT_OBEY': False,
        'FEEDS': {
            output_file: {
                'format': 'json',
                'overwrite': True
            }
        },
        'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7',
        'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'
    })
    
    process.crawl(ArticleSpider, url=target_url)
    process.start()
