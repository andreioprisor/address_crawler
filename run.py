import argparse
from crawler import Crawler
import threading
import os
import pandas as pd

def crawl(urls, file_path):
    c = Crawler(urls)
    c.crawl_to_file(file_path)

def main(csv_file_name):
    df = pd.read_csv(csv_file_name)
    urls_list = df['domain'].tolist()
    crawl(urls_list, 'addresses.csv')

	# # Uncomment the following code to run the crawler in multi-threading mode
    # num_threads = 3
    # threads = []
    # for i in range(num_threads):
    #     if i == num_threads - 1:
    #         segment = urls_list[i*len(urls_list)//num_threads:]
    #     else:
    #         segment = urls_list[i*len(urls_list)//num_threads:(i+1)*len(urls_list)//num_threads]
        
    #     t = threading.Thread(target=crawl, args=(segment, f"crawl_{i}.csv"))
    #     threads.append(t)
    #     t.start()

    # for t in threads:
    #     t.join()

    # # Combine the crawled data and clean up
    # for i in range(num_threads):
    #     df = pd.read_csv(f'crawl_{i}.csv')
    #     df.to_csv('addresses.csv', mode='a', header=False, index=False)
    #     os.remove(f'crawl_{i}.csv')    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a web crawler on a list of URLs.')
    parser.add_argument('csv_file_name', type=str, help='The name of the CSV file containing the URLs to crawl.')
    args = parser.parse_args()

    main(args.csv_file_name)
