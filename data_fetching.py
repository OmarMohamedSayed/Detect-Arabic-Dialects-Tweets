import requests
import logging
import argparse
import pandas as pd
import json
import time

log = logging.getLogger(__name__)

return_data = {"ids":[],"title":[]}

def formate_data(data):
    for i in data:
        return_data['ids'].append(i)
        return_data['title'].append(data[i])


def make_requests(ids,request_numbers, remember_of_records, request_ids_limit, url, outputfile):
    """
        Send ids requests to the server and save the respond in outputfile 

        ids : ids of tweeter tweet
        request_numbers : number of requests send to the server
        remember_of_records: number of records remember from divided ids size / request_ids_limit
        request_ids_limit: Max number of ids send to server in one request
        url : server url
        outputfile : name of output file that store the data 
    """
    start = 0
    end = 0
    for i in range(0,request_numbers-1):
        end = request_ids_limit+end        
        r = requests.post(url,json=ids[start:end-1].astype(str).tolist()) 
        res = json.loads(r.text)
        formate_data(res)
        start = end
        print(f'Loading data with records numbers: {end}')

    if(remember_of_records!=0):
        start = end
        end = remember_of_records + start -1
        r = requests.post(url,json=ids[start:end].astype(str).tolist())
        res = json.loads(r.text)
        formate_data(res)
        

    with open(outputfile, mode='a+',encoding='utf-8') as outfile:
            json.dump(return_data, outfile, ensure_ascii = False)
    
    print(f'Success Loading data with records numbers: {end}')

def read_file(file_path, request_ids_limit, url, outputfile):
    """
        Read CSV file that contains tweety IDs
        send this ids to make_requests function

        file_path : CSV path file 
        request_ids_limit : Max number of ids send to server in one request
        url : server url
        outputfile : name of output file that store the data
    """
    df = pd.read_csv(file_path)
    ids = df['id']
    ids_size = df['id'].size
    if(ids_size%request_ids_limit)!=0:
        request_numbers = int(ids_size/request_ids_limit)
        remember_of_records= ids_size%request_ids_limit
        make_requests(ids,request_numbers,remember_of_records,request_ids_limit,url, outputfile)
    else:
        make_requests(ids,request_numbers,0,request_ids_limit, url, outputfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", default="./dialect_dataset.csv", help="File Path with extension CSV that contains Ids. Default is ./dialect_dataset.csv")
    parser.add_argument("--request-ids-limit", default=1000, help="The Number of IDs in each request. Default is 1000")
    parser.add_argument("--url", default='https://recruitment.aimtechnologies.co/ai-tasks', help="The Server url. Default is https://recruitment.aimtechnologies.co/ai-tasks")
    parser.add_argument("--output-file-name", default='output.json' ,help="Name of Output file with extension JSON. Default is output.json")
    
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(filename='logfile.log', filemode='w', 
                        level=logging.DEBUG if args.debug else logging.INFO)
    

    read_file(args.file_path, args.request_ids_limit, args.url, args.output_file_name)
