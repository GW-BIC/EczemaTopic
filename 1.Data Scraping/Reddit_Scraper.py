
#https://rareloot.medium.com/using-pushshifts-api-to-extract-reddit-submissions-fb517b286563

import pandas as pd
import requests
import json
import csv
import time
import datetime

start_time = int(datetime.datetime.utcnow().timestamp())
#start_time=1401558893
today= datetime.date.today().strftime("%Y-%m-%d")
rurl = "https://api.pushshift.io/reddit/{}/search?subreddit={}&limit=1000&sort=desc&before="
subreddit = "eczema"
subCount = 0
location = "../Reddit Data/"

def getPushshiftData(object_type, subreddit, before):
    url = rurl.format(object_type, subreddit)+str(before)
    print(url)
    r = requests.get(url)
    if r.status_code != 200:
        i = 1
        for i in range(5):
            print("Server didn't return an 'OK' response.  Content was: {!r}".format(r.text))
            #print("Server didn't return an 'OK' response.")
            time.sleep(10)
            before = before-1
            url = rurl.format(object_type, subreddit) + str(before)
            r = requests.get(url)
            if r.status_code != 200:
                print("Server didn't return an 'OK' response.")
            else:
                data = json.loads((r.text).encode().decode("utf-8"))
                break
            i=i+1
    else:
        data = json.loads((r.text).encode().decode("utf-8"))
    return data['data']

def DownloadRedditData(object_type):
    subStats = {}
    def collectData(postm):
        if  object_type == "submission":
            subData = list()  # list to store data points
            title = postm['title']
            url = postm['full_link']
            author = postm['author']
            sub_id = postm['id']
            score = postm['score']
            created = datetime.datetime.fromtimestamp(postm['created_utc'])  # 1520561700.0
            numComms = postm['num_comments']
            permalink = postm['permalink']
            try:
                selftext = postm['selftext']
            except KeyError:
                selftext = "NaN"
            subData.append((sub_id, title, url, author, score, created, numComms, permalink, selftext))
            subStats[sub_id] = subData
        if object_type == "comment":
            subData = list()  # list to store data points
            author = postm['author']
            sub_id = postm['id']
            link_id = postm['link_id']
            score = postm['score']
            created = datetime.datetime.fromtimestamp(postm['created_utc'])
            try:
                permalink = postm['permalink']
            except KeyError:
                permalink = "NaN"
            try:
                body = postm['body']
            except KeyError:
                body = "NaN"
            subData.append((sub_id, link_id, author, score, created, permalink, body))
            subStats[sub_id] = subData

    data = getPushshiftData(object_type, subreddit,start_time)
    # Will run until all posts have been gathered
    while len(data) > 0:
        for post in data:
            collectData(post)
        # Calls getPushshiftData() with the created date of the last submission
        #print(len(data))
        # print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
        before = data[-1]['created_utc']
        #print(before)
        data = getPushshiftData(object_type, subreddit,before)

    print(len(data))
    print(str(len(subStats)) + " submissions have added to list")
    print("1st entry is:")
    print(list(subStats.values())[0][0][1] + " created: " + str(list(subStats.values())[0][0][5]))
    print(list(subStats.values())[-1][0][1] + " created: " + str(list(subStats.values())[-1][0][5]))

    def save_data(object_type):
        file = location  + subreddit + "-" + object_type + '-' +today + '.csv'
        upload_count = 0
        if object_type == "submission":
            with open(file, 'w', newline='', encoding='utf-8') as file:
                a = csv.writer(file, delimiter=',')
                headers = ["Post ID", "Title", "Url", "Author", "Score", "Publish Date", "Total No. of Comments",
                           "Permalink", "body"]
                a.writerow(headers)
                for sub in subStats:
                    a.writerow(subStats[sub][0])
                    upload_count += 1
                print(str(upload_count) + " submissions have been uploaded")
        if object_type == "comment":
            with open(file, 'w', newline='', encoding='utf-8') as file:
                a = csv.writer(file, delimiter=',')
                headers = ["Post ID","Linked ID","Author","Score","Publish Date","Permalink","body"]
                a.writerow(headers)
                for sub in subStats:
                    a.writerow(subStats[sub][0])
                    upload_count+=1
                print(str(upload_count) + " posts have been uploaded")
    save_data(object_type)

DownloadRedditData("submission")
#DownloadRedditData("comment")