import datetime
from pymongo import MongoClient
import pprint
# docker run -d -p 27017-27019:27017-27019  -v /home/ubuntu/shared/mongo_data:/data/db --name mongodb mongo
# hyperopt-mongo-worker --mongo=52.213.134.161:27017/foo_db --poll-interval=0.1
client = MongoClient('mongodb://52.213.134.161:27017/')
db = client.test_database

post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "pymongo"],
        "date": datetime.datetime.utcnow()}

posts = db.posts
post_id = posts.insert_one(post).inserted_id
ret = posts.find({"author": "Mike"})
for p in ret:
    print(p)
print('done!')