import subprocess, argparse

def spawn_hyperopt_worker(mongo_server='52.213.134.161:27017',
                          db_name='test_db'):
    cmd = ['hyperopt-mongo-worker',
           '--mongo=' + mongo_server + '/' + db_name,
           '--poll-interval=0.1']
    subprocess.Popen(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spawn a hyperopt worker')
    parser.add_argument('--mongo_server', help="ip:port of mongo", default='52.213.134.161:27017')
    parser.add_argument('--db_name', help="mongo db name", default='test')

    args = parser.parse_args()
    spawn_hyperopt_worker(args.mongo_server, args.db_name)
