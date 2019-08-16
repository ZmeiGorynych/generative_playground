from fabric import Connection


SOURCE_ROOT = '/home/ubuntu/generative_playground/generative_playground/src'
PYTHON_FILE = '{}/models/problem/mcts/mcts.py'.format(SOURCE_ROOT)
KEY_FILE = '~/keys/mark.pem'
job_assignments = {
    'ip': []
}


for ip, jobs in job_assignments.items():
    c = Connection(
        'ubuntu@{}'.format(ip),
        connect_kwargs={'key_filename': KEY_FILE}
    )
    for job in jobs:
        c.run(
            (
                'screen -d -m '
                'source activate pytorch_p36 && '
                'export PYTHONPATH=$PYTHONPATH:{src_root}/ && '
                'python {python_file} {job_id}'
            ).format(src_root=SOURCE_ROOT, python_file=PYTHON_FILE, job_id=job)
        )
        print('Started job {}'.format(job))
print('All jobs running')
