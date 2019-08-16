from typing import Dict, List

from fabric import Connection


def batch_run(
    source_root: str,
    python_file: str,
    key_file: str,
    job_assignments: Dict[str, List[str]]
) -> None:
    for ip, jobs in job_assignments.items():
        c = Connection(
            'ubuntu@{}'.format(ip),
            connect_kwargs={'key_filename': key_file}
        )
        for job in jobs:
            c.run(
                (
                    "screen -dmS mols; "
                    "screen -S mols -X stuff 'source activate pytorch_p36'$(echo -ne '\015'); "
                    "screen -S mols -X stuff 'export PYTHONPATH=$PYTHONPATH:{src_root}/'$(echo -ne '\015'); "
                    "screen -S mols -X stuff 'python {python_file} {job_id}'$(echo -ne '\015');"
                ).format(src_root=source_root, python_file=python_file, job_id=job)
            )
            print('Started job {}'.format(job))
    print('All jobs running')


if __name__ == '__main__':
    SOURCE_ROOT = '/home/ubuntu/shared/GitHub/generative_playground/src'
    PYTHON_FILE = '{}/generative_playground/molecules/train/pg/conditional/v2/train_conditional_v2_0.py'.format(SOURCE_ROOT)
    KEY_FILE = '/home/mark/keys/aws_second_key_pair.pem'
    job_assignments = {
        '34.242.186.107': ["1 --attempt test_fabric"],
    }
    batch_run(SOURCE_ROOT, PYTHON_FILE, KEY_FILE, job_assignments)
