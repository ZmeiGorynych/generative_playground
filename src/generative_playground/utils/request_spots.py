import argparse
import os
import uuid

import boto3


parser = argparse.ArgumentParser(description='Launch spot instances')
parser.add_argument(
    'spot_price',
    type=str,
    help="Max spot price"
)
parser.add_argument(
    '--number',
    type=int,
    help="Number of spot instances to request",
    default=1
)
parser.add_argument(
    '--type',
    type=str,
    help="Type of spot instances to request",
    default='p2.xlarge'
)
parser.add_argument(
    '--image_id',
    type=str,
    help="AMI Id",
    default=''
)
parser.add_argument(
    '--volume_size',
    type=int,
    help="EBS Volume Size",
    default=100
)
parser.add_argument(
    '--volume_delete_on_term',
    type=bool,
    help="EBS Volume Size",
    default=True
)
parser.add_argument(
    '--key_name',
    type=str,
    help="Name of an ssh key pair",
    default=''
)
parser.add_argument(
    '--security_group_id',
    type=str,
    help="ID of a security group",
    default=''
)
parser.add_argument(
    '--dry_run',
    type=bool,
    help="DryRun to check credentials and permissions",
    default=False
)

args = parser.parse_args()


if __name__ == '__main__':
    client = boto3.client(
        'ec2',
        aws_access_key_id=os.environ['MOLECULES_AWS_ACCESS_KEY'],
        aws_secret_access_key=os.environ['MOLECULES_AWS_SECRET_KEY'],
    )
    token = uuid.uuid4()

    response = client.request_spot_instances(
        DryRun=args.dry_run,
        SpotPrice=args.spot_price,
        ClientToken=token,
        InstanceCount=args.number,
        Type='one-time',
        LaunchSpecification={
            'ImageId': args.image_id,
            'KeyName': args.key_name,
            'InstanceType': args.spot_price,
            'BlockDeviceMappings': [
                {
                    'Ebs': {
                        'VolumeSize': args.volume_size,
                        'DeleteOnTermination': True,
                        'VolumeType': 'gp2',
                        'Encrypted': False
                    },
                },
            ],
            'EbsOptimized': True,
            'Monitoring': {
                'Enabled': False
            },
            'SecurityGroupIds': [
                args.security_group_id,
            ]
        }
    )

    response = client.describe_instances()
