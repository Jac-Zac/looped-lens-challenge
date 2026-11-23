"""The CDK entry point: `cdk <command> --app "python -m infra.app"`, which is
what `manage.py deploy` runs. The account comes from the AWS credentials in
use; the region from the organizers' configuration."""

import os

import aws_cdk as cdk

import competition as comp
from infra.stack import Leaderboard

competition = comp.load(comp.PROD_FILE)
app = cdk.App()
Leaderboard(app, competition, env=cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=competition.aws.region))
app.synth()
