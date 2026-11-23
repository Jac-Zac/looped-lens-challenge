"""The AWS side: where submissions run and get scored.

    stack.py              everything in AWS, as one CDK stack
    app.py                the CDK entry point
    runner.Dockerfile     the untrusted runner image
    evaluator.Dockerfile  the trusted evaluator image
    proxy/                the runner's only way out to the network

`python manage.py deploy` builds the images and deploys the stack. Nothing here
is imported by the runner, the evaluator, or a participant's local run.
"""
