"""The Hugging Face Space: the front door and the display.

    app.py      the HTTP API and the page
    ndif.py     who an NDIF key belongs to
    state.py    submissions and teams, kept in DynamoDB and S3
    board.py    turning submissions into what people are shown
    mirror.py   a copy of every submission in a private Hugging Face dataset

The Space accepts a submission, checks who sent it, stores it, and starts the
pipeline when a slot is free. It never runs a submission and never scores one;
those are the runner and evaluator tasks, driven by the state machine in infra/.
"""
