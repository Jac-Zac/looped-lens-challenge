# The egress proxy: mitmproxy with the competition's allowlist (allowlist.py).
FROM mitmproxy/mitmproxy:12.2.1
COPY allowlist.py /allowlist.py
EXPOSE 3128
# block_global=false: the proxy listens on a private subnet address, and the
# security group already limits who can reach it to the runner tasks.
# stream_large_bodies: a body over 1 MB is passed on as it arrives. Without it
# the proxy holds every download in memory (the model's tokenizer is 25 MB) and
# does not give the memory back, so it is killed after about ten runs.
CMD ["mitmdump", "--mode", "regular", "--listen-port", "3128", "--set", "block_global=false", "--set", "flow_detail=1", "--set", "stream_large_bodies=1m", "-s", "/allowlist.py"]
