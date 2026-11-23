# The egress proxy: mitmproxy with the competition's allowlist (allowlist.py).
FROM mitmproxy/mitmproxy:12.2.1
COPY allowlist.py /allowlist.py
EXPOSE 3128
# block_global=false: the proxy listens on a private subnet address, and the
# security group already limits who can reach it to the runner tasks.
CMD ["mitmdump", "--mode", "regular", "--listen-port", "3128", "--set", "block_global=false", "--set", "flow_detail=1", "-s", "/allowlist.py"]
