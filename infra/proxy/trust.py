"""The first thing to run in the runner container on AWS: send traffic through
the egress proxy and trust its certificate authority, then start the runner.

The proxy decrypts connections to read-only hosts (see allowlist.py) so that it
can refuse anything but GET and HEAD. For that to work the method's HTTP
libraries have to trust the proxy's certificate authority, which is fetched
here, from the proxy itself, and added to the usual list.

    LEADERBOARD_PROXY    the proxy's address, e.g. http://proxy.example.internal:3128
    LEADERBOARD_DIRECT   comma-separated hosts reached without the proxy
                         (the bucket the task's own links point at)

Without LEADERBOARD_PROXY this does nothing but start the runner.
"""

import os
import sys
import time
import urllib.request

import certifi

BUNDLE = "/tmp/ca-bundle.pem"


def proxy_certificate(proxy: str) -> bytes:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy}))
    for attempt in range(10):
        try:
            return opener.open("http://mitm.it/cert/pem", timeout=15).read()
        except OSError:
            time.sleep(3)
    raise SystemExit(f"could not reach the egress proxy at {proxy}")


if proxy := os.environ.get("LEADERBOARD_PROXY"):
    with open(BUNDLE, "wb") as bundle:
        bundle.write(open(certifi.where(), "rb").read() + b"\n" + proxy_certificate(proxy))
    direct = os.environ.get("LEADERBOARD_DIRECT", "")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[name] = proxy
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = direct
    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ[name] = BUNDLE

os.execv(sys.executable, [sys.executable, "-m", "runner.task"])
