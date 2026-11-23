"""Who an NDIF API key belongs to."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class NdifUnavailable(Exception):
    """NDIF could not be asked. Nothing is known about the key."""


def whoami(host: str, api_key: str, timeout: float = 15.0) -> dict:
    """Ask NDIF about a key. Returns ``{"email": str | None, "tags": [str]}``;
    ``email`` is ``None`` for a key NDIF does not recognise. Raises
    ``NdifUnavailable`` when NDIF gives no usable answer, so that an outage is
    never mistaken for a bad key or, worse, for a good one."""
    request = urllib.request.Request(host.rstrip("/") + "/whoami", headers={"ndif-api-key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code in (400, 401, 403):  # NDIF's reply to a malformed or unknown key
            return {"email": None, "tags": []}
        raise NdifUnavailable(f"NDIF answered {error.code}") from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise NdifUnavailable(str(error)) from error
    return {"email": answer.get("email"), "tags": list(answer.get("tags") or [])}
