"""Best-effort client for the ledger experiment-log HTTP API.

Ledger is optional infrastructure: every method here swallows its own failures
and returns ``None`` instead of raising, so callers can record a run without
ever risking the thing they are recording. Only the stdlib is used and there is
no ROS dependency, so the module is easy to test.

The API (``http://127.0.0.1:4317/api`` by default) takes JSON bodies only. File
attachment sends *paths*, which the ledger server copies from its own
filesystem -- so ledger has to run on the same machine as the caller.

Environment:
    LEDGER_API_URL    default ``http://127.0.0.1:4317``
    LEDGER_API_TOKEN  sent as a bearer token when set (matches the ledger's)

Failure policy:
    * Connection-level failures (refused, unreachable, timeout) log one
      warning and mark the client unavailable; every later call returns
      ``None`` immediately, so a dead ledger costs at most one short timeout.
    * HTTP errors (400, 404, ...) are logged and return ``None`` but do not
      disable the client -- one bad request should not block the others.
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Callable, Iterable

DEFAULT_URL = 'http://127.0.0.1:4317'
DEFAULT_TIMEOUT_S = 3.0
# Ledger copies attached files before it answers, so uploads get longer.
UPLOAD_TIMEOUT_S = 30.0


def _noop(_msg: str) -> None:
    pass


class LedgerClient:
    def __init__(self, base_url: str = DEFAULT_URL, token: str | None = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 log: Callable[[str], None] = _noop) -> None:
        self.base = (base_url or DEFAULT_URL).rstrip('/') + '/api'
        self.token = token
        self.timeout_s = timeout_s
        self._log = log
        self.available = True

    @classmethod
    def from_env(cls, url: str | None = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 log: Callable[[str], None] = _noop) -> 'LedgerClient':
        """URL from `url`, else ``LEDGER_API_URL``; token only from the env."""
        base = url or os.environ.get('LEDGER_API_URL') or DEFAULT_URL
        return cls(base, os.environ.get('LEDGER_API_TOKEN') or None,
                   timeout_s, log)

    # -- transport -------------------------------------------------------------
    def _call(self, method: str, path: str, body=None,
              timeout: float | None = None):
        """JSON round-trip. Returns the parsed response, or None on any failure."""
        if not self.available:
            return None
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header('Content-Type', 'application/json')
        if self.token:
            req.add_header('Authorization', f'Bearer {self.token}')
        try:
            with urllib.request.urlopen(
                    req, timeout=timeout or self.timeout_s) as resp:
                raw = resp.read()
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors='replace')
            try:
                detail = json.loads(detail).get('error', detail)
            except (ValueError, AttributeError):
                pass
            self._log(f'ledger {method} {path} -> HTTP {exc.code}: {detail}')
        except (urllib.error.URLError, OSError) as exc:
            # URLError wraps refused/unreachable; socket timeouts are OSError.
            reason = getattr(exc, 'reason', exc)
            self._log(f'ledger unreachable at {self.base} ({reason}); '
                      'skipping ledger for this run')
            self.available = False
        except ValueError as exc:  # server answered with non-JSON
            self._log(f'ledger {method} {path}: bad response ({exc})')
        except Exception as exc:  # never let ledger take the caller down
            self._log(f'ledger {method} {path} failed: {exc!r}')
        return None

    # -- operations ------------------------------------------------------------
    def ensure_experiment(self, name: str, subject: str = '',
                          objective: str = '',
                          description: str = '') -> int | None:
        """Id of the experiment called `name`, creating it if it is missing.

        The API only does substring search, so match the exact name here.
        """
        experiments = self._call('GET', '/experiments')
        if experiments is None:
            return None
        wanted = name.strip().lower()
        for exp in experiments:
            if str(exp.get('name', '')).strip().lower() == wanted:
                return exp.get('id')
        created = self._call('POST', '/experiments', {
            'name': name, 'subject': subject, 'objective': objective,
            'description': description})
        return created.get('id') if created else None

    def start_run(self, experiment_id: int, title: str, params: dict,
                  notes: str = '') -> dict | None:
        """Create a ``pending`` run; returns the run (``id``, ``seq``, ...)."""
        return self._call('POST', f'/experiments/{experiment_id}/runs', {
            'title': title,
            'params': _clean_params(params),
            'notes': notes,
            'outcome': 'pending',
        })

    def add_results(self, run_id: int, results: Iterable[dict]):
        """Attach numeric results; non-finite values are dropped (ledger 400s)."""
        good = []
        for r in results:
            try:
                value = float(r['value'])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                good.append({'label': r['label'], 'value': value,
                             'unit': r.get('unit', '')})
        if not good:
            return None
        return self._call('POST', f'/runs/{run_id}/results', {'results': good})

    def attach_files(self, run_id: int, paths: Iterable[str]):
        """Have ledger copy `paths` (absolute, on this machine) into the run."""
        paths = [os.path.abspath(p) for p in paths]
        if not paths:
            return None
        return self._call('POST', f'/runs/{run_id}/files', {'paths': paths},
                          timeout=max(self.timeout_s, UPLOAD_TIMEOUT_S))

    def update_run(self, run_id: int, **fields):
        """PATCH a run, e.g. ``outcome='success', notes='...'``."""
        return self._call('PATCH', f'/runs/{run_id}', fields)


def _clean_params(params: dict) -> dict:
    """Ledger stores params as strings; keep None out and values scalar."""
    out = {}
    for key, value in params.items():
        if value is None or key == '':
            continue
        out[str(key)] = value if isinstance(value, (str, int, float, bool)) \
            else str(value)
    return out
