"""Tests for the best-effort ledger client and the results payload helper.

No ROS needed: a stub ledger runs on a local ``http.server``.
"""
import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from crazybridge import metrics
from crazybridge.ledger import LedgerClient


class _Stub:
    """Tiny recording ledger. `routes` maps (method, path) -> (status, body)."""

    def __init__(self, routes):
        self.routes = routes
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self):
                n = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                stub.requests.append({
                    'method': self.command, 'path': self.path, 'body': body,
                    'auth': self.headers.get('Authorization')})
                status, payload = stub.routes.get(
                    (self.command, self.path), (404, {'error': 'not found'}))
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_PATCH = _handle

            def log_message(self, format, *args):  # silence request logging
                pass

        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub_factory():
    stubs = []

    def make(routes):
        s = _Stub(routes)
        stubs.append(s)
        return s
    yield make
    for s in stubs:
        s.close()


def _closed_port_url():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return f'http://127.0.0.1:{s.getsockname()[1]}'


# -- happy path ---------------------------------------------------------------
def test_ensure_experiment_reuses_exact_name(stub_factory):
    stub = stub_factory({('GET', '/api/experiments'): (200, [
        {'id': 1, 'name': 'Crazyflie OOT Controller Comparison extra'},
        {'id': 2, 'name': 'crazyflie oot controller comparison'},
    ])})
    client = LedgerClient(stub.url)
    assert client.ensure_experiment('Crazyflie OOT Controller Comparison') == 2
    assert [r['method'] for r in stub.requests] == ['GET']  # nothing created


def test_ensure_experiment_creates_when_missing(stub_factory):
    stub = stub_factory({
        ('GET', '/api/experiments'): (200, [{'id': 1, 'name': 'Other'}]),
        ('POST', '/api/experiments'): (201, {'id': 9, 'name': 'New'}),
    })
    client = LedgerClient(stub.url)
    assert client.ensure_experiment('New', subject='s', objective='o') == 9
    assert stub.requests[1]['body']['name'] == 'New'
    assert stub.requests[1]['body']['subject'] == 's'


def test_run_lifecycle_bodies_and_bearer(stub_factory):
    stub = stub_factory({
        ('POST', '/api/experiments/3/runs'): (201, {'id': 40, 'seq': 2}),
        ('POST', '/api/runs/40/results'): (201, []),
        ('POST', '/api/runs/40/files'): (201, []),
        ('PATCH', '/api/runs/40'): (200, {'id': 40}),
    })
    client = LedgerClient(stub.url, token='sekret')
    run = client.start_run(3, 'cfg (now)', {'a': 1.5, 'b': None, 'c': [1, 2]},
                           notes='n')
    assert run == {'id': 40, 'seq': 2}
    assert client.add_results(40, [{'label': 'x', 'value': 1.0, 'unit': 'm'}]) \
        is not None
    assert client.attach_files(40, ['/tmp/a.json']) is not None
    assert client.update_run(40, outcome='success') is not None

    start, results, files, patch = stub.requests
    assert start['body'] == {'title': 'cfg (now)',
                             'params': {'a': 1.5, 'c': '[1, 2]'},
                             'notes': 'n', 'outcome': 'pending'}
    assert results['body'] == {
        'results': [{'label': 'x', 'value': 1.0, 'unit': 'm'}]}
    assert files['body'] == {'paths': ['/tmp/a.json']}
    assert patch['body'] == {'outcome': 'success'}
    assert all(r['auth'] == 'Bearer sekret' for r in stub.requests)


def test_add_results_drops_non_finite(stub_factory):
    stub = stub_factory({('POST', '/api/runs/1/results'): (201, [])})
    client = LedgerClient(stub.url)
    client.add_results(1, [{'label': 'ok', 'value': 2.0},
                           {'label': 'nan', 'value': float('nan')},
                           {'label': 'inf', 'value': math.inf}])
    assert [r['label'] for r in stub.requests[0]['body']['results']] == ['ok']
    # nothing finite -> no request at all
    stub.requests.clear()
    assert client.add_results(1, [{'label': 'nan', 'value': float('nan')}]) \
        is None
    assert stub.requests == []


# -- failure handling ---------------------------------------------------------
def test_unreachable_ledger_never_raises_and_goes_quiet():
    logs = []
    client = LedgerClient(_closed_port_url(), timeout_s=0.5, log=logs.append)
    t0 = time.monotonic()
    assert client.ensure_experiment('x') is None
    assert client.start_run(1, 't', {}) is None
    assert client.add_results(1, [{'label': 'a', 'value': 1}]) is None
    assert client.attach_files(1, ['/tmp/x']) is None
    assert client.update_run(1, outcome='failure') is None
    assert time.monotonic() - t0 < 5
    assert client.available is False
    assert len(logs) == 1 and 'unreachable' in logs[0]  # one warning, not five


def test_http_error_returns_none_but_client_stays_available(stub_factory):
    logs = []
    stub = stub_factory({
        ('POST', '/api/runs/1/results'): (400, {'error': 'bad label'}),
        ('PATCH', '/api/runs/1'): (200, {'id': 1}),
    })
    client = LedgerClient(stub.url, log=logs.append)
    assert client.add_results(1, [{'label': 'a', 'value': 1}]) is None
    assert client.available is True
    assert 'bad label' in logs[0]
    assert client.update_run(1, outcome='success') == {'id': 1}


def test_non_json_response_returns_none():
    # The stub always sends JSON; use a raw socket server for a bad body.
    srv = socket.socket()
    srv.bind(('127.0.0.1', 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.recv(4096)
        conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nnope')
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    client = LedgerClient(f'http://127.0.0.1:{srv.getsockname()[1]}')
    assert client.update_run(1, outcome='success') is None
    srv.close()


def test_from_env_reads_url_and_token(monkeypatch):
    monkeypatch.setenv('LEDGER_API_URL', 'http://example.invalid:1234/')
    monkeypatch.setenv('LEDGER_API_TOKEN', 'tok')
    c = LedgerClient.from_env()
    assert c.base == 'http://example.invalid:1234/api'
    assert c.token == 'tok'
    assert LedgerClient.from_env('http://other:1').base == 'http://other:1/api'


# -- metrics payload ----------------------------------------------------------
def test_ledger_results_labels_units_and_skips_timestamps():
    ts = {'t_start': 0.0, 't_takeoff_end': 1.0, 't_regulate_end': 2.0,
          't_track_end': 3.0, 't_land_end': 4.0}
    m = metrics.compute_metrics({}, ts)
    results = metrics.ledger_results(m)
    labels = {r['label']: r for r in results}

    assert 'track.t_start' not in labels and 'track.t_end' not in labels
    assert labels['track.duration_s']['value'] == 1.0
    assert labels['track.duration_s']['unit'] == 's'
    assert labels['track.int_err_norm']['unit'] == 'm·s'
    assert labels['total.rmse_norm']['unit'] == 'm'
    assert labels['land.int_torque_norm']['unit'] == 'N·m·s'
    assert all(math.isfinite(r['value']) for r in results)
    # every metric of every segment, minus the two timestamps each
    expected = sum(len(seg) - 2 for seg in m.values())
    assert len(results) == expected
    # no metric is left without a declared unit mapping
    for seg in m.values():
        for k in seg:
            assert k in metrics.METRIC_UNITS or k in ('t_start', 't_end')
