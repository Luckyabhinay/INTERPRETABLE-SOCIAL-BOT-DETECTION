import io
import json
import logging
import socket
import time
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import patch

from xfetch_client import XFetchClient, _InstrumentedHTTPSConnection


class _FakeSocket:
    def __init__(self, family):
        self.family = family
        self.timeout = None
        self.address = None
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, address):
        self.address = address

    def close(self):
        self.closed = True


class _FakeTLSSocket(_FakeSocket):
    def version(self):
        return "TLSv1.3"


class _Response(io.BytesIO):
    status = 200


class _IPv6HandshakeTimeoutContext:
    def __init__(self):
        self.attempted_hosts = []

    def wrap_socket(self, raw_socket, server_hostname):
        self.attempted_hosts.append(server_hostname)
        if raw_socket.family == socket.AF_INET6:
            raise TimeoutError("simulated IPv6 TLS handshake timeout")
        return _FakeTLSSocket(raw_socket.family)


class XFetchTransportTests(unittest.TestCase):
    def test_tls_timeout_falls_back_to_next_address_before_http_request(self):
        diagnostics = {}
        context = _IPv6HandshakeTimeoutContext()
        addresses = [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("2001:db8::1", 443, 0, 0),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("192.0.2.1", 443),
            ),
        ]
        sockets = []

        def make_socket(family, socktype, proto):
            self.assertEqual(socktype, socket.SOCK_STREAM)
            self.assertIn(proto, (socket.IPPROTO_TCP, 0))
            instance = _FakeSocket(family)
            sockets.append(instance)
            return instance

        connection = _InstrumentedHTTPSConnection(
            "api.xfetch.io",
            timeout=1,
            context=context,
            diagnostics=diagnostics,
        )
        with patch("xfetch_client.socket.getaddrinfo", return_value=addresses), patch(
            "xfetch_client.socket.socket", side_effect=make_socket
        ):
            connection.connect()

        self.assertEqual(len(sockets), 2)
        self.assertTrue(sockets[0].closed)
        self.assertEqual(context.attempted_hosts, ["api.xfetch.io", "api.xfetch.io"])
        self.assertEqual(diagnostics["stage"], "tls_complete")
        self.assertFalse(diagnostics.get("http_request_sent", False))
        self.assertGreaterEqual(diagnostics["dns_seconds"], 0)
        self.assertGreaterEqual(diagnostics["tls_seconds"], 0)
        connection.close()

    def test_request_logs_do_not_include_api_key(self):
        secret = "private-test-api-key"
        payload = json.dumps(
            {"data": {"user": {"id": "public-id"}, "recent_tweets": []}}
        ).encode()
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        logger = logging.getLogger("xfetch_client")
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            def fake_open_request(req, timeout, diagnostics):
                self.assertEqual(req.full_url.split("/")[2], "api.xfetch.io")
                self.assertEqual(timeout, 15)
                diagnostics.update(
                    {
                        "stage": "response_received",
                        "http_status": 200,
                        "http_request_sent": True,
                        "request_sent_at": time.perf_counter() - 0.01,
                        "response_headers_seconds": 0.005,
                        "response_body_seconds": 0.001,
                    }
                )
                return _Response(payload)

            with patch(
                "xfetch_client._open_request",
                side_effect=fake_open_request,
            ) as open_request:
                result, _, error, status = XFetchClient(api_key=secret).fetch_profile(
                    "public_user"
                )
            request = open_request.call_args.args[0]
            self.assertTrue(request.get_header("Authorization").endswith(secret))
            self.assertEqual(status, 200)
            self.assertIsNone(error)
            self.assertTrue(result)
            self.assertNotIn(secret, output.getvalue())
            self.assertIn("authentication_stage=response_received", output.getvalue())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

    def test_http_401_is_identified_as_authentication_rejection(self):
        secret = "private-test-api-key"
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        logger = logging.getLogger("xfetch_client")
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        error = urllib.error.HTTPError(
            "https://api.xfetch.io/v1/profiles/by-username/public_user",
            401,
            "Unauthorized",
            {},
            BytesIO(b'{"detail":"Unauthorized"}'),
        )
        try:
            def reject_after_request(req, timeout, diagnostics):
                self.assertEqual(req.full_url.split("/")[2], "api.xfetch.io")
                self.assertEqual(timeout, 15)
                diagnostics.update(
                    {
                        "http_request_sent": True,
                        "request_sent_at": time.perf_counter() - 0.01,
                        "response_headers_seconds": 0.005,
                    }
                )
                raise error

            with patch(
                "xfetch_client._open_request", side_effect=reject_after_request
            ):
                result, _, message, status = XFetchClient(api_key=secret).fetch_profile(
                    "public_user"
                )
            self.assertIsNone(result)
            self.assertEqual(status, 401)
            self.assertIn("Authentication failed", message)
            self.assertIn("authentication_stage=rejected", output.getvalue())
            self.assertNotIn(secret, output.getvalue())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)


if __name__ == "__main__":
    unittest.main()
