# Hive API Guide

The Hive REST API lets customers create pick tasks, watch the fleet, and receive events. This guide
covers version 2. The production base URL is `https://api.hive.mossgrove.example/v2`, and the sandbox
is `https://sandbox.hive.mossgrove.example/v2`.

## Authentication

Send your API key in the `X-Hive-Key` header. Keys start with `hk_live_` for production and
`hk_test_` for the sandbox.

```bash
curl -H "X-Hive-Key: hk_test_example" https://sandbox.hive.mossgrove.example/v2/robots
```

### Key rotation

Rotate API keys every 90 days. You may hold two active keys at the same time, so rotation has a
24-hour overlap: create the new key, deploy it, then revoke the old one. A revoked key returns
`401 key_revoked` immediately.

Integrators that act on behalf of several customers should use OAuth 2.0 client credentials instead.
Access tokens live for 15 minutes; request a new one before expiry rather than waiting for a 401.

## Rate limits

Each key may make 600 requests per minute with a burst allowance of 100 requests. Over the limit the
API answers `429 Too Many Requests` and includes a `Retry-After` header with the number of seconds to
wait. Clients should back off exponentially with jitter and honor `Retry-After`.

## Pagination

List endpoints use cursor pagination. Pass `limit` (default 50, maximum 200) and the opaque `cursor`
from the previous response's `next_cursor` field. A missing `next_cursor` means the last page.

## Core endpoints

| Method | Path                         | Purpose                                  |
|--------|------------------------------|------------------------------------------|
| POST   | `/tasks`                     | Create a pick or transport task          |
| GET    | `/tasks/{id}`                | Fetch a task and its current state       |
| GET    | `/robots`                    | List robots with battery and status      |
| POST   | `/robots/{id}/commands`      | Send `pause`, `resume` or `return_to_dock` |

### Idempotency

Send an `Idempotency-Key` header with every `POST`. Hive remembers the key for 24 hours and returns
the original response for a repeated request, so retries never create duplicate tasks.

### Errors

Errors share one shape:

```json
{"error": {"code": "invalid_zone", "message": "Zone Z-14 does not exist", "request_id": "req_8f2c"}}
```

Always log the `request_id`; support uses it to find your request.

## Webhooks

Hive can call your HTTPS endpoint for these events: `task.completed`, `task.failed`, and
`robot.fault`. Every delivery carries an `X-Hive-Signature` header: the hex HMAC-SHA256 of the raw
request body, keyed with your webhook secret. Verify the signature before trusting the payload, and
compare it in constant time.

If your endpoint returns anything other than a 2xx, Hive retries with exponential backoff, up to 8
attempts over 24 hours, and then marks the delivery as failed in the dashboard.

## Versioning and deprecation

Breaking changes ship only in a new major version. A deprecated version keeps working for at least 12
months after the announcement, and every response from a deprecated endpoint includes a `Sunset`
header with the removal date.
