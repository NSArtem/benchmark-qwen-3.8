"""Default runtime settings. Environment overrides live in app.settings.env."""

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5
USER_AGENT = "lantern-client/2.1"
