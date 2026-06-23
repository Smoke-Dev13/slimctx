# ── Build stage ───────────────────────────────────────────────────────────────
# Downloads tiktoken encoding files into the source tree so they are bundled
# as wheel artifacts when pip builds the package.
FROM python:3.12-slim AS builder

WORKDIR /build

# Download script uses only stdlib (urllib, hashlib) — no pre-install needed.
COPY scripts/ ./scripts/
COPY src/    ./src/
COPY pyproject.toml ./
# pyproject declares readme=README.md and license=LICENSE; both must be present
# in the build context for hatchling to build the wheel.
COPY LICENSE README.md ./

RUN python scripts/download_encodings.py && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[nlp,ast,mcp-server]"

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages (includes contextly with bundled tiktoken data).
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/contextly /usr/local/bin/contextly

# Non-root user for defence-in-depth.
RUN useradd --system --uid 1001 --gid root contextly && \
    chmod g+rwx /app
USER contextly

EXPOSE 4000

# Liveness check via the /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:4000/health')" \
        || exit 1

ENTRYPOINT ["contextly", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "4000"]
