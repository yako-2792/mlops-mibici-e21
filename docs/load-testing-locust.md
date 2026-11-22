# Load Testing Guide (`Locust`)

This document serves as an operational and architectural guide for running load tests against the inference API using **Locust**.

## 1. Overview & Execution Model

Locust is used as an external load generator to simulate concurrent traffic against the containerized inference service (`mibici-api`).

* **User Modeling (`locustfile.py`):** Each ML model (e.g., `P1`, `P2`, `P3`) is defined as its own `HttpUser` class containing weighted HTTP tasks (`/health`, single `/predict`, and batch `/predict`) along with simulated user think time (`wait_time`).
* **Concurrency Engine (`Locust Runner`):** When launched in standalone mode, Locust runs in a **single process and single OS thread**, using `gevent` *greenlets* (lightweight coroutines) with non-blocking network I/O.
  * When spawning $N$ virtual users, each user instance yields control back to the event loop immediately after dispatching an HTTP request.
  * This allows a single thread to sustain dozens of concurrent in-flight TCP connections without OS thread-switching overhead, effectively simulating parallel client load.

---

## 2. Prerequisites

Before running any load test, ensure the following requirements are met:

1. **Target Services Running:** The inference stack (`mibici-api` and its dependencies) must be up and reachable (e.g., via `docker compose up -d`), with `/health` returning `200 OK`.
2. **Locust Installed:** Locust must be installed in your active Python environment (`pip install locust` or via the project's dependency manager).
3. **Output Directory:** Ensure the target directory for test reports exists (e.g., `reports/` or `docs/load_testing/`), or pass the appropriate relative path in the CLI flags.

---

## 3. Running Load Tests

### Headless Mode (CLI / Automated Runs)

Use headless mode to run deterministic, time-boxed load tests targeting a specific model user class (e.g., `P1User`):

```bash
locust -f locustfile.py P1User \
  --headless \
  --host http://localhost:8000 \
  --users 20 \
  --spawn-rate 5 \
  --run-time 30s \
  --csv=reports/locust_p1 \
  --html=reports/locust_p1_report.html
```

#### Key CLI Flags

| Flag | Description |
| :--- | :--- |
| `-f` | Path to the Locust test definition file (`locustfile.py`). |
| `P1User` | Optional positional argument specifying which `User` class to instantiate. If omitted, Locust spawns all defined user classes proportionally. |
| `--headless` | Runs the test without starting the web interface. |
| `--host` | Base URL of the target inference API. |
| `-u` / `--users` | Peak number of concurrent virtual users to simulate. |
| `-r` / `--spawn-rate` | Number of virtual users spawned per second during ramp-up. |
| `-t` / `--run-time` | Total duration of the test (e.g., `30s`, `2m`). |

### Interactive Web UI Mode

Use the web interface when you want to monitor latency charts in real time or manually ramp users up and down:

```bash
locust -f locustfile.py --host http://localhost:8000
```

Once started, open **`http://localhost:8089`** in your browser to configure the user count, spawn rate, and target host interactively.

---

## 4. Output Artifacts & Where Results Go

When running with `--csv` and `--html` (or when exporting from the Web UI's **Download Data** tab), Locust automatically generates five output files at the specified path prefix (e.g., `reports/locust_p1*`):

| File | Description |
| :--- | :--- |
| `<prefix>_report.html` | Self-contained interactive HTML report with response time percentiles, throughput graphs, and failure summaries. |
| `<prefix>_stats.csv` | Aggregated per-endpoint summary table (request counts, failure counts, median, average, min, max, P95/P99 latencies, and RPS). |
| `<prefix>_stats_history.csv` | Time-series metrics recorded second-by-second throughout the test run. |
| `<prefix>_failures.csv` | Detailed breakdown of any HTTP errors or connection failures encountered. |
| `<prefix>_exceptions.csv` | Python tracebacks for any unhandled exceptions raised inside virtual user tasks. |

---

## 5. Resource Monitoring During Tests

While Locust measures client-side latency and throughput, server-side resource saturation should be monitored simultaneously in a separate terminal using Docker stats:

```bash
docker stats
```

* **CPU %:** In Docker, `100%` corresponds to 1 full vCPU core. Sustained usage near or above `100%` on a single-worker API container indicates a compute bottleneck.
* **MEM USAGE / %:** Tracks model memory footprint and helps determine headroom for scaling API worker processes.