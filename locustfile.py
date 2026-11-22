"""Run Locust load tests against the served P1 demand prediction API.

Exercises single-instance predictions, mini-batch requests, and the health probe
with realistic think times between calls.
"""

from random import randint

from locust import HttpUser, between, task


class P1DemandAPIUser(HttpUser):
    """A simulated client querying the P1 hourly station demand API."""

    wait_time = between(0.5, 2.0)
    failure_message = lambda self, response: (
        f"Unexpected status {response.status_code}: {response.text}"
    )

    def _build_instance_query(self) -> dict[str, int]:
        """Build a synthetic station-hour feature payload.

        Values are sampled uniformly within the active MiBici network bounds
        (`station_id` 1..370, `hour` 0..23, `day_of_week` 0..6).
        """
        return {
            "station_id": randint(1, 370),
            "hour": randint(0, 23),
            "day_of_week": randint(0, 6),
        }

    @task(3)
    def predict_single(self) -> None:
        """Query `POST /predict` with a single station-hour instance.

        Weighted 3x relative to batch and health requests to reflect real-time
        point inference traffic.
        """
        endpoint = "/predict"
        test_name = "POST /predict (single)"
        payload = {"instances": [self._build_instance_query()]}

        with self.client.post(
            url=endpoint,
            json=payload,
            name=test_name,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(self.failure_message(response))
            elif "predictions" not in response.json():
                response.failure("Missing 'predictions' field in response")

    @task(1)
    def predict_batch(self) -> None:
        """Query `POST /predict` with a mini-batch of 5 to 25 instances.

        Validates that the response carries one expected-departure value per
        input instance under vectorized DataFrame scoring.
        """
        batch_size = randint(5, 25)
        endpoint = "/predict"
        test_name = "POST /predict (batch 5-25)"
        payload = {"instances": [self._build_instance_query() for _ in range(batch_size)]}

        with self.client.post(
            url=endpoint,
            json=payload,
            name=test_name,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(self.failure_message(response))
            elif len(response.json().get("predictions", [])) != batch_size:
                response.failure("Prediction count does not match batch size")

    @task(1)
    def health_check(self) -> None:
        """Probe `GET /health` to verify service readiness.

        Confirms both a healthy status flag and a non-empty `model_version`
        so silent artifact load failures are caught during load runs.
        """
        endpoint = "/health"
        test_name = "GET /health"

        with self.client.get(
            url=endpoint,
            name=test_name,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(self.failure_message(response))
                return

            data = response.json()
            if data.get("status") != "ok":
                response.failure(f"Unhealthy status: {data.get('status')}")
            elif not data.get("model_version"):
                response.failure("Missing 'model_version' in health response")
