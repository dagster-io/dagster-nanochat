import os
import time
from typing import Any

import dagster as dg
import requests
from pydantic import Field

REST_API_BASE = "https://rest.runpod.io/v1"
INFERENCE_API_BASE = "https://api.runpod.ai/v2"


class ServerlessResource(dg.ConfigurableResource):
    api_key: str = Field(
        default_factory=lambda: os.getenv("RUNPOD_API_KEY", ""),
        description="RunPod API key",
    )
    timeout: int = Field(
        default=120,
        description="Maximum time to wait for response (seconds)",
    )

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with API key."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        api_base: str = None,
        json_data: dict | None = None,
        timeout: int = 30,
        handle_404: bool = False,
        accept_status_codes: tuple[int, ...] = (200,),
    ) -> dict | None:
        """
        Helper method to make API requests with consistent error handling.

        Args:
            method: HTTP method (GET, POST, PATCH, etc.)
            endpoint: API endpoint path (e.g., "/templates" or "/templates/{id}")
            api_base: Base URL to use (defaults to REST_API_BASE)
            json_data: JSON payload for POST/PATCH requests
            timeout: Request timeout in seconds
            context: Optional Dagster context for logging
            handle_404: If True, return None for 404 responses instead of raising
            accept_status_codes: Tuple of acceptable status codes (default: (200,))

        Returns:
            Response JSON dict, or None if 404 and handle_404=True

        Raises:
            RuntimeError: If request fails or returns unexpected status code
        """
        if api_base is None:
            api_base = REST_API_BASE

        url = f"{api_base}{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                json=json_data,
                headers=self._get_headers(),
                timeout=timeout,
            )

            # Handle 404 if requested
            if response.status_code == 404 and handle_404:
                return None

            # Check if status code is acceptable
            if response.status_code not in accept_status_codes:
                error_detail = response.text
                raise RuntimeError(
                    f"Request failed (status {response.status_code}): {error_detail}"
                )

            response.raise_for_status()

            # Return JSON if available, otherwise return empty dict
            if response.content:
                return response.json()
            return {}

        except requests.exceptions.RequestException as e:
            raise

    def create_template(
        self,
        name: str,
        image_name: str,
        container_disk_gb: int = 10,
        is_serverless: bool = True,
        context: dg.AssetExecutionContext | None = None,
    ) -> str:
        """
        Create a new RunPod template for serverless deployment.

        Templates define the Docker image and configuration for endpoints.

        Args:
            name: Template name (must be unique)
            image_name: Docker image (e.g., "username/image:tag")
            container_disk_gb: Container disk size in GB
            is_serverless: Whether template is for serverless (default: True)
            context: Optional Dagster context for logging

        Returns:
            Created template ID

        Reference:
            https://docs.runpod.io/api-reference/templates/POST/templates
        """
        payload = {
            "name": name,
            "imageName": image_name,
            "containerDiskInGb": container_disk_gb,
            "isServerless": is_serverless,
            "isPublic": False,
        }

        try:
            data = self._make_request(
                method="POST",
                endpoint="/templates",
                json_data=payload,
                timeout=30,
                context=context,
                accept_status_codes=(200, 201),
            )
        except RuntimeError as e:
            # Check if error is "Template name must be unique"
            error_message = str(e)
            if "Template name must be unique" in error_message:
                # Try to find the existing template by name
                existing_template = self.find_template_by_name(name, context=context)

                if existing_template:
                    template_id = existing_template.get("id")
                    return template_id
                else:
                    # Template exists but we can't find it - this shouldn't happen
                    raise RuntimeError(
                        f"Template '{name}' exists but could not be found. "
                        f"Please delete it manually at https://www.runpod.io/console/serverless "
                        f"and re-run this asset."
                    )
            # Re-raise other errors
            raise

        template_id = data.get("id")

        if not template_id:
            raise RuntimeError(f"No template ID returned: {data}")

        return template_id

    def update_template(
        self,
        template_id: str,
        image_name: str | None = None,
        container_disk_gb: int | None = None,
        context: dg.AssetExecutionContext | None = None,
    ) -> None:
        """
        Update an existing RunPod template.

        Args:
            template_id: Template ID to update
            image_name: New Docker image (optional)
            container_disk_gb: New container disk size (optional)
            context: Optional Dagster context for logging

        Reference:
            https://docs.runpod.io/api-reference/templates/PATCH/templates/templateId
        """
        # Build update payload with only provided fields
        payload = {}
        if image_name is not None:
            payload["imageName"] = image_name
        if container_disk_gb is not None:
            payload["containerDiskInGb"] = container_disk_gb

        if not payload:
            return

        self._make_request(
            method="PATCH",
            endpoint=f"/templates/{template_id}",
            json_data=payload,
            timeout=30,
            context=context,
        )

    def get_template(
        self,
        template_id: str,
        context: dg.AssetExecutionContext | None = None,
    ) -> dict | None:
        """
        Get information about a template by ID.
        Returns None if the template does not exist (404).

        Reference:
            https://docs.runpod.io/api-reference/templates/GET/templates/templateId
        """
        return self._make_request(
            method="GET",
            endpoint=f"/templates/{template_id}",
            timeout=10,
            context=context,
            handle_404=True,
        )

    def list_templates(
        self,
        context: dg.AssetExecutionContext | None = None,
    ) -> list[dict]:
        """
        List all templates accessible to this API key.

        Returns:
            List of template dictionaries

        Reference:
            https://docs.runpod.io/api-reference/templates/GET/templates
        """
        return self._make_request(
            method="GET",
            endpoint="/templates",
            timeout=10,
            context=context,
        )

    def find_template_by_name(
        self,
        name: str,
        context: dg.AssetExecutionContext | None = None,
    ) -> dict | None:
        """
        Find a template by name.
        Returns the template dict if found, None otherwise.
        """
        templates = self.list_templates(context=context)

        for template in templates:
            if template.get("name") == name:
                return template

        return None

    def run_inference(
        self,
        endpoint_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        context: dg.AssetExecutionContext | None = None,
    ) -> dict[str, Any]:
        """
        Run inference on the serverless endpoint (async with polling).

        Args:
            endpoint_id: RunPod Serverless endpoint ID
            messages: List of conversation messages with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            context: Optional Dagster context for logging

        Returns:
            Response dictionary from custom handler
        """
        # Submit job
        payload = {
            "input": {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
            }
        }

        job_data = self._make_request(
            method="POST",
            endpoint=f"/{endpoint_id}/run",
            api_base=INFERENCE_API_BASE,
            json_data=payload,
            timeout=30,
            context=context,
        )

        job_id = job_data.get("id")

        if not job_id:
            raise ValueError(f"No job ID returned: {job_data}")

        # Poll for results
        start_time = time.time()

        while True:
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Inference timed out after {self.timeout}s")

            status_data = self._make_request(
                method="GET",
                endpoint=f"/{endpoint_id}/status/{job_id}",
                api_base=INFERENCE_API_BASE,
                timeout=10,
                context=context,
            )

            status = status_data.get("status")

            if status == "COMPLETED":
                output = status_data.get("output", {})
                return output

            elif status == "FAILED":
                error = status_data.get("error", "Unknown error")
                raise RuntimeError(f"Inference failed: {error}")

            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                # Still processing, wait and retry
                time.sleep(1)

            else:
                raise RuntimeError(f"Unknown status: {status}")

    def get_endpoint(
        self,
        endpoint_id: str,
        context: dg.AssetExecutionContext | None = None,
    ) -> dict[str, Any] | None:
        """
        Get endpoint information by ID.

        Args:
            endpoint_id: Endpoint ID to retrieve
            context: Optional Dagster context for logging

        Returns:
            Endpoint info dict if exists, None if not found (404)

        Reference:
            https://docs.runpod.io/api-reference/endpoints/GET/endpoints/endpointId
        """
        return self._make_request(
            method="GET",
            endpoint=f"/endpoints/{endpoint_id}",
            timeout=30,
            context=context,
            handle_404=True,
        )

    def list_endpoints(
        self,
        context: dg.AssetExecutionContext | None = None,
    ) -> list[dict[str, Any]]:
        """
        List all serverless endpoints accessible to this API key.

        Returns:
            List of endpoint dictionaries

        Reference:
            https://docs.runpod.io/api-reference/endpoints/GET/endpoints
        """
        return self._make_request(
            method="GET",
            endpoint="/endpoints",
            timeout=10,
            context=context,
        )

    def find_endpoint_by_name(
        self,
        name: str,
        context: dg.AssetExecutionContext | None = None,
    ) -> dict[str, Any] | None:
        """
        Find a serverless endpoint by name.

        Note: RunPod may append suffixes to endpoint names (e.g., "nanochat-d4" → "nanochat-d4 -fb"),
        so we match either exact name or name with a space suffix.

        Returns the endpoint dict if found, None otherwise.
        """
        endpoints = self.list_endpoints(context=context)

        for endpoint in endpoints:
            endpoint_name = endpoint.get("name", "")
            # Match exact name or name with RunPod suffix (starts with name followed by space)
            if endpoint_name == name or endpoint_name.startswith(f"{name} "):
                return endpoint

        return None

    def create_endpoint(
        self,
        name: str,
        template_id: str,
        workers_min: int = 0,
        workers_max: int = 3,
        gpu_count: int = 1,
        idle_timeout: int = 5,
        context: dg.AssetExecutionContext | None = None,
    ) -> str:
        """
        Create a new serverless endpoint using the REST API.

        Note: You must create a template in the RunPod UI first (https://www.runpod.io/console/serverless)
        with your Docker image and GPU configuration. Then use that template ID here.

        Args:
            name: Endpoint name
            template_id: RunPod template ID (create this in the UI first)
            workers_min: Minimum workers (0 for scale-to-zero)
            workers_max: Maximum workers
            gpu_count: GPUs per worker (default: 1)
            idle_timeout: Idle timeout in seconds (1-3600)
            context: Optional Dagster context for logging

        Returns:
            Created endpoint ID

        Reference:
            https://docs.runpod.io/api-reference/endpoints/POST/endpoints
        """
        payload = {
            "name": name,
            "templateId": template_id,
            "gpuCount": gpu_count,
            "workersMin": workers_min,
            "workersMax": workers_max,
            "idleTimeout": idle_timeout,
        }

        data = self._make_request(
            method="POST",
            endpoint="/endpoints",
            json_data=payload,
            timeout=30,
            context=context,
            accept_status_codes=(200, 201),
        )

        endpoint_id = data.get("id")

        if not endpoint_id:
            raise RuntimeError(f"No endpoint ID returned: {data}")

        return endpoint_id

    def update_endpoint(
        self,
        endpoint_id: str,
        template_id: str | None = None,
        workers_min: int | None = None,
        workers_max: int | None = None,
        context: dg.AssetExecutionContext | None = None,
    ) -> None:
        """
        Update an existing serverless endpoint.

        Args:
            endpoint_id: Endpoint ID to update
            template_id: New template ID (optional, to update image/config)
            workers_min: New minimum workers (optional)
            workers_max: New maximum workers (optional)
            context: Optional Dagster context for logging

        Reference:
            https://docs.runpod.io/api-reference/endpoints/POST/endpoints/endpointId/update
        """
        # Build update payload with only provided fields
        payload = {}
        if template_id is not None:
            payload["templateId"] = template_id
        if workers_min is not None:
            payload["workersMin"] = workers_min
        if workers_max is not None:
            payload["workersMax"] = workers_max

        self._make_request(
            method="POST",
            endpoint=f"/endpoints/{endpoint_id}/update",
            json_data=payload,
            timeout=30,
            context=context,
        )
