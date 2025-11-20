import os
import time
from typing import Any

import dagster as dg
import requests
from pydantic import Field


class ServerlessResource(dg.ConfigurableResource):
    """
    Resource for running inference on RunPod Serverless endpoints.

    Attributes:
        api_key: RunPod API key (from environment variable RUNPOD_API_KEY)
        timeout: Maximum time to wait for a response (seconds)
    """

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
        if context:
            context.log.info(f"Creating RunPod template: {name}")

        url = "https://rest.runpod.io/v1/templates"

        payload = {
            "name": name,
            "imageName": image_name,
            "containerDiskInGb": container_disk_gb,
            "isServerless": is_serverless,
            "isPublic": False,
        }

        response = requests.post(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=30,
        )

        # Accept both 200 (OK) and 201 (Created) as success
        if response.status_code not in (200, 201):
            error_detail = response.text

            # Check if error is "Template name must be unique"
            if (
                response.status_code == 500
                and "Template name must be unique" in error_detail
            ):
                if context:
                    context.log.warning(
                        f"Template '{name}' already exists, searching for existing template..."
                    )

                # Try to find the existing template by name
                existing_template = self.find_template_by_name(name, context=context)

                if existing_template:
                    template_id = existing_template.get("id")
                    if context:
                        context.log.info(f"Found existing template: {template_id}")
                    return template_id
                else:
                    # Template exists but we can't find it - this shouldn't happen
                    raise RuntimeError(
                        f"Template '{name}' exists but could not be found. "
                        f"Please delete it manually at https://www.runpod.io/console/serverless "
                        f"and re-run this asset."
                    )

            # Other errors
            if context:
                context.log.error(
                    f"Failed to create template. Status: {response.status_code}"
                )
                context.log.error(f"Response: {error_detail}")
                context.log.error(f"Payload sent: {payload}")
            raise RuntimeError(
                f"Failed to create template (status {response.status_code}): {error_detail}"
            )

        response.raise_for_status()

        data = response.json()
        template_id = data.get("id")

        if not template_id:
            raise RuntimeError(f"No template ID returned: {data}")

        if context:
            context.log.info(f"Template created: {template_id}")

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
        if context:
            context.log.info(f"Updating RunPod template: {template_id}")

        url = f"https://rest.runpod.io/v1/templates/{template_id}"

        # Build update payload with only provided fields
        payload = {}
        if image_name is not None:
            payload["imageName"] = image_name
        if container_disk_gb is not None:
            payload["containerDiskInGb"] = container_disk_gb

        if not payload:
            if context:
                context.log.info("No update parameters provided")
            return

        response = requests.patch(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=30,
        )
        response.raise_for_status()

        if context:
            context.log.info(f"Template updated successfully")

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
        if context:
            context.log.info(f"Checking for template: {template_id}")
        url = f"https://rest.runpod.io/v1/templates/{template_id}"
        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                if context:
                    context.log.info(f"Template {template_id} not found (404).")
                return None
            raise
        except Exception as e:
            if context:
                context.log.error(f"Error getting template {template_id}: {e}")
            raise

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
        if context:
            context.log.info("Listing all templates")
        url = "https://rest.runpod.io/v1/templates"
        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if context:
                context.log.error(f"Error listing templates: {e}")
            raise

    def find_template_by_name(
        self,
        name: str,
        context: dg.AssetExecutionContext | None = None,
    ) -> dict | None:
        """
        Find a template by name.
        Returns the template dict if found, None otherwise.
        """
        if context:
            context.log.info(f"Searching for template: {name}")

        templates = self.list_templates(context=context)

        for template in templates:
            if template.get("name") == name:
                if context:
                    context.log.info(f"Found template: {template.get('id')}")
                return template

        if context:
            context.log.info(f"Template '{name}' not found")
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
            endpoint_id: RunPod Serverless endpoint ID (e.g., "9xfrqwkqdnm449")
            messages: List of conversation messages with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            context: Optional Dagster context for logging

        Returns:
            Response dictionary from your custom handler

        Example:
            ```python
            response = serverless.run_inference(
                endpoint_id="9xfrqwkqdnm449",
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=100,
            )
            print(response["response"])
            ```
        """
        if context:
            context.log.info(f"Calling serverless endpoint: {endpoint_id}")

        # Submit job
        url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        payload = {
            "input": {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
            }
        }

        response = requests.post(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=30,
        )
        response.raise_for_status()

        job_data = response.json()
        job_id = job_data.get("id")

        if not job_id:
            raise ValueError(f"No job ID returned: {job_data}")

        if context:
            context.log.info(f"Job submitted: {job_id}")

        # Poll for results
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
        start_time = time.time()

        while True:
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Inference timed out after {self.timeout}s")

            status_response = requests.get(
                status_url,
                headers=self._get_headers(),
                timeout=10,
            )
            status_response.raise_for_status()

            status_data = status_response.json()
            status = status_data.get("status")

            if status == "COMPLETED":
                output = status_data.get("output", {})
                if context:
                    context.log.info(
                        f"Inference complete. Tokens: {output.get('tokens_generated', 'N/A')}"
                    )
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
        url = f"https://rest.runpod.io/v1/endpoints/{endpoint_id}"

        response = requests.get(
            url,
            headers=self._get_headers(),
            timeout=30,
        )

        if response.status_code == 404:
            if context:
                context.log.info(f"Endpoint {endpoint_id} not found")
            return None

        response.raise_for_status()
        return response.json()

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
        if context:
            context.log.info("Listing all endpoints")

        url = "https://rest.runpod.io/v1/endpoints"

        response = requests.get(
            url,
            headers=self._get_headers(),
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

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
        if context:
            context.log.info(f"Searching for endpoint: {name}")

        endpoints = self.list_endpoints(context=context)

        for endpoint in endpoints:
            endpoint_name = endpoint.get("name", "")
            # Match exact name or name with RunPod suffix (starts with name followed by space)
            if endpoint_name == name or endpoint_name.startswith(f"{name} "):
                endpoint_id = endpoint.get("id")
                if context:
                    context.log.info(
                        f"Found endpoint '{endpoint_name}' (ID: {endpoint_id})"
                    )
                return endpoint

        if context:
            context.log.info(f"Endpoint '{name}' not found")
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
        if context:
            context.log.info(f"Creating serverless endpoint: {name}")
            context.log.info(f"Using template: {template_id}")

        url = "https://rest.runpod.io/v1/endpoints"

        payload = {
            "name": name,
            "templateId": template_id,
            "gpuCount": gpu_count,
            "workersMin": workers_min,
            "workersMax": workers_max,
            "idleTimeout": idle_timeout,
        }

        response = requests.post(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=30,
        )

        # Accept both 200 (OK) and 201 (Created) as success
        if response.status_code not in (200, 201):
            error_detail = response.text
            if context:
                context.log.error(
                    f"Failed to create endpoint. Status: {response.status_code}"
                )
                context.log.error(f"Response: {error_detail}")
                context.log.error(f"Payload sent: {payload}")
            raise RuntimeError(
                f"Failed to create endpoint (status {response.status_code}): {error_detail}"
            )

        response.raise_for_status()

        data = response.json()
        endpoint_id = data.get("id")

        if not endpoint_id:
            raise RuntimeError(f"No endpoint ID returned: {data}")

        if context:
            context.log.info(f"Created endpoint: {endpoint_id}")

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
        if context:
            context.log.info(f"Updating serverless endpoint: {endpoint_id}")

        url = f"https://rest.runpod.io/v1/endpoints/{endpoint_id}/update"

        # Build update payload with only provided fields
        payload = {}
        if template_id is not None:
            payload["templateId"] = template_id
        if workers_min is not None:
            payload["workersMin"] = workers_min
        if workers_max is not None:
            payload["workersMax"] = workers_max

        response = requests.post(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=30,
        )
        response.raise_for_status()

        if context:
            context.log.info(f"Endpoint updated successfully")
