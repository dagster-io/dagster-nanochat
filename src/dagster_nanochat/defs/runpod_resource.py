import time
from typing import Any

import dagster as dg
import runpod


class RunPodResource(dg.ConfigurableResource):
    """
    Dagster resource for managing RunPod GPU instances.

    Provides methods to create and terminate pods using a Docker image
    that contains all code, dependencies, and the rustbpe tokenizer.

    Configuration:
    - api_key: RunPod API key
    - gpu_type_id: GPU type (e.g., 'NVIDIA A40', 'RTX A4000')
    - gpu_count: Number of GPUs (1-8, default 2 for DDP training)
    - cloud_type: SECURE, COMMUNITY, or ALL
    - env_variables: Optional dict of environment variables to pass to container
    """

    api_key: str
    gpu_type_id: str = "NVIDIA A40"
    gpu_count: int = 2
    cloud_type: str = "SECURE"
    env_variables: dict[str, str] = {}

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 1 <= self.gpu_count <= 8:
            raise ValueError(f"gpu_count must be between 1 and 8, got {self.gpu_count}")

    def run_pod(
        self,
        pod_name: str,
        image_name: str,
        command: str,
        context: dg.AssetExecutionContext,
        volume_in_gb: int = 300,
        gpu_count: int | None = None,
    ) -> dict[str, Any]:
        """
        Run a command on a new RunPod GPU instance.

        Creates a pod and executes the specified command. The pod will run
        the command and can be terminated when complete.

        Args:
            pod_name: Name for the pod
            image_name: Docker image to use
            command: Command to execute when pod starts
            context: Dagster context for logging
            volume_in_gb: Storage size in GB (default 300)
            gpu_count: Number of GPUs (default: use resource config)

        Returns:
            Pod metadata dictionary with id, ip, etc.
        """
        runpod.api_key = self.api_key

        # Use provided gpu_count or fall back to resource config
        actual_gpu_count = gpu_count if gpu_count is not None else self.gpu_count

        # Command will run automatically when container starts
        # Pass environment variables to container
        pod = runpod.create_pod(
            name=pod_name,
            image_name=image_name,
            gpu_type_id=self.gpu_type_id,
            gpu_count=actual_gpu_count,
            cloud_type=self.cloud_type,
            support_public_ip=True,
            docker_args=command,
            env=self.env_variables if self.env_variables else None,
            volume_in_gb=volume_in_gb,  # Allocate storage for Docker image and training data
        )

        pod_id = pod["id"]
        context.log.info(f"Pod created: {pod_id}")
        context.log.info("Waiting for pod to be ready...")

        # Wait for pod to be running
        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            pod_response = runpod.get_pod(pod_id)

            # Handle list response
            if isinstance(pod_response, list):
                if len(pod_response) == 0:
                    context.log.info(f"Pod not found yet, waiting...")
                    time.sleep(10)
                    continue
                pod_status = pod_response[0]
            else:
                pod_status = pod_response

            desired_status = pod_status.get("desiredStatus")

            elapsed = int(time.time() - start_time)
            context.log.info(f"[{elapsed}s] Pod status: {desired_status}")

            # Check if pod is running
            if desired_status == "RUNNING":
                context.log.info("Pod is running and ready!")
                # Small delay to ensure pod is fully ready
                time.sleep(10)
                return pod_status

            time.sleep(10)

        # If we time out, log final state for debugging
        context.log.error("Timeout waiting for pod. Final status:")
        context.log.error(f"Pod structure: {pod_status}")
        raise TimeoutError(f"Pod {pod_id} did not start within {max_wait}s")

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        """
        Get pod status and metadata.

        Args:
            pod_id: Pod identifier

        Returns:
            Pod metadata dictionary
        """
        runpod.api_key = self.api_key
        return runpod.get_pod(pod_id)

    def terminate_pod(
        self,
        pod_id: str,
        context: dg.AssetExecutionContext,
    ) -> None:
        """
        Terminate and cleanup a pod.

        Args:
            pod_id: Pod identifier
            context: Dagster context for logging
        """
        runpod.api_key = self.api_key
        runpod.terminate_pod(pod_id)
        context.log.info("Pod terminated")
