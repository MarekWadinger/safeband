"""Entry point for running the RPC outlier detection client."""

import logging

from rpc_server import RpcOutlierDetector
from safeband.parse import get_params

RPC_ENDPOINT = "rpc_online_outlier_detection"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = get_params()

    client: RpcOutlierDetector = RpcOutlierDetector()
    assert config.client is not None
    client.start(
        client=config.client,
        io=config.io,
        model_params=config.model,
        setup=config.setup,
        email=config.email,
    )
