from functions.parse import get_params
from rpc_server import RpcOutlierDetector

RPC_ENDPOINT = "rpc_online_outlier_detection"

if __name__ == '__main__':
    config = get_params()

    client: RpcOutlierDetector = RpcOutlierDetector()
    client.start(
        client=config["client"],
        io=config["io"],
        model_params=config["model"],
        setup=config["setup"],
        email=config["email"],
        )
