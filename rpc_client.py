from functions.parse import get_argparser, get_config
from rpc_server import RpcOutlierDetector

RPC_ENDPOINT = "rpc_online_outlier_detection"

if __name__ == '__main__':
    parser = get_argparser()
    parser.add_argument("-d", "--debug",
                        help="Debug the file using loop as source",
                        default=False, type=bool)
    args = parser.parse_args()

    config = get_config(args.config_file)

    client: RpcOutlierDetector = RpcOutlierDetector()
    client.start(config, args.topic, args.key_path, args.debug)
