from server import OutlierDetector
from functions.parse import get_argparser, get_config


if __name__ == '__main__':
    parser = get_argparser()
    parser.add_argument("-d", "--debug",
                        help="Debug the file using loop as source",
                        default=False, type=bool)
    args = parser.parse_args()

    config = get_config(args.config_file)

    client = OutlierDetector()

    client.start(config, args.topic, args.key_path, args.debug)
