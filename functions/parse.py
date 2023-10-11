from argparse import ArgumentParser, FileType
from configparser import ConfigParser


def get_argparser():  # pragma: no cover
    """
    Parse command-line arguments and return the parsed namespace object.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-f', '--config-file', type=FileType('r'),
                        default='config.ini')
    parser.add_argument('-k', '--key-path', help='Path to RSA keys',
                        default='.security')
    # "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/power"
    parser.add_argument("-t", "--topic", nargs='*', type=str,
                        help="Topic of MQTT or Column of pd.DataFrame")
    return parser


def get_config(config_file):  # pragma: no cover
    config_parser = ConfigParser()
    config_parser.read_file(config_file)

    if (config_parser.has_option('file', 'path') and
            config_parser.get('file', 'path') and
            config_parser.has_option('file', 'output') and
            config_parser.get('file', 'output')):
        config = dict(config_parser['file'])
    elif (config_parser.has_section('mqtt') and
          config_parser.has_option('mqtt', 'host') and
          config_parser.has_option('mqtt', 'port') and
          config_parser.get('mqtt', 'host') and
          config_parser.get('mqtt', 'port')):
        config = dict(config_parser['mqtt'])
        config['port'] = int(config['port'])
    elif (config_parser.has_section('kafka') and
          config_parser.has_option('kafka', 'bootstrap.servers') and
          config_parser.get('kafka', 'bootstrap.servers')):
        config = dict(config_parser['kafka'])
    elif (config_parser.has_section('pulsar') and
          config_parser.has_option('pulsar', 'service_url') and
          config_parser.get('pulsar', 'service_url')):
        config = dict(config_parser['pulsar'])
    else:
        # TODO: Handle possible errorous scenarios
        raise ValueError("Missing configuration.")
    return config
