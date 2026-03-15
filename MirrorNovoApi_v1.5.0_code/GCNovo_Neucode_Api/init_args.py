import configparser


class Argu:
    train_dir: str
    batch_size: int
    MAX_NUM_PEAK: int
    MZ_MAX: float
    num_ions: int
    n_classes: int
    units: int
    beam_size: int
    max_mass_tolerance: float
    mass_tolerance_ppm: int
    knapsack: str
    denovo_input_spectrum_file: str
    denovo_input_feature_file: str
    denovo_output_file: str
    accuracy_file: str
    denovo_only_file: str
    scan2fea_file: str
    multifea_file: str
    MAX_LEN: int


def init_args(file):
    config = configparser.ConfigParser()
    config.read(file)
    args = Argu()
    args.train_dir = config["train"]["train_dir"]
    args.engine_model = config["train"]["engine_model"]
    args.batch_size = int(config["train"]["batch_size"])
    args.cuda_device = int(config["train"]["cuda_device"])
    args.type = int(config["train"]["type"])
    args.MAX_NUM_PEAK = int(config["train"]["MAX_NUM_PEAK"])
    args.MAX_LEN = int(config["train"]["MAX_LEN"])
    args.MZ_MAX = float(config["train"]["MZ_MAX"])
    args.num_ions = int(config["train"]["num_ions"])
    args.n_classes = int(config["model"]["n_classes"])
    args.units = int(config["model"]["units"])
    args.input_dim = int(config["model"]["input_dim"])
    args.output_dim = int(config["model"]["output_dim"])
    args.embedding_size = int(config["model"]["embedding_size"])
    args.num_lstm_layers = int(config["model"]["num_lstm_layers"])
    args.edges_classes = int(config["model"]["edges_classes"])
    args.lstm_hidden_units = int(config["model"]["lstm_hidden_units"])
    args.dropout = float(config["model"]["dropout"])
    args.use_lstm = config["model"]["use_lstm"]

    args.beam_size = int(config["search"]["beam_size"])
    args.max_mass_tolerance = float(config["search"]["max_mass_tolerance"])
    args.mass_tolerance_ppm = int(config["search"]["mass_tolerance_ppm"])

    args.knapsack = config["search"]["knapsack"]
    args.denovo_input_spectrum_file = config["data"]["denovo_input_spectrum_file"]
    args.denovo_input_feature_file = config["data"]["denovo_input_spectrum_file"] + ".feature.csv"
    args.denovo_output_file = config["data"]["denovo_output_file"]
    args.accuracy_file = config["data"]["denovo_output_file"] + "_model"
    args.denovo_only_file = config["data"]["denovo_output_file"] + ".denovo_only"
    args.scan2fea_file = config["data"]["denovo_output_file"] + ".scan2fea"
    args.multifea_file = config["data"]["denovo_output_file"] + ".multifea"

    return args
