import configparser
class Argu:
    train_dir:str
    batch_size:int
    MAX_NUM_PEAK:int
    MZ_MAX:float
    num_ions:int
    n_classes:int
    units:int
    beam_size:int
    knapsack:str
    denovo_input_spectrum_file:str
    denovo_input_mirror_spectrum_file:str
    denovo_input_feature_file:str
    denovo_output_file:str
def init_args(file):
    config = configparser.ConfigParser()
    config.read(file)
    args = Argu()
    args.train_dir = config["train"]["train_dir"]
    args.batch_size = int(config["train"]["batch_size"])
    args.processes = int(config["train"]["processes"])
    args.MAX_NUM_PEAK = int(config["train"]["MAX_NUM_PEAK"])
    args.MZ_MAX = float(config["train"]["MZ_MAX"])
    args.num_ions = int(config["train"]["num_ions"])
    args.n_classes = int(config["model"]["n_classes"])
    args.units = int(config["model"]["units"])
    args.beam_size = int(config["search"]["beam_size"])
    args.knapsack = config["search"]["knapsack"]
    args.denovo_input_spectrum_file = config["data"]["denovo_input_spectrum_file"]
    args.denovo_input_mirror_spectrum_file = config["data"]["denovo_input_mirror_spectrum_file"]
    args.denovo_input_feature_file = config["data"]["denovo_input_feature_file"]
    args.denovo_output_file = config["data"]["denovo_output_file"]

    return args


