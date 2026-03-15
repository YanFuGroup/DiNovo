from dataclasses import dataclass
import numpy as np
from enum import Enum
class Direction(Enum):
    forward = 1
    backward = 2

@dataclass
class DDAFeature:
    feature_id: str
    category: str
    mz: float
    z: float
    peptide: list
    mass: float

    lys_mz: float
    lys_z: float
    lys_peptide: list
    lys_mass: float


@dataclass
class DenovoData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    # spectrum_representation: np.ndarray
    mirror_peak_location: np.ndarray
    mirror_peak_intensity: np.ndarray
    original_dda_feature: DDAFeature

@dataclass
class MGFfeature:
    PEPMASS: float
    CHARGE: int
    SCANS: str
    SEQ: str
    RTINSECONDS: float
    MOZ_LIST: list
    INTENSITY_LIST: list

@dataclass
class TrainData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    lys_peak_location: np.ndarray
    lys_peak_intensity: np.ndarray

    forward_id_target: list
    backward_id_target: list
    forward_ion_location_index_list: list
    backward_ion_location_index_list: list
    forward_id_input: list
    backward_id_input: list

    lys_forward_ion_location_index_list: list
    lys_backward_ion_location_index_list: list


@dataclass
class DDAFeature:
    feature_id: str
    category: str
    mz: float
    z: float
    peptide: list
    mass: float

    lys_mz: float
    lys_z: float
    lys_peptide: list
    lys_mass: float


@dataclass
class DenovoData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    # spectrum_representation: np.ndarray
    mirror_peak_location: np.ndarray
    mirror_peak_intensity: np.ndarray
    original_dda_feature: DDAFeature

@dataclass
class MGFfeature:
    PEPMASS: float
    CHARGE: int
    SCANS: str
    SEQ: str
    RTINSECONDS: float
    MOZ_LIST: list
    INTENSITY_LIST: list

@dataclass
class TrainData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    lys_peak_location: np.ndarray
    lys_peak_intensity: np.ndarray

    forward_id_target: list
    backward_id_target: list
    forward_ion_location_index_list: list
    backward_ion_location_index_list: list
    forward_id_input: list
    backward_id_input: list

    lys_forward_ion_location_index_list: list
    lys_backward_ion_location_index_list: list

@dataclass
class BeamSearchedSequence:
    sequence: list  # list of aa id
    position_score: list
    score: float  # average by length score
    direction: Direction
#denovo
@dataclass
class BeamSearchStartPoint:
    prefix_mass: float
    suffix_mass: float
    mass_tolerance: float
    direction: Direction




@dataclass
class BeamSearchResult:
    dda_feature: DDAFeature
    beam_search_batch: list
#writer

@dataclass
class Candidate:
    SEQUENCE: []  # = ["PEPTIDEMK", ...]      # item: string - only pure AA
    MODIFICATION: []  # = ["8,Oxidation[M];9,QWER[K];", ...]  # item: string - modName
    CALCMASS: []  # = [xxxx.xx, ...]      # item: float - calculate peptide mass
    AASCORE: []  # = [[x1,x2,x3,x4,x5,x6,x7,x8,x9], ...]      # item: list
    PEPSCORE: [] # = []  # item: float
@dataclass
class BeamSearchedSequence:
    sequence: list  # list of aa id
    position_score: list
    score: float  # average by length score
    direction: Direction
@dataclass
class DenovoResult:
    dda_feature: DDAFeature
    best_beam_search_sequence: BeamSearchedSequence
@dataclass
class SearchPath:
    aa_id_list: list
    # mirror_aa_id_list: list
    aa_seq_mass: float
    # mirror_aa_seq_mass: float
    score_list: list
    score_sum: float
    # lstm_state: tuple  # state tupe store in search path is of shape [num_lstm_layers, num_units]
    direction: Direction


@dataclass
class SearchEntry:
    feature_index: int
    current_path_list: list  # list of search paths
    spectrum_state: tuple  # tuple of (peak_location, peak_intensity)
    error_mass: float
    error_aa_id: int




@dataclass
class BeamSearchStartPoint:
    prefix_mass: float
    suffix_mass: float
    mass_tolerance: float
    direction: Direction


@dataclass
class DenovoResult:
    dda_feature: DDAFeature
    best_beam_search_sequence: BeamSearchedSequence

@dataclass
class BeamSearchResult:
    dda_feature: DDAFeature
    beam_search_batch: list

@dataclass
class SearchPath:
    aa_id_list: list
    # mirror_aa_id_list: list
    aa_seq_mass: float
    # mirror_aa_seq_mass: float
    score_list: list
    score_sum: float
    # lstm_state: tuple  # state tupe store in search path is of shape [num_lstm_layers, num_units]
    direction: Direction


@dataclass
class SearchEntry:
    feature_index: int
    current_path_list: list  # list of search paths
    spectrum_state: tuple  # tuple of (peak_location, peak_intensity)
    error_mass: float
    error_aa_id: int