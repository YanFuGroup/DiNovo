from dataclasses import dataclass
from collections import defaultdict
from data_reader import DDAFeature
from model_gcn import Direction
import config
import logging

logger = logging.getLogger(__name__)


@dataclass
class BeamSearchedSequence:
    sequence: list  # list of aa id
    position_score: list
    score: float  # average by length score
    direction: Direction

class DenovoWriter(object):
    def __init__(self, args):
        self.args = args
        self.output_handle = open(self.args.denovo_output_file, 'w')
        header_list = ["feature_id",
                       "feature_area",
                       "predicted_sequence",
                       "predicted_score",
                       "predicted_position_score",
                       "precursor_mz",
                       "precursor_charge",
                       "protein_access_id",
                       "scan_list_middle",
                       "scan_list_original",
                       "predicted_score_max"]
        header_row = "\t".join(header_list)
        # print(header_row, file=self.output_handle, end='\n')
        self.output_handle2 = open(args.denovo_output_file + '.beamsearch', 'w')

    def close(self):
        self.output_handle.close()
        self.output_handle2.close()
    def write(self, dda_original_feature: DDAFeature, searched_sequence: BeamSearchedSequence):
        """
        keep the output in the same format with the tensorflow version
        :param dda_original_feature:
        :param searched_sequence:
        :return:
        """
        feature_id = dda_original_feature.feature_id
        feature_area = dda_original_feature.feature_area
        precursor_mz = str(dda_original_feature.mz)
        precursor_charge = str(dda_original_feature.z)
        scan_list_middle = dda_original_feature.scan
        scan_list_original = dda_original_feature.scan
        if searched_sequence.sequence:
            predicted_sequence = ','.join([GCnovo_config.vocab_reverse[aa_id] for
                                           aa_id in searched_sequence.sequence])
            predicted_score = "{:.2f}".format(searched_sequence.score)
            predicted_score_max = predicted_score
            predicted_position_score = ','.join(['{0:.2f}'.format(x) for x in searched_sequence.position_score])
            protein_access_id = 'DENOVO'
        else:
            predicted_sequence = ""
            predicted_score = ""
            predicted_score_max = ""
            predicted_position_score = ""
            protein_access_id = ""
        predicted_row = "\t".join([feature_id,
                                   feature_area,
                                   predicted_sequence,
                                   predicted_score,
                                   predicted_position_score,
                                   precursor_mz,
                                   precursor_charge,
                                   protein_access_id,
                                   scan_list_middle,
                                   scan_list_original,
                                   predicted_score_max])
        # print(predicted_row, file=self.output_handle, end="\n")

    def write_beamsearch(self, dda_original_feature: DDAFeature, beam_search_batch: list):
        # print('BEGIN', file=self.output_handle2, end='\n')
        feature_id = dda_original_feature.feature_id
        feature_area = dda_original_feature.feature_area
        precursor_mz = str(dda_original_feature.mz)
        precursor_charge = str(dda_original_feature.z)
        scan_list_middle = dda_original_feature.scan
        scan_list_original = dda_original_feature.scan
        predicted_row = "\t".join([feature_id,
                                   feature_area,
                                   precursor_mz,
                                   precursor_charge])
        # print(predicted_row, file=self.output_handle2, end="\n")
        header_list2 = ["index",
                        "direction",
                        "predicted_sequence",
                        "predicted_score",
                        "predicted_position_score"
                        ]
        header_row2 = "\t".join(header_list2)
        # print(header_row2, file=self.output_handle2, end='\n')
        new_beam_search_batch = []
        temp_sequence_list = []
        temp_sequence_dict = defaultdict(BeamSearchedSequence)
        for searched_sequence in beam_search_batch:
            if not searched_sequence.sequence:
                new_beam_search_batch.append(searched_sequence)
                # print(feature_id)
                continue
            key = '_'.join(map(str, searched_sequence.sequence))
            if key in temp_sequence_list:
                if searched_sequence.score > temp_sequence_dict[key].score:
                    temp_sequence_dict[key] = searched_sequence
            else:
                temp_sequence_dict[key] = searched_sequence
                temp_sequence_list.append(key)
        new_beam_search_batch.extend(list(temp_sequence_dict.values()))

        for id, searched_sequence in enumerate(new_beam_search_batch):
            if searched_sequence.sequence:
                predicted_sequence = ','.join([GCnovo_config.vocab_reverse[aa_id] for
                                               aa_id in searched_sequence.sequence])
                predicted_score = "{:.2f}".format(searched_sequence.score)
                predicted_score_max = predicted_score
                predicted_position_score = ','.join(['{0:.2f}'.format(x) for x in searched_sequence.position_score])
                protein_access_id = 'DENOVO'
                direction = searched_sequence.direction
            else:
                predicted_sequence = ""
                predicted_score = ""
                predicted_score_max = ""
                predicted_position_score = ""
                protein_access_id = ""
                direction = ""
            predicted_row = "\t".join([str(id),
                                       str(direction),
                                       predicted_sequence,
                                       predicted_score,
                                       predicted_position_score
                                       ])
            # print(predicted_row, file=self.output_handle2, end="\n")
        # print('END', file=self.output_handle2, end="\n")
    def __del__(self):
        self.close()
