from collections import defaultdict
import config
from member import Candidate,BeamSearchedSequence,DDAFeature,Direction

MirrorFormat = {
    "A_KK":"A1:K-K",
    "A_RR":"A2:R-R",
    "B":"B: R-K",
    "C":"C: K-R",
    "D":"D: K-X",
    "E":"E: R-X",
    "F":"F: X-K",
    "G":"G: X-R",
    "H":"H: X-X",#MultiNovo
}
class DenovoWriter(object):
    def __init__(self, args,denovo_output_file):
        self.args = args
        self.output_handle = open(denovo_output_file, 'w')

        header = "A_TITLE\tB_TITLE\tMIRROR_TYPE\n"
        self.output_handle.write(header)
        # header = "\tCAND_RANK\tTRY_CALC_MH+\tSEQUENCE\tMODIFICATIONS\tPEPTIDE_SCORE\tAA_SCORE\n"
        header = "\tCAND_RANK\tSEQUENCE\tMODIFICATIONS\tPEPTIDE_SCORE\tAA_SCORE\n"
        self.output_handle.write(header)

    def close(self):
        self.output_handle.close()

    def write_beamsearch(self, dda_original_feature: DDAFeature, beam_search_batch: list):
        feature_id = dda_original_feature.feature_id
        precursor_mz = dda_original_feature.mz
        precursor_charge = dda_original_feature.z
        feature_category = MirrorFormat[dda_original_feature.category]
        new_beam_search_batch = []
        temp_sequence_list = []
        temp_sequence_dict = defaultdict(BeamSearchedSequence)
        for searched_sequence in beam_search_batch:
            if len(temp_sequence_list) >= self.args.beam_size*2:
                break
            if not searched_sequence.sequence:
                new_beam_search_batch.append(searched_sequence)

                continue
            key = '_'.join(map(str, searched_sequence.sequence))
            if key in temp_sequence_list:
                if searched_sequence.score > temp_sequence_dict[key].score:
                    temp_sequence_dict[key] = searched_sequence
            else:
                temp_sequence_dict[key] = searched_sequence
                temp_sequence_list.append(key)

        new_beam_search_batch.extend(list(temp_sequence_dict.values()))

        SEQUENCE, MODIFICATION, CALCMASS, AASCORE, PEPSCORE = [], [], [], [], []
        for id, searched_sequence in enumerate(new_beam_search_batch):
            if searched_sequence.sequence:
                predicted_sequence = '' if feature_category[-1] == 'X' else feature_category[-1]
                modification = ''
                for aa_id in searched_sequence.sequence:
                    if aa_id == 7: aa,site = 'C',len(predicted_sequence)
                    elif  aa_id == 15: aa,site = 'M',len(predicted_sequence)
                    else:aa,site = config.vocab_reverse[aa_id],''
                    predicted_sequence += aa
                    if site!='':#["8,Oxidation[M];9,QWER[K];", ...]
                        modification+=str(site+1)+","+("Oxidation[M]" if aa == 'M' else "Carbamidomethyl[C]")+";"
                predicted_score = "{:.2f}".format(searched_sequence.score)
                predicted_score_max = predicted_score
                predicted_position_score = ['{0:.2f}'.format(x) for x in searched_sequence.position_score]
                protein_access_id = 'DENOVO'
                direction = searched_sequence.direction
            else:
                predicted_sequence = ""
                predicted_score = ""
                predicted_score_max = ""
                predicted_position_score = ""
                protein_access_id = ""
                direction = ""
                modification = ""

            SEQUENCE.append(predicted_sequence)
            MODIFICATION.append(modification)
            CALCMASS.append('{0:.5f}'.format(precursor_mz*precursor_charge))
            AASCORE.append(",".join(predicted_position_score))
            PEPSCORE.append(predicted_score)
        RANK = [rank+1 for rank in range(len(SEQUENCE))]

        merged_list = list(zip(RANK,CALCMASS,SEQUENCE,MODIFICATION,PEPSCORE,AASCORE))
        # 将每一行的元素用\t连接起来

        merged_str = feature_id.replace("@", "\t") + "\t" + feature_category+ "\n"
        merged_str += '\n'.join("\t" + '\t'.join(map(str, row)) for row in merged_list) + "\n"

        self.output_handle.write(merged_str)

        return Candidate(SEQUENCE,MODIFICATION,CALCMASS,AASCORE,PEPSCORE)

    def __del__(self):
        self.close()
