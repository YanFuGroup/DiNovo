import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


class CFunctionNeuCodeDetection:

    def __init__(self, inputDP):
        self.dp = inputDP
        self.model_path = self.dp.myCFG.B10_CLASSIFICATION_MODEL_PATH
        self.model = self.__captainLoadModel()

    def __captainLoadModel(self):
        tmpModel = lgb.Booster(model_file=self.model_path)
        return tmpModel


    def detect(self, inputMOZ, inputINT) -> list:
        tmpFeature = self.__captainGetFeature(inputMOZ, inputINT)
        y_prob = self.model.predict(tmpFeature, num_iteration=self.model.best_iteration)
        y_pred = [list(x).index(max(x)) for x in y_prob]
        # 最终的输出列表
        return y_pred

    # emmmmm...
    def __captainGetFeature(self, peak_mz, peak_inten):
        peak_inten_dict = {}
        ii = 0
        while ii < len(peak_mz):
            peak_inten_dict.setdefault(peak_mz[ii], []).append(peak_inten[ii])
            ii += 1

        fet_mass = []
        fet_inten = []
        fet_inten_ratio2 = []
        ii = 0
        while ii < len(peak_mz) - 2:
            flag = 0
            diff = peak_mz[ii + 1] - peak_mz[ii]
            diff2 = peak_mz[ii + 2] - peak_mz[ii]

            if diff > 0.026 and diff < 0.047:
                flag = 1
                # 特征3
                fet_mass.append(diff)
                # 特征4
                fet_inten.append(round(peak_inten_dict[peak_mz[ii]][0] / peak_inten_dict[peak_mz[ii + 1]][0], 3))
                # 特征7
                inten_diff = peak_inten_dict[peak_mz[ii]][0] - peak_inten_dict[peak_mz[ii + 1]][0]
                fet_inten_ratio2.append(round(inten_diff / peak_inten_dict[peak_mz[ii]][0], 3))

            elif diff2 > 0.026 and diff2 < 0.047:
                flag = 2
                # 特征3
                fet_mass.append(diff)
                # 特征4
                fet_inten.append(round(peak_inten_dict[peak_mz[ii]][0] / peak_inten_dict[peak_mz[ii + 2]][0], 3))
                # 特征7
                inten_diff = peak_inten_dict[peak_mz[ii]][0] - peak_inten_dict[peak_mz[ii + 2]][0]
                fet_inten_ratio2.append(round(inten_diff / peak_inten_dict[peak_mz[ii]][0], 3))

            else:
                fet_mass.append(0)
                fet_inten.append(0)
                fet_inten_ratio2.append(0)

            ii += 1

        # ∵ ii<len(peak)-2
        fet_mass.append(0)
        fet_inten.append(0)
        fet_inten_ratio2.append(0)
        fet_mass.append(0)
        fet_inten.append(0)
        fet_inten_ratio2.append(0)

        inten_mean_num = np.mean(fet_inten)
        inten_ratio2_mean_num = np.mean(fet_inten_ratio2)
        inten_ratio2_std_num = np.std(fet_inten_ratio2)

        fet_inten_mean = []
        fet_inten_mean_ratio = []
        fet_inten_ratio2_mean = []
        fet_inten_ratio2_std = []

        jj = 0
        while jj < len(peak_mz):
            # 特征5
            fet_inten_mean.append(inten_mean_num)
            # 特征6
            fet_inten_mean_ratio.append(fet_inten[jj] / inten_mean_num)
            # 特征8.1
            fet_inten_ratio2_mean.append(inten_ratio2_mean_num)
            # 特征8.2
            fet_inten_ratio2_std.append(inten_ratio2_std_num)

            jj += 1

        label = pd.DataFrame()
        label['peak'] = peak_mz
        label['inten'] = peak_inten
        label['mass_diff'] = fet_mass

        label['inten_ratio'] = fet_inten
        # label['inten_ratio_mean'] = fet_inten_mean
        # # 特征6
        # label['inten_ratio_mean_ratio'] = fet_inten_mean_ratio
        #
        # label['inten_ratio2'] = fet_inten_ratio2
        # label['inten_ratio2_mean'] = fet_inten_ratio2_std
        # label['inten_ratio2_std'] = fet_inten_ratio2_mean

        return label
