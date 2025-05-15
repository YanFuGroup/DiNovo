# -*- coding: utf-8 -*-
"""
Author: Zixuan Cao, Piyu Zhou
Email: yfu@amss.ac.cn
Academy of Mathematics and Systems Science
Chinese Academy of Sciences
Project: DiNovo
File: MSData.py
Time: 2022.09.05 Monday
ATTENTION: none
"""

#  主要放和生物相关的一些文件

class CINI:

    # 这几个东东，只要爱因斯坦不被批斗，不太可能会变
    MASS_ELECTRON = 0.0005485799      # float
    MASS_PROTON_MONO = 1.00727645224  # float  # 1.00782503214-0.0005485799
    MASS_PROTON_ARVG = 1.0025         # float
    MASS_NEUTRON_AVRG = 1.003         # float
    # 而最高峰与相邻峰间距小于 1.0 且方差较大

    DICT0_ELEMENT_MASS = {}           # map <string, vector<float>>
    DICT0_ELEMENT_ABDC = {}           # map <string, vector<float>>

    DICT1_AA_COM = {}                 # map <char, vector<string>>
    DICT1_AA_MASS = {}                # map <char, vector<>>
    DICT2_MOD_INFO = {}               # map <string, vector<CModInfo>>
    DICT2_MOD_MASS = {}               # map <string, vector<float>>


class Config:
    # [INI FILE]
    I0_INI_PATH_ELEMENT = "element.ini"    # string
    I1_INI_PATH_AA = "aa.ini"              # string
    I2_INI_PATH_MOD = "modification.ini"  # string

    # [DATA PARAM]
    A0_PATH_CFG_FILE = ""                  # string
    A1_PATH_MGF_TRY = ""                   # string
    A2_PATH_MGF_LYS = ""                   # string
    A3_FIX_MOD = ""                        # string
    A4_VAR_MOD = ""                        # string
    # if you want to compare with pNovo etc
    # maybe you have to
    # mass range of precursor
    A5_MIN_PRECURSOR_MASS = 300.0            # float
    A6_MAX_PRECURSOR_MASS = 3500.0           # float
    A7_LABEL_TYPE = 0
    # 0: no label
    # 1: NeuCode (neutron-encoding, double peaks)
    # 2: ...
    A8_NEUCODE_OUTPUT_TYPE = 1
    # when neucode labeled data, use it
    # 1: write mgf with new line to record neucode or not by 0/1
    # 2: write mgf with generated double peaks(only gene neucode-class peaks shifted 0.036Da)
    A9_DOUBLE_PEAK_GAP = 0.036
    # (K080 - K602)(~0.0360164Da)   R040 - R004(~0.0369676Da)
    # average (K_d + R_d) / 2 = 0.036492 Da
    # 先不考虑其他的了嗷，我先写在这里
    # 后面如果要动态生成的话，就在INI读取之后
    # 质量 gap 的计算应该是用 Emass 算法做的
    # 或者直接使用元素的质量来计算
    A10_NEUCODE_PEAK_MAX_RATIO = 2.0

    A11_MIRROR_PROTEASES_APPROACH = 1    # int flag
    # 1  trypsin & lysargiNase
    # 2  lysC  &   lysN

    # [Preprocess]
    # 丕宇的预处理部分
    B1_ADD_ISOTOPE_INTENSITY = 1         # int flag
    B2_CHECK_NATURAL_LOSS = 1            # int flag
    B3_REMOVE_PRECURSOR_ION = 0          # int flag
    B4_CHECK_PRECURSOR_CHARGE = 1        # int flag
    B5_ROUND_ONE_HOLD_PEAK_NUM = 200     # int
    # 子璇的预处理方法
    B6_ROUND_TWO_REMOVE_IMMONIUM = 1     # int flag
    B7_ROUND_TWO_PEAK_NUM_PER_BIN = 4    # int
    B8_ROUND_TWO_MASS_BIN_LENGTH = 100   # int  maybe float, but just be simple
    # 每隔100Da的质量窗口，取2根最强的峰保留下来

    # 20230214
    # 接入新明的离子类型识别脚本
    B9_NEUCODE_DETECT_APPROACH = 1       # int
    B10_CLASSIFICATION_MODEL_PATH = ""  # string

    # [Pairing]
    C0_MIRROR_DELTA_RT = 900.0    # Retention time threshold for judging spectral pair, default: 300s.(15min * 60s/min = 900s)
    C1_MIRROR_TYPE_A1 = 1         # xxK - Kxx    0.0 Da    -128.09Da|+128.09Da
    C2_MIRROR_TYPE_A2 = 1         # xxR - Rxx    0.0 Da    -156.10Da|+156.10Da
    C3_MIRROR_TYPE_A3 = 0         # xx  -  xx    0.0 Da       0.00Da
    C4_MIRROR_TYPE_B  = 1         # xxR - Kxx    +28 Da    -128.09Da|+156.10Da
    C5_MIRROR_TYPE_C  = 1         # xxK - Rxx    -28 Da    -156.10Da|+128.09Da
    C6_MIRROR_TYPE_D  = 1         # xxK -  xx   +128 Da       0.00Da|+128.09Da
    C7_MIRROR_TYPE_E  = 1         # xxR -  xx   +156 Da       0.00Da|+156.10Da
    C8_MIRROR_TYPE_F  = 1         # xx  - Kxx   -128 Da    -128.09Da|   0.00Da
    C9_MIRROR_TYPE_G  = 1         # xx  - Rxx   -156 Da    -156.10Da|   0.00Da
    # 0: do not consider this mirror type.
    # 1: consider this mirror type;
    # [MIRROR TYPE INTRODUCTION](Trypsin - LysargiNase)
    # TYPE A1: xxxxK - Kxxxx    E(delta(PT-PL)) = (K - K) =    0.00Da, E(delta(FT-FL)) = -K|+K = -128.09Da|+128.09Da 
    # TYPE A2: xxxxR - Rxxxx    E(delta(PT-PL)) = (R - R) =    0.00Da, E(delta(FT-FL)) = -R|+R = -156.10Da|+156.10Da 
    # TYPE A3: xxxx  -  xxxx    E(delta(PT-PL)) = (0 - 0) =    0.00Da, E(delta(FT-FL)) =  0|0  =    0.00Da 
    # TYPE B:  xxxxR - Kxxxx    E(delta(PT-PL)) = (R - K) =  +28.01Da, E(delta(FT-FL)) = -K|+R = -128.09Da|+156.10Da 
    # TYPE C:  xxxxK - Rxxxx    E(delta(PT-PL)) = (K - R) =  -28.01Da, E(delta(FT-FL)) = -R|+K = -156.10Da|+128.09Da 
    # TYPE D:  xxxxK -  xxxx    E(delta(PT-PL)) = (K - 0) = +128.09Da, E(delta(FT-FL)) =  0|+K =    0.00Da|+128.09Da 
    # TYPE E:  xxxxR -  xxxx    E(delta(PT-PL)) = (R - 0) = +156.10Da, E(delta(FT-FL)) =  0|+R =    0.00Da|+156.10Da 
    # TYPE F:  xxxx  - Kxxxx    E(delta(PT-PL)) = (0 - K) = -128.09Da, E(delta(FT-FL)) = -K|0  = -128.09Da|   0.00Da 
    # TYPE G:  xxxx  - Rxxxx    E(delta(PT-PL)) = (0 - R) = -156.10Da, E(delta(FT-FL)) = -R|0  = -156.10Da|   0.00Da 
    # If neuron-encoded sample: Mass(K) should add 8Da, and Mass(R) should add 4Da.

    C10_PAIR_FILTER_APPROACH = 3     # how to filer spectral pairs
    # 0: filtered by direct judge
    # 1: filtered by p-value
    # 2: filtered by FDR (simulated null distribution)
    # 3: filtered by p-value, then filtered by FDR (simulated null distribution)

    C11_PAIR_P_VALUE_THRESHOLD = 0.05 # only if C10 if 1 (using E-Value)

    C12_PAIR_DECOY_APPROACH = 2           # test mode: generate decoy spectral pair
    # 0: no decoy gene
    # 1: shifted precursor mass with +X Da
    # 2: shifted fragment delta with +X Da, within target-decoy competition
    # 3: shifted preprocessed peaks with +X Da

    C13_PAIR_DECOY_SHIFTED_IN_DA = 15.0    # test mode: X = 15.0 by default

    C14_PAIR_FDR_THRESHOLD = 0.02       # test mode: FDR := (D + 1) / max(T, 1)

    # [De Novo]
    D0_WORK_FLOW_NUMBER = 1     # int flag
    # 1: process .mgf files only
    # 2: process .mgf files, find the spectra pair, but do not de novo sequencing
    # 3: process .mgf files, find the spectra pair, and de novo spectral pair
    # 4: process .mgf files, and de novo single spectrum (allow only one mgf path param is not empty)
    # 5: process .mgf files, find the spectra pair, and de novo by way 3... ...etc
    D1_MEMORY_LOAD_MODE = 1       # int flag  大内存方案，把所有二级谱图数据都load到内存，避免 a * b 次的load
    #  do not use this mode  -->  # 0: 关闭大内存模式，改为通过 a * b 次 IO 来降低内存耗费，但这样也会增加运行的时间，但这也主要是IO的时间
    #  do not use this mode  -->  # 1: 开启大内存模式，即直接把所有的谱图数据一次性load进内存中，IO 次数为 a + b，所有谱图数据一直占用内存
    # 1  [less-load mode]  Load two data files into memory every time.
    # 2  [semi-load mode]  Load half data files into memory, default: load all LysargiNase data.
    # 3  [full-load mode]  Load total data files into memory once at the beginning.

    D2_MULTIPROCESS_NUM = 1      # int
    # 根据Python的并行计算原理，并行运行时所用数据会额外占用内存，可以认为直接复制了一整份所用的数据，互相也不会产生读写数据冲突
    # 如果在大内存模式下，即C1_BIG_MEMORY_MODE = 1，则谱图数据占用内存情况为 oriMemory + n * (singleTrypsin + singleLysargiNase)
    # 如果关闭大内存模式，即C1_BIG_MEMORY_MODE = 0，则谱图数据占用内存情况为 oriMemory + n * (singleTrypsin + singleLysargiNase)
    # 考虑并行时 IO 冲突，循环在双 for 内部进行，第一层 for 会先把对应的 Trypsin 数据 load 进去，所以会多 n//a 个 singleTrypsin
    # 所以严格说来，在开启并行时，实际上会多使用的内存占用是 n * (singleTrypsin + singleLysargiNase) + (n // a) * singleTrypsin

    D3_MS_TOL = 20                 # float
    D4_MS_TOL_PPM = 1              # int
    D5_MSMS_TOL = 20               # float
    D6_MSMS_TOL_PPM = 1            # int
    D7_HOLD_CANDIDATE_NUM = 400    # int  具体干什么等以后再看看，先把位置占下来，估计剪枝时候可以用！
    D8_REPORT_PEPTIDE_NUM = 10     # int   最多报告这么多的候选肽段嗷
    # 20230220
    D9_DE_NOVO_APPROACH = 1        # int
    D10_MIRROR_NOVO_MODEL_PATH = ""    # string
    D11_PNOVOM_EXE_PATH = ""           # string
    # -- de novo sequencing for single spectrum
    D12_DE_NOVO_SINGLE_SPECTRUM = 0 # int
    D13_BATCH_SIZE = 2


    # [EXPORT]
    E1_PATH_EXPORT = ".\\test\\"  # string    一个用于存放所有结果的地址，不过我觉得地址
    E2_EXPORT_ROUND_ONE_MGF = 0    # int flag  输出经过丕宇预处理的谱图
    E3_EXPORT_ROUND_TWO_MGF = 0    # int flag  输出经过子璇预处理的谱图
    E4_EXPORT_SPECTRAL_PAIR = 1    # int flag  输出寻找镜像谱图对的结果
    E5_COMBINE_SPLIT_RESULT = 0    # int flag  把单个trypsin文件分散的鉴定结果给汇总起来，并把文件写入到trpsin文件夹下
    E6_COMBINE_TOTAL_RESULT = 1    # int flag  把分散的所有文件都给汇总起来，并把文件写入到export文件夹中去
    E7_EXPORT_FEATURE_MGF = 0    # int flag  输出学力需要的特征标记

    # [VALIDATION]
    V0_FLAG_DO_VALIDATION = 0      # int flag  是否进行 validation 环节，for 谱图对识别与测序
                                   # 0 no validation
                                   # 1 do validation with database search result
                                   # 2 do validation with mapping of fasta file
    V1_PATH_TRY_PFIND_RES = ""     # string    pFind 鉴定结果的路径 of try 酶切数据
    V2_PATH_LYS_PFIND_RES = ""     # string    pFind 鉴定结果的路径 of lys 酶切数据
    V3_PATH_FASTA_FILE    = ""     # string    fasta 文件的路径


# 这是整合的二级谱图文件
# 里面对应了多张二级谱图的信息
# 可以认为就是mgf文件
# [ATTENTION] 这里因为不读取
class CFileMS2:  # 注意：这是按列存储，每个属性都是list。如果按列搞，这个行是一个对象，不太好管理。
    # INDEX_SCAN 是索引用的，所有scan的数值都会存在这里
    INDEX_SCAN = []               # vector<int>    # 这个大小和大家不一样，方便快速索引
    INDEX_RT = []                 # vector<float>

    # LIST: 每个scan只对应一个，因此单独存放数值即可
    LIST_RET_TIME = []            # vector<float>
    LIST_ION_INJECTION_TIME = []  # vector<float>
    LIST_ACTIVATION_CENTER = []   # vector<float>
    LIST_PRECURSOR_SCAN = []      # vector<float>

    # MATRIX: 每一行是个list
    MATRIX_FILE_NAME = []         # list/[] of vector<string>  这是两层的，内部用vector，外部只索引scan，所以用列表存也行
    MATRIX_PEAK_MOZ = []          # list/[] of vector<float>   这是两层的，内部用vector，外部只索引scan，所以用列表存也行
    MATRIX_PEAK_INT = []          # list/[] of vector<float>   这是两层的，内部用vector，外部只索引scan，所以用列表存也行
    # MATRIX: 每一行是个list      # 相同的scan，可能有多个母离子状态（质量+电荷）
    MATRIX_PRECURSOR_CHARGE = []  # list/[] of vector<int>     这是两层的，内部用vector，外部只索引scan，所以用列表存也行
    MATRIX_PRECURSOR_MOZ = []     # list/[] of vector<float>   这是两层的，内部用vector，外部只索引scan，所以用列表存也行


# 单张二级谱图，这个是对应了一张scan
# 虽然pParse把一张scan拆了几个spectra文件
# 但是我不想把moz和int重复存好几遍
# 所以把int和moz合并了，这样filename，charge，precursor信息都需要排序整理起来
# 但是保留时间是唯一的，毕竟是同一个scan
class CMS2Spectrum:
    LIST_FILE_NAME = []         # vector<string>
    LIST_PRECURSOR_CHARGE = []  # vector<int>  # 母离子电荷信息
    LIST_PRECURSOR_MOZ = []     # vector<float>
    LIST_PRECURSOR_MASS = []    # vector<float>
    LIST_PEAK_MOZ = []          # vector<float>
    LIST_PEAK_INT = []          # vector<float>
    SCAN_RET_TIME = 0.0         # float
    NEUCODE_LABEL = []          # vector<float>

# ban
class CSinglePrecursorSpectrum:

    FILE_NAME = ""
    PRECURSOR_CHARGE = 0.0
    PRECURSOR_MASS = 0.0
    PRECURSOR_MOZ = 0.0
    LIST_PEAK_MOZ = []
    LIST_PEAK_INT = []
    SCAN_RET_TIME = 0.0


class CModInfo:
    COMP = ""         # string  # 代表元素的组成
    SITES = ""        # string  # 可能会有多个位点，按字符串形式构建即可
    POSITION = ""     # string  # 氨基酸序列上特定位置，如NORMAL
    COMMON = False    # Boolean  # 标记为True或False
    MASS = -1         # float  # 修饰的质量信息


class CTagDAGNode:
    MASS = -1.0       # float  # 代表对应谱峰的质量
    INTENSITY = -1.0  # float  # 节点的信号强度（暂）
    IN = 0            # int    # 节点入度IN DEGREE，随着配对+1
    OUT = 0           # int    # 节点出度OUT DEGREE，随着配对+1
    VISITED = 0       # int    # 控制循环，以及查找边使用
    LIST_EDGE = []    # vector<CTagDAGEdge>  # 存储CTagDAGEdge，每个元素是边的列表


class CTagDAGEdge:

    WEIGHT_INT_AND_DEGREE = -1  # float  # 这里是半截儿打分，因为tolerance的打分需要实时算，所以就只有半截儿
    TOL = -1                    # float  # 两峰间距，用于后续生成tag时MASS_TOLERANCE的计算
    MASS_TOLERANCE = 100.0      # float  # 对应氨基酸的质量偏差，用于打分，目前没啥用。。但是留着吧
    AA_MARK = ""                # int    # 这里是一个标记，可以对应氨基酸解释的集合，即该边对应的氨基酸、名称与质量信息的标记
    LINK_NODE_INDEX = 1000      # int    # 有向边的边头，谱峰节点的下标，是个偏移地址


class CTagDAG:

    LIST_NODE = []  # vector<CTagDAGNode>  # 存储CTagDAGNode，一个元素是一个节点
    NUM_NODE = 0  # int  # 一共有多少个结点信息
    IN_ZERO_NODE = []  # vector<int>  # 存储氨基酸路径的起点


# 这个得用构造函数了，输入五个参数直接赋值
class CTagInfo:
    def __init__(self, inputLM, inputRM, inputSeq, inputScore, inputMod):
        self.lfMass = inputLM       # 左边质量  float
        self.rfMass = inputRM       # 右边质量  float
        self.tagSeq = inputSeq      # 序列标签  string
        self.tagScore = inputScore  # 打分      float
        self.tagMod = inputMod      # 修饰信息  string


class CDataPack:  # 这个类必须放到最后

    myCFG = Config()  # 这是 Config 对象。。。
    myINI = CINI()  # 这是 CINI 对象。。。
    # EXE_FILE_PATH = ""       # 20230221  exe 文件的绝对路径，for新明的脚本，好烦，还要单独记录一个东西，绝了
    # [ATTENTION]  如果在脚本中直接用os库的获取绝对路径的方案，反而会跳转到C盘的Temp临时文件夹，很迷惑的行为
    #  从config里面搞出来的
    LIST_PATH_MGF_TRY = []   # vector<string>  目前没啥用，先留着
    LIST_PATH_MGF_LYS = []   # vector<string>  二级谱图的路径列表，到时候挨个处理文件的循环得用它
    LIST_OUTPUT_PATH  = []   # vector<string>  输出路径的列表，cfg中的export路径，和raw的文件名的产物
    # ======================== 废 ============================
    # # [作废] 如TRY有a个mgf，LYS有b个mgf，则该项共计有a * b 个路径
    # # [作废] 名字的格式为[TryFileName]@[LysFileName]，用[]分离，并用@字符连接起来
    # ======================== 废 ============================
    # 因为文件夹的长度，过长了可能会有问题，所以考虑直接采用二级菜单的形式把文件给存放在文件夹内
    # 如Trypsin有两个mgf，t1.mgf和t2.mgf，则直接生成2个文件夹，分别名为t1和它，存放镜像谱测序结果
    # 而如果对应的LysargiNase也有两个mgf，则在t1和t2下生成2个文件夹，同时分别以他们的文件名来命名
    # 这样之后也可以直接把测序结果汇总起来，寻址则直接根据列表中的路径+LIST_MS2_NAME_LYS[idx]来找
    # 寻址的方法为LIST_OUTPUT_PATH[i] + LIST_MS2_NAME_LYS[j] + "\\" + result_file_name

    LIST_MGF_NAME_TRY = []   # vector<string>  二级谱图的名字，和TRY的二级谱图文件路径有对应关系
    LIST_MGF_NAME_LYS = []   # vector<string>  二级谱图的名字，和LYS的二级谱图文件路径有对应关系

    logPath = ""

    # 路径，cfg中的export路径，和raw的文件名的产物
    # 当运行多个raw的时候，便于将不同raw的结果分开

# 候选肽的结果
# class Candidate:
#
#     MS_MIRROR_TYPE = 0
#
#     LIST_SEQ = []
#     LIST_MOD = []
#
#     LIST_SPEC = []


class Candidate:
    SEQUENCE : list
    MODIFICATION : list
    CALCMASS : list
    AASCORE : list
    PEPSCORE : list

# 含有 top-10 的结果
class CPSM:

    MS_SPEC_IDX_TRY = []
    MS_SPEC_IDX_LYS = []


    ID_PEP_SEQUENCE = []
    ID_MOD_INFO = []
    ID_CALC_MASS = []


# for xueli

class CPrepLabel:
    FILTERED = [] # 0:fitered low-intensity peaks, 1: top-350 peaks
    CLUSTER = [] # 0 is no cluster, 1,2,...,n is cluster number, if some peaks share same n(n!=0), they belong to one cluster.
    CHARGE = []  # 0 is no idea of charge, 1,2,3,...,n is the charge number
