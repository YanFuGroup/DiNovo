import pandas as pd

num_str_set = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

def mgftofeature(path):
    # print(path)
    mgf = open(path, 'r')
    line = mgf.readline()
    data = []
    while line:
        if line.startswith('BEGIN IONS'):
            line = mgf.readline()
            while line:
                if line[0] == "T" and line.startswith('TITLE'):
                    spec_group_id = line[6:-1]
                elif line[0] == "C" and line.startswith('CHARGE'):
                    z = line.split('=')[1][0]
                elif line[0] == "R" and line.startswith('RTINSECONDS'):
                    rt_mean = line.split('=')[1][:-1]
                elif line[0] == "P" and line.startswith('PEPMASS'):
                    mz = line.split('=')[1][:-1]
                elif line[0] in num_str_set:
                    data.append([spec_group_id,mz,z,rt_mean,spec_group_id])
                    break
                line = mgf.readline()
        line = mgf.readline()
    mgf.close()
    # print(len(data))
    feature = pd.DataFrame(data,columns=['spec_group_id', 'm/z', 'z', 'rt_mean', 'scans'])
    feature["profile"]=0
    feature["feature area"]=0
    feature["seq"]=0
    feature.to_csv(path + '.feature.csv', index=False)
    return feature

if __name__ == '__main__':
    paths = [
        # "Ecoli_lysargiNase_lysN.mgf",
        # "Ecoli_lysC.mgf",
        # "Ecoli_lysN.mgf",
        # "Ecoli_trypsin_lysC.mgf",
        # "Yeast_lysargiNase_lysN.mgf",
        # "Yeast_lysC.mgf",
        # "Yeast_lysN.mgf",
        # "Yeast_trypsin_lysC.mgf",
    ]
    for path in paths:
        feature = mgftofeature(path)

    '''
Ecoli_lysargiNase_lysN.mgf
1364202
Ecoli_lysC.mgf
1829193
Ecoli_lysN.mgf
1630987
Ecoli_trypsin_lysC.mgf
1378808
Yeast_lysargiNase_lysN.mgf
725829
Yeast_lysC.mgf
1163099
Yeast_lysN.mgf
873543
Yeast_trypsin_lysC.mgf
1111519
    '''
