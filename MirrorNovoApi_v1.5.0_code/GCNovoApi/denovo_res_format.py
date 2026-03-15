def seq2seq(seq: str):
    """
    :return:
    """
    seq = seq.replace('I', 'L')
    sequence = seq.split(',')
    modify = ''
    count = 1
    for s in sequence:
        if s == 'M(Oxidation)':
            modify += str(count) + ',Oxidation[M];'
            sequence[count - 1] = 'M'
        elif s == 'N(Deamidation)':
            modify += str(count) + ',Deamidated[N];'
            sequence[count - 1] = 'N'
        elif s == 'Q(Deamidation)':
            modify += str(count) + ',Deamidated[Q];'
            sequence[count - 1] = 'Q'
        elif s == 'C(Carbamidomethylation)':
            modify += str(count) + ',Carbamidomethyl[C];'
            sequence[count - 1] = 'C'
        count += 1
    return ''.join(sequence) + '\t' + modify


def change_format(file_in, file_out, n=1000000):
    """
    修改输出格式
    :return:
    """
    fw = open(file_out, 'w')
    fw.write('TITLE\n')
    fw.write('\tCAND_RANK\tSEQUENCE\tMODIFICATIONS\tPEPTIDE_SCORE\tAA_SCORE\n')
    with open(file_in, 'r') as fr:
        line = fr.readline()
        while line:
            if line.startswith('BEGIN'):
                line = fr.readline()
                titles = line.split("\t")[0]
                fw.write(titles + '\n')
                fr.readline()
                line = fr.readline()
                while not line.startswith('END'):
                    line_split = line.split()
                    rank = str(int(line_split[0]) + 1)
                    if int(rank) <= n:
                        fw.write('\t' + rank + '\t' + seq2seq(line_split[2]) + '\t' + line_split[3] + '\t' + line_split[4] + '\n')
                    line = fr.readline()
            line = fr.readline()
    fw.close()
