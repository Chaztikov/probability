
for i in range(4):
    outfile = 'paramslist.txt'
    params = [0, 1, 2]
    with open(outfile, 'a+') as f:
        for param in params:
            f.write(str(param))
            f.write(' ')
        f.write('\n')
