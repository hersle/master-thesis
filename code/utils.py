import numpy as np

def writecols(cols, headers, filename):
    maxlen = max(len(col) for col in cols)
    for col in cols:
        while len(col) < maxlen:
            col.append(np.nan)
        
    file = open(filename, "w")
    file.write(" ".join(headers) + "\n")
    for r in range(0, maxlen):
        for col in cols:
            file.write(str(col[r]) + " ")
        file.write("\n")
    file.close()
