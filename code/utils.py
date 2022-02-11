import numpy as np

def writecols(cols, headers, filename, skipevery=-1):
    print(f"Wrote {headers} to {filename}")
    maxlen = max(len(col) for col in cols)
    for col in cols:
        while len(col) < maxlen:
            col.append(np.nan)
        
    file = open(filename, "w")
    file.write(" ".join(headers) + "\n")
    for r in range(0, maxlen):
        for col in cols:
            if len(str(col[r])) > 0:
                file.write(str(col[r]) + " ")
        file.write("\n")
        if skipevery > 0 and r % skipevery == skipevery - 1:
            file.write("\n")
    file.close()
