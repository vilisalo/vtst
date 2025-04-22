import numpy as np

def frequencies(eigval):
    conversion=5140.399168648488
    eigenvalue_sign=[]
    for i in range(len(eigval)):
        if eigval[i] < 0:
            eigenvalue_sign.append("i")
        else:
            eigenvalue_sign.append("")
    eigval = np.sqrt(np.abs(eigval))
    return (eigval*conversion)


def frequencies_print(eigval):
    conversion=5140.399168648488  # Converts the Hessian eigenvalues to cm-1
    eigenvalue_sign=[]
    for i in range(len(eigval)):
        if eigval[i] < 0:
            eigenvalue_sign.append("i")
        else:
            eigenvalue_sign.append("")
    eigval = np.sqrt(np.abs(eigval))
    print("-----------------------------------")
    print("--Vibrational-Frequencies--(cm-1)--")
    print("-----------------------------------")
    print("#  Frequencies".rjust(14))
    freqcount=1
    for i in range(len(eigval)):
        freq_str = eigenvalue_sign[i]+str(round(eigval[i]*conversion,2))
        whole_str = str(freqcount)+"."+freq_str.rjust(12)
        print(whole_str)
        freqcount+=1
    print("-----------------------------------")
    
def proj_freq_print(eigval,eigval_proj):
    conversion=5140.399168648488  # Converts the Hessian eigenvalues to cm-1
    eigenvalue_sign=[]
    for i in range(len(eigval)):
        if eigval[i] < 0:
            eigenvalue_sign.append("i")
        else:
            eigenvalue_sign.append("")
    eigval = np.sqrt(np.abs(eigval))    
    eigenvalue_sign_proj=[]
    for i in range(len(eigval_proj)):
        if eigval_proj[i] < 0:
            eigenvalue_sign_proj.append("i")
        else:
            eigenvalue_sign_proj.append("")
    eigval_proj = np.sqrt(np.abs(eigval_proj))
    print("-----------------------------------")
    print("--Vibrational-Frequencies--(cm-1)--")
    print("-----------------------------------")
    print("#  Frequencies".rjust(14)+" | "+"Reaction path projected")
    freqcount=1
    for i in range(len(eigval)):
        freq_str = eigenvalue_sign[i]+str(round(eigval[i]*conversion,2))
        projfreq_str = eigenvalue_sign_proj[i]+str(round(eigval_proj[i]*conversion,2))
        whole_str = str(freqcount)+"."+freq_str.rjust(12)+" | "+projfreq_str.ljust(10)
        print(whole_str)
        freqcount+=1
    print("-----------------------------------")