import numpy as np

def alpha_weak(pos, neg, C=0.):
    sup_pos = np.sum(pos[1] <= len(neg[0]) * C) / len(pos[0])
    sup_neg = np.sum(neg[1] <= len(pos[0]) * C) / len(neg[0])
    
    if sup_pos > sup_neg:
        return 1
    elif sup_pos == sup_neg:
        return -1
    else:
        return 0
    
def alpha_weak_support(pos, neg, C=0.):
    sup_pos = np.sum(pos[0][pos[1] <= len(neg[0]) * C]) / len(pos[0])**2
    sup_neg = np.sum(neg[0][neg[1] <= len(pos[0]) * C]) / len(neg[0])**2
    
    if sup_pos > sup_neg:
        return 1
    elif sup_pos == sup_neg:
        return -1
    else:
        return 0

def ratio_support(pos, neg, C=1.):
    sup_pos = len(neg[0])*np.sum( pos[0][pos[0]/len(pos[0]) >= C * pos[1]/len(neg[0])] )
    cont_pos = len(pos[0])*np.sum( pos[1][pos[0]/len(pos[0]) >= C * pos[1]/len(neg[0])] ) + 1e-6
    sup_neg = len(pos[0])*np.sum( neg[0][neg[0]/len(neg[0]) >= C * neg[1]/len(pos[0])] )
    cont_neg = len(neg[0])*np.sum( neg[1][neg[0]/len(neg[0]) >= C * neg[1]/len(pos[0])] ) + 1e-6
    
    if sup_pos/cont_pos > sup_neg/cont_neg:
        return 1
    elif sup_pos/cont_pos == sup_neg/cont_neg:
        return -1
    else:
        return 0
    