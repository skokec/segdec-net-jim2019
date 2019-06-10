import os, sys
import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

def calc_confusion_mat(D, Y):
    FP = (D != Y) & (Y.astype(np.bool) == False)
    FN = (D != Y) & (Y.astype(np.bool) == True)
    TN = (D == Y) & (Y.astype(np.bool) == False)
    TP = (D == Y) & (Y.astype(np.bool) == True)

    return FP, FN, TN, TP

def get_performance_eval(P,Y):
    precision_, recall_, thresholds = precision_recall_curve(Y.astype(np.int32), P)
    FPR, TPR, _ = roc_curve(Y.astype(np.int32), P)
    AUC = auc(FPR, TPR)
    AP = average_precision_score(Y.astype(np.int32), P)

    f_measure = 2 * (precision_ * recall_) / (precision_ + recall_ + 0.0000000001)

    best_idx = np.argmax(f_measure)

    f_measure[best_idx]
    thr = thresholds[best_idx]

    FP, FN, TN, TP = calc_confusion_mat(P >= thr, Y)

    FP_, FN_, TN_, TP_ = calc_confusion_mat(P >= thresholds[np.where(recall_ >= 1)], Y)

    F_measure = (2 * TP.sum()) / float(2 * TP.sum() + FP.sum() + FN.sum())

    return TP, FP, FN, TN, TP_, FP_, FN_, TN_, F_measure, AUC, AP

def evaluate_decision(data_dir, folds_list = [0,1,2]):

    PD_decision_net = None

    num_params_list = []

    for f in folds_list:
        if f >= 0:
            fold_name = 'fold_%d' % f
        else:
            fold_name = ''

        sample_outcomes = np.load(os.path.join(data_dir, fold_name, 'test', 'results_decision_net.npy'))

        if len(sample_outcomes) > 0:
            PD_decision_net = np.concatenate((PD_decision_net, sample_outcomes)) if PD_decision_net is not None else sample_outcomes

        num_params_filename = os.path.join(data_dir, fold_name, 'test', 'decision_net_num_params.npy')
        if os.path.exists(num_params_filename):
            num_params_list.append(np.load(num_params_filename))

    results = None

    if PD_decision_net is not None:

        TP, FP, FN, TN, TP_, FP_, FN_, TN_, F_measure, AUC, AP = get_performance_eval(PD_decision_net[:,0], PD_decision_net[:,1])

        print "AP: %.03f, FP/FN: %d/%d, FP@FN=0: %d" % (AP, FP.sum(), FN.sum(), FP_.sum())

        results = {'TP': TP.sum(),
                   'FP': FP.sum(),
                   'FN': FN.sum(),
                   'TN': FN.sum(),
                   'FP@FN=0': FP_.sum(),
                   'f-measure': F_measure,
                   'AUC': AUC,
                   'AP': AP}

    return results


if __name__ == "__main__":

    evaluate_decision(sys.argv[1], folds_list = [0,1,2])

