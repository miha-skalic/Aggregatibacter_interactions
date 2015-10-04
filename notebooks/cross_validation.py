import numpy as np
from sklearn.cross_validation import KFold
from sklearn import metrics


def matrix_cv_setup(my_matrix, n_fold=10, alternative=0, by_row=False):
    """
    Generator that prepairs matrix for cross validation.
    :return: matrix with replaced values, selected matrix elements and their values
    """
    dims = my_matrix.shape
    
    if by_row:
        for _, sel_rows in KFold(my_matrix.shape[0], n_folds=n_fold, shuffle=True, random_state=0):
            my_newmat = np.copy(my_matrix)
            my_newmat[sel_rows, :] = alternative
            yield my_newmat, sel_rows
    else:
        
        for _, elements in KFold(my_matrix.size, n_folds=n_fold, shuffle=True, random_state=0):
            my_flat = my_matrix.flatten()
            my_flat[elements] = alternative
            yield my_flat.reshape(dims), elements


def auc_value(orig_matrix, reconst_matrix, elements, treshold=2.5, by_row=False):
    """
    Calcucales AUC score of reconstruction matrix on given elements
    :return: AUC value
    """
    if by_row:
        true_vals = (orig_matrix[elements, :] > treshold).flatten()
        pred_vals = (reconst_matrix[elements, :]).flatten()
        fpr, tpr, thresholds = metrics.roc_curve(true_vals, pred_vals)
    else:
        true_vals = orig_matrix.flatten()[elements] > treshold
        pred_vals = reconst_matrix.flatten()[elements]
        fpr, tpr, thresholds = metrics.roc_curve(true_vals, pred_vals)
    return metrics.auc(fpr, tpr)


def avr_res(orig_matrix, reconst_matrix, elements, by_row=False):
    """
    Calculates average deviation for prediction
    :return:
    """
    if by_row:
        return np.absolute(orig_matrix[elements, :] - reconst_matrix[elements, :]).sum() / (elements.size * orig_matrix.shape[1]) 
    else:
        return np.absolute(orig_matrix.flatten()[elements] - reconst_matrix.flatten()[elements]).sum() / elements.size

