

import sys
import pandas as pd
from collections import defaultdict
import scipy.misc


def evaluate(groundtruth, parsedresult):
    """ Evaluation function to benchmark log parsing accuracy
    
    Arguments
    ---------
        groundtruth : str
            file path of groundtruth structured csv file 
        parsedresult : str
            file path of parsed structured csv file

    Returns
    -------
        f_measure : float
        accuracy : float
    """ 
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult)
    # Remove invalid groundtruth event Ids
    non_empty_log_ids = df_groundtruth[~df_groundtruth['EventId'].isnull()].index
    df_groundtruth = df_groundtruth.loc[non_empty_log_ids]
    df_parsedlog = df_parsedlog.loc[non_empty_log_ids]
    
    
    df_groundtruth_et= df_groundtruth['EventTemplate']
    df_groundtruth_list=df_groundtruth_et.values.tolist()
    df_groundtruth_list = list(map(str.split, df_groundtruth_list))

    df_parsedlog_et = df_parsedlog['EventTemplate']
    df_parsedlog_list = df_parsedlog_et.values.tolist()
    df_parsedlog_list = list(map(str.split, df_parsedlog_list))

    a = df_parsedlog_list
    b = df_groundtruth_list
    edit_dist=0
    # calculate Levenshtein distance between two arrays
    for c in range(0,len(df_groundtruth_et)):
        a= df_groundtruth_list[c]
        b = df_parsedlog_list[c]
        for i, k in zip(a, b):
            edit_dist+=(lev_dist(i,k))

    edit_dist=float(edit_dist)/len(df_groundtruth_et)
    total_log_token=0
    acc_pa=0.0
    correct_log=0
     # calculate parsing accuracy between two arrays
    for i in range(0,len(df_groundtruth_et)):
        length_list=len(df_groundtruth_list[i])
        a = df_groundtruth_list[i]
        b = df_parsedlog_list[i]
        if (a==b):
            correct_log += 1

    acc_pa = float (correct_log)/len(df_groundtruth_et)

    
    (precision, recall, f_measure, accuracy) = get_accuracy(df_groundtruth['EventId'], df_parsedlog['EventId'])
    print('Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Parsing_Accuracy: %.4f'%(precision, recall, f_measure, accuracy))
    return accuracy,  acc_pa, edit_dist

def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    """ Compute accuracy metrics between log parsing results and ground truth
    
    Arguments
    ---------
        series_groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        series_parsedlog : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.misc.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.misc.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0 # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.misc.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy

def lev_dist(a, b):
        '''
        This function will calculate the levenshtein distance between ground truth and generated output
        strings a and b

        params:
            a (String) : The first string you want to compare
            b (String) : The second string you want to compare

        returns:
            This function will return the distnace between string a and b.

        example:
            a = 'stamp'
            b = 'stomp'
            lev_dist(a,b)
            >> 1.0
        '''

        def min_dist(s1, s2):

            if s1 == len(a) or s2 == len(b):
                return len(a) - s1 + len(b) - s2

            # no change required
            if a[s1] == b[s2]:
                return min_dist(s1 + 1, s2 + 1)

            return 1 + min(
                min_dist(s1, s2 + 1),  # insert character
                min_dist(s1 + 1, s2),  # delete character
                min_dist(s1 + 1, s2 + 1),  # replace character
            )

        return min_dist(0, 0)






