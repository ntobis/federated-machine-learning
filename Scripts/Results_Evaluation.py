import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, 'Results', 'Thesis')


def results_table(experiment_path, metric, view_by, subjects, pivot):
    sessions = True if view_by in 'sessions' else False

    # Sort according to name not timestamp
    df_concat = pd.DataFrame()
    list_dir = os.listdir(experiment_path)
    f_paths = [file.split("PAIN_")[1] for file in list_dir if 'PAIN' in file]
    list_dir = [file for file in list_dir if 'PAIN' in file]
    folders = [x for _, x in sorted(zip(f_paths, list_dir))]

    for file in folders:
        if os.path.isfile(os.path.join(experiment_path, file)):
            # Read in one file per experiment
            df = pd.read_csv(os.path.join(experiment_path, file))
            df = df.rename(columns={'Unnamed: 0': 'Epoch'})

            # Compute Means per subject
            df_mean = weighted_mean_SD(df, subjects, metric, sessions, pivot)
            # noinspection PyTypeChecker
            df_mean['Experiment'] = file.split('PAIN_')[1].split('_TEST')[0]

            # Concatenate all experiments
            df_concat = pd.concat((df_concat, df_mean))
    df_concat.set_index('Experiment', inplace=True)

    # Sort rows according to experiment number
    df_concat['indexNumber'] = [int(i.split('-')[0]) for i in df_concat.index]
    df_concat = df_concat.sort_values(by='indexNumber')
    df_concat.drop('indexNumber', inplace=True, axis=1)

    # Calculate Regular Mean
    cols = [col for col in df_concat.columns.values if type(col) is int]
    df_concat['Mean'] = df_concat[cols].mean(axis=1)
    df_concat['SD'] = df_concat[cols].std(axis=1)
    return df_concat


def session_examples_total(df, subjects, metric):
    df_subjects = pd.DataFrame()
    for subject in subjects:
        df_subjects['subject_{}_{}'.format(subject, metric)] = df['subject_{}_true_positives'.format(subject)] + \
                                                               df['subject_{}_true_negatives'.format(subject)] + \
                                                               df['subject_{}_false_positives'.format(subject)] + \
                                                               df['subject_{}_false_negatives'.format(subject)]
    return df_subjects


def mask_df(df, metric, subjects, pivot):
    # Get column names for each subject of the format "subject_43_accuracy"
    columns = ['subject_{}_{}'.format(subject, metric) for subject in subjects]

    # Filter DataFrame for the columns specified
    df = df[columns].reset_index()

    # Change Zero-Based Session Indexing to 1-Based indexing
    df['index'] += 1

    # Rename Index Column to Session Column
    df = df.rename(columns={'index': 'Session'})

    # Rename columns to only include subject number, e.g. '43' or '59'
    df = df.rename(columns={col: int(col.split('_')[1].split('_')[0]) for col in df.columns if 'subject' in col})

    # 1. Reset index of pivot to have 1-dimensional index
    # 2. Filter Pivot table to only include same columns as df ['Session', 43, 48, 52, 59, ..., 120]
    # 3. Drop Session '0'
    # 4. Reset index to be zero-based
    pivot = pivot.reset_index()[df.columns].drop(0).reset_index(drop=True)

    # Return a df, where:
    # - Rows are sessions (starting with '1')
    # - Columns are ['Session', 43, 48, 52, 59, ..., 120]
    # - Values are "TOTAL FP/FN/TN/TP", or NULL, when no positive examples exist
    return df.where(pivot != '')


def weighted_mean_SD(df, subjects, metric, sessions, pivot):
    # Compute Sum of all TP, TN, FP, FN
    df_total = session_examples_total(df, subjects, metric)

    # Drop 'Session' column
    weights = mask_df(df_total, metric, subjects, pivot).drop('Session', axis=1)

    # Get column names for each subject of the format "subject_43_accuracy"
    columns = ['subject_{}_{}'.format(subject, metric) for subject in subjects]

    # Filter df down to those columns, e.g. for accuracy, the new df will have accuracy per subject, per session
    df_new = df[columns]

    # Change column names to [43, 48, 52, 59, ..., 120]
    df_new = df_new.rename(
        columns={col: int(col.split('_')[1].split('_')[0]) for col in df_new.columns if 'subject' in col})

    # If sessions, simply transpose the data frames
    if sessions:
        weights = weights.T
        df_new = df_new.T

    # Calculate average (e.g. accuracy) accross subjects OR sessions
    # Weighted by number of observations:
    # E.g. ACC [1.0, 0.5] & Obs. [10, 5] == weighted AVG 12.5 / 15 == 0.833
    # Returns np.array of weighted AVG for all subjects OR sessions
    weighted_avg = (df_new * weights).sum() / weights.sum()

    if sessions:
        weighted_avg.index += 1

    # Compute Mean of subjects OR sessions, weighted by number of observations across subject OR session
    weighted_avg['Weighted Mean'] = (df_new * weights).sum().sum() / weights.sum().sum()

    # Compute standard deviation between subjects OR session, weighted by number of observations in that session
    variance = ((df_new - weighted_avg) ** 2 * weights).sum().sum() / weights.sum().sum()
    weighted_avg['Weighted SD'] = np.sqrt(variance)

    # Returns weighted mean performance for each subject OR session, and mean/std performance of entire model
    # DF columns: [43, 48, 52, 59, 64, 80, 92, 96, 107, 109, 115, 120, 'Weighted Mean', 'Weighted SD']
    return pd.DataFrame(weighted_avg).T


def compute_avg_df(metric, view_by, subjects, pivot, folders, path):
    res = {}
    for folder in folders:
        experiment_path = os.path.join(path, folder)
        res[folder] = results_table(experiment_path, metric, view_by, subjects, pivot)
    df_sum = pd.DataFrame()
    seeds = 0
    for key, val in res.items():
        if 'Seed' in key:
            df_sum = df_sum.add(val, fill_value=0)
            seeds += 1
    return df_sum / seeds


def prep_col(df):
    pd.options.display.float_format = '{:,.1f}'.format
    df_str = (df[['Weighted Mean', 'Weighted SD']] * 100).round(0).astype(int)
    df_str['Mean + STD'] = df_str['Weighted Mean'].astype(str) + ' ± ' + df_str['Weighted SD'].astype(str)
    return df_str[['Mean + STD']]


def rename_index(df, exp_names):
    df['indexNumber'] = [int(i.split('-')[0]) for i in df.index]
    df = df.sort_values(by='indexNumber')
    df.drop('indexNumber', inplace=True, axis=1)
    df.rename(index=exp_names, inplace=True)
    return df


def prepare_top_experiments(df, exp_names, top_exp):
    df = rename_index(df, exp_names)
    df = df[df.index.isin(top_exp)]
    df = (df[[col for col in df]] * 100)
    df = df.fillna('NA').drop(['Mean', 'SD'], axis=1)
    df = df.fillna('NA').drop(['Weighted Mean', 'Weighted SD'], axis=1)
    cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
    df[cols] = df[cols].round(0).astype(int)
    df['wt. Mean ± SD'] = df['Weighted Mean'].astype(str) + ' ± ' + df['Weighted SD'].astype(str)
    df = df.drop(['Weighted Mean', 'Weighted SD'], axis=1)
#     df['Mean ± SD'] = df['Mean'].astype(str) + ' ± ' + df['SD'].astype(str)
#     df = df.drop(['Mean', 'SD'], axis=1)
    return df.reset_index()


def compute_average_metrics(view_by, subjects, pivot, path):
    return_metrics = {}
    folders = [folder for folder in os.listdir(path) if 'Seed' in folder]
    return_metrics['accuracy'] = compute_avg_df('accuracy', view_by, subjects, pivot, folders, path)
    return_metrics['recall'] = compute_avg_df('recall', view_by, subjects, pivot, folders, path)
    return_metrics['precision'] = compute_avg_df('precision', view_by, subjects, pivot, folders, path)
    return_metrics['roc'] = compute_avg_df('auc', view_by, subjects, pivot, folders, path)
    return_metrics['TP'] = compute_avg_df('true_positives', view_by, subjects, pivot, folders, path)
    return_metrics['TN'] = compute_avg_df('true_negatives', view_by, subjects, pivot, folders, path)
    return_metrics['FP'] = compute_avg_df('false_positives', view_by, subjects, pivot, folders, path)
    return_metrics['FN'] = compute_avg_df('false_negatives', view_by, subjects, pivot, folders, path)
    return_metrics['MCC'] = mcc(return_metrics)
    return_metrics['pr'] = compute_avg_df('pr', view_by, subjects, pivot, folders, path)
    return_metrics['f1_score'] = 2 * return_metrics['recall'] * \
                                 return_metrics['precision'] / (return_metrics['recall'] + return_metrics['precision'])
    return return_metrics


def mcc(return_metrics):
    return (return_metrics['TP'] * return_metrics['TN'] - return_metrics['FP'] * return_metrics['FN']) / np.sqrt(
        (return_metrics['TP'] + return_metrics['FP']) * (return_metrics['TP'] + return_metrics['FN']) * (
                    return_metrics['TN'] + return_metrics['FP']) * (return_metrics['TN'] + return_metrics['FN']))


def generate_overview_table(return_metrics, exp_names):
    # Concatenate overview table and rename index
    overview_table = pd.concat((prep_col(return_metrics['accuracy']).rename(columns={'Mean + STD': 'ACC'}),
                                prep_col(return_metrics['pr']).rename(columns={'Mean + STD': 'PR-AUC'}),
                                prep_col(return_metrics['f1_score']).rename(columns={'Mean + STD': 'F1'})), axis=1)
    overview_table = rename_index(overview_table, exp_names)

    # Rename Columns
    overview_table.reset_index(inplace=True)
    cols = np.array(
        [np.array(['', 'Weighted AVG + STD', 'Weighted AVG + STD', 'Weighted AVG + STD']), overview_table.columns])
    tuples = list(zip(*cols))
    return pd.DataFrame(overview_table.values, columns=pd.MultiIndex.from_tuples(tuples))


def concat_validation_metrics(experiment_folder):
    if not os.path.isdir(os.path.join(experiment_folder, 'Plotting')):
        os.mkdir(os.path.join(experiment_folder, 'Plotting'))
    for folder in sorted(os.listdir(experiment_folder)):
        if folder != 'Plotting' and os.path.isdir(os.path.join(experiment_folder, folder)):
            folder_path = os.path.join(experiment_folder, folder)
            df_concat = pd.DataFrame()
            for file in sorted(os.listdir(folder_path)):
                if file != '.DS_Store':
                    file_path = os.path.join(experiment_folder, folder_path, file)
                    df = pd.read_csv(file_path)
                    df['Session'] = os.path.splitext(file.split('shard-')[1])[0]
                    df_concat = pd.concat((df_concat, df), ignore_index=True, sort=False)
            df_concat = df_concat.rename(columns={'Unnamed: 0': 'Epoch'})
            df_concat.to_excel(os.path.join(experiment_folder, 'Plotting', folder + '.xlsx'))


def create_detailed_metric_table(results_path, subjects, metric):
    folder_paths = [os.path.join(results_path, folder) for folder in os.listdir(results_path) if 'Seed' in folder]

    count = 0
    for folder_path in folder_paths:

        # Sort according to name not timestamp
        list_dir = os.listdir(folder_path)
        f_paths = [file.split("PAIN_")[1] for file in list_dir if 'PAIN' in file]
        list_dir = [file for file in list_dir if 'PAIN' in file]
        folders = [x for _, x in sorted(zip(f_paths, list_dir))]

        df_concat = pd.DataFrame()
        for file in folders:
            if os.path.isfile(os.path.join(folder_path, file)):

                # Read in df
                df = pd.read_csv(os.path.join(folder_path, file))

                # Create table
                columns = ['subject_{}_{}'.format(subject, metric) for subject in subjects]
                df_new = df[columns]
                df_new = df_new.rename(columns={col: int(col.split('_')[1].split('_')[0]) for col in df_new.columns if 'subject' in col})
                df_new['Experiment'] = file.split('PAIN_')[1].split('_TEST')[0]
                df_concat = pd.concat((df_concat, df_new))

        if count > 0:
            df_add[subjects] = df_add[subjects].add(df_concat[subjects])
        else:
            df_add = df_concat.copy()

        count += 1
    return df_add[subjects].divide(count)
