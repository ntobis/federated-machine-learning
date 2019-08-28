import os

import pandas as pd
import numpy as np

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
    columns = ['subject_{}_{}'.format(subject, metric) for subject in subjects]
    df = df[columns].reset_index()
    df['index'] += 1
    df = df.rename(columns={'index': 'Session'})
    df = df.rename(columns={col: int(col.split('_')[1].split('_')[0]) for col in df.columns if 'subject' in col})
    pivot = pivot.reset_index()[df.columns].drop(0).reset_index(drop=True)
    return df.where(pivot != '')


def weighted_mean_SD(df, subjects, metric, sessions, pivot):
    weights = mask_df(session_examples_total(df, subjects, metric), metric, subjects, pivot).drop('Session', axis=1)
    columns = ['subject_{}_{}'.format(subject, metric) for subject in subjects]
    df_new = df[columns]
    df_new = df_new.rename(
        columns={col: int(col.split('_')[1].split('_')[0]) for col in df_new.columns if 'subject' in col})
    if sessions:
        weights = weights.T
        df_new = df_new.T

    weighted_avg = (df_new * weights).sum() / weights.sum()
    variance = ((df_new - weighted_avg) ** 2 * weights).sum().sum() / weights.sum().sum()
    std = np.sqrt(variance)
    if sessions:
        weighted_avg.index += 1
    weighted_avg['Weighted Mean'] = (df_new * weights).sum().sum() / weights.sum().sum()
    weighted_avg['Weighted SD'] = std
    return pd.DataFrame(weighted_avg).T


def compute_avg_df(metric, view_by, subjects, pivot, folders):
    res = {}
    for folder in folders:
        experiment_path = os.path.join(RESULTS, folder)
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
    df_str = (df[['Weighted Mean', 'Weighted SD']] * 100).astype(int)
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
    df = (df[[col for col in df if type(col) is int]] * 100)
    df = df.fillna('NA')
    cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
    df[cols] = df[cols].astype(int)
    return df.reset_index()


def compute_average_metrics(view_by, subjects, pivot):
    return_metrics = {}
    folders = [folder for folder in os.listdir(RESULTS) if 'Seed' in folder]
    return_metrics['accuracy'] = compute_avg_df('accuracy', view_by, subjects, pivot, folders)
    return_metrics['recall'] = compute_avg_df('recall', view_by, subjects, pivot, folders)
    return_metrics['precision'] = compute_avg_df('precision', view_by, subjects, pivot, folders)
    return_metrics['auc'] = compute_avg_df('auc', view_by, subjects, pivot, folders)
    return_metrics['f1_score'] = 2 * return_metrics['recall'] * \
                                 return_metrics['precision'] / (return_metrics['recall'] + return_metrics['precision'])
    return return_metrics


def generate_overview_table(return_metrics, exp_names):
    # Concatenate overview table and rename index
    overview_table = pd.concat((prep_col(return_metrics['accuracy']).rename(columns={'Mean + STD': 'ACC'}),
                                prep_col(return_metrics['auc']).rename(columns={'Mean + STD': 'AUC'}),
                                prep_col(return_metrics['f1_score']).rename(columns={'Mean + STD': 'F1'})), axis=1)
    overview_table = rename_index(overview_table, exp_names)

    # Rename Columns
    overview_table.reset_index(inplace=True)
    cols = np.array(
        [np.array(['', 'Weighted AVG + STD', 'Weighted AVG + STD', 'Weighted AVG + STD']), overview_table.columns])
    tuples = list(zip(*cols))
    return pd.DataFrame(overview_table.values, columns=pd.MultiIndex.from_tuples(tuples))