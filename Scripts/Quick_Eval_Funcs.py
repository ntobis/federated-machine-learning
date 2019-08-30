import os
import time

import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Recall, \
    Precision, AUC

from Scripts import Data_Loader_Functions as dL, Model_Architectures as mA, Print_Functions as pF
from Scripts.Experiments import GROUP_2_PATH, find_newest_model_path, evaluate_session, RESULTS, \
    baseline_model_evaluation


def quick_model_evaluation_runner(dataset, experiment, df, optimizer, loss, metrics, f_path):
    df_history = pd.DataFrame()
    df_testing = dL.create_pain_df(GROUP_2_PATH, pain_gap=())

    for session in df_testing['Session'].unique():
        if session > 0:
            df_test = df[df['Shard'] == session - 1]
            if len(df_test) > 0:
                print('Loading:', df_test['paths'].iloc[0])
                model = tf.keras.models.load_model(find_newest_model_path(f_path, df_test['paths'].iloc[0]))
            else:
                print('Building')
                model = mA.build_model((215, 215, 1), 'CNN')
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            pF.print_session(session)
            df_history = evaluate_session(df_history, df_testing, model, 'CNN', session)
            del model

    # Save history to CSV
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + "_{}_{}.csv".format(dataset, experiment + "_TEST")
    df_history.to_csv(os.path.join(RESULTS, f_name))


def quick_model_evaluation(f_path):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = ['accuracy', TruePositives(), TrueNegatives(),
               FalsePositives(), FalseNegatives(), Recall(), Precision(), AUC(curve='ROC', name='auc'),
               AUC(curve='PR', name='pr')]

    paths = sorted(os.listdir(f_path))
    models = [file.split('_') for file in paths]
    df = pd.DataFrame(models, columns=['Date', 'Pain', 'Experiment', 'Seed', 'Shard'])
    df['paths'] = paths
    df['Shard'] = df['Shard'].apply(lambda x: x.split('-')[1].split('.')[0]).astype(int)

    df_res = pd.DataFrame([file.split('_') for file in sorted(os.listdir(RESULTS)) if '.csv' in file],
                          columns=['Date', 'Pain', 'Experiment', 'Seed', 'TEST'])
    df_res['Exp_Seed'] = df_res['Experiment'] + df_res['Seed']
    for seed, df_seed in df.groupby('Seed'):
        for experiment, df_experiment in df_seed.groupby('Experiment'):
            if experiment + seed in df_res['Exp_Seed'].values:
                pass
            else:
                print('Seed:', seed, 'Experiment:', experiment)

                quick_model_evaluation_runner(dataset="PAIN",
                                              experiment=experiment + '_' + str(seed),
                                              df=df_experiment,
                                              optimizer=optimizer,
                                              loss=loss,
                                              metrics=metrics,
                                              f_path=f_path
                                              )
            df_res = pd.DataFrame([file.split('_') for file in sorted(os.listdir(RESULTS)) if '.csv' in file],
                                  columns=['Date', 'Pain', 'Experiment', 'Seed', 'TEST'])
            df_res['Exp_Seed'] = df_res['Experiment'] + df_res['Seed']


def quick_baselines(f_path, learn_type):
    baselines = sorted([file for file in os.listdir(f_path) if '0.00' in file])

    df = pd.DataFrame([file.split('_') for file in baselines], columns=['Date', 'Pain', 'Experiment', 'Seed', 'Shard'])
    df['baseline'] = baselines

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = ['accuracy', TruePositives(), TrueNegatives(),
               FalsePositives(), FalseNegatives(), Recall(), Precision(), AUC(curve='ROC', name='auc'),
               AUC(curve='PR', name='pr')]

    for baseline, df_baseline in df.groupby('baseline'):
        print('Load model', baseline, 'Seed:', str(df_baseline['Seed'].iloc[0]))
        baseline_model_evaluation(dataset="PAIN",
                                  experiment="0-sessions-Baseline-" + learn_type + "-pre-training" + "_" + str(
                                      df_baseline['Seed'].iloc[0]),
                                  model_path=find_newest_model_path(f_path, baseline),
                                  optimizer=optimizer,
                                  loss=loss,
                                  metrics=metrics,
                                  model_type='CNN'
                                  )
