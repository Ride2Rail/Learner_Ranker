import configparser
from tasks import *
import warnings
warnings.filterwarnings('ignore')
input_file_path = 'input_file.txt'


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read(input_file_path)

    users_columns = [i.strip() for i in config['columns']['users_columns'].split(',')]
    one_hot_col_cat = [i.strip() for i in config['columns']['one_hot_categorical_columns'].split(',')]
    one_hot_col_cat_list = [i.strip() for i in config['columns']['one_hot_categorical_list_columns'].split(',')]
    classifier_columns = [i.strip() for i in config['columns']['classifier_columns'].split(',')]
    clustering_columns = [i.strip() for i in config['columns']['clustering_columns'].split(',')]
    target_column = config['columns']['target_column']

    users_profiles_path = config['path']['user_profile']
    if not os.path.exists(users_profiles_path):
        os.mkdir(users_profiles_path)

    users_classifier_path = config['path']['users_classifier']
    if not os.path.exists(users_classifier_path):
        os.mkdir(users_classifier_path)

    clusters_path = config['path']['clusters_path']
    if not os.path.exists(clusters_path):
        os.mkdir(clusters_path)

    selected_task = int(config['task']['selected_task'])

    if selected_task == 1:
        users_classifying = [i.strip() + '.csv' for i in config['task']['users_for_classification'].split(',')]

        make_classifier_for_user(users_classifying, classifier_columns, users_profiles_path, target_column,
                                 one_hot_col_cat, one_hot_col_cat_list, users_classifier_path)

    elif selected_task == 2:
        cluster_users(users_columns, users_profiles_path, clustering_columns, one_hot_col_cat, one_hot_col_cat_list,
                      classifier_columns, clusters_path)

    elif selected_task == 3:
        new_users_clustering = [i.strip() + '.csv' for i in config['task']['cluster_new_user'].split(',')]
        new_user_cluster_pred(new_users_clustering, clusters_path, users_columns, clustering_columns,
                              users_profiles_path, one_hot_col_cat, one_hot_col_cat_list, users_classifier_path)
    elif selected_task == 4:
        rank_travel_offers = [i.strip() for i in config['task']['rank_travel_offers'].split(',')]
        user_for_rank = rank_travel_offers[0]
        travel_offers = rank_travel_offers[1:]
        sorted_travel_offers = sort_offers(user_for_rank, travel_offers, users_classifier_path, one_hot_col_cat,
                                           one_hot_col_cat_list, classifier_columns)
