from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
from classifiers import *
from preprocessing import *


def clustering_grid_search(clustering_df2, clusters_path, users_columns, users_dict, classifier_columns,
                           users_profiles_path, one_hot_col_cat, one_hot_col_cat_list,
                           target_col='Bought Tag'):
    if target_col in list(clustering_df2.columns.values):
        clustering_df2 = clustering_df2.drop(target_col, axis=1)

    x_train, x_test = train_test_split(clustering_df2,  train_size=0.8, random_state=2021)
    kmeans_params = {'n_clusters': [2, 5, 7]}
    dbscan_params = {'eps': [0.001, 0.002], 'min_samples': [2, 5, 20]}

    dbscan_gs_params = [{'eps': e, 'min_samples': m} for e in dbscan_params['eps'] for m in
                        dbscan_params['min_samples']]

    k_means_results = []
    k_means_results_scores = []
    for k in kmeans_params['n_clusters']:
        kmeans = KMeans(n_clusters=k, n_init=15, max_iter=300, tol=1e-04, random_state=2021).fit(x_train)
        km_labels = kmeans.labels_
        k_means_results_scores.append(kmeans.score(x_test))
        if len(set(km_labels)) > 1:
            k_means_results.append(metrics.silhouette_score(x_train, km_labels))
        else:
            k_means_results.append(0)

    best_k = kmeans_params['n_clusters'][int(np.argmax(k_means_results_scores))]
    best_kmeans = KMeans(n_clusters=best_k, n_init=15, max_iter=300, tol=1e-04, random_state=2021).fit(clustering_df2)

    dbscan_results = []

    for para in dbscan_gs_params:
        db_cl_labels = DBSCAN(**para).fit_predict(x_train)
        if len(set(db_cl_labels)) > 1:

            dbscan_results.append(metrics.silhouette_score(x_train, db_cl_labels, metric='euclidean'))
        else:
            dbscan_results.append(0)

    best_db_param = dbscan_gs_params[int(np.argmax(dbscan_results))]
    best_db = DBSCAN(**best_db_param).fit(x_test)

    best_clusters = [best_kmeans, best_db]

    test_results = []
    for clu in best_clusters:
        y_pred = clu.fit_predict(x_test)
        if len(set(y_pred)) > 1:
            score = metrics.silhouette_score(x_test, y_pred)
        else:
            score = 0
        test_results.append(score)

    best_clu_id = np.argmax(test_results)
    best_cluster = best_clusters[int(best_clu_id)]

    file_name = os.path.join(clusters_path, "cluster.pkl")
    col_file_name = os.path.join(clusters_path, "cluster.txt")

    with open(file_name, "wb") as open_file:
        pickle.dump(best_cluster, open_file)

    cluster_col_list = list(x_test.columns.values)
    with open(col_file_name, 'w') as f:
        for item in cluster_col_list:
            f.write("%s\n" % item)

    user_labels = best_cluster.fit_predict(clustering_df2)

    for lab in list(set(user_labels)):
        this_users = list(np.where(user_labels == lab)[0])
        this_users_profile = [users_dict[i] for i in this_users]

        this_df = pd.DataFrame(columns=users_columns)
        for user in this_users_profile:
            user_add = os.path.join(users_profiles_path, user + '.csv')
            this_user_data = pd.read_csv(user_add)
            this_df = this_df.append(this_user_data, ignore_index=True)

        user_data_new, new_cols = preprocessing_user_profile(this_df, one_hot_col_cat, one_hot_col_cat_list)

        all_class_cols = classifier_columns + new_cols

        classifier_df = user_data_new[user_data_new.columns.intersection(all_class_cols)]

        columns = list(classifier_df.columns.values)
        classifier_np = classifier_df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        classifier_np_scaled = min_max_scaler.fit_transform(classifier_np)
        classifier_df = pd.DataFrame(data=classifier_np_scaled, columns=columns, index=range(len(classifier_df)))

        nunique = classifier_df.nunique()
        cols_to_drop = nunique[nunique == 1].index
        classifier_df2 = classifier_df.drop(cols_to_drop, axis=1)

        classifiers_grid_search(classifier_df2, clusters_path, 'cluster_' + str(lab), target_col='Bought Tag')
