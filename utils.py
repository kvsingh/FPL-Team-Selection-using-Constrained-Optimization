import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score, precision_score

def get_historical_data(filename, num_gws_before_wildcard, num_future_gws, fixtures_filename='fixtures.csv'):

    gw_level_points = pd.read_csv(filename)
    fixtures = pd.read_csv(fixtures_filename)

    fixture_difficulties = fixtures[
        ['id', 'team_h_difficulty', 'team_a_difficulty']
    ].rename(columns={'id': 'fixture'})

    gw_level_points = gw_level_points.merge(
        fixture_difficulties,
        on='fixture',
        how='left'
    )
    
    gw_level_points['fixture_difficulty'] = np.where(
        gw_level_points['was_home'],
        gw_level_points['team_h_difficulty'],
        gw_level_points['team_a_difficulty']
    )

    # import pdb
    # pdb.set_trace()
    #print (gw_level_points.columns)
    
    cols_to_keep = ['name', 'total_points', 'GW', 'fixture_difficulty', 'xP', 'was_home', 'value', 'expected_goal_involvements']
    df_gw = gw_level_points[cols_to_keep]

    # num_gws_before_wildcard = 12
    # num_future_gws = 12

    df_gw = df_gw[df_gw['GW'] <= (num_gws_before_wildcard + num_future_gws)]

    df_gw['total_points_so_far'] = df_gw.groupby('name')['total_points'].cumsum()

    for col in ['total_points', 'xP', 'expected_goal_involvements']:
        df_gw[col + '_last6'] = (df_gw.groupby('name')[col]
        .transform(lambda x: x.rolling(6, min_periods=6).sum()/6.0)
        )

    df_gw.drop(columns=['xP', 'expected_goal_involvements'], inplace=True)
    df_gw.rename(columns={'total_points_last6': 'form', 'xP_last6': 'xP', 'expected_goal_involvements_last6': 'xGI'}, inplace=True)

    #Get a single dataframe containing form, total_points for each player and average points in remaining games
    future = df_gw[(df_gw["GW"] >= num_gws_before_wildcard+1) & (df_gw["GW"] <= num_gws_before_wildcard+num_future_gws)]
    df_players = df_gw[df_gw["GW"] == num_gws_before_wildcard].copy()
    future = future.sort_values(['name', 'GW'], kind='stable')
    #next_12_fixtures = future.groupby('name', sort=False).head(12)
    next_points = future.groupby("name")["total_points"].sum()
    
    next_fixture_difficulty = future.groupby(
        'name'
    )['fixture_difficulty'].mean()
    next_home_ratio = future.groupby('name')['was_home'].mean()
    
    df_players["next_points"] = df_players["name"].map(next_points)
    df_players["next_fixture_difficulty"] = df_players["name"].map(
        next_fixture_difficulty
    )
    df_players["next_home_ratio"] = df_players["name"].map(next_home_ratio)
    #df_players["value"] = np.log(df_players["value"])

    return df_players

def convert_to_rank(y):
    # Convert the target variable to ranks for ranking models
    y_values = np.asarray(y).reshape(-1)
    label_values = np.sort(np.unique(y_values))
    y_rank = np.searchsorted(label_values, y_values).astype(int)
    return y_rank

def lgb_ranker(X_train, X_val, y_train, y_val, rank_feature_cols):

    #y_train and y_val are 'next_points'

    X_train_rank = X_train[rank_feature_cols]
    X_val_rank = X_val[rank_feature_cols]
    
    y_train_rank = convert_to_rank(y_train)
    y_val_rank = convert_to_rank(y_val)

    # print("raw train unique:", len(np.unique(y_train_values)))
    # print("raw val unique:", len(np.unique(y_val_values)))
    print("train rank range:", y_train_rank.min(), y_train_rank.max())
    print("val rank range:", y_val_rank.min(), y_val_rank.max())
    print("train rank labels:", np.unique(y_train_rank))
    print("val rank labels:", np.unique(y_val_rank))
    # import pdb
    # pdb.set_trace()
    
    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        verbosity=-1,
        label_gain=list(range(max(y_train_rank.max() + 1, y_val_rank.max() + 1)))
    )

    ranker.fit(
        X_train_rank,
        y_train_rank,
        group=np.array([[X_train_rank.shape[0]]], dtype=np.int32),
        eval_set=[(X_val_rank, y_val_rank)],
        eval_group=[[X_val_rank.shape[0]]],
        eval_at=[10],
        callbacks=[
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(0)
        ]
    )

    val_predictions = ranker.predict(X_val_rank)
    val_ndcg_at_10 = ndcg_score(
        [y_val_rank],
        [val_predictions],
        k=10
    )

    best_iteration = ranker.best_iteration_ or ranker.n_estimators
    X_full_rank = pd.concat([X_train_rank, X_val_rank])
    y_full_rank = convert_to_rank(pd.concat([y_train, y_val]))
    ranker.set_params(
        n_estimators=best_iteration,
        label_gain=list(range(y_full_rank.max() + 1))
    )
    ranker.fit(
        X_full_rank,
        y_full_rank,
        group=np.array([[X_full_rank.shape[0]]], dtype=np.int32)
    )

    #print(f'Validation NDCG@10: {val_ndcg_at_10:.4f}')
    return val_ndcg_at_10, ranker

def preprocessing(X_train, X_val, features):
    means = {}
    stds = {}
    for col in features:
        means[col] = X_train[col].mean()
        stds[col] = X_train[col].std()
        X_train[col + '_norm'] = (X_train[col] - means[col])/stds[col]
        X_val[col + '_norm'] = (X_val[col] - means[col])/stds[col]
    
    X_train['metric'] = X_train['total_points_so_far_norm'] * X_train['form_norm'].max() + X_train['form_norm'] * X_train['total_points_so_far_norm'].max()
    X_val['metric'] = X_val['total_points_so_far_norm'] * X_train['form_norm'].max() + X_val['form_norm'] * X_train['total_points_so_far_norm'].max()
    
    for df_foo in [X_train, X_val]:
        df_foo['rank_metric_base'] = df_foo['metric'].rank(ascending=False).astype(int)

    return X_train, X_val

#w1 and w2 : between 0 and 1
def optimal_weights(X_train, X_val, y_train, y_val):

    best = (-1, None)
    K = 20
    
    X_train['actual_rank'] = y_train['next_points'].rank(ascending=False).astype(int)
    true_topK = set(X_train.nsmallest(K, "actual_rank").index)
        
    for w1 in np.linspace(0, 1, 40):
        if w1 < 0.1:
            continue
        w2 = 1 - w1
        score = w1 * X_train['form_norm'] + w2 * X_train['total_points_so_far_norm']
        X_train["score"] = score
        X_train['rank_metric'] = X_train['score'].rank(ascending=False).astype(int)
        pred_topK = set(X_train.nsmallest(K, "rank_metric").index)
        
        acc = len(true_topK & pred_topK) / K
        
        if acc > best[0]:
            best = (acc, (w1, w2))
        
    print (f"Best w1 : {best[1][0]:.2f} and w2 : {best[1][1]:.2f}")
    
    pred_topK_base = set(X_train.nsmallest(K, "rank_metric_base").index)
    train_acc_base = len(true_topK & pred_topK_base) / K        
    
    X_val['actual_rank'] = y_val['next_points'].rank(ascending=False).astype(int)
    true_topK = set(X_val.nsmallest(K, "actual_rank").index)
        
    w1 = best[1][0]
    w2 = best[1][1]
    score = w1 * X_val['form_norm'] + w2 * X_val['total_points_so_far_norm']
    
    X_val["score"] = score
    X_val['rank_metric'] = X_val['score'].rank(ascending=False).astype(int)
    pred_topK = set(X_val.nsmallest(K, "rank_metric").index)
    acc = len(true_topK & pred_topK) / K
    
    pred_topK_base = set(X_val.nsmallest(K, "rank_metric_base").index)
    acc_base = len(true_topK & pred_topK_base) / K  

    #return (acc, best[0])

    y_val_rank = convert_to_rank(y_val)
    val_predictions = X_val['rank_metric'].values

    return ndcg_score(
            [y_val_rank],
            [val_predictions],
            k=10
        )
    
