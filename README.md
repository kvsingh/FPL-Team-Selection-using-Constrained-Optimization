# Approach

Selecting a team, with budget and position constaints, while maximizing a player performance-based metric, can be modelled as a constaint maximization problem

## Metric to Optimize

The custom metric to optimize is as follows:

$$ \text{ $metric = form * w1 + totalpoints * w2$} $$

learned using grid search on the current form and cumulative points through the season for a player.

## Grid Search

The search finds the values of w1 and w2 that maximizes overlap between top k players ranked on the metric and the average points earned in the next X games.

$$ \max_{w_1, w_2} \text{Overlap}_k \Big( \text{Rank}\big( \text{Metric}(w_1, w_2) \big) \, \text{Rank}\big( \text{AvgPoints}_{\text{next } x \text{ games}} \big) \Big) $$

That is, we use the metric as a monotonic proxy for projected points in the future for a player

## Data Used

Last 5 seasons' gameweek level FPL data is used to learn the weights, obtained from [vaastav's FPL dataset](https://github.com/vaastav/Fantasy-Premier-League/tree/master)

For prediction, the current season's player data is obtained from [Calvin Rostanto's kaggle dataset](https://www.kaggle.com/datasets/calvinrostanto/fantasy-premier-league-2025-2026)

## Application

This can be used to apply an FPL wildcard/free hit or for weekly transfers.

# Original Notebook Link 
https://www.kaggle.com/code/teamaker/team-selection-using-constraint-maximization
