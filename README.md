# Approach

Selecting a team, with budget and position constraints, while maximizing a player performance-based metric, can be modeled as a constraint maximization problem

## Metric to Optimize

The metric that we use for constraint maximization comes from the score output of a ranking model

## Ranking model

The ranking model is trained on the previous season's data. It utilizes the following features at an intermediate point in the season and uses the points scored by a player in the next N games as the y-label.

* Fixture difficulty in next N games
* Price
* Form
* Total Points so far

## Data Used

2025-26 gameweek-level FPL data is used to learn the weights, obtained from [vaastav's FPL dataset](https://github.com/vaastav/Fantasy-Premier-League/tree/master)

For prediction, the current season's player data is obtained from [Calvin Rostanto's Kaggle dataset](https://www.kaggle.com/datasets/calvinrostanto/fantasy-premier-league-2025-2026)

## Application

This can be used to apply an FPL wildcard/free hit or for weekly transfers.

# Original Notebook Link 
https://www.kaggle.com/code/teamaker/team-selection-using-constraint-maximization
