# Recommender Systems: Books

A book recommendation system built with the [Surprise](https://surpriselib.com/) library on user ratings for 1,000 books (500 users, about 24,400 ratings, scale 1-5).

## What's inside

- `book_recommendation_lab.ipynb`: the full notebook, with saved outputs
- `book_ratings.csv`: the ratings data

## Workflow

1. Load the ratings data into Surprise's format (75/25 train/test split)
2. Compare SVD(Singular Value Decomposition), KNN (user- and item-based), KNNWithMeans, and NMF using 5-fold cross-validation
3. Tune SVD with grid search over factors, epochs, learning rate, and regularization
4. Evaluate the tuned model on the test set
5. Generate top-N book recommendations for any user

## Results

| Model | RMSE |
|---|---|
| SVD (untuned, CV) | 1.650 |
| SVD (tuned, CV) | 1.533 |
| SVD (tuned, test set) | **1.478** |

Tuning improved SVD's cross-validated RMSE by about 7%.

## Tools

Python, pandas, NumPy, scikit-surprise, matplotlib, seaborn

## Running it

```bash
pip install scikit-surprise
```

Open the notebook in Jupyter and run all cells, with `book_ratings.csv` in the same folder.