## Summary
- Implement RBM recommender with reproducible data pipeline
- Add baselines, metrics, and CLI training/inference
- Generate artifacts for evaluation and reporting

## Test Plan
- `python src/train.py --data_dir data/movielens-20m --max_users 1000 --max_items 2000 --epochs 2`
- `python src/infer.py --data_dir data/movielens-20m --user_id 1 --topk 10`
