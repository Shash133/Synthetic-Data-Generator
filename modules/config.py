class Config:
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    NUM_FEATURES = 11  # All features including quality
    TARGET_COL = 'quality'
    CAT_FEATURES = 1  # Quality as categorical
    CAT_DIMS = [6]    # Quality values 3-8 (6 categories)
    LATENT_DIM = 128
    HIDDEN_DIM = 256
    BATCH_SIZE = 512
    EPOCHS = 1000
    CRITIC_ITER = 5
    LAMBDA_GP = 15
    SAMPLE_SIZE = 2000
    LR = 3e-5
    MIN_MAX_RANGE = (0, 1)
