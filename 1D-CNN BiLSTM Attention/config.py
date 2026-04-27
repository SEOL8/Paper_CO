CONFIG = {
    # Data
    'data_dir':   '.',
    'save_dir':   './results',
    'max_len':    2500,
    'batch_size': 64,

    # Model
    'hidden_size': 256,
    'lstm_layers': 3,
    'dropout':     0.3,

    # Training
    'epochs':                  200,
    'early_stopping_patience': 30,
    'learning_rate':           8e-4,
    'weight_decay':            1e-3,
    'grad_clip_norm':          0.3,

    # Scheduler (ReduceLROnPlateau)
    'scheduler_factor':   0.8,
    'scheduler_patience': 7,
    'scheduler_min_lr':   1e-5,
}
