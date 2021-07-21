K = 3   # how many patches extracted from one patient

raw_data_dir = 'data/raw/LITS'
processed_dir = 'data/processed'
patches_dir = 'data/patches'
png_dir = 'data/png'

patch_size = [32, 64, 64]
batch_size = 4
learning_rate = 0.01
learning_rate_decay = [30, 70]
Epoch = 100
alpha = 0.5
