# ======= PRE-IMPORT WORKAROUND =======
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# =====================================

# =======LIBRARIES=======
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset, Subset
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from torch.nn import L1Loss

# --- Imports for Evaluation ---
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =======GLOBAL VARIABLES=======

# ---  CONSTANTS WE *DO NOT* TUNE  ──────────────────────────────────────────
LR          = 1e-3         # learning-rate (fixed)
EPS         = 1e-11        # numerical guard (fixed)
FORWARD     = 5            # forecast horizon in minutes (fixed)
TRADING_DAYS_PER_YEAR = 252

# --- NEW: Annualization Factor Dictionary ---
# Based on a standard 390-minute trading day (e.g., 9:30-16:00)
TRADING_MINS_PER_DAY = 390

ANNUALIZATION_FACTORS = {
    # 5-min: 390/5 = 78 periods/day. Factor = sqrt(78 * 252)
    5: np.sqrt((TRADING_MINS_PER_DAY / 5) * TRADING_DAYS_PER_YEAR),  # ~140.20
    
    # 10-min: 390/10 = 39 periods/day. Factor = sqrt(39 * 252)
    10: np.sqrt((TRADING_MINS_PER_DAY / 10) * TRADING_DAYS_PER_YEAR), # ~99.14
    
    # 20-min: 390/20 = 19.5 periods/day. Factor = sqrt(19.5 * 252)
    20: np.sqrt((TRADING_MINS_PER_DAY / 20) * TRADING_DAYS_PER_YEAR), # ~70.09
    
    # 30-min: 390/30 = 13 periods/day. Factor = sqrt(13 * 252)
    30: np.sqrt((TRADING_MINS_PER_DAY / 30) * TRADING_DAYS_PER_YEAR)  # ~57.21
}

# ─── PARAMS WE *CAN* TUNE  ────────────────────────────────────────────────
DEFAULTS = dict(
    window_size       = 10,
    batch_size        = 512,
    max_epochs        = 100,
    patience          = 15, # EarlyStopping patience
    lr_patience       = 5,  # ReduceLROnPlateau patience
    kernel_size       = 3,
    num_blocks        = 4,
    num_layers        = 6,
    residual_channels = 16,
    dilation_channels = 32,
    skip_channels     = 64,
    end_channels      = 64,
)
# -----
# =======DATA PREP=======

def calculate_realized_volatility(log_returns, window=5):
    """Calculates rolling realized volatility (sqrt of sum of squares)."""
    return log_returns.rolling(window=window).apply(
        lambda x: np.sqrt(np.sum(x**2)), raw=True
    )

def create_sequences(data, target, sequence_length):
    """Creates sequences of past data (X) and corresponding targets (y)."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(target[i + sequence_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def prepare_rv_data(df, window_size: int = DEFAULTS['window_size']):
    """
    --- UPDATED ---
    Uses the static ANNUALIZATION_FACTORS dictionary
    to apply the correct factor to each horizon.
    """
    
    # 1. Use the 'return' column as our log_return
    df_market = df.copy()
    df_market['log_return'] = df_market['return'].fillna(0).astype(np.float32)
    
    # --- REMOVED: Dynamic Annualization Factor ---
    
    # 2. Calculate Realized Volatility (RV) and Annualize it
    grouped_returns = df_market.groupby(df_market.index.floor('D'))['log_return']
    
    # --- UPDATED: Apply correct factor from dictionary ---
    print("Applying annualization factors:")
    print(f"  5-min factor: {ANNUALIZATION_FACTORS[5]:.2f}")
    print(f" 10-min factor: {ANNUALIZATION_FACTORS[10]:.2f}")
    print(f" 20-min factor: {ANNUALIZATION_FACTORS[20]:.2f}")
    print(f" 30-min factor: {ANNUALIZATION_FACTORS[30]:.2f}")

    df_market['rv_5min'] = grouped_returns.apply(
        lambda x: calculate_realized_volatility(x, window=5)
    ).reset_index(level=0, drop=True) * ANNUALIZATION_FACTORS[5]
    
    df_market['rv_10min'] = grouped_returns.apply(
        lambda x: calculate_realized_volatility(x, window=10)
    ).reset_index(level=0, drop=True) * ANNUALIZATION_FACTORS[10]
    
    df_market['rv_20min'] = grouped_returns.apply(
        lambda x: calculate_realized_volatility(x, window=20)
    ).reset_index(level=0, drop=True) * ANNUALIZATION_FACTORS[20]
    
    df_market['rv_30min'] = grouped_returns.apply(
        lambda x: calculate_realized_volatility(x, window=30)
    ).reset_index(level=0, drop=True) * ANNUALIZATION_FACTORS[30]

    
    # 3. Create the TRAINING target variable: next 5-min RV
    df_market['target_rv_5min'] = df_market['rv_5min'].shift(-FORWARD)

    # 4. Clean data
    all_cols = ['rv_5min', 'rv_10min', 'rv_20min', 'rv_30min', 'target_rv_5min']
    df_processed = df_market[all_cols].dropna()
    
    print(f"Processed data has {len(df_processed)} rows after cleaning NaNs.")
    if len(df_processed) == 0:
        print("ERROR: No data left after dropna(). Check 'return' column in CSV.")
        return None, None, None

    # 5. Create sequences for the 5-MIN model
    features_data = df_processed['rv_5min'].values
    target_data = df_processed['target_rv_5min'].values
    
    features_data_log = np.log(features_data + EPS)
    target_data_log = np.log(target_data + EPS)
    
    X, y = create_sequences(features_data_log, target_data_log, sequence_length=window_size)
    
    X = np.expand_dims(X, axis=1) # -> (B, 1, T)
    
    print(f"Created {X.shape[0]} log-transformed 5-min sequences.")
    
    # 6. Get the non-sequenced, aligned data for plotting
    plot_data = {
        '5min_actual_target': target_data[window_size:],
        '10min_actual_cont': df_processed['rv_10min'].values[window_size:],
        '20min_actual_cont': df_processed['rv_20min'].values[window_size:],
        '30min_actual_cont': df_processed['rv_30min'].values[window_size:],
    }
    
    assert len(X) == len(plot_data['5min_actual_target']), "Alignment error in data prep"
    
    return X, y, plot_data

def load_set(csv_path: str, window_size = DEFAULTS['window_size']):
    """
    Loads the 1-min CSV, filters it, and calls the new RV prep function.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'")
        return None
        
    print(f"Loaded {len(df)} rows from {csv_path}...")
    
    if 'timestamp' not in df.columns or 'market_status' not in df.columns or 'return' not in df.columns:
        print("Error: CSV must contain 'timestamp', 'market_status', and 'return' columns.")
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()

    market_status_string = 'Trading' 
    df_market = df[df['market_status'] == market_status_string].copy()
    
    if df_market.empty:
        print(f"Error: No rows found with market_status == '{market_status_string}'.")
        return None
        
    print(f"Filtered down to {len(df_market)} '{market_status_string}' market rows...")

    X, y, plot_data = prepare_rv_data(df_market, window_size)
    
    if X is None:
        return None
        
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    
    full_dataset = TensorDataset(X, y)
    
    return full_dataset, plot_data


# =======DIALED CASUAL CONVOLUTION=======
# (This section is unchanged)
class GatedDCC(nn.Module):
    def __init__(self, in_ch, res_ch, dil_ch, skip_ch, k,dilation):
        super().__init__()
        pad = (dilation * (k-1)) // 2
        self.conv = nn.Conv1d(in_ch, dil_ch * 2, kernel_size=k,
                              dilation=dilation, padding=pad)
        self.residual = nn.Conv1d(dil_ch, res_ch, 1)
        self.skip     = nn.Conv1d(dil_ch, skip_ch, 1)

    def forward(self,x):
        h = self.conv(x)
        f,g = torch.chunk(h, 2, dim=1); z = torch.tanh(f) * torch.sigmoid(g)
        res = self.residual(z) + x; skip = self.skip(z)
        return res,skip

class DeepVol(nn.Module):
    def __init__(self, *, num_blocks:int, kernel_size:int, residual_channels:int,
                 dilation_channels:int, skip_channels:int, end_channels:int):
        super().__init__()
        self.input_proj = nn.Conv1d(1, residual_channels, 1)
        blocks, dilation = [], 1
        for _ in range(num_blocks):
            blocks.append(GatedDCC(residual_channels, residual_channels, 
                                   dilation_channels, skip_channels, 
                                   kernel_size, dilation))
            dilation *= 2 
        self.blocks = nn.ModuleList(blocks)
        self.post   = nn.Sequential(nn.ReLU(),
                                  nn.Conv1d(skip_channels, end_channels, 1),
                                  nn.ReLU(),
                                  nn.Conv1d(end_channels, 1, 1))
    def forward(self, x):
        x = self.input_proj(x); skip_sum = 0
        for blk in self.blocks: x, sk = blk(x); skip_sum = skip_sum + sk
        out = self.post(skip_sum)
        out = out.mean(dim=-1).squeeze(-1)
        return out

#=======TRAINING SCRIPT=======
# (This section is unchanged)
class VolModel(pl.LightningModule):
    def __init__(self, **h):
        super().__init__()
        self.save_hyperparameters()
        self.model = DeepVol(**{k: h[k] for k in (
            "num_blocks","kernel_size","residual_channels",
            "dilation_channels","skip_channels","end_channels")})
        self.loss  = L1Loss(reduction='mean')
        
    def forward(self, x):          
        return self.model(x)
    
    def _step(self,b):
        x, y = b
        pred_log_rv = self(x)
        return self.loss(pred_log_rv, y)
    
    def training_step(self,b, _):  
        l=self.loss(self(b[0]), b[1]); self.log("train_loss",l); return l
    
    def validation_step(self,b,_): 
        l=self.loss(self(b[0]), b[1]); self.log("val_loss",l,prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.1, 
            patience=self.hparams.lr_patience, verbose=True
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

# ======= EVALUATION FUNCTIONS (ADAPTED) =======
# (This section is unchanged)
def evaluate_model(model: pl.LightningModule,
                   test_loader: DataLoader,
                   device: torch.device,
                   eps: float = 1e-11):
    """
    Compute RMSE, MAE and QLIKE on our log-RV-to-log-RV model.
    """
    model.eval()
    model.to(device)
    all_preds_log_rv, all_targets_log_rv = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds_log_rv = model(x)
            all_preds_log_rv.append(preds_log_rv.cpu())
            all_targets_log_rv.append(y.cpu())
    
    y_pred_log_rv = torch.cat(all_preds_log_rv).numpy().astype(np.float64)
    y_true_log_rv = torch.cat(all_targets_log_rv).numpy().astype(np.float64)

    y_pred_rv = np.exp(y_pred_log_rv)
    y_true_rv = np.exp(y_true_log_rv) # This is the 5-min target

    # --- Calculate metrics ---
    rmse = np.sqrt(mean_squared_error(y_true_rv, y_pred_rv))
    mae = mean_absolute_error(y_true_rv, y_pred_rv)
    y_true_var = np.square(y_true_rv); y_pred_var = np.square(y_pred_rv)
    qlike = np.mean(np.log(y_pred_var + eps) + y_true_var / (y_pred_var + eps))

    print("\n––– Model Evaluation (Test Set) –––")
    print(f"Metrics vs. 5-min Target")
    print(f"RMSE (on Vol):  {rmse:.6f}")
    print(f"MAE (on Vol):   {mae:.6f}")
    print(f"QLIKE (on Var): {qlike:.6f}")

    metrics = {"rmse": rmse, "mae": mae, "qlike": qlike}
    # Return 5-min true, 5-min pred
    return y_true_rv, y_pred_rv, metrics

def plot_single_horizon(y_pred: np.ndarray,
                        y_true: np.ndarray,
                        horizon_label: str, # e.g., "5-min"
                        *,
                        start: int | None = None,
                        end:   int | None = None):
    """
    Plots the 5-min prediction vs. a single actual RV horizon.
    """
    N = len(y_pred)
    if end is None: end = N
    if start is None: start = 0
    if not (0 <= start < end <= N):
        raise ValueError("Invalid start/end slice for plotting range.")

    x_range = range(start, end)
    
    # Sliced data
    y_p_slice = y_pred[start:end]
    y_t_slice = y_true[start:end]

    plt.figure(figsize=(15, 6))
    
    plt.plot(x_range, y_t_slice, label=f"Actual ({horizon_label} RV)", linewidth=2, color='blue', alpha=0.8)
    plt.plot(x_range, y_p_slice, label="Model (5-min Prediction)", linewidth=2, linestyle="--", color='red')
    
    title = f"Model 5-min Forecast vs. Actual {horizon_label} RV (Indices {start}–{end-1})"
    filename = f"comparison_{horizon_label.replace('-', '')}.png"
    
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.show()


# --- Main execution block (replaces train_deepvol function) ---
if __name__ == "__main__":
    pl.seed_everything(42) # for reproducibility
    
    # 1. Load and prepare the data
    your_csv_filename = 'final.csv'
    full_dataset, plot_data = load_set(csv_path=your_csv_filename)
    
    if full_dataset:
        # 2. Sequential Train/Val/Test Split
        test_split = 0.15
        valid_split = 0.15
        
        test_size = int(len(full_dataset) * test_split)
        valid_size = int(len(full_dataset) * valid_split)
        train_size = len(full_dataset) - test_size - valid_size
        
        train_ds = Subset(full_dataset, range(0, train_size))
        valid_ds = Subset(full_dataset, range(train_size, train_size + valid_size))
        test_ds  = Subset(full_dataset, range(train_size + valid_size, len(full_dataset)))
        
        print(f"Dataset split: {train_size} train, {valid_size} val, {test_size} test")

        # 3. Create DataLoaders
        loader_train = DataLoader(train_ds, batch_size=DEFAULTS['batch_size'], shuffle=True,
                                  num_workers=4, persistent_workers=True, pin_memory=True)
        loader_val = DataLoader(valid_ds, batch_size=DEFAULTS['batch_size'] * 2, shuffle=False,
                                num_workers=4, persistent_workers=True, pin_memory=True)
        loader_test = DataLoader(test_ds, batch_size=DEFAULTS['batch_size'] * 2, shuffle=False,
                                 num_workers=4, persistent_workers=True, pin_memory=True)
        
        # 4. Initialize Model and Trainer
        hparams = {k: DEFAULTS[k] for k in (
                "num_blocks","kernel_size","residual_channels",
                "dilation_channels","skip_channels","end_channels", "lr_patience")}
        
        model = VolModel(**hparams)

        trainer = pl.Trainer(
            max_epochs=DEFAULTS['max_epochs'],
            log_every_n_steps=10,
            accelerator='auto',
            devices=1,
            callbacks=[
                TQDMProgressBar(refresh_rate=10),
                EarlyStopping(monitor="val_loss", patience=DEFAULTS['patience'], mode="min"),
            ],
        )

        # 5. Train the Model
        print("--- Starting Model Training (using L1 / MAE Loss) ---")
        trainer.fit(model, loader_train, loader_val)
        print("--- Training Complete ---")

        # 6. Evaluate on Test Set
        print("--- Starting Model Evaluation ---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        y_true_5min, y_pred_5min, test_metrics = evaluate_model(
            model, loader_test, device
        )
        
        # 7. --- UPDATED: Create separate plots ---
        print("Generating separate evaluation plots...")
        
        test_indices = test_ds.indices
        
        # Create a dict of the *test set* plotting data
        plot_data_test = {
            '5min_actual': plot_data['5min_actual_target'][test_indices],
            '10min_actual': plot_data['10min_actual_cont'][test_indices],
            '20min_actual': plot_data['20min_actual_cont'][test_indices],
            '30min_actual': plot_data['30min_actual_cont'][test_indices],
        }

        # Define the slice we want to see
        plot_start = 100
        plot_end = 400

        # Plot 1: 5-min Prediction vs 5-min Actual
        plot_single_horizon(
            y_pred=y_pred_5min,
            y_true=plot_data_test['5min_actual'],
            horizon_label="5-min",
            start=plot_start,
            end=plot_end
        )
        
        # Plot 2: 5-min Prediction vs 10-min Actual
        plot_single_horizon(
            y_pred=y_pred_5min,
            y_true=plot_data_test['10min_actual'],
            horizon_label="10-min",
            start=plot_start,
            end=plot_end
        )
        
        # Plot 3: 5-min Prediction vs 20-min Actual
        plot_single_horizon(
            y_pred=y_pred_5min,
            y_true=plot_data_test['20min_actual'],
            horizon_label="20-min",
            start=plot_start,
            end=plot_end
        )
        
        # Plot 4: Scatter (vs. 5-min target)
        plt.figure(figsize=(8, 8))
        max_val = max(y_true_5min.max(), y_pred_5min.max()) * 1.05
        min_val = min(y_true_5min.min(), y_pred_5min.min()) * 0.95
        plt.scatter(y_true_5min, y_pred_5min, alpha=0.3, label='Model vs Actual (5-min)')
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit (y=x)')
        plt.title('Prediction vs. Actual 5-min Scatter Plot')
        plt.xlabel('Actual 5-min Volatility')
        plt.ylabel('Predicted 5-min Volatility')
        plt.legend()
        plt.grid(True)
        plt.savefig('scatter_comparison.png')
        print("Saved scatter_comparison.png")
        plt.show()

        # 9. Save the final model
        save_path = "deepvol_rv_model_l1.pth"
        torch.save(model.state_dict(), save_path) # Save the state_dict directly
        print(f"💾 DeepVol weights saved → {save_path}")