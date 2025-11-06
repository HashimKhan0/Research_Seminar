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
from torch.nn import MSELoss

# --- Imports for Evaluation ---
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =======GLOBAL VARIABLES=======

# ---  CONSTANTS WE *DO NOT* TUNE  ──────────────────────────────────────────
LR          = 1e-3         # learning-rate (fixed)
EPS         = 1e-11        # numerical guard (fixed)
FORWARD     = 5            # forecast horizon in minutes (fixed)
ANNUALIZATION_FACTOR = np.sqrt(78 * 252)

# ─── PARAMS WE *CAN* TUNE  ────────────────────────────────────────────────
DEFAULTS = dict(
    window_size       = 10,
    batch_size        = 512,
    max_epochs        = 100,
    patience          = 15,
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
    """Calculates 5-minute realized volatility (sqrt of sum of squares)."""
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

# ---
# --- FUNCTION HAS BEEN RE-WRITTEN ---
# ---
def prepare_rv_data(df, window_size: int = DEFAULTS['window_size']):
    """
    It now creates (Past 5-min RVs) -> (Next 5-min RV)
    USING THE PRE-CALCULATED 'return' COLUMN.
    """
    
    # 1. Use the 'return' column as our log_return
    #    We must fill any NaNs (like the first row) with 0.
    df_market = df.copy()
    df_market['log_return'] = df_market['return'].fillna(0).astype(np.float32)
    
    # 2. Calculate 5-minute Realized Volatility (RV) and Annualize it
    #    We MUST group by date here to prevent the rolling window
    #    from crossing over the overnight gap.
    rv_5min_unscaled = df_market.groupby(df_market.index.floor('D'))['log_return'].apply(
        lambda x: calculate_realized_volatility(x, window=5)
    )
    df_market['rv_5min_annualized'] = rv_5min_unscaled.reset_index(level=0, drop=True) * ANNUALIZATION_FACTOR
    
    # 3. Create the target variable: next 5-min RV (also annualized)
    df_market['target_rv_5min'] = df_market['rv_5min_annualized'].shift(-FORWARD)

    # 4. Clean data
    feature_col = 'rv_5min_annualized'
    target_col = 'target_rv_5min'
    # Drop NaNs created by rolling() and shift()
    df_processed = df_market[[feature_col, target_col]].dropna()
    
    print(f"Processed data has {len(df_processed)} rows after cleaning NaNs.")
    if len(df_processed) == 0:
        print("ERROR: No data left after dropna(). Check 'return' column in CSV.")
        return None, None

    # 5. Create sequences
    features_data = df_processed[feature_col].values
    target_data = df_processed[target_col].values
    
    X, y = create_sequences(features_data, target_data, sequence_length=window_size)
    
    # Reshape X for Conv1d: [samples, features, timesteps]
    X = np.expand_dims(X, axis=1) # -> (B, 1, T)
    
    print(f"Created {X.shape[0]} sequences with shape {X.shape}.")
    
    return X, y

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
    
    # Check for required columns
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

    X, y = prepare_rv_data(df_market, window_size)
    
    if X is None:
        return None
        
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    return TensorDataset(X, y)


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
        return F.softplus(out)

#=======TRAINING SCRIPT=======
# (This section is unchanged)
class VolModel(pl.LightningModule):
    def __init__(self, **h):
        super().__init__()
        self.save_hyperparameters()
        self.model = DeepVol(**{k: h[k] for k in (
            "num_blocks","kernel_size","residual_channels",
            "dilation_channels","skip_channels","end_channels")})
        self.loss  = MSELoss()
        
    def forward(self, x):          
        return self.model(x)
    
    def _step(self,b):
        x, y = b
        pred_rv = self(x)
        return self.loss(pred_rv, y)
    
    def training_step(self,b, _):  
        l=self._step(b); self.log("train_loss",l); return l
    
    def validation_step(self,b,_): 
        l=self._step(b); self.log("val_loss",l,prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

# ======= EVALUATION FUNCTIONS (ADAPTED FROM YOURS) =======
# (This section is unchanged)
def evaluate_model(model: pl.LightningModule,
                   test_loader: DataLoader,
                   device: torch.device,
                   eps: float = 1e-11):
    """
    Compute RMSE, MAE and QLIKE on our RV-to-RV model.
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # Get model's RV prediction
            preds = model(x)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    
    # Concatenate all batches into single numpy arrays
    y_pred_rv = torch.cat(all_preds).numpy().astype(np.float64)
    y_true_rv = torch.cat(all_targets).numpy().astype(np.float64)

    # --- Calculate metrics ---
    
    # 1. RMSE and MAE on Volatility (RV)
    rmse = np.sqrt(mean_squared_error(y_true_rv, y_pred_rv))
    mae = mean_absolute_error(y_true_rv, y_pred_rv)

    # 2. QLIKE on Variance (RV^2)
    # We square the volatility to get variance for the QLIKE calculation
    y_true_var = np.square(y_true_rv)
    y_pred_var = np.square(y_pred_rv)
    
    qlike = np.mean(np.log(y_pred_var + eps) + y_true_var / (y_pred_var + eps))

    print("\n––– Model Evaluation (Test Set) –––")
    print(f"RMSE (on Vol):  {rmse:.6f}")
    print(f"MAE (on Vol):   {mae:.6f}")
    print(f"QLIKE (on Var): {qlike:.6f}")

    metrics = {"rmse": rmse, "mae": mae, "qlike": qlike}
    return y_true_rv, y_pred_rv, metrics


def plot_predictions(y_true: np.ndarray | torch.Tensor,
                     y_pred: np.ndarray | torch.Tensor,
                     *,
                     start: int | None = None,
                     end:   int | None = None,
                     title: str = "DeepVol: Actual vs. Predicted Volatility"):
    """Line‑plot of truth vs forecast over an index slice."""
    # Convert to numpy if they are tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    N = len(y_true)
    if end is None:
        end = N
    if start is None:
        start = 0
    if not (0 <= start < end <= N):
        raise ValueError("Invalid start/end slice for plotting range.")

    y_t = y_true[start:end]
    y_p = y_pred[start:end]
    x_range = range(start, end)

    plt.figure(figsize=(15, 6))
    plt.plot(x_range, y_t, label="Actual Volatility", linewidth=2, color='blue', alpha=0.8)
    plt.plot(x_range, y_p, label="Predicted Volatility", linewidth=2, linestyle="--", color='red')
    plt.title(f"{title} (Indices {start}–{end-1})")
    plt.xlabel("Sample Index")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('timeseries_comparison.png')
    print("Saved timeseries_comparison.png")
    plt.show()


# --- Main execution block ---
# (This section is unchanged)
if __name__ == "__main__":
    pl.seed_everything(42) # for reproducibility
    
    # 1. Load and prepare the data
    your_csv_filename = 'final.csv'
    full_dataset = load_set(csv_path=your_csv_filename)
    
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
        loader_train = DataLoader(train_ds,
                                  batch_size=DEFAULTS['batch_size'],
                                  shuffle=True,
                                  num_workers=4,
                                  persistent_workers=True,
                                  pin_memory=True)
        loader_val = DataLoader(valid_ds,
                                batch_size=DEFAULTS['batch_size'] * 2,
                                shuffle=False,
                                num_workers=4,
                                persistent_workers=True,
                                pin_memory=True)
        loader_test = DataLoader(test_ds,
                                 batch_size=DEFAULTS['batch_size'] * 2,
                                 shuffle=False,
                                 num_workers=4,
                                 persistent_workers=True,
                                 pin_memory=True)
        
        # 4. Initialize Model and Trainer
        hparams = {k: DEFAULTS[k] for k in (
                "num_blocks","kernel_size","residual_channels",
                "dilation_channels","skip_channels","end_channels")}
        
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
        print("--- Starting Model Training ---")
        trainer.fit(model, loader_train, loader_val)
        print("--- Training Complete ---")

        # 6. Evaluate on Test Set using your adapted function
        print("--- Starting Model Evaluation ---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        y_true_test, y_pred_test, test_metrics = evaluate_model(
            model, loader_test, device
        )
        
        # 7. Plot using your adapted function
        print("Generating evaluation plots...")
        plot_predictions(y_true_test, y_pred_test, start=100, end=400)
        
        # 8. Plot Scatter
        plt.figure(figsize=(8, 8))
        max_val = max(y_true_test.max(), y_pred_test.max()) * 1.05
        min_val = min(y_true_test.min(), y_pred_test.min()) * 0.95
        plt.scatter(y_true_test, y_pred_test, alpha=0.3, label='Model vs Actual')
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit (y=x)')
        plt.title('Prediction vs. Actual Scatter Plot')
        plt.xlabel('Actual Volatility')
        plt.ylabel('Predicted Volatility')
        plt.legend()
        plt.grid(True)
        plt.savefig('scatter_comparison.png')
        print("Saved scatter_comparison.png")
        plt.show()

        # 9. Save the final model
        save_path = "deepvol_rv_model.pth"
        torch.save(model.state_dict(), save_path) # Save the state_dict directly
        print(f"💾 DeepVol weights saved → {save_path}")