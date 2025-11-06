# =======LIBRARIES=======
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from torch.nn import MSELoss

# GLOBAL VARIABLES
 
# ───  CONSTANTS WE *DO NOT* TUNE  ──────────────────────────────────────────
LR          = 1e-3         # learning-rate (fixed)
EPS         = 1e-11         # numerical guard (fixed)
WINDOW_SIZE = 78 #78          # 78×5-min = 1 trading day (fixed)
FORWARD     = 5    #5      # forecast horizon in minutes (fixed)

# ─── PARAMS WE *CAN* TUNE  (paper defaults shown) ────────────────────────
DEFAULTS = dict(
    batch_size        = 512,
    max_epochs        = 100,
    patience          = 15,
    kernel_size       = 3,
    num_blocks        =4,
    num_layers        = 6,        # note: in this implementation = num_blocks
    residual_channels = 16,
    dilation_channels = 32,
    skip_channels     = 64,
    end_channels      = 64,
)
# -----
# =======DATA PREP=======
# we need intraday_returns (should be done in data_loading.py)

def prepare_minute_level_data(df, window_size: int =WINDOW_SIZE):
    # creates  training data(X) and true value (Y) of log returns given a past window(1 full day, past 30, past 15, past 1) {minutes}
    # creates a forward window of 5 minutes i.e creates a true value over the next 5 mins ENSURE MODEL IS ALSO PREDICTING NEXT VOL
    r = df['return'].values.astype(np.float32)
    
    X,y_logvar = [], []
    scale = np.std(r[:10_000])
    for i in range(len(r) - window_size - FORWARD):
        past_slice = r[i:i + window_size] /1e-1
        fwd_var = (r[i + window_size: i + window_size+FORWARD ] **2).mean() # the average of squared log-returns over the next FORWARD minutes
        X.append(past_slice )
        y_logvar.append(np.log(fwd_var + EPS)) # we add epsilonn to convert out small positive number into a real valued target the network can predict

    return np.stack(X, dtype=np.float32), np.array(y_logvar, dtype=np.float32) 

def load_set(csv_path: str, window_size = WINDOW_SIZE):
    df = pd.read_csv(csv_path)
    X, y = prepare_minute_level_data(df, window_size)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    return TensorDataset(X,y)



# =======DIALED CASUAL CONVOLUTION=======
#construct basic DCC block class

class GatedDCC(nn.Module):
    """A gated dilated‑causal convolution with residual & skip connections."""
    def __init__(self, in_ch, res_ch, dil_ch, skip_ch, k,dilation):
        super().__init__()
        pad = (dilation * (k-1)) // 2
        self.conv = nn.Conv1d(in_ch, 
                              dil_ch * 2, 
                              kernel_size=k,
                              dilation=dilation,
                              padding=pad)
        
        self.residual = nn.Conv1d(dil_ch, res_ch, 1)
        self.skip     = nn.Conv1d(dil_ch, skip_ch, 1)


    def forward(self,x):
        h = self.conv(x)
        f,g = torch.chunk(h, 2, dim=1) #gate split
        z = torch.tanh(f) * torch.sigmoid(g)
        res = self.residual(z) + x  # residual connection
        skip = self.skip(z)
        return res,skip
        

class DeepVol(nn.Module):
    def __init__(self, *,
                 num_blocks:int,
                 kernel_size:int,
                 residual_channels:int,
                 dilation_channels:int,
                 skip_channels:int,
                 end_channels:int):
        super().__init__()
        
        self.input_proj = nn.Conv1d(1, residual_channels, 1)


        blocks, dilation = [], 1
        for _ in range(num_blocks):
            blocks.append(
                GatedDCC(in_ch = residual_channels, 
                         res_ch= residual_channels, 
                         dil_ch= dilation_channels,
                         skip_ch=skip_channels, 
                         k= kernel_size, 
                         dilation=dilation)
            )
            dilation *= 2
        
        self.blocks = nn.ModuleList(blocks)
        self.post   = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, end_channels, 1),
            nn.ReLU(),
            nn.Conv1d(end_channels, 1, 1),
        )

    def forward(self, x):                # x:(B,T)
        x = self.input_proj(x.unsqueeze(1))              # ->(B,1,T)
        skip_sum = 0
        for blk in self.blocks:
            x, sk = blk(x)
            skip_sum = skip_sum + sk
        out = self.post(skip_sum)
        out = out.mean(dim=-1).squeeze(-1)
        return out


#=======TRAINING SCRIPT=======
class QLIKE(nn.Module):
    def __init__(self, eps: float = 1e-10, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        pred:   torch.Tensor,   # σ̂²  (already ≥ 0)  OR   log σ̂² if you pass log=True
        target: torch.Tensor,   # σ²   (true realised variance)
        *,
        pred_is_log: bool = False,
    ) -> torch.Tensor:

        # Flatten only the *last* dimensions we care about; keeps batch dims intact
        pred   = pred.view(-1)
        target = target.view(-1)

        if pred_is_log:
            # If the network outputs log(σ̂²) directly, transform once
            sigma2_hat = torch.exp(pred).clamp_min(self.eps)
            log_hat    = pred.clamp_min(torch.log(torch.tensor(self.eps)))
        else:
            sigma2_hat = pred.clamp_min(self.eps)
            log_hat    = torch.log(sigma2_hat)

        loss_vec = log_hat + target / sigma2_hat

        if self.reduction == "mean":
            return loss_vec.mean()
        elif self.reduction == "sum":
            return loss_vec.sum()
        else:                    # "none"
            return loss_vec
    

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
        return self.loss(self(b[0]), b[1])
    def training_step(self,b, _):  
        l=self._step(b); self.log("tr",l); return l
    def validation_step(self,b,_): 
        l=self._step(b); self.log("val_loss",l,prog_bar=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

#training LOOP / helper

def train_deepvol(train_ds, valid_ds, batch_size = 512, save_path = None, verbose=True):
    loader_train = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=11,
                              persistent_workers=True,
                              pin_memory=True)
    loader_val = DataLoader(valid_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=11,
                              persistent_workers=True,
                              pin_memory=True)
    
    model = VolModel(**DEFAULTS)

    trainer = pl.Trainer(
        max_epochs=150,
        log_every_n_steps=10,
        accelerator='cuda',
        devices=1,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            EarlyStopping(monitor="val_loss", patience=15, mode="min"),
        ],
    )

    trainer.fit(model, loader_train, loader_val)
    if save_path is not None:
        torch.save(model.model.state_dict(), save_path)
        if verbose:
            print(f"💾 DeepVol weights saved → {save_path}")

    return model
