import scipy.io as sio
import numpy as np
import torch
from torch._higher_order_ops import flex_attention
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import argparse
import torch.nn as nn
import optuna
import os
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    pred_mode = trial.suggest_categorical("pred_mode", ["rolling", "direct"])
    model_choice = trial.suggest_categorical("model_choice", ["1L-LSTM", "2L-LSTM", "2L-LSTM-Attn"])
    current_seq_len = trial.suggest_categorical("seq_len", [5, 10, 15])
    
    n_layers = 2 if "2L" in model_choice else 1
    use_attn = True if "Attn" in model_choice else False
    train_pred_len = 1 if pred_mode == "rolling" else 10

    train_ds = MultiCycleDataset(file_path, seq_len=current_seq_len, pred_len=train_pred_len, mode='train')
    val_ds = MultiCycleDataset(file_path, seq_len=current_seq_len, pred_len=10, mode='test', scaler=train_ds.scaler)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SpeedPredModel(
        seq_len=current_seq_len,
        pred_len=train_pred_len,
        hidden_size=hidden_size,
        dropout=0.3,
        n_layers=n_layers,
        use_attn=use_attn
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    num_bo_epochs = 15
    for epoch in range(15):
        model.train()
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device).squeeze(-1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(args.device)
            if pred_mode == "direct":
                preds = model(x).cpu().numpy()
            else:
                curr_input = x
                temp_preds = []
                for _ in range(10):
                    pred = model(curr_input)
                    temp_preds.append(pred.cpu().numpy())
                    curr_input = torch.cat((curr_input[:, 1:, :], pred.unsqueeze(2)), dim=1)
                preds=np.concatenate(temp_preds, axis=1)
            
            all_preds.append(preds.flatten())
            all_targets.append(y.squeeze(-1).cpu().numpy().flatten())
    full_preds = np.concatenate(all_preds, axis=0)
    full_targets = np.concatenate(all_targets, axis=0)
    final_mse = mean_squared_error(full_preds, full_targets)

    return final_mse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=10, help='the length of the history sequence')
    parser.add_argument('--pred_len', type=int, default=10, help='the length of the prediction sequence')
    parser.add_argument('--n_layers', type=int, default=2, help='the number of lstm layers in the model')
    parser.add_argument('--hidden_size', type=int, default=128, help='the hidden size of the lstm')
    parser.add_argument('--dropout', type=float, default=0.3, help='the dropout rate')
    parser.add_argument('--device', type=str, default='cuda', help='the device to use')
    parser.add_argument('--batch_size', type=int, default=64, help='the batch size')
    parser.add_argument('--epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='the weight decay')
    parser.add_argument('--seed', type=int, default=42, help='the seed for the random number generator')
    parser.add_argument('--wandb', type=bool, default=True, help='whether to use wandb')
    parser.add_argument('--wandb_project', type=str, default='EvPred', help='the name of the wandb project')
    parser.add_argument('--wandb_entity', type=str, default='EvPred', help='the name of the wandb entity')
    parser.add_argument('--wandb_name', type=str, default='EvPred', help='the name of the wandb run')
    parser.add_argument('--wandb_id', type=str, default='EvPred', help='the id of the wandb run')
    parser.add_argument('--wandb_version', type=str, default='EvPred', help='the version of the wandb run')
    parser.add_argument('--wandb_group', type=str, default='EvPred', help='the group of the wandb run')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='the path to the dataset')
    parser.add_argument('--test_dataset_path', type=str, default='./test_dataset', help='the path to save the model')
    parser.add_argument('--pred_mode', type=str, default='direct', help='the mode of the prediction')

    return parser.parse_args()

def print_metrics(preds, targets):
    mse = mean_squared_error(preds, targets)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(preds, targets)
    r2 = r2_score(preds, targets)
    
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

def direct_evaluation_pipeline(model, test_data_normalized, scaler, device, seq_len=5, pred_len=10):
    model.load_state_dict(torch.load('best_speed_pred_model.pth'))
    model.eval()
    
    inputs = []
    targets = []
    
    # 构建测试对
    for i in range(len(test_data_normalized) - seq_len - pred_len + 1):
        inputs.append(test_data_normalized[i : i + seq_len])
        targets.append(test_data_normalized[i + seq_len : i + seq_len + pred_len])
    
    inputs = torch.FloatTensor(np.array(inputs).reshape(-1, seq_len, 1)).to(device)
    targets = np.array(targets) # [N, pred_len]
    
    with torch.no_grad():
        # 一次性预测出 10 步
        preds = model(inputs).cpu().numpy() # [N, pred_len]
    
    # 反归一化
    real_preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1, pred_len)
    real_targets = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(-1, pred_len)
    
    return real_preds.flatten(), real_targets.flatten()

def rolling_evaluation_pipeline(model, test_data_normalized, scaler, device, seq_len=5, pred_steps=10):
    model.load_state_dict(torch.load('best_speed_pred_model.pth'))
    model.eval()
    
    inputs = []
    targets_10s = []

    for i in range(len(test_data_normalized) - seq_len - pred_steps + 1):
        inputs.append(test_data_normalized[i : i + seq_len])
        targets_10s.append(test_data_normalized[i + seq_len : i + seq_len + pred_steps])
    
    inputs = torch.FloatTensor(np.array(inputs).reshape(-1, seq_len, 1)).to(device) # [N, seq_len, 1]
    
    print(inputs.shape)
    targets_10s = np.array(targets_10s).squeeze()           # [N, pred_steps]
    print(targets_10s.shape)
    all_preds = []
    
    with torch.no_grad():
        curr_input = inputs
        for _ in range(pred_steps):
            pred = model(curr_input) # [N, 1]
            all_preds.append(pred.cpu().numpy())
            curr_input = torch.cat((curr_input[:, 1:, :], pred.unsqueeze(2)), dim=1)
            
    all_preds = np.concatenate(all_preds, axis=1) # [N, 10]
    
    N = all_preds.shape[0]
    real_preds = scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(N, pred_steps)
    real_targets = scaler.inverse_transform(targets_10s.reshape(-1, 1)).reshape(N, pred_steps)
    
    flat_preds = real_preds.flatten()
    
    flat_targets = real_targets.flatten()
    
    return flat_preds, flat_targets

# 调用并打印指标

class MultiCycleDataset(Dataset):
    def __init__(self, file_path, seq_len, pred_len, mode='train', scaler=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = []

        all_raw_data = []
        processed_segments = []

        for path in file_path:
            data = sio.loadmat(path)['speed_vector'].flatten().astype(np.float32)

            split_idx = int(len(data) * 0.8)
            if mode == 'train':
                segment = data[:split_idx]
            else:
                segment = data[split_idx:]

            processed_segments.append(segment)
            if mode == 'train':
                all_raw_data.append(segment)
        
        if mode == 'train':
            self.scaler = MinMaxScaler()
            concatenated_train = np.concatenate(all_raw_data).reshape(-1, 1)
            self.scaler.fit(concatenated_train)
        else:
            self.scaler = scaler
        
        for segment in processed_segments:
            scaled_segment = self.scaler.transform(segment.reshape(-1, 1)).flatten()

            for i in range(len(scaled_segment) - self.seq_len - self.pred_len + 1):
                x = scaled_segment[i:i+self.seq_len]
                y = scaled_segment[i+self.seq_len:i+self.seq_len+self.pred_len]
                self.samples.append((x, y))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]

        return torch.tensor(x).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)



class SpeedPredModel(nn.Module):
    def __init__(self, seq_len, pred_len, hidden_size, dropout, n_layers, use_attn=False):
        super(SpeedPredModel, self).__init__()
        self.use_attn = use_attn
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=hidden_size, 
                            num_layers=n_layers, 
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        if use_attn:
            self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)


    def forward(self, x):
        # x: [batch_size, seq_len, 1]
        x = x.permute(1, 0, 2)
        # x: [seq_len, batch_size, 1]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [seq_len, batch_size, hidden_size]
        if self.use_attn:
            lstm_out = lstm_out.permute(1, 0, 2)
            # lstm_out: [batch_size, seq_len, hidden_size] [B,L,D]
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # attn_out: [batch_size, seq_len, hidden_size] [B,L,D]
            feat = attn_out[:, -1, :] # [B,D]
            # feat: [batch_size, hidden_size] [B,D]
        else:
            feat = lstm_out[-1, :, :] # [B,D]

        x = self.fc(feat)
        # x: [batch_size, pred_len] [B,P]
        return x

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    dataset_path = args.dataset_path
    train_files = ['real_speed_vector1.mat', 'real_speed_vector2.mat', 'real_speed_vector3.mat', 'real_speed_vector4.mat', 'real_speed_vector5.mat',
                'Standard_ChinaCity.mat','Standard_HWFET.mat','Standard_IM240.mat','Standard_WVUCITY.mat','Standard_WVUSUB.mat']
    file_path = [os.path.join(dataset_path, path) for path in train_files]

    print("Starting the optimization process...")
    study = optuna.create_study(
        direction="minimize",
        study_name="speed_optimization",
        storage="sqlite:///speed_optimization.db",
        load_if_exists=True,
    )

    #study.optimize(objective, n_trials=30)
    print("Optimization completed!")
    print("Best trial:")
    best_params = study.best_params
    for k,v in best_params.items():
        print(f"{k}: {v}")
    
    print("\n >>> using the best parameters to train the model...")
    train_pred_len = 1 if best_params["pred_mode"] == "rolling" else 10
    n_layers = 2 if "2L" in best_params["model_choice"] else 1
    use_attn = True if "Attn" in best_params["model_choice"] else False
    
    best_seq = best_params["seq_len"]
    train_dataset = MultiCycleDataset(file_path, seq_len=best_seq, pred_len=train_pred_len, mode='train')
    val_dataset = MultiCycleDataset(file_path, seq_len=best_seq, pred_len=train_pred_len, mode='test', scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    set_seed(args.seed)

    model = SpeedPredModel(
        seq_len=best_seq,
        pred_len = train_pred_len,
        hidden_size=best_params["hidden_size"],
        n_layers=n_layers,
        dropout=0.3,
        use_attn=use_attn
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    save_path = 'best_speed_pred_model.pth'

    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            speed_pred = model(x)
            target_speed = y
            target_speed = target_speed.squeeze(-1)
            speed_loss = criterion(speed_pred, target_speed)
            loss = speed_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad(): 
            for batch in val_loader:
                x, y = batch
                x = x.to(args.device)
                y = y.to(args.device)
                speed_pred = model(x)
                target_speed = y
                target_speed = target_speed.squeeze(-1)
                speed_loss = criterion(speed_pred, target_speed)
                val_loss += speed_loss.item()
        avg_val_loss = val_loss / len(val_loader)

        # --- 保存逻辑 ---
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", end="")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print("  <-- 最佳模型已保存!")
        else:
            print("") # 换行

    
    print("Training completed!")
    print("Loading the best model...")
    model.load_state_dict(torch.load(save_path))
    print("Model loaded!")

    #test_files = ['Guiyang_City.mat']
    test_files = ['Merged_real_cycle.mat']
    #test_files = ['real_speed_vector6.mat']
    #test_files = ['Standard_WVUCITY.mat']
    test_path = [os.path.join(args.test_dataset_path, path) for path in test_files]
    test_data = sio.loadmat(test_path[0])['speed_vector'].flatten().astype(np.float32)
    test_data = train_dataset.scaler.transform(test_data.reshape(-1, 1)).flatten()

    if best_params["pred_mode"] == 'direct':
        real_preds, real_targets = direct_evaluation_pipeline(model, test_data, train_dataset.scaler, args.device, seq_len=best_seq, pred_len=10)
    else:
        real_preds, real_targets = rolling_evaluation_pipeline(model, test_data, train_dataset.scaler, args.device, seq_len=best_seq, pred_steps=10)   
    print("\n" + "="*30)
    print(f"模式: {best_params['pred_mode']} | 模型: {best_params['model_choice']}")
    real_preds[real_preds < 0] = 0
    print_metrics(real_preds, real_targets)
    print("="*30)
