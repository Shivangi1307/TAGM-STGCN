import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging, os, gc, argparse, math, random, warnings, tqdm
import numpy as np, pandas as pd
from sklearn import preprocessing
import torch, torch.nn as nn, torch.optim as optim, torch.utils as utils
from code.script import dataloader, utility, earlystopping
from code.model import model_regime


# ================= SEED =================
def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

# ================= ARGS =================
def get_parameters():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lambda_graph', type=float, default=0.00005)
    parser.add_argument('--use_dynamic',type=int,default=1)
    parser.add_argument('--use_alpha',type=int,default=1)
    parser.add_argument('--num_heads',type=int,default=4)
    parser.add_argument('--use_memory', type=int, default=1)
    parser.add_argument('--use_regime', type=int, default=1)
    parser.add_argument('--dataset',type=str,default='metr-la')
    parser.add_argument('--seed',type=int,default=42)

    parser.add_argument('--n_his',type=int,default=12)
    parser.add_argument('--n_pred',type=int,default=12)
    parser.add_argument('--Kt',type=int,default=3)
    parser.add_argument('--stblock_num',type=int,default=2)
    parser.add_argument('--act_func',type=str,default='glu')
    parser.add_argument('--Ks',type=int,default=3)
    parser.add_argument('--graph_conv_type',type=str,default='cheb_graph_conv')
    parser.add_argument('--gso_type',type=str,default='sym_norm_lap')
    parser.add_argument('--enable_bias',type=bool,default=True)

    parser.add_argument('--droprate',type=float,default=0.05)
    parser.add_argument('--lr',type=float,default=0.0007)
    parser.add_argument('--batch_size',type=int,default=50)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--patience',type=int,default=50)


    args=parser.parse_args()
    print("Training configs:",args)
    set_env(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Ko=args.n_his-(args.Kt-1)*2*args.stblock_num
    blocks=[[1]]
    for _ in range(args.stblock_num):
        blocks.append([64,16,64])
    if Ko>0: blocks.append([128,128])
    else: blocks.append([128])
    blocks.append([1])

    return args,device,blocks

# ================= DATA =================
def data_preparate(args,device):

    adj,n_vertex=dataloader.load_adj(args.dataset)
    gso=utility.calc_gso(adj,args.gso_type)
    gso=utility.calc_chebynet_gso(gso)
    gso=torch.from_numpy(gso.toarray().astype(np.float32)).to(device)
    args.gso=gso

    data_path = f'./data/{args.dataset}/vel.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    data = pd.read_csv(data_path)
    total=len(data)
    train_len=int(total*0.7)
    val_len=int(total*0.1)

    train=data[:train_len]
    val=data[train_len:train_len+val_len]
    test=data[train_len+val_len:]

    scaler=preprocessing.StandardScaler()
    train=scaler.fit_transform(train)
    val=scaler.transform(val)
    test=scaler.transform(test)


    x_train,y_train=dataloader.data_transform(train,args.n_his,args.n_pred,device)
    x_val,y_val=dataloader.data_transform(val,args.n_his,args.n_pred,device)
    x_test,y_test=dataloader.data_transform(test,args.n_his,args.n_pred,device)


    train_iter=utils.data.DataLoader(utils.data.TensorDataset(x_train,y_train),
                                     batch_size=args.batch_size,shuffle=True)


    val_iter=utils.data.DataLoader(utils.data.TensorDataset(x_val,y_val),
                                   batch_size=args.batch_size,shuffle=False)


    test_iter=utils.data.DataLoader(utils.data.TensorDataset(x_test,y_test),
                                    batch_size=args.batch_size,shuffle=False)


    return n_vertex,scaler,train_iter,val_iter,test_iter



# ================= MODEL =================
def prepare_model(args,blocks,n_vertex,device):
    loss=nn.MSELoss()

    es=earlystopping.EarlyStopping(
        patience=args.patience,
        path=f"STGCN_{args.use_memory}_{args.use_regime}_{args.dataset}.pt"
    )

    model=model_regime.TAGM_STGCN(args,blocks,n_vertex).to(device)

    # STABLE optimizer for dynamic model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )

    return loss,es,model,optimizer,scheduler


# ================= TRAIN =================
def train(args,model,loss,optimizer,scheduler,es,train_iter,val_iter):


    train_losses=[]
    val_losses=[]

    for epoch in range(args.epochs):

        model.train()
        total=0;n=0

        for x,y in tqdm.tqdm(train_iter):

            optimizer.zero_grad()

            pred, (dyn_graphs, mem_graphs) = model(x)
            pred_loss = loss(pred,y)

            if args.use_dynamic==1:

                graph_loss=0
                for i in range(1,len(dyn_graphs)):
                    graph_loss += torch.mean((dyn_graphs[i] - dyn_graphs[i-1])**2)

                l = pred_loss + args.lambda_graph*graph_loss
            else:
                l = pred_loss

            l.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),5)

            optimizer.step()

            total+=l.item()*y.size(0)
            n+=y.size(0)

        scheduler.step()

        val_loss = validate(model,val_iter,loss)

        train_losses.append(total/n)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:03d} | Train {total/n:.4f} | Val {val_loss:.4f}")

        es(val_loss,model)

        if es.early_stop:
            print("Early stopping")
            break

    return train_losses,val_losses


@torch.no_grad()
def validate(model,val_iter,loss):
    model.eval()
    tot=0;n=0
    for x,y in val_iter:
        pred, _ = model(x)
        l=loss(pred,y)
        tot+=l.item()*y.size(0)
        n+=y.size(0)
    return tot/n


# ================= TEST + ALL PLOTS =================
@torch.no_grad()
def evaluate_and_plot(model,test_iter,scaler,args,EXP_DIR):

    all_preds = []
    all_true = []
    noise_list = []
    graph_plotted = False

    for x,y in test_iter:

      output = model(x)

      if args.use_dynamic == 1:
          pred, (dyn, mem) = output
      else:
          pred = output
          dyn, mem = [], []

      # ===== Noise Test =====
      NOISE_STD = 0.05
      x_noisy = x + NOISE_STD * torch.randn_like(x)
      pred_noisy, _ = model(x_noisy)

      pred_noisy = pred_noisy.cpu().numpy()

      # plotting graph
      if not graph_plotted:
          A_dyn = dyn[0][0].cpu().numpy()

          if len(mem) > 0:
              A_mem = mem[0][0].cpu().numpy()
          else:
              A_mem = None

          plt.figure(figsize=(12,5))

          plt.subplot(1,2,1)
          sns.heatmap(A_dyn, cmap="viridis")
          plt.title("Dynamic Graph")

          if A_mem is not None:
            plt.subplot(1,2,2)
            sns.heatmap(A_mem, cmap="viridis")
            plt.title("Memory Graph")

          plt.savefig(f"{EXP_DIR}/dynamic_vs_memory_{args.use_memory}_{args.use_regime}.png", dpi=300)
          plt.close()

          graph_plotted = True

      pred = pred.cpu().numpy()
      y = y.cpu().numpy()

      B,H,N = pred.shape

      pred = scaler.inverse_transform(pred.reshape(-1,N)).reshape(B,H,N)
      y = scaler.inverse_transform(y.reshape(-1,N)).reshape(B,H,N)

      pred_noisy = scaler.inverse_transform(pred_noisy.reshape(-1,N)).reshape(B,H,N)

      noise_error = np.mean(np.abs(pred_noisy - y))
      noise_list.append(noise_error)  

      all_preds.append(pred)
      all_true.append(y)

    all_preds = np.concatenate(all_preds, axis=0)
    all_true  = np.concatenate(all_true, axis=0)

    # ===== 1. Horizon MAE =====
    horizons = [5*(i+1) for i in range(12)]
    mae_list = []

    for i in range(12):

        pred_h = all_preds[:,i,:]
        true_h = all_true[:,i,:]

        mae = np.mean(np.abs(pred_h - true_h))
        mae_list.append(mae)


    plt.figure()
    plt.plot(horizons, mae_list, marker='o', linewidth=2)
    plt.grid(True)
    plt.title("MAE vs Horizon")
    plt.xlabel("Prediction Horizon (5-minute steps)")
    plt.ylabel("MAE")
    plt.savefig(f"{EXP_DIR}/mae_vs_horizon_{args.use_memory}_{args.use_regime}_{args.dataset}.png", dpi=300)
    plt.close()

    # ===== 2. Prediction vs GT =====
    plt.figure()
    plt.plot(all_true[:300,0], label="Ground Truth")
    plt.plot(all_preds[:300,0], label="Prediction")
    plt.legend()
    plt.title("Prediction vs Ground Truth")
    plt.savefig(f"{EXP_DIR}/prediction_vs_gt_{args.use_memory}_{args.use_regime}_{args.dataset}.png", dpi=300)
    plt.close()

    # ===== 3. Error Distribution =====
    errors = np.abs(all_preds - all_true).flatten()
    plt.figure()
    plt.hist(errors, bins=50)
    plt.title("Absolute Error Distribution")
    plt.savefig(f"{EXP_DIR}/error_distribution_{args.use_memory}_{args.use_regime}_{args.dataset}.png", dpi=300)
    plt.close()

    # ===== 4. Confusion Matrix =====
    y_true_c = np.zeros_like(all_true)
    y_pred_c = np.zeros_like(all_preds)

    y_true_c[all_true > 20] = 1
    y_true_c[all_true > 40] = 2
    y_pred_c[all_preds > 20] = 1
    y_pred_c[all_preds > 40] = 2

    cm = confusion_matrix(y_true_c.flatten(), y_pred_c.flatten())

    plt.figure()
    sns.heatmap(cm,annot=True,fmt='d',cmap="Blues",
            xticklabels=["Low","Medium","High"],
            yticklabels=["Low","Medium","High"])
    plt.title("Traffic Regime Confusion Matrix")
    plt.savefig(f"{EXP_DIR}/confusion_matrix_{args.use_memory}_{args.use_regime}_{args.dataset}.png", dpi=300)
    plt.close()


    avg_noise_mae = float(np.mean(noise_list))
    print("Average Noise MAE:", avg_noise_mae)
    errors = np.abs(all_preds - all_true)
    mean_mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((all_preds - all_true)**2)))
    print("RMSE:", rmse)
    print("All plots saved successfully.")
    print("Final MAE:", mean_mae)

    return mean_mae, rmse, avg_noise_mae


def run_multiple_times(num_runs=3):
    import json
    maes = []

    for i in range(num_runs):
        print(f"\n===== RUN {i+1} =====")

        args,device,blocks=get_parameters()
        args.seed = 42 + i
        set_env(args.seed)

        n_vertex,scaler,train_iter,val_iter,test_iter=data_preparate(args,device)
        loss,es,model,optimizer,scheduler=prepare_model(args,blocks,n_vertex,device)

        train(args,model,loss,optimizer,scheduler,es,train_iter,val_iter)
        mae, _, _ = evaluate_and_plot(model,test_iter,scaler,args,"experiments")

        maes.append(mae)

    mean = float(np.mean(maes))
    std  = float(np.std(maes))

    print(f"\nFINAL: {mean:.4f} ± {std:.4f}")

    with open("experiments/stability_results.json","w") as f:
        json.dump({
            "mean": mean,
            "std": std,
            "runs": maes
        }, f)

# ================= RUN =================
if __name__=="__main__":

    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")

    args,device,blocks=get_parameters()
    n_vertex,scaler,train_iter,val_iter,test_iter=data_preparate(args,device)
    loss,es,model,optimizer,scheduler=prepare_model(args,blocks,n_vertex,device)

    train_losses, val_losses = train(args,model,loss,optimizer,scheduler,es,train_iter,val_iter)

    # ===== Loss Curve =====
    EXP_DIR="experiments"
    os.makedirs(EXP_DIR,exist_ok=True)

    plt.figure()
    plt.plot(train_losses,label="Train")
    plt.plot(val_losses,label="Validation")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.savefig(f"{EXP_DIR}/loss_curve_{args.use_memory}_{args.use_regime}_{args.dataset}.png", dpi=300)
    plt.close()

    final_mae, final_rmse, final_noise = evaluate_and_plot(model,test_iter,scaler,args,EXP_DIR)

    import json
    results = {
        "model": "regime",
        "dataset": args.dataset,
        "mae": final_mae,
        "rmse": final_rmse,
        "noise_mae": final_noise
    }

    model_name = "regime"

    if args.use_memory == 0 and args.use_regime == 0:
        model_name = "dynamic"
    elif args.use_memory == 0:
        model_name = "no_memory"
    elif args.use_regime == 0:
        model_name = "no_regime"

    with open(f"experiments/stability_{args.use_memory}_{args.use_regime}.json","w") as f:
        json.dump({
            "model": model_name,
            "dataset": args.dataset,
            "mae": final_mae,
            "rmse": final_rmse,
            "noise_mae": final_noise
        }, f)

    print("Results JSON saved.")
    print("Experiment complete.")

    # ===== STABILITY EXPERIMENT =====
    print("\n=== Running Stability Experiment ===")
    run_multiple_times(3)

