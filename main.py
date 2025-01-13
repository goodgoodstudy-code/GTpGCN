import torch
import argparse
import os
import numpy as np
import torch.optim as optim
from model import graph_Transformer, GCN, ChebGCNv2, ChebGCNv1
from train import test_transformer, train_transformer, train_gcn, test_gcn
from dataloader import get_dataloader, get_gcn_data_v2, get_gcn_data_val, get_gcn_data_v3

# Flags to control phase one (imaging) and phase two (non-imaging)
imaging = False
non_imaging = True
get_weight = False
grad_cam = False

# Argument parser for configuration
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='848MDDvs794NC', help='Name of the task')  # Example: 232FEDNvs394NC
parser.add_argument('--altas_name', type=str, default='AAL', help='Name of the atlas')  # Options: CC200, Dosenbach, HO, AAL
base_path = '/home/rendy/fMRI/DATASETS/MDD'

args = parser.parse_args()

# Paths for data and CSV files
data_path = f"{base_path}/{args.task_name}/{args.task_name}_{args.altas_name}.npy"
csv_path = f"{base_path}/{args.task_name}/{args.task_name}_{args.altas_name}.csv"

parser.add_argument('--data_path', type=str, default=data_path, help='Path to the data file')
parser.add_argument('--csv_path', type=str, default=csv_path, help='Path to the CSV file')

# Experiment configuration
parser.add_argument('--n_split', type=int, default=5, help='n-fold cross-validation')
parser.add_argument('--seed', type=int, default=2024, help='Random seed for reproducibility')
parser.add_argument('--device', type=str, default='cuda:0', help='Specify CUDA device')

# Transformer configuration
parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--saved_model_dir', type=str, default='./saved_model/', help='Directory for saving models')

# Extended feature configuration
parser.add_argument('--ext_featuresize', type=int, default=16, help='Size of the extended feature')

# GNN configuration
parser.add_argument('--gcn_num_epochs', type=int, default=500, help='Number of epochs for GCN')
parser.add_argument('--numeric_features', type=str, default=['Age', 'Edu'], help='Numeric features (e.g., Age, Education)')
parser.add_argument('--text_features', type=str, default=['Site', 'Sex'], help='Text features (e.g., Site, Sex)')
parser.add_argument('--num_features', type=int, default=16, help='Number of features (same as ext_featuresize)')
parser.add_argument('--nhid', type=int, default=256, help='Hidden layer size')
parser.add_argument('--dropout_ratio', type=float, default=0.3, help='Dropout ratio')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')

args = parser.parse_args()

# Print configuration details
print(args.task_name)
print(f"data_path: {args.data_path}")
print(f"csv_path: {args.csv_path}")

# Set random seed for reproducibility
torch.manual_seed(args.seed)

# Create directories for saving models and features
saved_model_dir = os.path.join(args.saved_model_dir, f"{args.task_name}_{args.altas_name}")
os.makedirs(saved_model_dir, exist_ok=True)

feature_dir = os.path.join('./data/', f"{args.task_name}_{args.altas_name}")
os.makedirs(feature_dir, exist_ok=True)

# Load data loaders
dataloaders_list = get_dataloader(args)

if imaging:
    # Define the loss function for classification
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    all_best_metrics = []  # Store the best metrics for all folds
    for fold, (train_dataloader, test_dataloader) in enumerate(dataloaders_list):

        # Initialize the model
        model = graph_Transformer(args).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000, eta_min=1.0e-5)

        best_acc = 0  # Best accuracy for the current fold

        for epoch in range(args.num_epochs):
            # Training and testing
            train_loss, train_ACC, train_SEN, train_SPE, train_AUC = train_transformer(model, train_dataloader, loss_fn, optimizer, args)
            test_loss, test_ACC, test_SEN, test_SPE, test_AUC = test_transformer(model, test_dataloader, loss_fn, args)
            lr_scheduler.step()

            # Print metrics for each epoch
            print(f'Epoch {epoch+1}/{args.num_epochs}, '
                  f'Train Loss: {train_loss:.3f}, Train ACC: {train_ACC:.3f}, '
                  f'Test Loss: {test_loss:.3f}, Test ACC: {test_ACC:.3f}, '
                  f'Test AUC: {test_AUC:.3f}, SEN: {test_SEN:.3f}, SPE: {test_SPE:.3f}')

            # Save the best model based on accuracy
            if test_ACC > best_acc:
                best_acc = test_ACC
                best_acc_epoch = epoch
                best_model_path = os.path.join(saved_model_dir, f'best_acc_model_{fold}.pth')
                torch.save(model.state_dict(), best_model_path)
                best_metrics = {
                    'ACC': best_acc,
                    'AUC': test_AUC,
                    'SEN': test_SEN,
                    'SPE': test_SPE
                }

        # Print the best results for the current fold
        print(f'Best Epoch for Accuracy in Fold {fold}: {best_acc_epoch+1}, '
              f'Best ACC: {best_metrics["ACC"]:.3f}, Best AUC: {best_metrics["AUC"]:.3f}, '
              f'Best SEN: {best_metrics["SEN"]:.3f}, Best SPE: {best_metrics["SPE"]:.3f}')

        all_best_metrics.append(best_metrics)

        # ------------------- Feature extraction ------------------- #
        # Load the best model
        model = graph_Transformer(args).to(args.device)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()  # Set the model to evaluation mode

        # Initialize lists to store features and labels
        all_features_list = []
        all_labels_list = []
        all_indices_list = []  # Store indices corresponding to the original data
        train_test_list = []  # Indicate whether the data belongs to the training (0) or test (1) set

        with torch.no_grad():
            # Process training data
            for time_series, node_feature, labels, indices in train_dataloader:
                labels = labels.float()
                time_series, node_feature, labels = time_series.to(args.device), node_feature.to(args.device), labels.to(args.device)
                _, extract_feature = model(node_feature)
                all_features_list.append(extract_feature.cpu().numpy())
                all_labels_list.append(labels.cpu().numpy())
                all_indices_list.append(indices.cpu().numpy())
                train_test_list.append(np.zeros(indices.size(0)))  # Add zeros for training data

            # Process test data
            for time_series, node_feature, labels, indices in test_dataloader:
                labels = labels.float()
                time_series, node_feature, labels = time_series.to(args.device), node_feature.to(args.device), labels.to(args.device)
                _, extract_feature = model(node_feature)
                all_features_list.append(extract_feature.cpu().numpy())
                all_labels_list.append(labels.cpu().numpy())
                all_indices_list.append(indices.cpu().numpy())
                train_test_list.append(np.ones(indices.size(0)))  # Add ones for test data

            # Save features, labels, indices, and train/test flags
            all_features_array = np.concatenate(all_features_list, axis=0)
            all_labels_array = np.concatenate(all_labels_list, axis=0)
            all_indices_array = np.concatenate(all_indices_list, axis=0)
            train_test_array = np.concatenate(train_test_list, axis=0)

            np.save(os.path.join(feature_dir, f'all_features_fold{fold}_X.npy'), all_features_array)
            np.save(os.path.join(feature_dir, f'all_labels_fold{fold}_Y.npy'), all_labels_array)
            np.save(os.path.join(feature_dir, f'all_indices_fold{fold}.npy'), all_indices_array)
            np.save(os.path.join(feature_dir, f'train_test_fold{fold}.npy'), train_test_array)

        print("------ Feature extraction completed. ------")

    # Calculate average metrics across all folds
    average_metrics = {
        'ACC': np.mean([m['ACC'] for m in all_best_metrics]),
        'SEN': np.mean([m['SEN'] for m in all_best_metrics]),
        'SPE': np.mean([m['SPE'] for m in all_best_metrics]),
        'AUC': np.mean([m['AUC'] for m in all_best_metrics]),
    }
    print(f"All best ACC: {[m['ACC'] for m in all_best_metrics]}, Average ACC: {average_metrics['ACC']:.3f}")
    print(f"All best SEN: {[m['SEN'] for m in all_best_metrics]}, Average SEN: {average_metrics['SEN']:.3f}")
    print(f"All best SPE: {[m['SPE'] for m in all_best_metrics]}, Average SPE: {average_metrics['SPE']:.3f}")
    print(f"All best AUC: {[m['AUC'] for m in all_best_metrics]}, Average AUC: {average_metrics['AUC']:.3f}")

    print("------ Starting GCN fusion with non-imaging features. -------")


# ---------------可解释分析---------------
# if grad_cam:

if get_weight:
    attention_weights_dir = os.path.join(saved_model_dir, 'attention_weight')
    if not os.path.exists(attention_weights_dir):
        os.makedirs(attention_weights_dir)
    for fold, (train_dataloader, test_dataloader) in enumerate(dataloaders_list):
        model = graph_Transformer(args).to(args.device)
        fold_model_path = os.path.join(saved_model_dir, f'best_acc_model_{fold}.pth')
        model.load_state_dict(torch.load(fold_model_path))
        model.eval()

        train_attention_weights = []
        train_labels = []
        for time_series, node_feature, labels, indices in train_dataloader:
            labels = labels.float()
            time_series, node_feature, labels = time_series.to(args.device), node_feature.to(args.device), labels.to(args.device)
            model(node_feature)# time_series, 
            attention_weights = model.transformer1.get_attention_weights().detach()
            train_attention_weights.append(attention_weights.cpu())
            train_labels.extend(labels.cpu().tolist())
        train_attention_weights = torch.cat(train_attention_weights, dim=0)
        train_attention_weights_path = os.path.join(attention_weights_dir, f'train_attention_weights_fold{fold}.npy')
        np.save(train_attention_weights_path, train_attention_weights.numpy())
        train_labels_path = os.path.join(attention_weights_dir, f'train_labels_fold{fold}.npy')
        np.save(train_labels_path, np.array(train_labels))

        test_attention_weights = []
        test_labels = []
        for time_series, node_feature, labels, indices in test_dataloader:
            labels = labels.float()
            time_series, node_feature, labels = time_series.to(args.device), node_feature.to(args.device), labels.to(args.device)
            model( node_feature)# time_series,
            attention_weights = model.transformer1.get_attention_weights().detach()
            test_attention_weights.append(attention_weights.cpu())
            test_labels.extend(labels.cpu().tolist()) 
        test_attention_weights = torch.cat(test_attention_weights, dim=0)
        test_attention_weights_path = os.path.join(attention_weights_dir, f'test_attention_weights_fold{fold}.npy')
        np.save(test_attention_weights_path, test_attention_weights.numpy())
        test_labels_path = os.path.join(attention_weights_dir, f'test_labels_fold{fold}.npy')
        np.save(test_labels_path, np.array(test_labels))
    print("------Attention weights extraction completed.-------")

    
from torch_geometric.data import DataLoader



if non_imaging:
    gcn_all_best_metrics = []
    for fold in range(args.n_split):
        
        data = get_gcn_data_v3(args,fold,feature_dir)
        data_file = os.path.join('/home/rendy/rdyBrainNetTransformer/WithGNN/graphdata/', f'gcn_data_fold{fold}.pt')
        torch.save(data, data_file)
        data_loader = DataLoader([data], batch_size=1, shuffle=True)
        model = GCN(args.num_features, num_classes=2, dropout=0.2, hgc=16, lg=3).to(args.device)
        # model = ChebGCNv1(args.num_features, num_classes=2, dropout=0.2, hgc=16, lg=3,K = 3).to(args.device) #  K = 3
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        gcn_best_epoch = 0
        gcn_best_acc = 0
        gcn_best_auc = 0
        for epoch in range(args.gcn_num_epochs):
            train_loss, train_ACC, train_SEN, train_SPE, train_AUC = train_gcn(data_loader, model, optimizer, args)
            test_loss, test_ACC, test_SEN, test_SPE, test_AUC = test_gcn(data_loader, model, args, is_test_phase=True)

            

            print(f'Epoch {epoch+1}/{args.gcn_num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train ACC: {train_ACC:.4f}, '
              f'Test Loss: {test_loss:.4f}, ACC: {test_ACC:.4f}, '
              f'AUC: {test_AUC:.4f}, SEN: {test_SEN:.4f}, SPE: {test_SPE:.4f}')
            
            if test_ACC > gcn_best_acc:
                gcn_best_acc = test_ACC
                best_acc_epoch = epoch
                gcn_best_metrics = {
                        'ACC': gcn_best_acc,
                        'AUC': test_AUC,
                        'SEN': test_SEN,
                        'SPE': test_SPE
                    }
                best_model_path = os.path.join('/home/rendy/rdyBrainNetTransformer/WithGNN/graphdata/', f'gcn_best_model{fold}.pt')
                torch.save(model.state_dict(), best_model_path)
               
        print(f'Best Epoch for Accuracy in Fold {fold}: {best_acc_epoch+1}, '
        f'Best ACC: {gcn_best_metrics["ACC"]:.4f}, Best AUC: {gcn_best_metrics["AUC"]:.4f}, '
        f'Best SEN: {gcn_best_metrics["SEN"]:.4f}, Best SPE: {gcn_best_metrics["SPE"]:.4f}')
        gcn_all_best_metrics.append(gcn_best_metrics)

        
    gcn_average_metrics = {
        'ACC': np.mean([m['ACC'] for m in gcn_all_best_metrics]),
        'SEN': np.mean([m['SEN'] for m in gcn_all_best_metrics]),
        'SPE': np.mean([m['SPE'] for m in gcn_all_best_metrics]),
        'AUC': np.mean([m['AUC'] for m in gcn_all_best_metrics]),
    }
    print(f"All best ACC: {[m['ACC'] for m in gcn_all_best_metrics]}, Average ACC: {gcn_average_metrics['ACC']:.3f}")
    print(f"All best SEN: {[m['SEN'] for m in gcn_all_best_metrics]}, Average SEN: {gcn_average_metrics['SEN']:.3f}")
    print(f"All best SPE: {[m['SPE'] for m in gcn_all_best_metrics]}, Average SPE: {gcn_average_metrics['SPE']:.3f}")
    print(f"All best AUC: {[m['AUC'] for m in gcn_all_best_metrics]}, Average AUC: {gcn_average_metrics['AUC']:.3f}")

    