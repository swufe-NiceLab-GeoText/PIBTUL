import pickle
import os
import random

from utils import TrajAugmenterWrapper, aug_collate_fn
from torch.optim.lr_scheduler import StepLR
from utils import accuracy_at_k, calculate_macro_metrics
from utils import read_processed_tra, get_embedding_vector, read_trajectories
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from data_load import TrajDataset
from models import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PIBTUL: Trajectory User Linking with Multi-view Learning')
    
    # Data parameters
    parser.add_argument('--processed_file', type=str, 
                       default='data/gowalla_traj_200.pkl',
                       help='Path to processed trajectory data file')
    parser.add_argument('--train_file', type=str,
                       default='data/processed_data/Gowalla_200.txt',
                       help='Path to raw training data file')
    parser.add_argument('--vec_file', type=str,
                       default='data/processed_data/gowalla200_embedding_node2vec.dat',
                       help='Path to embedding vector file')
    parser.add_argument('--city', type=str, default='gowalla',
                       help='City name for model identification')
    parser.add_argument('--processed_flag', type=bool, default=False,
                       help='Whether to use preprocessed dataset')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=250,
                       help='Embedding dimension size')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden layer dimension size')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout_prob', type=float, default=0.5,
                       help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_step_size', type=int, default=10,
                       help='Step size for learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                       help='Gamma for learning rate scheduler')
    
    # Loss weights
    parser.add_argument('--kl_weight', type=float, default=0.01,
                       help='Weight for KL divergence loss')
    parser.add_argument('--cluster_weight', type=float, default=1.0,
                       help='Weight for clustering loss')
    
    # Early stopping parameters
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001,
                       help='Minimum improvement threshold for early stopping')
    
    # Data split parameters
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=2024,
                       help='Random seed for reproducibility')
    parser.add_argument('--print_freq', type=int, default=100,
                       help='Print frequency during training')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Set random seed
torch.manual_seed(args.seed)
random.seed(args.seed)

# Set device
if args.device == 'auto':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device(args.device)

print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Data preprocessing
if not args.processed_flag:
    users, traj = read_trajectories(args.train_file)
    data_set = TrajDataset(traj_data=traj,
                           traj_user=users,
                           padding_idx=0,
                           use_sos_eos=None)
    with open(args.processed_file, 'wb') as f:
        pickle.dump(data_set, f)

origin_dataset = read_processed_tra(args.processed_file)

# Split dataset (key modification: split first, then augment)
train_size = int(args.train_ratio * len(origin_dataset))
test_size = len(origin_dataset) - train_size
train_sub, test_sub = random_split(origin_dataset, [train_size, test_size])

# Apply augmentation wrapper
train_dataset = TrajAugmenterWrapper(train_sub, augment=True)
test_dataset = TrajAugmenterWrapper(test_sub, augment=True)  # Test set also augmented

output_traj_size = torch.max(origin_dataset.poi_list).item() + 1
output_user_size = torch.max(origin_dataset.user_label).item() + 1

embeddings = get_embedding_vector(args.vec_file, embed_size=args.embed_size)


# Model generation
enc = Encoder(embed_size=args.embed_size, hidden_size=args.hidden_size, dropout_prob=args.dropout_prob, 
              num_layers=args.num_layers, embeddings=embeddings, output_user_size=output_user_size, device=device)
model = Model(encoder=enc, hidden_size=args.hidden_size, output_user_size=output_user_size).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

# Best model tracking
best_acc1 = 0.0
best_acc5 = 0.0
best_macro_f = 0.0
best_macro_r = 0.0
best_macro_p = 0.0
best_epoch = 0
best_model_state = None

best_map = {
    'best_acc1': 0.0,
    'best_acc5': 0.0,
    'best_macro_f': 0.0,
    'best_macro_r': 0.0,
    'best_macro_p': 0.0,
}

# Initialize list to record evaluation results
eval_results = []

# Early stopping parameters
early_stopping_counter = 0  # Early stopping counter

if __name__ == '__main__':
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=aug_collate_fn,
        shuffle=True
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=aug_collate_fn,  # Use same collate
        shuffle=False
    )

    Loss_fun = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        total_kl = 0.0
        for i, batch_data in enumerate(train_data_loader):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            output, kl_loss, z = model(batch_data)
            user_label = batch_data['users']

            # Clustering loss and prototype update
            model.update_prototypes(z.detach(), user_label)
            loss_cluster = model.cluster_loss(z, user_label)

            # Classification results and metrics
            predict_userlabel = torch.argmax(F.softmax(output, dim=-1), -1)
            _, predict_userlabel1 = torch.topk(F.softmax(output, dim=-1), 1, dim=-1)
            _, predict_userlabel5 = torch.topk(F.softmax(output, dim=-1), 5, dim=-1)
            acc1 = accuracy_at_k(predict_userlabel1.tolist(), user_label.tolist(), 1)
            acc5 = accuracy_at_k(predict_userlabel5.tolist(), user_label.tolist(), 5)
            macro_p, macro_r, macro_f = calculate_macro_metrics(predict_userlabel.tolist(), user_label.tolist())

            # Loss combination
            loss_ce = Loss_fun(output, user_label)
            total_loss = loss_ce + args.kl_weight * kl_loss + args.cluster_weight * loss_cluster

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)  # Gradient clipping
            optimizer.step()

            if i % args.print_freq == 0:
                print(f'Epoch {epoch + 1}, Batch {i}')
                print(
                    f'Acc@1: {acc1:.4f}, Acc@5: {acc5:.4f}, Macro_F: {macro_f:.4f}, Macro_R: {macro_r:.4f}, Macro_P: {macro_p:.4f}')
                print(
                    f'Loss CE: {loss_ce.item():.4f}, KL Loss: {kl_loss.item():.4f}')
        end_time = time.time()  # Record epoch end time
        epoch_time = end_time - start_time  # Calculate epoch duration

        print(f'Epoch {epoch + 1} training time: {epoch_time:.2f} seconds')  # Print training time
        scheduler.step()

        # --------- Validation Phase --------- #
        model.eval()
        TestPredict, TestPredict1, TestPredict5, UserLabel = [], [], [], []
        total_loss = 0
        epoch_eval_loss = 0
        epoch_eval_acc1 = 0

        with torch.no_grad():
            for i, batch_data in enumerate(test_data_loader):
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                output, kl_loss, _ = model(batch_data)
                user_label = batch_data['users']

                predict_userlabel = torch.argmax(F.softmax(output, dim=-1), -1)
                _, predict_userlabel1 = torch.topk(F.softmax(output, dim=-1), 1, dim=-1)
                _, predict_userlabel5 = torch.topk(F.softmax(output, dim=-1), 5, dim=-1)

                loss = Loss_fun(output, user_label)
                epoch_eval_loss += loss.item()
                epoch_eval_acc1 += accuracy_at_k(predict_userlabel1.tolist(), user_label.tolist(), 1)
                total_loss += loss.item()

                TestPredict.extend(predict_userlabel.tolist())
                TestPredict1.extend(predict_userlabel1.tolist())
                TestPredict5.extend(predict_userlabel5.tolist())
                UserLabel.extend(user_label.tolist())

            acc1 = accuracy_at_k(TestPredict1, UserLabel, 1)
            acc5 = accuracy_at_k(TestPredict5, UserLabel, 5)
            macro_p, macro_r, macro_f = calculate_macro_metrics(TestPredict1, UserLabel)

            print(f'---Test acc@1: {acc1:.4f}, Test acc@5: {acc5:.4f}, Test Loss: {total_loss / (i + 1):.4f}')
            print(f'Macro_F: {macro_f:.4f}, Macro_R: {macro_r:.4f}, Macro_P: {macro_p:.4f}')

            eval_results.append(acc1)

        # Early stopping logic
        if acc1 > best_acc1 + args.early_stopping_min_delta:
            # Performance improved, reset early stopping counter
            best_acc1 = acc1
            best_acc5 = acc5
            best_macro_f = macro_f
            best_macro_r = macro_r
            best_macro_p = macro_p
            best_map = {
                "best_acc1": acc1,
                "best_acc5": acc5,
                "best_macro_f": macro_f,
                "best_macro_r": macro_r,
                "best_macro_p": macro_p
            }
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            early_stopping_counter = 0

            print(f"✓ Performance improved! Reset early stopping counter")
            print(f"✓ New best model found at epoch {epoch + 1} with Acc@1: {acc1:.4f}")
        else:
            # Performance not improved, increase early stopping counter
            early_stopping_counter += 1
            print(f"⚠ Performance not improved, early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")

        print(f'Epoch {epoch + 1}, Accuracy: {acc1:.4f}')
        print(f"Best Accuracy: {best_acc1:.4f}")

        # Check if early stopping is needed
        if early_stopping_counter >= args.early_stopping_patience:
            print(f"\n🛑 Early stopping triggered! No improvement for {args.early_stopping_patience} consecutive epochs")
            print(f"Best performance achieved at epoch {epoch + 1 - args.early_stopping_patience}")
            print(f"Training stopped early to save computational resources")
            break

    # Print best results after all epochs
    print("\n----- Training Completed -----")
    if early_stopping_counter >= args.early_stopping_patience:
        print(f"Training ended due to early stopping, actually trained {epoch + 1} epochs")
    else:
        print(f"Training completed normally, trained {args.epochs} epochs")

    print("\n----- Best Results -----")
    print(f"Best Acc@1: {best_acc1:.4f}")
    print(f"Best Acc@5: {best_acc5:.4f}")
    print(f"Best Macro_F: {best_macro_f:.4f}")
    print(f"Best Macro_R: {best_macro_r:.4f}")
    print(f"Best Macro_P: {best_macro_p:.4f}")

    # Save the best model found during training
    if best_model_state is not None:
        # Ensure models directory exists
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"✅ Directory created: {models_dir}")
        
        best_model_path = f'{models_dir}/best_model_{args.city}_acc{best_acc1:.4f}_epoch{best_epoch}.pth'
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc1': best_acc1,
            'best_acc5': best_acc5,
            'best_macro_f': best_macro_f,
            'best_macro_r': best_macro_r,
            'best_macro_p': best_macro_p,
            'city': args.city,
            'hyperparameters': {
                'embed_size': args.embed_size,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'dropout_prob': args.dropout_prob,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate
            },
            'training_completed': True
        }, best_model_path)
        print(f"✅ Best model saved: {best_model_path}")
        print(f"✅ Best model was found at epoch {best_epoch}")
    else:
        print("⚠ No best model state found to save")

    # Save results to JSON file, filename also includes city
    results_file = f'acc_data_{args.city}_PIBTUL.json'
    with open(results_file, 'w') as f:
        json.dump(eval_results, f)
    print(f"✅ Training results saved: {results_file}")

    # Ensure data directory exists
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"✅ Directory created: {data_dir}")

    best_results_file = f'data/best_results_{args.city}_PIBTUL.json'
    with open(best_results_file, 'w') as f:
        json.dump(best_map, f)
    print(f"✅ Best results saved: {best_results_file}")




