import argparse
import logging
import os
import torch
from Biotokenizer import BioTokenizerV2
from Dataset_Construct import TokenizedBioDataset, collate_fn
from StandardGPT import GPTModelStandard
from StandardGPT_Training import save_checkpoint as save_standard_checkpoint
from pHGate_Training import GPTModelPHGate, save_checkpoint as save_phgate_checkpoint
from torch.utils.data import DataLoader
import json
import math
from tqdm import tqdm


def setup_logging(log_dir: str):
    """Set up logging for the experiment."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'experiment.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logging.info("Logging setup complete.")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run key experiments for pHGateGPT and StandardGPT.")
    parser.add_argument('--task', type=str, choices=['tokenize', 'train', 'evaluate'], required=True,
                        help="Task to perform: 'tokenize', 'train', or 'evaluate'.")
    parser.add_argument('--model', type=str, choices=['phgate', 'standard'], required=True,
                        help="Model to use: 'phgate' or 'standard'.")
    parser.add_argument('--data_folder', type=str, default='data',
                        help="Path to the folder containing the data.")
    parser.add_argument('--output_file', type=str, default='tokenized_data.json',
                        help="Path to save the tokenized data.")
    parser.add_argument('--log_dir', type=str, default='logs',
                        help="Directory to save logs.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for training and evaluation.")
    parser.add_argument('--num_epochs', type=int, default=3,
                        help="Number of epochs for training.")
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help="Learning rate for training.")
    return parser.parse_args()


def tokenize_data(data_folder, output_file):
    """Tokenize data using BioTokenizerV2."""
    tokenizer = BioTokenizerV2()
    tokenizer.preprocess(data_folder, window_size=1024, stride=512, output_file=output_file)
    logging.info(f"Tokenization complete. Tokenized data saved to {output_file}.")


def train_model(data_file, model_type, batch_size, num_epochs, learning_rate, log_dir):
    """Train the specified model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = TokenizedBioDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize the model
    vocab_size = 50257
    embed_size = 768
    num_heads = 12
    num_layers = 12
    dropout = 0.1
    max_seq_len = 1024
    forward_expansion = 4

    if model_type == 'phgate':
        model = GPTModelPHGate(vocab_size, embed_size, num_heads, num_layers, dropout, max_seq_len, forward_expansion)
        save_checkpoint = save_phgate_checkpoint
    elif model_type == 'standard':
        model = GPTModelStandard(vocab_size, embed_size, num_heads, num_layers, dropout, max_seq_len)
        save_checkpoint = save_standard_checkpoint
    else:
        raise ValueError("Invalid model type. Choose 'phgate' or 'standard'.")

    model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    total_steps = len(dataloader)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            batch = {key: val.to(device) for key, val in batch.items()}
            inputs = batch['input_ids'][:, :-1]
            targets = batch['input_ids'][:, 1:]
            mask = torch.tril(torch.ones(inputs.size(1), inputs.size(1))).unsqueeze(0).unsqueeze(0).to(device)

            optimizer.zero_grad()
            output = model(inputs)
            logits = output["logits"]
            loss = criterion(logits.view(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0 or step == total_steps:
                logging.info(f"Epoch {epoch}, Step {step}/{total_steps}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / total_steps
        logging.info(f"Epoch {epoch} complete. Average Loss: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, filename=os.path.join(log_dir, f'checkpoint_{model_type}_epoch_{epoch}.pt'))


def evaluate_model(data_file, model_type, batch_size, log_dir):
    """Evaluate the specified model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = TokenizedBioDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize the model
    vocab_size = 50257
    embed_size = 768
    num_heads = 12
    num_layers = 12
    dropout = 0.1
    max_seq_len = 1024
    forward_expansion = 4

    if model_type == 'phgate':
        model = GPTModelPHGate(vocab_size, embed_size, num_heads, num_layers, dropout, max_seq_len, forward_expansion)
        checkpoint_path = os.path.join(log_dir, 'checkpoint_phgate_epoch_3.pt')
    elif model_type == 'standard':
        model = GPTModelStandard(vocab_size, embed_size, num_heads, num_layers, dropout, max_seq_len)
        checkpoint_path = os.path.join(log_dir, 'checkpoint_standard_epoch_3.pt')
    else:
        raise ValueError("Invalid model type. Choose 'phgate' or 'standard'.")

    model.to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        logging.error("Checkpoint not found. Please train the model first.")
        return

    # Evaluation loop
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {key: val.to(device) for key, val in batch.items()}
            inputs = batch['input_ids'][:, :-1]
            targets = batch['input_ids'][:, 1:]
            mask = torch.tril(torch.ones(inputs.size(1), inputs.size(1))).unsqueeze(0).unsqueeze(0).to(device)

            output = model(inputs)
            logits = output["logits"]
            loss = criterion(logits.view(-1, vocab_size), targets.reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    logging.info(f"Evaluation complete. Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")


def main():
    args = parse_arguments()
    setup_logging(args.log_dir)

    if args.task == 'tokenize':
        tokenize_data(args.data_folder, args.output_file)
    elif args.task == 'train':
        train_model(args.output_file, args.model, args.batch_size, args.num_epochs, args.learning_rate, args.log_dir)
    elif args.task == 'evaluate':
        evaluate_model(args.output_file, args.model, args.batch_size, args.log_dir)
    else:
        logging.error("Invalid task selected.")


if __name__ == "__main__":
    main()
