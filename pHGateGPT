import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
from collections import Counter

class TokenizedBioDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.label2id = {
            "pH_enzyme": 0,
            "control": 1,
            "novel": 2
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item['tokens']
        label = self.label2id[item['label']]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (padded_input_ids != 0).long()

    return {
        'input_ids': padded_input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def main():
    dataset_path = 'tokenized_data.json'
    dataset = TokenizedBioDataset(dataset_path)

    all_labels = [int(item['label']) for item in dataset]
    label_counts = Counter(all_labels)
    print("Label distribution:", label_counts)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        print("input_ids:", batch['input_ids'].shape)
        print("attention_mask:", batch['attention_mask'].shape)
        print("labels:", batch['labels'])
        break

    torch.save(dataset, "bio_dataset.pt")

if __name__ == '__main__':
    main()
