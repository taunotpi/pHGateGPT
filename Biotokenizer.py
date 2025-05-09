import os
import json
import gzip
from transformers import GPT2Tokenizer
from bs4 import BeautifulSoup


class BioTokenizerV2:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.eos_token = "<|endoftext|>"

    def sliding_window(self, tokens, window_size, stride):
        windows = []
        for i in range(0, len(tokens) - window_size + 1, stride):
            window = tokens[i: i + window_size]
            windows.append(window)
        return windows

    def process_fasta(self, filepath):
        sequences = []
        with open(filepath, "r") as f:
            current_sequence = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_sequence:
                        sequences.append("".join(current_sequence))
                        current_sequence = []
                else:
                    current_sequence.append(line)
            if current_sequence:
                sequences.append("".join(current_sequence))
        return sequences

    def process_text(self, filepath, encoding="utf-8"):
        with open(filepath, "r", encoding=encoding) as f:
            text = f.read()
        paragraphs = text.split("\n\n")
        return paragraphs

    def process_html(self, filepath):
        paragraphs = []
        with open(filepath, "r", encoding="utf-8") as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        paragraphs = text.split("\n\n")
        return paragraphs

    def preprocess(self, data_folder, window_size=1024, stride=512, output_file="tokenized_data.json"):
        data_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(data_folder)
            for file in files if file.endswith((".txt", ".gz", ".fasta", ".html"))
        ]

        print(f"Found {len(data_files)} files to process.")

        tokenized_data = []
        for file in data_files:
            label = self.get_label_from_filename(file)
            print(f"Processing file: {file} with label: {label}")
            try:
                if file.endswith(".fasta"):
                    sequences = self.process_fasta(file)
                    for seq in sequences:
                        tokenized_seq = self.tokenizer.encode(seq, add_special_tokens=True)
                        windows = self.sliding_window(tokenized_seq, window_size, stride)
                        for w in windows:
                            if len(w) == window_size:
                                tokenized_data.append({"tokens": w, "label": label})
                elif file.endswith(".txt"):
                    paragraphs = self.process_text(file)
                    for para in paragraphs:
                        tokenized_text = self.tokenizer.encode(para, add_special_tokens=True)
                        windows = self.sliding_window(tokenized_text, window_size, stride)
                        for w in windows:
                            if len(w) == window_size:
                                tokenized_data.append({"tokens": w, "label": label})
                elif file.endswith(".gz"):
                    paragraphs = self.process_gz(file)
                    for para in paragraphs:
                        tokenized_text = self.tokenizer.encode(para, add_special_tokens=True)
                        windows = self.sliding_window(tokenized_text, window_size, stride)
                        for w in windows:
                            if len(w) == window_size:
                                tokenized_data.append({"tokens": w, "label": label})
                elif file.endswith(".html"):
                    paragraphs = self.process_html(file)
                    for para in paragraphs:
                        tokenized_text = self.tokenizer.encode(para, add_special_tokens=True)
                        windows = self.sliding_window(tokenized_text, window_size, stride)
                        for w in windows:
                            if len(w) == window_size:
                                tokenized_data.append({"tokens": w, "label": label})
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        with open(output_file, 'w') as out_f:
            json.dump(tokenized_data, out_f)
        print(f"Tokenized data with labels saved to {output_file} with {len(tokenized_data)} entries.")

    def get_label_from_filename(self, filename):
        if "vatpase" in filename or "cathepsin_L" in filename or "NHE9" in filename:
            return "pH_enzyme"
        elif "control" in filename:
            return "control"
        else:
            return "novel"

    def encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=True)

    def decode(self, list_of_ids):
        return self.tokenizer.decode(list_of_ids)


if __name__ == "__main__":
    tokenizer = BioTokenizerV2()
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenized_data.json")
    tokenizer.preprocess(data_folder, window_size=1024, stride=512, output_file=output_file)
