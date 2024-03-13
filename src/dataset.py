from pathlib import Path
import numpy as np
import pandas as pd
import re
import datetime

# NOTE: Regex taken from RAPID paper original implementation

class Hdfs:
    def __init__(self, file_path: Path, labels_path: Path):
        self.data = file_path.read_text().splitlines()
        self.df = self.clean_parse_dataset()
        self.add_labels(labels_path)

    def regex_sub_line(self, line: str):
        """
        @returns block_id, cleaned_line
        """
        blk_regex = re.compile('blk_-?\d+')
        block_id = re.findall(blk_regex, line)[0]

        ip_regex = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
        num_regex = re.compile("\d*\d")

        line = re.sub(blk_regex, 'BLK', line)
        line = re.sub(ip_regex, 'IP', line)
        line = re.sub(num_regex, 'NUM', line)

        return block_id, line

    def clean_parse_dataset(self):
        cleaned_logs = []
        block_ids = []
        for line in self.data:
            block_id, line = self.regex_sub_line(line)
            block_ids.append(block_id)
            cleaned_logs.append(line)
        return pd.DataFrame({'line': cleaned_logs, 'block_id': block_ids})

    def add_labels(self, labels_path: Path):
        true_labels = pd.read_csv(labels_path)
        true_labels['Label'] = (true_labels['Label'] == 'Anomaly').astype('int64')
        self.df = pd.merge(self.df, true_labels, left_on='block_id', right_on='BlockId').drop(columns=['BlockId']).rename(columns={'Label': 'is_anomaly'})

    def save(self, output_path):
        """
        Exports to csv with gzip compression
        """
        self.df.to_csv(output_path, compression='gzip')

    def train_test_split(self, train_size = 446_578):
        df = self.df

        df = df.groupby(by='block_id')['line'].unique()
        df = df.apply(lambda x: ' '.join(x))
        train = df[df['is_anomaly'] == 0].iloc[:train_size]
        train_X = train['line']

        test = pd.concat([df[df['is_anomaly'] == 0].iloc[train_size:], df[df['is_anomaly'] == 1]])
        test_X = test['line']
        test_y = test['is_anomaly']

        return train_X, test_X, test_y

class BGL:
    def __init__(self, file_path: Path):
        self.data = file_path.read_text().splitlines()
        self.df = self.clean_parse_dataset()

    def regex_sub_line(self, line: str):
        """
        @returns is_alert, cleaned_line
        """
        is_anomaly = re.search('^- ', line) is None
        date_time_regex = re.compile(
            "\d{1,4}\-\d{1,2}\-\d{1,2}-\d{1,2}.\d{1,2}.\d{1,2}.\d{1,6}"
        )
        date_regex = re.compile("\d{1,4}\.\d{1,2}\.\d{1,2}")
        ip_regex = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
        server_regex = re.compile("\S+(?=.*[0-9])(?=.*[a-zA-Z])(?=[:]+)\S+")
        server_regex2 = re.compile("\S+(?=.*[0-9])(?=.*[a-zA-Z])(?=[-])\S+")
        ecid_regex = re.compile("[A-Z0-9]{28}")
        serial_regex = re.compile("[a-zA-Z0-9]{48}")
        memory_regex = re.compile("0[xX][0-9a-fA-F]\S+")
        path_regex = re.compile(".\S+(?=.[0-9a-zA-Z])(?=[/]).\S+")
        iar_regex = re.compile("[0-9a-fA-F]{8}")
        num_regex = re.compile("(\d+)")
        
        timestamp = datetime.strptime(re.findall(date_time_regex, line)[0], '%Y-%m-%d-%H.%M.%S.%f')

        line = re.sub(date_time_regex, 'DT', line)
        line = re.sub(date_regex, 'DATE', line)
        line = re.sub(ip_regex, 'IP', line)
        line = re.sub(server_regex, 'NODE', line)
        line = re.sub(server_regex2, 'NODE', line)
        line = re.sub(ecid_regex, 'ECID', line)
        line = re.sub(serial_regex, 'SERIAL', line)
        line = re.sub(memory_regex, 'MEM', line)
        line = re.sub(path_regex, 'PATH', line)
        line = re.sub(iar_regex, 'IAR', line)
        line = re.sub(num_regex, 'NUM', line)

        return is_anomaly, timestamp, line

    def clean_parse_dataset(self):
        cleaned_logs = []
        labels = []
        timestamps = []
        for line in self.data:
            is_anomaly, timestamp, line = self.regex_sub_line(line)
            labels.append(is_anomaly)
            timestamps.append(timestamp)
            cleaned_logs.append(line)
        return pd.DataFrame({'line': cleaned_logs, 'timestamp': timestamps, 'is_anomaly': labels})

    def save(self, output_path):
        """
        Exports to csv with gzip compression
        """
        self.df.to_csv(output_path, compression='gzip')

    def train_test_split(self, train_size = 3_519_602, test_normal_size = 879_910, test_abnormal_size = 348_460):
        df = self.df
        train = df[df['is_anomaly'] == 0].iloc[:train_size]
        train_X = train['line']

        test = pd.concat([df[df['is_anomaly'] == 0].iloc[train_size:train_size + test_normal_size], df[df['is_anomaly'] == 1].iloc[:test_abnormal_size]])
        test_X = test['line']
        test_y = test['is_anomaly']

        return train_X, test_X, test_y

class ThunderBird:
    def __init__(self, file_path: Path):
        self.data = file_path.read_text().splitlines()
        self.df = self.clean_parse_dataset()

    def regex_sub_line(self, line: str):
        """
        @returns is_alert, cleaned_line
        """
        is_anomaly = re.search('^- ', line) is None
        date_regex = re.compile("\d{2,4}\.\d{1,2}\.\d{1,2}\s")
        date_regex2 = re.compile(
            "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+"
        )
        time_regex = re.compile("\d{1,2}\:\d{1,2}\:\d{1,2}")
        id_regex = re.compile(r"DATE\s.*\sDATE")

        account_regex = re.compile("(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)")
        account_regex2 = re.compile("(\w+[\w\.]*)@(\w+[\w\.]*)")
        account_regex3 = re.compile(r"TIME\s\S+")

        dir_regex = re.compile(r'[a-zA-Z0-9_\-\.\/]+\/[a-zA-Z0-9_\-\.\/]+\/[a-zA-Z0-9_\-\.\/]*')
        dir_regex2 = re.compile(r'\/[a-zA-Z0-9_\-\.\/]+\/[a-zA-Z0-9_\-\.\/]*')
        iar_regex = re.compile("[0-9a-fA-F]{10}")
        ip_regex = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
        num_regex = re.compile("(\[\d+\])")

        date_time_str = re.findall(date_regex, line)[0] + " " + re.findall(time_regex, line)[0]
        timestamp = datetime.strptime(date_time_str,'%Y.%m.%d %H:%M:%S')
        
        line = re.sub(date_regex, 'DATE', line)
        line = re.sub(date_regex2, 'DATE', line)
        line = re.sub(time_regex, 'TIME', line)
        line = re.sub(id_regex, 'ID', line)
        line = re.sub(account_regex, 'ACC', line)
        line = re.sub(account_regex2, 'ACC', line)
        line = re.sub(account_regex3, 'ACC', line)
        line = re.sub(dir_regex, 'DIR', line)
        line = re.sub(dir_regex2, 'DIR', line)
        line = re.sub(iar_regex, 'IAR', line)
        line = re.sub(ip_regex, 'IP', line)
        line = re.sub(num_regex, 'NUM', line)

        return is_anomaly, timestamp, line

    def clean_parse_dataset(self):
        cleaned_logs = []
        labels = []
        timestamps = []
        for line in self.data:
            is_anomaly, timestamp, line = self.regex_sub_line(line)
            labels.append(is_anomaly)
            timestamps.append(timestamp)
            cleaned_logs.append(line)
        return pd.DataFrame({'line': cleaned_logs, 'timestamp': timestamps, 'is_anomaly': labels})

    def save(self, output_path):
        """
        Exports to csv with gzip compression
        """
        self.df.to_csv(output_path, compression='gzip')

    def train_test_split(self, train_size = 3_938_483, test_normal_size = 984_621, test_abnormal_size = 76_895):
        df = self.df
        train = df[df['is_anomaly'] == 0].iloc[:train_size]
        train_X = train['line']

        test = pd.concat([df[df['is_anomaly'] == 0].iloc[train_size:train_size + test_normal_size], df[df['is_anomaly'] == 1].iloc[:test_abnormal_size]])
        test_X = test['line']
        test_y = test['is_anomaly']

        return train_X, test_X, test_y