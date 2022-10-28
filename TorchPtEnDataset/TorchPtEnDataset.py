from torch.utils.data import Dataset, DataLoader
import os
import re

TEST_FILES = [('IWSLT17.TED.tst2010.br-en.br.xml', 'IWSLT17.TED.tst2010.br-en.en.xml'),
              ('IWSLT17.TED.tst2011.br-en.br.xml', 'IWSLT17.TED.tst2011.br-en.en.xml'),  
              ('IWSLT17.TED.tst2012.br-en.br.xml', 'IWSLT17.TED.tst2012.br-en.en.xml'),  
              ('IWSLT17.TED.tst2013.br-en.br.xml', 'IWSLT17.TED.tst2013.br-en.en.xml'),  
              ('IWSLT17.TED.tst2014.br-en.br.xml', 'IWSLT17.TED.tst2014.br-en.en.xml'),  
              ('IWSLT17.TED.tst2015.br-en.br.xml', 'IWSLT17.TED.tst2015.br-en.en.xml'),  
              ('IWSLT17.TED.tst2016.br-en.br.xml', 'IWSLT17.TED.tst2016.br-en.en.xml'),  
              ('IWSLT17.TED.tst2017.br-en.br.xml', 'IWSLT17.TED.tst2017.br-en.en.xml')]
TRAIN_FILES = [('train.tags.br-en.br', 'train.tags.br-en.en')]

TRAIN_SMALL_FILES = TEST_FILES
TEST_SMALL_FILES = [('IWSLT17.TED.dev2010.br-en.br.xml', 'IWSLT17.TED.dev2010.br-en.en.xml')]

class TorchPtEnDataset(Dataset):

    def __init__(self, path, split='train', transform=None):
        """
        Args:
            path (string): Path to dataset files
            split (string): 'train' or 'test'. 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.corpus = []
        self.transform = transform

        if split == 'train':
            file_list = TRAIN_FILES
        elif split == 'test':
            file_list = TEST_FILES
        elif split == 'train-small':
            file_list = TRAIN_SMALL_FILES
        elif split == 'test-small':
            file_list = TEST_SMALL_FILES

        for pt_file, en_file in file_list:
            pt_path = os.path.join(path, pt_file)
            en_path = os.path.join(path, en_file)
            with open(pt_path, "r") as pt_f, open(en_path) as en_f:
                for pt_line, en_line in zip (pt_f, en_f):
                    pt_text = ''
                    en_text = ''
                    if pt_line.startswith('<seg') and en_line.startswith('<seg'):
                        pt_text = re.sub('<[^<]+>', "", pt_line).strip()
                        en_text = re.sub('<[^<]+>', "", en_line).strip()
                    if not pt_line.startswith('<') and not en_line.startswith('<'):
                        pt_text = pt_line.strip()
                        en_text = en_line.strip()
                    if pt_text != '' and en_text != '':
                        self.corpus.append([pt_text, en_text])

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.corpus[idx]
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        #sample = self.corpus.iloc[[idx]]
        #if self.transform:
        #    sample = self.transform(sample)
        #sample = list(sample.itertuples(index=False, name=None))
        #return [val for sublist in sample for val in sublist]
