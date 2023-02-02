import pandas as pd
import os
import sys

if __name__ == '__main__':

	args = sys.argv[1:]
	task_dir = args[0]
	out_dir = args[1]

	listing = os.listdir(task_dir)

	vocab = set()
	for f in listing:
		df = pd.read_csv(os.path.join(task_dir,f))
		df.dropna(inplace=True)
		for idx in range(len(df)):
			w1 = df.word1.iloc[idx]
			w2 = df.word2.iloc[idx]
			w1 = w1.split('-')[0]
			w2 = w2.split('-')[0]
			vocab.add(w1)
			vocab.add(w2)
	
	print(len(vocab),' unique words found')

	with open(os.path.join(out_dir, 'sim_vocab.txt'), 'w') as outf:
		for v in vocab:
			outf.write(v+'\n')