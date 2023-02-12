import pandas as pd
import os
import sys
from gensim.models import KeyedVectors
if __name__ == '__main__':

	args = sys.argv[1:]
	
	if len(args) == 3:

		task_dir = args[0]
		embeddings_dir = args[1]
		out_dir = args[2]

		embeddings_listing = os.listdir(embeddings_dir)

		results_rows = []
		for embeddings_file in embeddings_listing:
			embeddings_file = os.path.join(embeddings_dir, embeddings_file)
			print('Processing vectors: ',embeddings_file)

			# load embeddings
			try:
				model = KeyedVectors.load_word2vec_format(embeddings_file, binary=True)
			except Exception as e:
				print('Error 1: ',e)
				try:
					model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
				except Exception as e:
					print('Error 2: ',e)
					sys.exit('cant load embeddings')

			datarows = []
			listing = os.listdir(task_dir)
			# build dataset for target task (atm only tested for sim)
			for f in listing:
				df = pd.read_csv(os.path.join(task_dir,f))
				df.dropna(inplace=True)
				for idx in range(len(df)):
					# get ground truth similarity
					sim = df.similarity.iloc[idx]
					# get words and strip PoS and any suffixed metadata
					w1 = df.word1.iloc[idx]
					w2 = df.word2.iloc[idx]
					w1 = w1.split('-')[0]
					w2 = w2.split('-')[0]
					if w1 in model and w2 in model:
						datarows.append({
							'dataset':f,
							'w1':w1,
							'w2':w2,
							'similarity':sim
							})
			
			df = pd.DataFrame(datarows)

			methods = {'pearson', 'kendall', 'spearman'}
			embeddings_filename = embeddings_file.split('/')[1]
			for f in listing:
				thisdf = df[df.dataset==f]
				#gold_sims = thisdf.similarity.values.tolist()
				pred_sims = thisdf.apply(lambda x: model.similarity(x.w1, x.w2), axis=1)
				thisdf['pred_sims'] = pred_sims
				for method in methods:
					results_rows.append({
						'dataset':f,
						'corr-method':method,
						'embeddings':embeddings_filename,
						'value': thisdf.similarity.corr(thisdf.pred_sims, method=method)
						})

		resdf = pd.DataFrame(results_rows)
		task_name = task_dir.replace('/','_')
		outfilename = f'results__task={task_name}.csv'
		resdf.to_csv(os.path.join(out_dir,outfilename))