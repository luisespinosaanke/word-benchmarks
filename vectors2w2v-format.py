import pickle
import sys
import os
import numpy as np

if __name__ == '__main__':

	args = sys.argv[1:]

	if len(args) == 3:
		
		inp_file = args[0]
		dims = int(args[1])
		out_file = args[2]

		nb_vecs = 0
		is_pickle = False
		if inp_file.endswith('pkl'):
			is_pickle = True
			vectors = pickle.load(open(inp_file, 'rb'))
			for _ in vectors:
				nb_vecs += 1
		else:
			for _ in open(inp_file):
				nb_vecs += 1

		with open(out_file,'w') as outf:
			outf.write(f'{nb_vecs} {dims}\n')
			if is_pickle:
				for w,v in vectors.items():
					w = '_'.join(w.split())
					vstr = ' '.join([str(k) for k in v])
					outf.write(f'{w} {vstr}\n')
			else:
				for line in open(inp_file):
					# do not deal with mwes, lets assume they are single-tokenized already
					outf.write(line)
