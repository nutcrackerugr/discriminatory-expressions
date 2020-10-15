import numpy as np
import pandas as pd
import string
import swifter

from collections import Counter, defaultdict, deque
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from scipy.sparse import lil_matrix

from joblib import Parallel, delayed
import multiprocessing as mp


class DEVectorizer(BaseEstimator, TransformerMixin):
	def __init__(self, tokenizer=TweetTokenizer(), stemmer=SnowballStemmer("spanish"),
		sent_tokenizer=sent_tokenize, stopw=None, include_stopw=False, punct=None,
		min_cfidf=0, max_depth=3, min_precision=.9, min_recall=.001, recall_percentile=1,
		export=None, n_jobs=1, uniclass=False, max_features=1000, boolean_features=True
		):
		self.vocabulary = None
		self.docscontainingword = None
		self.maxs = None
		self.dlen = None
		self.exprs = None

		# Params
		self._recalls = None
		self._class_size = None
		self.tokenizer = tokenizer
		self.stemmer = stemmer
		self.sent_tokenizer = sent_tokenizer
		self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
		self._CHUNKS_PER_THREAD = 1
		self.uniclass = uniclass
		self.include_stopw = include_stopw
		self._boolean_features = boolean_features

		self.stopw = stopw
		self.punct = punct
		self.export = export

		self.min_cficf = min_cfidf
		self.max_depth = max_depth
		self.min_precision = min_precision
		self.min_recall = min_recall
		self.recall_percentile = recall_percentile
		self.max_features = max_features

		if not self.stopw:
			self.stopw = set(stopwords.words("spanish"))
			self.stopw.update(['q', 't', 's', 'h', "rt", "via", "vía", "tt"])

		if not self.punct:
			self.punct = set((p for p in string.punctuation))
			self.punct.update(['¡', '¿', "...", '…', '“', '”', '©', '×', '–', '’', '−', '‘'])
	
	def get_params(self, **kwargs):
		return {
			"max_depth": self.max_depth, "min_precision": self.min_precision,
			"min_recall": self.min_recall, "stopw": self.stopw,
			"recall_percentile": self.recall_percentile,
			"min_cfidf": self.min_cficf, "tokenizer": self.tokenizer,
			"punct": self.punct, "n_jobs": self.n_jobs,
			"uniclass": self.uniclass, "include_stopw": self.include_stopw,
			"sent_tokenizer": self.sent_tokenizer, "stemmer": self.stemmer,
			"max_features": self.max_features
		}
	
	def _fit_chunk(self, X, y, chunk):
		# Note: recall_targets may need to be copied or protected. If threads
		# share the reference, popleft() will misbehave. Thats why we compute
		# it here rather than in self.fit()
		recall_targets = deque(np.percentile(self._recalls.unique(), [60, 40, 30, 15, 5]))
		
		de = defaultdict(list)
		de_list = []
		recalls = []
		count_exprs = 0
		limit_exprs = self.max_features / (self.n_jobs * self._CHUNKS_PER_THREAD)

		# Compute expression for each row
		remaining_docs = [i for i in X.index.values if i in chunk] # In order to process them sorted by cficf
		
		if self.uniclass:
			remaining_docs = [i for i in remaining_docs if y[i]]
		
		next_docs = []
		trecall = recall_targets.popleft()
		
		while remaining_docs and trecall and count_exprs <= limit_exprs:
			for i in remaining_docs:
				expr, recall = self._calcde(i, X, y, trecall)

				# Save expression in de_list for uniqueness check
				# Save expression and recall in a per-class dict for feature selection
				# Save recall to compute percentiles
				if expr:
					if not self._isin(expr, de_list):
						de_list.append(expr)
						de[y.loc[i]].append((expr, recall))
						recalls.append(recall)
						count_exprs += 1
				else:
					next_docs.append(i)

			remaining_docs = next_docs
			next_docs = []

			trecall = recall_targets.popleft() if recall_targets else None

		return de

	def indexChunks(self, l, n):
		size = len(l) / n
		last = 0

		while last < len(l):
			yield l[int(last):int(last + size)]
			last += size

	def fit(self, X, y=None, **kwargs):
		if self.export:
			export_file = open(self.export, "w+")
			
		# Tokenize and compute variables for cfidf
		text_nostopw = X.str.lower().map(lambda abstract: [[self.stemmer.stem(token) for token in self.tokenizer.tokenize(sentence) if token not in self.stopw and token not in self.punct] for sentence in self.sent_tokenizer(abstract)])
		text_nostopw = text_nostopw.sample(frac=1) # Random shuffle

		self._class_size = y.value_counts()
		self._computeFrequencies(text_nostopw, y)

		self.maxs = {cname:counter.most_common(1)[0][1] for cname, counter in self.vocabulary.items()}
		self.dlen = {cname:np.sum(y != cname) for cname in self.vocabulary.keys()}

		# Order by cficf
		sorted_text = pd.concat([text_nostopw.rename("X"), y.rename("y")], axis="columns")
		sorted_text["cficf"] = sorted_text.apply(lambda document: np.sum([self._cficf(token, document.y) for sentence in document.X for token in sentence]), axis="columns")
		sorted_text = sorted_text.sort_values("cficf", ascending=False)

		# Compute target recalls
		wcounts = pd.DataFrame(self.vocabulary).fillna(0)

		divisors = y.value_counts()
		wcounts = wcounts.divide(divisors)

		self._recalls = wcounts.max(axis=1)

		# Compute expression for each row
		if self.n_jobs and self.n_jobs != 1:
			n_chunks = self.n_jobs * self._CHUNKS_PER_THREAD
			dicts = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_chunk)(sorted_text.X, y, chunk) for chunk in self.indexChunks(list(text_nostopw.index), n_chunks))
		else:
			dicts = [self._fit_chunk(sorted_text.X, y, list(text_nostopw.index))]

		# Initialization
		self.exprs = defaultdict(list)
		
		de_list = []
		de = defaultdict(list)
		recalls = []
		
		for job_dict in dicts:
			for cname, er_list in job_dict.items():
				for er in er_list:
					expr, recall = er

					if not self._isin(expr, de_list):
						de_list.append(expr)
						de[cname].append(er)
						recalls.append(recall)

						if self.export:
							export_file.write("{}\t{}\t{}\n".format(expr, cname, recall))

		del dicts

		# Compute percentile
		p = np.percentile(recalls, self.recall_percentile, interpolation="lower")

		# Save expressions with recall >= percentile
		for cname, clist in de.items():
			for expr, recall in clist:
				if recall >= p:
					self.exprs[cname].append(expr)

		if self.export:
			export_file.close()

		return self
	
	def transform(self, X, y=None, **kwargs):
		text_nostopw = X.str.lower().map(lambda abstract: [[self.stemmer.stem(token) for token in self.tokenizer.tokenize(sentence) if token not in self.stopw and token not in self.punct] for sentence in self.sent_tokenizer(abstract)])

		n = sum([len(cfeatures) for _, cfeatures in self.exprs.items()])

		if self._boolean_features:
			transformedX = np.zeros((len(X), n)).astype("bool")
			
			for i_text in range(len(text_nostopw)):
				for sentence in text_nostopw[i_text]:
					sentence_features = np.zeros(n)

					i_prev = 0
					for cname, cfeatures in self.exprs.items():
						for i_feature in range(len(cfeatures)):
							sentence_features[i_prev + i_feature] = self._isexpr(cfeatures[i_feature], sentence)

						i_prev += len(cfeatures)

					transformedX[i_text,] = np.logical_or(transformedX[i_text,], sentence_features)
		else:
			transformedX = np.zeros((len(X), n))
			
			for i_text in range(len(text_nostopw)):
				for sentence in text_nostopw[i_text]:
					sentence_features = np.zeros(n)

					i_prev = 0
					for cname, cfeatures in self.exprs.items():
						for i_feature in range(len(cfeatures)):
							sentence_features[i_prev + i_feature] += self._isexpr(cfeatures[i_feature], sentence) * (1 if cname else -1)

						i_prev += len(cfeatures)
					
					transformedX[i_text,] = transformedX[i_text,] + sentence_features

		return transformedX


	def _calcde(self, i, text, y, trecall):
		# Get target recall for this iteration
		trecall = max(self.min_recall, trecall)
		cname = y.loc[i]

		for sentence in text.loc[i]:
			# Add words to queue in relevance order
			rowrank = self._getimportancelist(sentence, cname)
			opened = deque([word] for word, rank in rowrank if rank >= self.min_cficf and word not in self.stopw and word not in self.punct and self._recalls[word] >= trecall)

			de = None
			de_recall = None
			while opened and not de:
				expr = opened.popleft()

				# Compute precision and recall
				tp = text.loc[y == cname].apply(lambda x: self._isexpr(expr, x)).sum()
				fp = text.loc[y != cname].apply(lambda x: self._isexpr(expr, x)).sum()

				recall = tp / self._class_size[cname]
				precision = tp / (tp + fp)

				# If expr's recall is higher than the target recall, accept the expr
				if recall >= trecall:
					exprset = set(expr)
					if precision >= self.min_precision and not exprset.issubset(self.stopw) and not exprset.issubset(self.punct):
						de = expr
						de_recall = recall
					else:
						if len(expr) < self.max_depth:
							append_opened = [sorted(expr + [word], key=lambda x: sentence.index(x)) for word, rank in rowrank if word not in expr and (self.include_stopw or word not in self.stopw) and word not in self.punct and rank >= self.min_cficf and (word in self.stopw or self._recalls[word] >= trecall)]
							opened.extendleft(append_opened)

			# Stop when found an expr for the doc. Awful...
			if de:
				break
			
		return de, de_recall

	def _isexpr(self, expr, text):
		isexpr = False
		
		for words in text:
			last_index = -1
			status = True
			for w in expr:
				try:
					new_index = words.index(w)

					if last_index >= new_index:
						status = False
						break
					else:
						last_index = new_index
						
				except ValueError:
					status = False
					break

			if status:
				isexpr = True
				break

		return isexpr and not set(expr).issubset(self.stopw)

	def _computeFrequencies(self, X, y, **kwargs):
		self.vocabulary = defaultdict(Counter)
		self.docscontainingword = defaultdict(Counter)
		
		def count(row):
			unique = set()
			
			for sentence in row.X:
				self.vocabulary[row.y].update(sentence)
				unique.update(sentence)

			self.docscontainingword[row.y].update(unique)

		pd.concat([X.rename("X"), y.rename("y")], axis="columns").apply(count, axis="columns")

	def _cf(self, word, cname):
		return self.vocabulary[cname].get(word) or 0 / self.maxs[cname]

	def _icf(self, word, cname):
		doc_count = 0
		
		for key in self.vocabulary.keys():
			if key != cname:
				if word in self.docscontainingword[key].keys():
					doc_count += self.docscontainingword[key][word]
		
		return self.dlen[cname] / (1 + doc_count)
