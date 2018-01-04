import random
import numpy as np
import h5py

# from dec (https://github.com/piiswrong/dec/tree/master/dec)
def save_hdf5(X, y, name):
    import h5py
    with h5py.File('./{}.h5'.format(name), 'w') as f:
        f['data'] = X
        f['labels'] = y

def make_reuters_data():
  np.random.seed(1234)
  random.seed(1234)
  from sklearn.feature_extraction.text import CountVectorizer
  did_to_cat = {}
  cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
  with open('../reuters/rcv1-v2.topics.qrels') as fin:
    for line in fin.readlines():
      line = line.strip().split(' ')
      cat = line[0]
      did = int(line[1])
      if cat in cat_list:
        did_to_cat[did] = did_to_cat.get(did, []) + [cat]
    for did in list(did_to_cat):
      if len(did_to_cat[did]) > 1:
        del did_to_cat[did]

  dat_list = ['lyrl2004_tokens_test_pt0.dat',
              'lyrl2004_tokens_test_pt1.dat',
              'lyrl2004_tokens_test_pt2.dat',
              'lyrl2004_tokens_test_pt3.dat',
              'lyrl2004_tokens_train.dat']
  data = []
  target = []
  cat_to_cid = {'CCAT':0, 'GCAT':1, 'MCAT':2, 'ECAT':3}
  del did
  for dat in dat_list:
    with open('../reuters/'+dat) as fin:
      for line in fin.readlines():
        if line.startswith('.I'):
          if 'did' in locals():
            assert doc != ''
            if did in did_to_cat:
              data.append(doc)
              target.append(cat_to_cid[did_to_cat[did][0]])
          did = int(line.strip().split(' ')[1])
          doc = ''
        elif line.startswith('.W'):
          assert doc == ''
        else:
          doc += line

  assert len(data) == len(did_to_cat)

  X = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
  Y = np.asarray(target)

  from sklearn.feature_extraction.text import TfidfTransformer
  X = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(X)
  X = np.asarray(X.todense())*np.sqrt(X.shape[1])

  p = np.random.permutation(X.shape[0])
  X = X[p]
  Y = Y[p]

  N = X.shape[0]
  save_hdf5(X[:N], Y[:N], 'reutersidf_train')
  save_hdf5(X[int(N*4/5):N], Y[int(N*4/5):N], 'reutersidf_test')
  save_hdf5(X[:N], Y[:N], 'reutersidf_total')

if __name__ == '__main__':
    make_reuters_data()
