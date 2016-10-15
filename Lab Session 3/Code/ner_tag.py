import numpy as np
from scipy.optimize import minimize as mymin
from my_maxent import MyMaxEnt
import feature_functions as f_fn

hist_list = eval(open('history.txt').read())
feature_fn_list = [f_fn.f1,f_fn.f2,f_fn.f3,f_fn.f4,f_fn.f5,f_fn.f6,f_fn.f7,f_fn.f8,f_fn.f9,f_fn.f10,f_fn.f11]

ner_tagger = MyMaxEnt(hist_list,feature_fn_list)
ner_tagger.train()
print ner_tagger.model
# ner_tagger.save("ner_iphone5.pickle")
# ner_tagger = ner_tagger.load("ner_iphone5.pickle")
h = ('OTHER', 'OTHER', ('Apple', 'iPhone', '5', ':', 'First', 'look', 'As', 'I', 'played', 'around', 'with', 'the', 'iPhone', '5', 'on', 'Wednesday', ',', 'I', 'wondered', 'what', 'the', 'late', 'Steve', 'Jobs', 'would', 'have', 'thought', 'about', 'the', 'latest', 'twist', 'on', 'Apple', "'s", 'best-selling', 'device.It', 'did', "n't", 'take', 'long', 'to', 'conclude', 'Jobs', 'would', 'have', 'been', 'delighted', 'with', 'the', 'iPhone', '5', "'s", 'blend', 'of', 'beauty', ',', 'utility', 'and', 'versatility.Add', 'in', 'the', 'more', 'advanced', 'technology', 'and', 'new', 'features', 'that', 'went', 'into', 'this', 'iPhone', ',', 'and', 'it', "'s", 'clear', 'Apple', 'has', 'come', 'up', 'with', 'another', 'product', 'that', 'will', 'compel', 'hordes', 'of', 'people', 'to', 'line', 'up', 'outside', 'its', 'stores', 'before', 'its', 'Sept.', '21', 'release', 'in', 'the', 'U.S.', ',', 'Japan', ',', 'Britain', ',', 'Germany', ',', 'France', 'and', 'four', 'other', 'countries', '.'), 77, 'GPE')
print h[2][77]
print ner_tagger.classify(h)