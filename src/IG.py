import numpy as np
import math

from sklearn.pipeline import FeatureUnion
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
class IG:
    def __init__(self,X,classes,threshold=0.25):
        self.X = X
        self.classes = classes
        self.threshold = threshold
    def calulate_ig(self,X,classes):
        ig = []
        for i in range(len(X[0])):
            feature_items = sorted([x[i] for x in X])

            # feature_divide = (max(feature_items) - min(feature_items))/
            feature_divide = (feature_items[0]+feature_items[1])/2
            feature_items = [((x[i]-min(feature_items))//feature_divide)*(feature_divide) + feature_divide for x in X]


            ig.append(self.ig(self.classes,feature_items ))
        return ig

    def ig(self,class_, feature):
        classes = set(class_)
        print(classes)

        Hc = 0
        for c in classes:
            pc = list(class_).count(c)/len(class_)
            Hc += - pc * math.log(pc, 2)
        print('Overall Entropy:', Hc)
        feature_values = set(feature)

        Hc_feature = 0
        for feat in feature_values:
            pf = list(feature).count(feat)/len(feature)
            indices = [i for i in range(len(feature)) if feature[i] == feat]
            clasess_of_feat = [class_[i] for i in indices]
            
            for c in classes:
                pcf = clasess_of_feat.count(c)/len(clasess_of_feat)
                if pcf != 0:
                
                    temp_H = - pf * pcf * math.log(pcf, 2)
                    Hc_feature += temp_H
            
        ig = Hc - Hc_feature
        return ig