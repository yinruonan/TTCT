import SimpleITK as sitk
import numpy as np 

class MMetric:
    
    def __init__(self) :
        self.IOU = 0.
        self.DICE = 0.
        self.Recall = 0.
        self.Precision = 0.
        self.Fmeasure = 0.
        

    def update(self, Ps, GTs):
        sitk_GTs = sitk.GetImageFromArray(GTs.astype(np.int8))
        sitk_PVs = sitk.GetImageFromArray((Ps > 0.5).astype(np.int8))

        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(sitk_PVs, sitk_GTs)
        Fp = overlap_measures_filter.GetFalsePositiveError()
        Fn = overlap_measures_filter.GetFalseNegativeError()
        Tp = 1 - Fn
        self.IOU = overlap_measures_filter.GetJaccardCoefficient()
        self.DICE = overlap_measures_filter.GetDiceCoefficient()
        self.Recall = Tp / (Tp + Fn + 1e-10)
        self.Precision = Tp / (Tp + Fp + 1e-10)
        self.Fmeasure = (2 * Tp) / (2 * Tp + Fp + Fn + 1e-10)
        
    def iou(self):
        return self.IOU
    
    def dice(self):
        return self.DICE
    
    def recall(self):
        return self.Recall
    
    def precision(self):
        return self.Precision
    
    def f1score(self):
        return self.Fmeasure
