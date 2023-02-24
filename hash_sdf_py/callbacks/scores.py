
import torchnet
import numpy as np
import torch
import csv

            
class Scores():
    def __init__(self):
        # self.TPs=None
        # self.FPs=None
        # self.FNs=None
        # self.Total=None

        #attempt 2
        self.clear()


        # self.logger_iou = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_iou'}, port=port, env=node_name)

    #adapted from https://github.com/NVlabs/splatnet/blob/f7e8ca1eb16f6e1d528934c3df660bfaaf2d7f3b/splatnet/semseg3d/eval_seg.py
    def accumulate_scores(self, pred_softmax, gt, unlabeled_idx):
        # print("pred_softmax has shape, ", pred_softmax.shape)
        self.nr_classes=pred_softmax.shape[1]
        # self.nr_classes=nr_classes-1 #we dont take the background class into considertation for the evaluation
        # print("computing and accumulating iou")
        # print("pred_softmax after squeeze has shape, ", pred_softmax.shape)
        pred=pred_softmax.argmax(1)
        # print("pred has shape, ", pred.shape)
        gt=gt.detach()
        pred=pred.detach()

        # self.labels = np.unique(gt)
        self.labels = torch.unique(gt) # vector containing an index of the classes that are in the cloud. The order of them is not defined
        # assert np.all([(v in self.labels) for v in np.unique(pred)])

        #instantiate the tps, fps, and so one to the nr of classes we have
        # if( self.TPs==None):
        #     self.TPs = [0] * self.nr_classes
        #     self.FPs = [0] * self.nr_classes
        #     self.FNs = [0] * self.nr_classes
        #     self.Total = [0] * self.nr_classes

        if( self.intersection_per_class==None):
            self.intersection_per_class = [0] * self.nr_classes
            self.union_per_class = [0] * self.nr_classes

        for i in range(len(self.labels)):
            l=self.labels[i]
            if not l==unlabeled_idx:
                # print("accumulating statistics for class ", l)
                # self.TPs[i]+=sum((gt == l) * (pred == l))
                # self.FPs[i]+=sum((gt != l) * (pred == l))
                # self.FNs[i]+=sum((gt == l) * (pred != l))
                # self.Total[i]+=sum(gt == l)

                #attempt 2
                # self.TPs[l]+=((gt == l) * (pred == l)).sum().item()
                # self.FPs[l]+=((gt != l) * (pred == l)).sum().item()
                # self.FNs[l]+=((gt == l) * (pred != l)).sum().item()
                # self.Total[l]+=(gt == l).sum().item()

                #attempt 3
                current_intersection=((pred==gt)*(gt==l)).sum().item()
                self.intersection_per_class[l]+= current_intersection
                self.union_per_class[l]+=  (gt==l).sum().item() + (pred==l).sum().item()  -  current_intersection

    def compute_stats(self, print_per_class_iou=False):
        valid_classes=0
        iou_sum=0

        # for i in range(self.nr_classes):
        #     if( not self.Total[i]==0 ):
        #         valid_classes+=1
        #         iou=self.TPs[i] / (self.TPs[i] + self.FNs[i] + self.FPs[i])
        #         iou_sum+=iou
        #         if print_per_class_iou:
        #             print("class iou for idx", i, " is ", iou )
        # avg_iou=iou_sum/valid_classes
        # return avg_iou

        #attempt 2
        iou_dict={}
        for i in range(self.nr_classes):
            if( self.union_per_class[i]>0 ):
                valid_classes+=1
                iou=self.intersection_per_class[i] / self.union_per_class[i]
                iou_sum+=iou
                if print_per_class_iou:
                    print("class iou for idx", i, " is ", iou )
                iou_dict[i]=iou
        avg_iou=iou_sum/valid_classes
        return avg_iou, iou_dict


    def avg_class_iou(self, print_per_class_iou=False):
        avg_iou, iou_dict= self.compute_stats(print_per_class_iou) 
        return avg_iou

    def iou_per_class(self, print_per_class_iou=False):
        avg_iou, iou_dict= self.compute_stats(print_per_class_iou) 
        return iou_dict

    def update_best(self):
        avg_iou, iou_dict= self.compute_stats(print_per_class_iou=False) 
        if avg_iou>self.best_iou:
            self.best_iou=avg_iou
            self.best_iou_dict=iou_dict




    def show(self, epoch_nr):
        # self.scores['accuracy'] = sum(gt == pred) / len(gt)
        # # self.scores['confusion'] = confusion_matrix(gt, pred)
        # self.scores['class_accuracy'] = [self.TPs[i] / (self.TPs[i] + self.FNs[i]) for i in range(len(labels))]
        # self.scores['avg_class_accuracy'] = sum(scores['class_accuracy']) / len(labels)
        # self.scores['class_iou'] = [self.TPs[i] / (self.TPs[i] + self.FNs[i] + self.FPs[i]) for i in range(len(self.labels))]
        # self.scores['avg_class_iou'] = sum(self.scores['class_iou']) / len(self.labels)
        # self.scores['num_points'] = Total 

        #the iou per class (only those that have any points in the gt)
        avg_iou=self.avg_class_iou(print_per_class_iou=True)
        # self.logger_iou.log(epoch_nr, avg_iou, name='Avg IoU')



    #     for i in range( len(self.labels) ):
    #         print("class iou for idx", i, " is ", self.scores['class_iou'][i] )
    #         print("true positives", self.TPs[i]  )

    # #     print('-------------------- Summary --------------------')
    # #     print('   Overall accuracy: {:.4f}'.format(scores['accuracy']))
    #     # print('Avg. class accuracy: {:.4f}'.format(scores['avg_class_accuracy']))
    #     print('                IoU: {:.4f}'.format(self.scores['avg_class_iou']))
    # #     print('-------------------- Breakdown --------------------')
    # #     print('  class      count(ratio) accuracy   IoU')
    # #     total_points = sum(scores['num_points'])
    # #     for i in range(len(classes)):
    # #         print('{:10} {:7d}({:4.1f}%) {:.4f}   {:.4f}'.format(classes[i], scores['num_points'][i],
    # #                                                             100 * scores['num_points'][i] / total_points,
    # #                                                             scores['class_accuracy'][i], scores['class_iou'][i]))
    # #     print('-------------------- Confusion --------------------')
    # #     print('        {}'.format(' '.join(['{:>7}'.format(v) for v in classes])))
    # #     for i, c in enumerate(classes):
    # #         print('{:7} {}'.format(c, ' '.join(['{:7d}'.format(v) for v in scores['confusion'][i]])))

    #     ##log stuff 
    #     logger_iou.log(epoch_nr, self.scores['avg_class_iou'], name='Avg IoU')
    def clear(self):
        # self.TPs=None
        # self.FPs=None
        # self.FNs=None
        # self.Total=None
        # self.labels = None

        self.intersection_per_class=None
        self.union_per_class=None


        self.labels = None
        self.nr_classes =None

        #storing the best iou we got
        self.best_iou=-99999999
        self.best_iou_dict={}

    def start_fresh_eval(self):
        self.intersection_per_class=None
        self.union_per_class=None
        self.labels = None
        self.nr_classes =None

    def write_iou_to_csv(self,filename):
        iou_dict=self.iou_per_class(print_per_class_iou=False)
        avg_iou= self.avg_class_iou(print_per_class_iou=False)
        w = csv.writer(open(filename, "w"))
        for key, val in iou_dict.items():
            w.writerow([key, val])
        w.writerow(["mean_iou", avg_iou])

    def write_best_iou_to_csv(self,filename):
        iou_dict=self.best_iou_dict
        best_iou=self.best_iou
        w = csv.writer(open(filename, "w"))
        for key, val in iou_dict.items():
            w.writerow([key, val])
        w.writerow(["best_iou", best_iou])

