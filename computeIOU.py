import numpy as np
# from MIoUData import MIoU_dataloader
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

# if __name__ == "__main__":
#     best_miou = 0.0
#     for i in range(1):
#         miou = Evaluator(256)
#         miouVal = 0
#         accVal = 0
#         for index, (predict, label) in enumerate(MIoU_dataloader):
#             predict = predict.cpu().numpy()
#             label = label.cpu().numpy()
#             miou.add_batch(label, predict)
#             accVal += miou.Pixel_Accuracy()
#             miouVal += miou.Mean_Intersection_over_Union()
#             print(index)
#             print('acc and miou are {},{}'.format(miou.Pixel_Accuracy(),miou.Mean_Intersection_over_Union()))
#         now_miou = miouVal/len(MIoU_dataloader)
#         if now_miou > best_miou:
#             best_miou = now_miou
#         print('all acc and miou are {},{}'.format(accVal/len(MIoU_dataloader),miouVal/len(MIoU_dataloader)))
#     print('best miou are{}'.format(best_miou))