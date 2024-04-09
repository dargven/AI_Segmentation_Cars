from sklearn.metrics import f1_score, jaccard_score, recall_score, precision_score
import numpy as np


def metrics(predicted_masks, true_masks):
    miou_scores = []
    for predicted, true in zip(predicted_masks, true_masks):
        miou_scores.append(jaccard_score(true.flatten(), predicted.flatten(), average='macro'))
    ####
    f1_scores = []
    for predicted, true in zip(predicted_masks, true_masks):
        f1_scores.append(f1_score(true.flatten(), predicted.flatten(), average='macro'))
    ####
    dice_coefficients = []
    for predicted, true in zip(predicted_masks, true_masks):
        predicted_flat = predicted.flatten()
        true_flat = true.flatten()

        intersection = np.sum(predicted_flat * true_flat)
        union = np.sum(predicted_flat) + np.sum(true_flat)
        dice = (2. * intersection) / (union + 1e-7)
        dice_coefficients.append(dice)
    return miou_scores, f1_scores, dice_coefficients
