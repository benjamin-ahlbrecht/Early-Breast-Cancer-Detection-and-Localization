import numpy as np
import matplotlib.pyplot as plt


class Evaluator():
    def __init__(self, probabilities:np.ndarray, truth:np.ndarray):
        """Evaluate model predictions against the ground truth labels
            
        Args:
            probabilities (torch.Tensor): The output predictions of the model in
                the form of probabilities
            truth (torch.Tensor): The respective ground truth labels (0 or 1)
        """
        self.probabilitiess = probabilities
        self.truth = truth
        
        self.predictions = np.round(probabilities).astype(int)
        
        self.tp = np.sum((truth == 1) & (self.predictions == 1))
        self.fn = np.sum((truth == 1) & (self.predictions == 0))
        self.fp = np.sum((truth == 0) & (self.predictions == 1))
        self.tn = np.sum((truth == 0) & (self.predictions == 0))
        
        # Probabilistic true and false positives
        self.c_tp = np.sum(probabilities[truth == 1])
        self.c_fp = np.sum(probabilities[truth == 0])
    
    def recall(self) -> float:
        """Compute the standard recall evaluation metric
        """
        recall = self.tp / (self.tp + self.fn)
        return recall
    
    def precision(self) -> float:
        """Compute the standard precision evaluation metric
        """
        precision = self.tp / (self.tp + self.fp)
        return precision
    
    def f_beta(self, beta:float) -> float:
        """Compute the F_beta score, a harmonic mean balancing the recall and
        precision metrics such that recall is considered beta times more
        important than precision
        """
        beta2 = beta * beta
        precision = self.precision()
        recall = self.recall()
        
        f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        return f_beta
    
    def f1(self) -> float:
        """Compute the F1-socre, a harmonic mean equally balancing the recall
        and the precision.
        """
        f1 = self.f_beta(1)
        return f1
    
    def recall_prob(self) -> float:
        """Computes the probabilistic version of the recall evaluation metric
        """
        recall = self.c_tp / (self.tp + self.fn)
        return recall
    
    def precision_prob(self) -> float:
        """Computes the probabilistic version of the precision evaluation metric
        """
        precision = self.c_tp / (self.c_tp + self.c_fp)
        return precision
    
    def f_beta_prob(self, beta:float) -> float:
        """Compute the probabilistic version of the f beta score
        """
        beta2 = beta * beta
        precision = self.precision_prob()
        recall = self.recall_prob()
        
        f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        return f_beta
    
    def f1_prob(self) -> float:
        """Compute the probabilistic version of the F1 score
        """
        f1 = self.f_beta_prob(1)
        return f1
    
    def confusion_matrix_plot(self):
        """Generate a confusion matrix plot
        """
        confusion_matrix = np.array([[self.tp, self.fp], [self.fn, self.tn]])
        labels = ["Positive", "Negative"]

        fig, ax = plt.subplots()
        sns.heatmap(
            data=confusion_matrix,
            vmin=0,
            cmap=sns.color_palette("Blues", as_cmap=True),
            annot=True,
            fmt=".0f",
            linewidths=2,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels
        )

        ax.set_title("Actual Values", fontsize=14)
        ax.set_ylabel("Predicted values", fontsize=14)
        
        return (fig, ax)