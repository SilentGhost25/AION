import type { Syllabus } from "@/types";

export const MOCK_SYLLABI: Record<string, Syllabus> = {
  "21AI51": {
    subjectCode: "21AI51",
    updatedAt: "2024-07-01",
    modules: [
      {
        number: 1,
        title: "Introduction to Machine Learning",
        topics: [
          "What is Machine Learning?",
          "Types of ML: Supervised, Unsupervised, Reinforcement",
          "Bias-Variance Tradeoff",
          "Model Evaluation Metrics",
          "Overfitting and Underfitting",
          "Cross-Validation techniques",
        ],
        co: "CO1",
        bloomLevels: ["L1", "L2"],
      },
      {
        number: 2,
        title: "Regression and Classification Algorithms",
        topics: [
          "Linear Regression and OLS",
          "Logistic Regression",
          "Gradient Descent Optimization",
          "Support Vector Machines",
          "Kernel Trick and RBF Kernel",
          "Decision Trees and Random Forests",
          "Naïve Bayes Classifier",
        ],
        co: "CO2",
        bloomLevels: ["L3"],
      },
      {
        number: 3,
        title: "Unsupervised Learning and Dimensionality Reduction",
        topics: [
          "K-Means Clustering",
          "Elbow Method",
          "Hierarchical Clustering",
          "DBSCAN",
          "Principal Component Analysis (PCA)",
          "Linear Discriminant Analysis (LDA)",
        ],
        co: "CO3",
        bloomLevels: ["L4"],
      },
      {
        number: 4,
        title: "Neural Networks and Deep Learning",
        topics: [
          "Perceptron and Multi-Layer Perceptron",
          "Backpropagation Algorithm",
          "Activation Functions: ReLU, Sigmoid, Tanh",
          "Convolutional Neural Networks (CNN)",
          "Recurrent Neural Networks (RNN)",
          "Dropout and Regularization",
        ],
        co: "CO3",
        bloomLevels: ["L4"],
      },
      {
        number: 5,
        title: "Model Evaluation and Ensemble Methods",
        topics: [
          "k-Fold Cross Validation",
          "Precision, Recall, F1-Score",
          "ROC-AUC Curve",
          "Boosting: AdaBoost, Gradient Boosting",
          "XGBoost",
          "Bagging and Random Forests",
        ],
        co: "CO2",
        bloomLevels: ["L3"],
      },
    ],
  },
};
