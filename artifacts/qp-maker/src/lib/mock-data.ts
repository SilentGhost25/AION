export const departments = [
  "Computer Science & Engineering",
  "Information Science & Engineering",
  "Artificial Intelligence & Machine Learning",
  "Electronics & Communication Engineering",
  "Electrical & Electronics Engineering",
  "Mechanical Engineering",
  "Civil Engineering"
];

export const subjects: Record<string, { code: string; name: string }[]> = {
  "Artificial Intelligence & Machine Learning": [
    { code: "21AI51", name: "Machine Learning" },
    { code: "21AI52", name: "Deep Learning" },
    { code: "21AI53", name: "Natural Language Processing" },
    { code: "21AI54", name: "Computer Vision" },
    { code: "21AI55", name: "Data Structures and Algorithms" }
  ],
  "Computer Science & Engineering": [
    { code: "21CS51", name: "Operating Systems" },
    { code: "21CS52", name: "Database Management Systems" },
    { code: "21CS53", name: "Computer Networks" },
    { code: "21CS54", name: "Software Engineering" },
  ],
  "Information Science & Engineering": [
    { code: "21IS51", name: "Information Security" },
    { code: "21IS52", name: "Cloud Computing" },
  ]
};

// Fixed CO-Bloom mapping — enforced as an iron-clad rule during generation
export const MODULE_CO_BLOOM: Record<number, { co: string; bloom: string }> = {
  1: { co: "CO1", bloom: "L1/L2" },
  2: { co: "CO2", bloom: "L3" },
  3: { co: "CO3", bloom: "L4" },
  4: { co: "CO3", bloom: "L4" },
  5: { co: "CO2", bloom: "L3" },
};

export const studyMaterials = [
  {
    id: "mat-1",
    name: "Module1_IntroToML.pdf",
    type: "notes" as const,
    module: 1,
    subject: "Machine Learning",
    subjectCode: "21AI51",
    size: "3.2 MB",
    uploadedOn: "2024-07-01",
    status: "processed" as const,
    pages: 42,
    images: 7,
    chunks: 138,
  },
  {
    id: "mat-2",
    name: "Module2_Regression_SVM.pdf",
    type: "notes" as const,
    module: 2,
    subject: "Machine Learning",
    subjectCode: "21AI51",
    size: "5.1 MB",
    uploadedOn: "2024-07-02",
    status: "processed" as const,
    pages: 65,
    images: 12,
    chunks: 203,
  },
  {
    id: "mat-3",
    name: "PatternRecognition_Bishop.pdf",
    type: "textbook" as const,
    module: "all" as const,
    subject: "Machine Learning",
    subjectCode: "21AI51",
    size: "18.4 MB",
    uploadedOn: "2024-07-03",
    status: "processed" as const,
    pages: 738,
    images: 94,
    chunks: 1820,
  },
  {
    id: "mat-4",
    name: "Module3_NeuralNetworks.pdf",
    type: "notes" as const,
    module: 3,
    subject: "Machine Learning",
    subjectCode: "21AI51",
    size: "4.8 MB",
    uploadedOn: "2024-07-05",
    status: "processing" as const,
    pages: 0,
    images: 0,
    chunks: 0,
  },
];

export const syllabusData = [
  {
    subjectCode: "21AI51",
    subjectName: "Machine Learning",
    department: "Artificial Intelligence & Machine Learning",
    modules: [
      {
        id: "m1",
        number: 1,
        title: "Introduction to Machine Learning",
        topics: [
          "Definition, scope and limitations of Machine Learning",
          "Types of Machine Learning: Supervised, Unsupervised, Reinforcement",
          "Key concepts: Features, Labels, Training and Test sets",
          "Evaluation metrics: Accuracy, Precision, Recall, F1-score",
          "Bias-Variance Tradeoff and Overfitting / Underfitting",
        ],
        coMapping: "CO1",
        bloomLevel: "L1/L2",
        hours: 10,
      },
      {
        id: "m2",
        number: 2,
        title: "Regression and Classification Algorithms",
        topics: [
          "Linear Regression and Gradient Descent",
          "Logistic Regression",
          "Decision Trees and Random Forests",
          "Support Vector Machines (SVM) with kernels",
          "k-Nearest Neighbors (k-NN)",
          "Naïve Bayes Classifier",
        ],
        coMapping: "CO2",
        bloomLevel: "L3",
        hours: 12,
      },
      {
        id: "m3",
        number: 3,
        title: "Unsupervised Learning",
        topics: [
          "K-Means Clustering",
          "Hierarchical Clustering",
          "DBSCAN",
          "Principal Component Analysis (PCA)",
          "Singular Value Decomposition (SVD)",
          "Autoencoders for dimensionality reduction",
        ],
        coMapping: "CO3",
        bloomLevel: "L4",
        hours: 10,
      },
      {
        id: "m4",
        number: 4,
        title: "Neural Networks and Deep Learning Basics",
        topics: [
          "Perceptron and Multi-layer Perceptron",
          "Backpropagation Algorithm",
          "Activation functions: ReLU, Sigmoid, Tanh, Softmax",
          "Convolutional Neural Networks (CNN) — architecture overview",
          "Recurrent Neural Networks (RNN) — LSTM basics",
          "Regularization techniques: Dropout, Batch Normalization",
        ],
        coMapping: "CO3",
        bloomLevel: "L4",
        hours: 12,
      },
      {
        id: "m5",
        number: 5,
        title: "Model Evaluation and Ensemble Methods",
        topics: [
          "Cross-validation techniques",
          "Hyperparameter tuning: Grid Search, Random Search",
          "Bagging and Boosting — AdaBoost, Gradient Boosting, XGBoost",
          "Model interpretability and explainability (LIME, SHAP)",
          "Deployment considerations and ML pipelines",
        ],
        coMapping: "CO2",
        bloomLevel: "L3",
        hours: 8,
      },
    ],
  },
];

export const historyData = [
  {
    id: "PPR-2024-001",
    subject: "Machine Learning",
    examType: "IAT-1",
    semester: "5th",
    generatedOn: "2024-07-10",
    status: "Downloaded",
    coverageScope: "Modules 1 and 2 — up to SVM with kernels"
  },
  {
    id: "PPR-2024-002",
    subject: "Deep Learning",
    examType: "IAT-2",
    semester: "6th",
    generatedOn: "2024-07-18",
    status: "Generated",
    coverageScope: "Modules 3 and 4 entirely"
  },
  {
    id: "PPR-2024-003",
    subject: "Data Structures and Algorithms",
    examType: "End-Sem",
    semester: "4th",
    generatedOn: "2024-07-22",
    status: "Downloaded",
    coverageScope: "All 5 modules"
  }
];
