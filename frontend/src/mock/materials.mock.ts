import type { Material } from "@/types";

export const MOCK_MATERIALS: Material[] = [
  {
    id: "m1", name: "Module1_IntroToML.pdf", type: "notes", module: 1,
    subject: "Machine Learning", subjectCode: "21AI51",
    size: "2.4 MB", uploadedOn: "2024-07-01", status: "processed",
    pages: 28, images: 6, chunks: 42,
  },
  {
    id: "m2", name: "Module2_Regression_SVM.pdf", type: "notes", module: 2,
    subject: "Machine Learning", subjectCode: "21AI51",
    size: "3.1 MB", uploadedOn: "2024-07-02", status: "processed",
    pages: 34, images: 9, chunks: 58,
  },
  {
    id: "m3", name: "PatternRecognition_Bishop.pdf", type: "textbook", module: "all",
    subject: "Machine Learning", subjectCode: "21AI51",
    size: "18.7 MB", uploadedOn: "2024-07-03", status: "processed",
    pages: 738, images: 124, chunks: 1420,
  },
  {
    id: "m4", name: "Module3_Clustering_PCA.pdf", type: "notes", module: 3,
    subject: "Machine Learning", subjectCode: "21AI51",
    size: "1.8 MB", uploadedOn: "2024-07-04", status: "processing",
    pages: 0, images: 0, chunks: 0,
  },
  {
    id: "m5", name: "DL_Goodfellow_Textbook.pdf", type: "textbook", module: "all",
    subject: "Deep Learning", subjectCode: "21AI52",
    size: "22.1 MB", uploadedOn: "2024-07-05", status: "processed",
    pages: 800, images: 210, chunks: 2100,
  },
];
