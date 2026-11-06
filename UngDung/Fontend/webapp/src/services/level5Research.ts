// Official SemRAG benchmarks with Multi-Layer Knowledge Base SemKB (KB V2).
// MS COCO 2014 Karpathy Test (5,000 images, 3 Seeds: 42, 44, 46).
export const LEVEL5_RESEARCH = {
  seed: 46,
  modelName: 'SemRAG (Proposed Model)',
  kbName: 'Kho Tri Thức Đa Tầng SemKB (KB V2)',
  source: 'https://drive.google.com/drive/folders/1Ays7jb7w1tcg6J2fzq5XpMLcAMb39DFI',
  metrics: [
    { name: 'CIDEr', value: 115.49, mean: 115.41, std: 0.07, v1: 111.63, delta: 3.78 },
    { name: 'BLEU-4', value: 34.96, mean: 34.90, std: 0.06, v1: 34.66, delta: 0.24 },
    { name: 'SPICE', value: 20.78, mean: 20.76, std: 0.02, v1: 20.35, delta: 0.41 },
    { name: 'METEOR', value: 27.87, mean: 27.85, std: 0.02, v1: 27.14, delta: 0.71 },
    { name: 'ROUGE-L', value: 56.03, mean: 56.01, std: 0.01, v1: 55.67, delta: 0.34 },
    { name: 'BLEU-1', value: 74.86, mean: 74.92, std: 0.05, v1: 74.94, delta: -0.02 },
    { name: 'BLEU-2', value: 58.55, mean: 58.57, std: 0.03, v1: 58.64, delta: -0.07 },
    { name: 'BLEU-3', value: 45.15, mean: 45.13, std: 0.04, v1: 45.10, delta: 0.03 },
  ],
  progressiveLevels: [
    { level: 'Level 1', name: 'VOB: Visual-Only Baseline', tag: 'Visual Only', cider: 107.15, bleu4: 33.68, spice: 19.73, description: 'Chỉ dùng vector ảnh CLIP ViT-B/32, không có tri thức bổ trợ.' },
    { level: 'Level 2', name: 'WF-RAG: Word-Filtered RAG', tag: 'Basic RAG', cider: 112.52, bleu4: 34.34, spice: 20.16, description: 'Truy hồi FAISS Top-5 với biểu diễn gộp trung bình (Mean-Pool).' },
    { level: 'Level 3', name: 'RR-RAG: Re-Ranked RAG', tag: 'RelTR + Rerank', cider: 114.13, bleu4: 34.71, spice: 20.68, description: 'Tái xếp hạng kết hợp đồ thị cảnh RelTR và bộ ba quan hệ SPO.' },
    { level: 'Level 4', name: 'GM-RAG: Gated-Memory RAG', tag: 'Gated 5-Token', cider: 114.81, bleu4: 34.73, spice: 20.67, description: 'Bộ nhớ nguyên mẫu 5-Token và cổng hòa trộn thích nghi g (chưa có MMR).' },
    { level: 'Level 5', name: 'SemRAG: Semantic RAG', tag: 'SemRAG (Proposed)', cider: 115.41, bleu4: 34.90, spice: 20.76, description: 'Mô hình đề xuất hoàn chỉnh với Kho Tri Thức Đa Tầng SemKB + MMR (λ=0.7).' },
  ],
} as const;
