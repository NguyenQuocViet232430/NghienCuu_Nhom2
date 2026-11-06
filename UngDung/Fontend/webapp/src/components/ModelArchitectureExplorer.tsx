import { ArrowRight } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { BenchmarkTable } from './BenchmarkTable';

export function ModelArchitectureExplorer() {
  const { i18n } = useTranslation();
  const vi = i18n.language.startsWith('vi');
  const stages = [
    [vi ? 'Ảnh RGB' : 'RGB Image', vi ? 'Ảnh đầu vào' : 'Input image'],
    ['CLIP & RelTR', vi ? 'Đặc trưng ảnh 512d & Đồ thị cảnh SPO' : '512d visual feature & SPO graph'],
    ['SemKB (FAISS)', vi ? 'Truy hồi 50 ứng viên tri thức' : 'Retrieve 50 knowledge candidates'],
    ['Hybrid + MMR', vi ? 'Chọn 5 câu đa dạng (λ=0.7)' : 'Select 5 diverse captions (λ=0.7)'],
    ['Gated Fusion', vi ? 'Cổng thích nghi động g' : 'Adaptive dynamic gate g'],
    ['Transformer', vi ? 'Giải mã Beam Search (k=3)' : 'Dual-branch Beam Search (k=3)'],
    ['Caption', vi ? 'Chú thích giàu ngữ nghĩa' : 'Rich semantic caption'],
  ];
  return (
    <div className="research-page">
      <header className="page-heading">
        <h1>{vi ? 'Kiến trúc Mô hình SemRAG' : 'SemRAG Architecture'}</h1>
        <p>
          {vi
            ? 'Quy trình sinh chú thích ảnh tăng cường truy hồi dẫn dắt bởi đồ thị cảnh và cổng hợp nhất thích ứng.'
            : 'Scene-graph guided retrieval-augmented image captioning with adaptive gated fusion.'}
        </p>
      </header>
      <ol className="architecture-pipeline" aria-label={vi ? 'Pipeline từ ảnh đến chú thích' : 'Image-to-caption pipeline'}>
        {stages.map(([name, detail], index) => (
          <li key={name}>
            <div className="pipeline-block">
              <h2>{name}</h2>
              <p>{detail}</p>
            </div>
            {index < stages.length - 1 && <ArrowRight size={16} aria-hidden="true" className="pipeline-arrow" />}
          </li>
        ))}
      </ol>
      <div className="architecture-notes">
        <section>
          <h2>{vi ? 'Kho Tri Thức Đa Tầng SemKB & Tái xếp hạng' : 'SemKB Knowledge Base & Reranking'}</h2>
          <p>
            {vi
              ? 'Kho Tri Thức Đa Tầng SemKB (566.747 bản ghi COCO train/restval) tích hợp sẵn thực thể, hành động, thuộc tính và quan hệ SPO cùng cơ chế khử trùng lặp. Thuật toán tái xếp hạng lai 0,6 CLIP + 0,4 SPO và lọc đa dạng hóa MMR (λ = 0,7) giúp chọn ra 5 ứng viên giàu thông tin và giảm trùng lặp.'
              : 'Multi-Layer SemKB (566,747 COCO train/restval records) integrates structured entities, actions, attributes, SPO triples, and duplicate pruning. Hybrid reranking (0.6 CLIP + 0.4 SPO) with MMR diversity (λ = 0.7) selects 5 diverse and highly informative candidates.'}
          </p>
        </section>
        <section>
          <h2>{vi ? 'Bộ giải mã Dual-Branch Transformer' : 'Dual-Branch Transformer Decoder'}</h2>
          <p>
            {vi
              ? 'Duy trì 5 vector nguyên mẫu tri thức độc lập qua Linear(512, 256) (Multi-Token Memory). Bộ giải mã 3 lớp, 8 heads, d_model = 256, FFN = 1024. Giải mã Beam Search k = 3, α = 0,6, độ dài chuỗi tối đa 30 tokens, giữ nguyên 100% trọng số huấn luyện (Zero Retraining).'
              : 'Retains 5 independent prototype vectors via Linear(512, 256) (Multi-Token Memory). 3-layer decoder, 8 heads, d_model = 256, FFN = 1024. Beam Search decoding with k = 3, α = 0.6, max_len = 30 tokens, strictly preserving 100% trained weights (Zero Retraining).'}
          </p>
        </section>
      </div>
      <section className="gate-formula">
        <h2>{vi ? 'Cơ chế Cổng Hòa Trộn Tự Thích Ứng (Confidence-Aware Gated Fusion)' : 'Confidence-Aware Gated Fusion'}</h2>
        <p className="formula">context = (1 − g) × context_visual + g × context_knowledge</p>
        <p>
          {vi
            ? 'g = sigmoid(Linear([v; k̄; s̄])). Cổng g học được dựa trên độ tương đồng và mức tin cậy của ảnh và tri thức. Khi hình ảnh rõ nét, mô hình tập trung vào thị giác; khi bối cảnh phức tạp, cổng tự động nâng cao tỷ trọng tri thức từ SemKB để bổ sung chi tiết.'
            : 'g = sigmoid(Linear([v; k̄; s̄])). Learned gate g dynamically balances visual and knowledge branches based on confidence and relevance. The visual-only ablation is obtained with g = 0.'}
        </p>
      </section>
      <BenchmarkTable />
    </div>
  );
}
