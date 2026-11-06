import { useTranslation } from 'react-i18next';
import { LEVEL5_RESEARCH } from '../services/level5Research';

export function BenchmarkTable() {
  const { i18n } = useTranslation();
  const vi = i18n.language.startsWith('vi');
  return (
    <section className="benchmark-section">
      <div className="section-heading">
        <h2>{vi ? 'Kết quả Benchmark Chính thức (SemRAG)' : 'Official Benchmark Results (SemRAG)'}</h2>
        <a className="text-link" href={LEVEL5_RESEARCH.source} target="_blank" rel="noreferrer">
          {vi ? 'Xem nguồn dữ liệu thực nghiệm' : 'View experimental data source'}
        </a>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <caption>MS COCO 2014 · Karpathy Test (5.000 ảnh) · {vi ? 'Thang điểm ×100' : 'Scores ×100'}</caption>
          <thead>
            <tr>
              <th scope="col">{vi ? 'Chỉ số' : 'Metric'}</th>
              <th scope="col">Seed 46</th>
              <th scope="col">{vi ? 'Trung bình ± SD (3 Seeds)' : 'Mean ± SD (3 Seeds)'}</th>
              <th scope="col">{vi ? 'Kho Cũ V1' : 'KB V1 Baseline'}</th>
              <th scope="col">{vi ? 'Tăng trưởng (Δ)' : 'Improvement (Δ)'}</th>
            </tr>
          </thead>
          <tbody>
            {LEVEL5_RESEARCH.metrics.map(metric => (
              <tr key={metric.name}>
                <th scope="row">{metric.name}</th>
                <td>{metric.value.toFixed(2)}</td>
                <td>{metric.mean.toFixed(2)} ± {metric.std.toFixed(2)}</td>
                <td className="muted">{metric.v1.toFixed(2)}</td>
                <td style={{ color: metric.delta > 0 ? '#10b981' : undefined, fontWeight: metric.delta > 0 ? 600 : undefined }}>
                  {metric.delta > 0 ? `+${metric.delta.toFixed(2)}` : metric.delta.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="section-note">
        {vi
          ? 'Số liệu thực nghiệm chuẩn hóa trên 3 Seeds chuẩn [42, 44, 46] với Kho Tri Thức Đa Tầng SemKB (KB V2). Điểm CIDEr tăng vượt bậc từ 111.63 lên 115.41 (+3.78 điểm) trong cùng điều kiện kiểm soát nghiêm ngặt 100% trọng số mô hình đã huấn luyện (Zero Retraining).'
          : 'Standardized experimental results across 3 standard seeds [42, 44, 46] with Multi-Layer Knowledge Base SemKB (KB V2). CIDEr improves significantly from 111.63 to 115.41 (+3.78 points) under strict 100% frozen model weights (Zero Retraining).'}
      </p>
    </section>
  );
}
