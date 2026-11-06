import { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { AnalysisResult } from '../types';
import { copyCaptionToClipboard } from '../services/exportService';

export function BaselineComparisonView({ result, onCreate }: { result: AnalysisResult; onCreate: () => void }) {
  const { i18n } = useTranslation();
  const vi = i18n.language.startsWith('vi');
  const [copied, setCopied] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const native5 = result.level_info.level === 'Level 5A' || result.level_info.level === 'SemRAG';
  const captions = [
    { id: 'initial', label: native5 ? (vi ? 'Chỉ dùng ảnh · g = 0' : 'Visual only · g = 0') : 'Baseline', text: vi && result.baseline_caption_vi ? result.baseline_caption_vi : result.baseline_caption },
    ...(!native5 && showAll ? [
      { id: 'level2', label: 'Level 2 (Basic RAG)', text: vi && result.level2_caption_vi ? result.level2_caption_vi : result.level2_caption },
      { id: 'level3', label: 'Level 3 (RelTR Rerank)', text: vi && result.level3_caption_vi ? result.level3_caption_vi : result.level3_caption },
    ] : []),
    { id: 'final', label: vi ? 'SemRAG (Tri thức SemKB + Cổng động g)' : 'SemRAG (SemKB Knowledge + Dynamic Gate)', text: vi && result.raic_caption_vi ? result.raic_caption_vi : result.raic_caption },
  ];
  const copy = async (id: string, text: string) => { if (await copyCaptionToClipboard(text)) { setCopied(id); setTimeout(() => setCopied(null), 2000); } };
  return <section><header className="page-heading page-heading-with-action"><div><h1>{vi ? 'So sánh' : 'Compare'}</h1>
    <p>{vi ? 'Đối chiếu chú thích khi tắt và bật tri thức trên cùng một ảnh.' : 'Compare captions with knowledge disabled and enabled for the same image.'}</p></div>
    <button type="button" className="studio-primary-button" onClick={onCreate}>{vi ? 'Ảnh mới' : 'New image'}</button></header>
    {result.metadata.is_mock && <p className="notice">{vi ? 'Dữ liệu mô phỏng.' : 'Simulated data.'}</p>}
    <div className="comparison-layout"><figure><img src={result.image_url} alt={vi ? 'Ảnh được so sánh' : 'Comparison image'} /><figcaption>{result.image_name}</figcaption></figure>
      <div>{!native5 && <div className="segmented-control"><button aria-pressed={!showAll} onClick={() => setShowAll(false)}>{vi ? 'Ban đầu / Kết quả' : 'Initial / Result'}</button><button aria-pressed={showAll} onClick={() => setShowAll(true)}>{vi ? 'Tất cả cấu hình' : 'All configurations'}</button></div>}
        {captions.map(item => <section className="comparison-caption" key={item.id}><h2>{item.label}</h2><p>{item.text || '—'}</p><button type="button" disabled={!item.text} className="subtle-button" onClick={() => copy(item.id, item.text || '')}>
          {copied === item.id ? <Check size={14} /> : <Copy size={14} />}{copied === item.id ? (vi ? 'Đã sao chép' : 'Copied') : (vi ? 'Sao chép' : 'Copy')}</button></section>)}
      </div></div>
    {native5 && <p className="section-note">{vi ? 'Mô hình sinh chú thích SemRAG với Kho Tri Thức Đa Tầng SemKB (Seed 46). Câu "Chỉ dùng ảnh" thu được khi vô hiệu hóa nhánh tri thức (g = 0).' : 'SemRAG captioning model with SemKB Knowledge Base (Seed 46). The visual-only caption is obtained by disabling the knowledge branch (g = 0).'}</p>}
    {result.evaluation.status !== 'not_evaluated' && <p className="section-note">CIDEr · Baseline: {result.evaluation.metrics.cider.baseline.toFixed(2)} · {vi ? 'Kết quả SemRAG' : 'SemRAG Result'}: {result.evaluation.metrics.cider.raic.toFixed(2)}</p>}
  </section>;
}

