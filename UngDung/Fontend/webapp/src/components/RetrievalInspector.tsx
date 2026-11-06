import { useState } from 'react';
import { Database, Network, Tag } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { RetrievedCandidate } from '../types';

interface RetrievalInspectorProps {
  retrievedCandidates: RetrievedCandidate[];
  topKUsed: number;
  searchedCount?: number;
  rerankingMethod?: string;
  isMock?: boolean;
}

export function RetrievalInspector({
  retrievedCandidates,
  topKUsed,
  searchedCount,
  rerankingMethod,
  isMock = false
}: RetrievalInspectorProps) {
  const { i18n } = useTranslation();
  const vi = (i18n.resolvedLanguage || i18n.language).startsWith('vi');
  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState(retrievedCandidates[0]?.id);
  const selected = retrievedCandidates.find(candidate => candidate.id === selectedId) || retrievedCandidates[0];
  const visible = retrievedCandidates.filter(candidate => candidate.caption.toLowerCase().includes(query.toLowerCase()));

  return (
    <section className="space-y-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 sm:p-5">
      <div className="flex items-start gap-3">
        <Database className="mt-1 h-5 w-5 shrink-0 text-blue-500" />
        <div>
          <h3 className="font-bold">{vi ? 'Truy hồi và Tái xếp hạng SemKB' : 'SemKB Retrieval and Reranking'}</h3>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            {isMock
              ? (vi ? 'Dữ liệu mô phỏng' : 'Simulated data')
              : 'Kho Tri Thức Đa Tầng SemKB (MS COCO) · 566.747 bản ghi · 512d CLIP'}
          </p>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 text-xs">
        {searchedCount != null && (
          <span className="rounded-lg bg-slate-100 dark:bg-slate-800 px-3 py-2">
            {vi ? 'Ứng viên FAISS' : 'FAISS Candidates'}: {searchedCount}
          </span>
        )}
        <span className="rounded-lg bg-blue-50 dark:bg-blue-950/40 px-3 py-2 text-blue-600 dark:text-blue-300 font-medium">
          Top-K = {topKUsed} (MMR λ = 0.7)
        </span>
        {rerankingMethod && (
          <span className="rounded-lg bg-slate-100 dark:bg-slate-800 px-3 py-2">
            {rerankingMethod}
          </span>
        )}
      </div>

      <p className="text-xs text-slate-500 dark:text-slate-400">
        {vi
          ? 'Điểm Relevance kết hợp giữa độ tương đồng ảnh-câu CLIP (0.6) và mức trùng khớp thực thể đồ thị cảnh SPO (0.4), sau đó lọc đa dạng qua MMR (λ = 0.7).'
          : 'Relevance score balances CLIP image-text similarity (0.6) with SPO scene-graph entity matching (0.4), followed by MMR diversity pruning (λ = 0.7).'}
      </p>

      <input
        aria-label={vi ? 'Tìm trong caption đã chọn' : 'Search returned captions'}
        value={query}
        onChange={event => setQuery(event.target.value)}
        placeholder={vi ? 'Tìm trong caption đã chọn…' : 'Search returned captions…'}
        className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 px-3 py-2 text-sm"
      />

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-slate-500 dark:text-slate-400">
            <tr>
              <th className="p-2">{vi ? 'Hạng cuối' : 'Final rank'}</th>
              <th className="p-2">Caption Tri thức</th>
              <th className="p-2">FAISS</th>
              <th className="p-2">CLIP</th>
              <th className="p-2">Relevance</th>
            </tr>
          </thead>
          <tbody>
            {visible.map(candidate => (
              <tr
                key={candidate.id}
                className={`border-t border-slate-100 dark:border-slate-800 ${
                  selected?.id === candidate.id ? 'bg-blue-50 dark:bg-blue-950/30' : ''
                }`}
              >
                <td className="p-2 font-medium">#{candidate.final_rank ?? candidate.rank}</td>
                <td className="min-w-48 p-2">
                  <button onClick={() => setSelectedId(candidate.id)} className="text-left leading-relaxed hover:text-blue-500">
                    {candidate.caption}
                  </button>
                </td>
                <td className="p-2 text-slate-500">#{candidate.rank}</td>
                <td className="p-2 tabular-nums">{candidate.faiss_similarity.toFixed(4)}</td>
                <td className="p-2 tabular-nums font-medium text-blue-600 dark:text-blue-400">
                  {candidate.rerank_score?.toFixed(4) ?? '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {visible.length === 0 && (
        <p className="text-sm text-slate-500">{vi ? 'Không có caption phù hợp.' : 'No matching captions.'}</p>
      )}

      {selected && (
        <div className="rounded-lg border border-blue-200 dark:border-blue-900 bg-blue-50/30 dark:bg-blue-950/20 p-4 text-sm space-y-2">
          <p className="font-semibold text-slate-900 dark:text-slate-100">{selected.caption}</p>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            {selected.id} · {selected.source}
          </p>

          {selected.objects && selected.objects.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 pt-1 text-xs">
              <span className="flex items-center gap-1 text-slate-500 font-medium">
                <Tag size={13} /> {vi ? 'Thực thể' : 'Entities'}:
              </span>
              {selected.objects.map((obj, i) => (
                <span key={i} className="rounded bg-slate-200/80 dark:bg-slate-800 px-2 py-0.5 text-slate-700 dark:text-slate-300">
                  {obj}
                </span>
              ))}
            </div>
          )}

          {selected.triples && selected.triples.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 pt-1 text-xs">
              <span className="flex items-center gap-1 text-slate-500 font-medium">
                <Network size={13} /> {vi ? 'Quan hệ SPO' : 'SPO Triples'}:
              </span>
              {selected.triples.map((rel, i) => (
                <span key={i} className="rounded bg-indigo-100 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 px-2 py-0.5">
                  ({rel})
                </span>
              ))}
            </div>
          )}

          {selected.gate_score != null && (
            <p className="pt-1 text-xs text-blue-600 dark:text-blue-400 font-medium">
              {vi ? 'Hệ số cổng tri thức thích nghi' : 'Adaptive Knowledge Gate'}: g = {selected.gate_score.toFixed(4)}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
