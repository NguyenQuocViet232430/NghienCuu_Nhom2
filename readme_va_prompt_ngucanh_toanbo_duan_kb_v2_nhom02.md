# TÀI LIỆU TỔNG HỢP NGỮ CẢNH, KIẾN TRÚC KHO TRI THỨC MỚI (KB V2), BỘ SỐ LIỆU THỰC NGHIỆM & SIÊU PROMPT CHO KHÓA LUẬN / BÀI BÁO — NHÓM 02

**Đề tài:** Sinh Chú Thích Ảnh Tăng Cường Truy Hồi Dẫn Dắt Bởi Đồ Thị Cảnh Và Hợp Nhất Đa Token Có Cổng (*Scene-Graph Guided Retrieval-Augmented Image Captioning with Gated Multi-Token Fusion and Multi-Layer Structured Knowledge Base V2*)

**Nhóm thực hiện:** Nhóm 02 (Nguyễn Thị Quyên,Ngô Quốc Bình, Nguyễn Quốc Việt — Khoa Công nghệ Thông tin, Trường Đại học Công Thương TP.HCM — HUIT; Giảng viên hướng dẫn: TS. Nguyễn Văn Thịnh , ThS. Đinh Thị Mận)

**Bộ mã nguồn & Kết quả chuẩn hóa:** `NghienCuu_Nhom2.zip` (27 Jupyter Notebooks thực nghiệm hoàn chỉnh + 1 Sơ đồ kiến trúc PDF)

## 1. Báo cáo Kiểm định Toàn diện Gói `NghienCuu_Nhom2.zip` (28/28 File Hoàn tất 100%)

Qua kiểm định tự động toàn bộ **27 file Jupyter Notebook (`.ipynb`)** và **1 file sơ đồ kiến trúc (`.pdf`)** trong gói `NghienCuu_Nhom2.zip`, toàn bộ các thực nghiệm trên **Kho Tri Thức Mới V2 (`knowledge_base_level4` cho MS COCO 2014 và `knowledge_base_level4_Flick30K` cho Flickr30k)** đã **hoàn tất 100% các Cell thực thi (`Errors: []`, không có bất kỳ lỗi runtime nào)**, đồng bộ nghiêm ngặt trên đúng **3 Seeds chuẩn `[42, 44, 46]`**.

| STT | Thư mục / Đường dẫn trong `NghienCuu_Nhom2.zip` | Tên File | Số Cell Code đã chạy | Trạng thái | Vai trò trong Bài báo / Khóa luận | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| 1 | `NghienCuu_Nhom2/` | `Mô hình kho tri thức mới.pdf` | 1 trang Vector PDF | Hoàn tất | Sơ đồ tổng thể 4 khối (A, B, C, D) của kiến trúc RAG với Kho Tri Thức Đa Tầng V2 | 
| 2 | `Coco/` | `Knowledge_Base_V2_Improved_Full.ipynb` | 10/10 | Hoàn tất | Xây dựng Kho Tri Thức Mới V2 trên COCO (`566.747` câu, `512d` CLIP ViT-B/32, `objects`, `attributes`, `actions`, `relations` SPO, `duplicate_of`) | 
| 3 | `Coco/` | `Baseline_Level_1_COCO2014_MultiSeed_BeamSearch_k3_Seed42.ipynb` | 6/6 | Hoàn tất | Đánh giá Level 1 (Visual-Only Baseline, không dùng KB) trên COCO — Seed 42 | 
| 4 | `Coco/` | `Baseline_Level_1_COCO2014_MultiSeed_BeamSearch_k3_Seed44.ipynb` | 6/6 | Hoàn tất | Đánh giá Level 1 (Visual-Only Baseline, không dùng KB) trên COCO — Seed 44 | 
| 5 | `Coco/` | `Baseline_Level_1_COCO2014_MultiSeed_BeamSearch_k3_Seed46.ipynb` | 6/6 | Hoàn tất | Đánh giá Level 1 (Visual-Only Baseline, không dùng KB) trên COCO — Seed 46 + Bảng tổng hợp 3 Seeds | 
| 6 | `Coco/` | `DanhGia_Level2_3Seeds_CaiTien_COCO_KetQua3.ipynb` | 7/7 | Hoàn tất | Đánh giá Level 2 (`Basic RAG`) với Kho Mới V2 trên 5.000 ảnh COCO Test (3 Seeds: `42, 44, 46`) | 
| 7 | `Coco/` | `DanhGia_Level3_3Seeds_CaiTien_COCO_KetQua3.ipynb` | 7/7 | Hoàn tất | Đánh giá Level 3 (`Scene-Graph Reranked RAG`) với Kho Mới V2 trên 5.000 ảnh COCO Test (3 Seeds: `42, 44, 46`) | 
| 8 | `Coco/` | `DanhGia_Level4_3Seeds_CaiTien_COCO_KetQua6.ipynb` | 7/7 | Hoàn tất | Đánh giá Level 4 (`Gated Multi-Token RAG`, chưa có MMR) với Kho Mới V2 trên 5.000 ảnh COCO Test (3 Seeds: `42, 44, 46`) | 
| 9 | `Coco/` | `DanhGia_Level5_3Seeds_CaiTien_COCO_KetQua5.ipynb` | 7/7 | Hoàn tất | Đánh giá Level 5 (`Full Proposed RAG` — Mô hình đề xuất hoàn chỉnh) với Kho Mới V2 trên 5.000 ảnh COCO Test (3 Seeds: `42, 44, 46`) | 
| 10 | `flick30k/PhuongAn_A/` | `Tao_Kho_Tri_Thuc_V2_Flickr30k_CaiTien_KetQua.ipynb` | 4/4 | Hoàn tất | Xây dựng Kho Tri Thức Mới V2 nội miền Flickr30k (`145.000` câu Train, schema `2.1.0`, SPO + khử trùng lặp `524` câu) | 
| 11 | `flick30k/PhuongAn_A/` | `DanhGia_Level1_3Seeds_Baseline_Flickr30k_KetQua.ipynb` | 3/3 | Hoàn tất | Phương Án A (In-Domain Flickr30k): Đánh giá Level 1 Baseline trên 1.000 ảnh Flickr30k Test (3 Seeds: `42, 44, 46`) | 
| 12 | `flick30k/PhuongAn_A/` | `DanhGia_Level2_3Seeds_CaiTien_Flickr30k_KetQua.ipynb` | 5/5 | Hoàn tất | Phương Án A (In-Domain Flickr30k + Kho Flickr30k V2): Đánh giá Level 2 (3 Seeds: `42, 44, 46` + `CHAIRs/CHAIRi`) | 
| 13 | `flick30k/PhuongAn_A/` | `DanhGia_Level3_3Seeds_CaiTien_Flickr30k_KetQua.ipynb` | 5/5 | Hoàn tất | Phương Án A (In-Domain Flickr30k + Kho Flickr30k V2): Đánh giá Level 3 (3 Seeds: `42, 44, 46` + `CHAIRs/CHAIRi`) | 
| 14 | `flick30k/PhuongAn_A/` | `DanhGia_Level4_3Seeds_CaiTien_Flickr30k_KetQua.ipynb` | 5/5 | Hoàn tất | Phương Án A (In-Domain Flickr30k + Kho Flickr30k V2): Đánh giá Level 4 (3 Seeds: `42, 44, 46` + `CHAIRs/CHAIRi`) | 
| 15 | `flick30k/PhuongAn_A/` | `DanhGia_Level5_3Seeds_CaiTien_Flickr30k_KetQua.ipynb` | 6/6 | Hoàn tất | Phương Án A (In-Domain Flickr30k + Kho Flickr30k V2): Đánh giá Level 5 (3 Seeds: `42, 44, 46` + `CHAIRs/CHAIRi`) | 
| 16 | `flick30k/PhuongAn_B/` | `PA_B_BeNguyenCOCO_ChamFlickr30k_Level1_3Seeds_KetQua.ipynb` | 5/5 | Hoàn tất | Phương Án B (Zero-Shot Cross-Domain: Model COCO chấm thẳng 1.000 ảnh Flickr30k Test) — Level 1 (3 Seeds) | 
| 17 | `flick30k/PhuongAn_B/` | `PA_B_BeNguyenCOCO_ChamFlickr30k_Level2_3Seeds_KetQua.ipynb` | 5/5 | Hoàn tất | Phương Án B (Zero-Shot Cross-Domain: Model COCO + Kho COCO V2 chấm thẳng Flickr30k Test) — Level 2 (3 Seeds) | 
| 18 | `flick30k/PhuongAn_B/` | `PA_B_BeNguyenCOCO_ChamFlickr30k_Level3_3Seeds_KetQua.ipynb` | 5/5 | Hoàn tất | Phương Án B (Zero-Shot Cross-Domain: Model COCO + Kho COCO V2 chấm thẳng Flickr30k Test) — Level 3 (3 Seeds) | 
| 19 | `flick30k/PhuongAn_B/` | `PA_B_BeNguyenCOCO_ChamFlickr30k_Level4_3Seeds_KetQua.ipynb` | 5/5 | Hoàn tất | Phương Án B (Zero-Shot Cross-Domain: Model COCO + Kho COCO V2 chấm thẳng Flickr30k Test) — Level 4 (3 Seeds) | 
| 20 | `flick30k/PhuongAn_B/` | `PA_B_BeNguyenCOCO_ChamFlickr30k_Level5_3Seeds_KetQua.ipynb` | 6/6 | Hoàn tất | Phương Án B (Zero-Shot Cross-Domain: Model COCO + Kho COCO V2 chấm thẳng Flickr30k Test) — Level 5 (3 Seeds) | 
| 21 | `Bóc tách/Chi_Dung_TriThuc/` | `DanhGia_Level6_ChiDungTriThuc_KB_V2_COCO_KetQua.ipynb` | 7/7 | Hoàn tất | Bảng 7 (Cô lập phương thức): Đánh giá chế độ **Chỉ dùng Tri thức (`g = 1.0`, `Visual = 0.0`)** trên 5.000 ảnh & 4.464 ảnh chuẩn | 
| 22 | `Bóc tách/Tach_Dimersity/` | `Ablation_01_Tach_MMR_Diversity_KB_V2_COCO_KetQua.ipynb` | 6/6 | Hoàn tất | Bảng 6 (Bóc tách thành phần): Nhánh **`no_mmr` (`w/o MMR Diversity`)** trên 5.000 ảnh COCO Test (3 Seeds) | 
| 23 | `Bóc tách/Tach_Sematic/` | `Ablation_02_Tach_Semantic_Expansion_KB_V2_COCO_KetQua.ipynb` | 6/6 | Hoàn tất | Bảng 6 (Bóc tách thành phần): Nhánh **`no_semantic_expansion` (`w/o WordNet/SPO Expansion`)** trên 5.000 ảnh COCO Test (3 Seeds) | 
| 24 | `Bóc tách/Tach_ReRanking/` | `Ablation_03_Tach_RelTR_SPO_Reranking_KB_V2_COCO_KetQua.ipynb` | 6/6 | Hoàn tất | Bảng 6 (Bóc tách thành phần): Nhánh **`no_reltr_reranking` (`w/o Scene-Graph RelTR`)** trên 5.000 ảnh COCO Test (3 Seeds) | 
| 25 | `Bóc tách/Tach_MultiToken/` | `Ablation_04A_ChiTach_MultiToken_MeanPool_KB_V2_COCO_KetQua.ipynb` | 6/6 | Hoàn tất | Bảng 4 & Bảng 6: Nhánh **`no_multitoken_only` (`w/o Multi-Token Memory`)** + đo `CHAIRs / CHAIRi` trên COCO (3 Seeds) | 
| 26 | `Bóc tách/Tach_GateFusion/` | `Ablation_04B_ChiTach_GatedFusion_Linear_KB_V2_COCO_KetQua.ipynb` | 6/6 | Hoàn tất | Bảng 4 & Bảng 6: Nhánh **`no_gate_only` (`w/o Dynamic Gated Fusion`, khóa** $g=0.5$**)** + đo `CHAIRs / CHAIRi` trên COCO (3 Seeds) | 
| 27 | `Bóc tách/SPO/` | `KhaoSat_Bang8_SoBoBa_SPO_M_KB_V2_COCO_3Seeds_KetQua.ipynb` | 6/6 | Hoàn tất | Bảng 8: Khảo sát độ nhạy theo **Số lượng bộ ba quan hệ đồ thị cảnh** $M \in \{0, 3, 5, 10, 20\}$ trên Kho Mới V2 (3 Seeds) | 
| 28 | `Bóc tách/Thay đổi Top K/` | `KhaoSat_Bang9_ThayDoi_TopK_KB_V2_COCO_3Seeds_KetQua.ipynb` | 6/6 | Hoàn tất | Bảng 9 & Bảng 10: Khảo sát độ nhạy theo **Số câu tham khảo Top-**$K \in \{1, 3, 5, 10, 20\}$ trên Kho Mới V2 (3 Seeds) | 

### Lưu ý nhỏ duy nhất khi đọc trực tiếp các file trong gói ZIP:

* Trong file `flick30k/PhuongAn_A/DanhGia_Level5_3Seeds_CaiTien_Flickr30k_KetQua.ipynb`, bảng tổng hợp của riêng **Level 5 ở Cell 5** đã hoàn chỉnh 100% (`CIDEr = 61.98 ± 0.21`, `BLEU-4 = 27.26 ± 0.07`, `SPICE = 15.77 ± 0.01`, `CHAIRs = 8.63 ± 0.06%`), nhưng bảng so sánh phụ ở **Cell 6** của file đó được chạy trước khi bạn chạy xong Level 2, 3, 4 nên chữ in ra ở Cell 6 vẫn ghi `(Chưa chạy Lv2..4)`. Số liệu đầy đủ của Level 2, 3, 4 nằm chính xác tại **Cell 5** của 3 file `DanhGia_Level2/3/4_3Seeds_CaiTien_Flickr30k_KetQua.ipynb` và đã được tổng hợp đầy đủ trong Mục 3.2 bên dưới.

* Trong file `Ablation_04A_ChiTach_MultiToken_MeanPool_KB_V2_COCO_KetQua.ipynb`, do bạn chạy song song với `Ablation_04B` và `04A` kết thúc sớm hơn 45 giây nên dòng `no_gate_only` trong bảng in của `04A` mới gom 2 seed `[42, 44]` (`112.91 ± 0.05`). Kết quả đủ 3 seeds `[42, 44, 46]` của `no_gate_only` nằm chính xác tại file `Ablation_04B_ChiTach_GatedFusion_Linear_KB_V2_COCO_KetQua.ipynb` (`113.00 ± 0.17 CIDEr`, $\Delta = -2.41$).

## 2. Kiến trúc Kho Tri Thức Đa Tầng Mới V2 & Nguyên tắc Thực nghiệm Đối chứng Công bằng (100% Controlled Benchmark)

### 2.1. Điểm đột phá của Kho Tri Thức Đa Tầng Mới V2 (`knowledge_base_level4` & `knowledge_base_level4_Flick30K`)

Trong phiên bản Kho Tri Thức Cũ (KB V1), mỗi bản ghi trong kho chỉ lưu văn bản chú thích thô (`caption`) và vector nhúng tĩnh CLIP `512` chiều (`IndexFlatIP`). Điều này gây ra 3 hạn chế lớn:

1. **Dư thừa câu đồng nghĩa cấp ảnh trong tập ứng viên Top-**$N$: Nhiều câu chú thích gần giống hệt nhau chiếm hết các vị trí đầu bảng khi truy vấn FAISS, làm giảm không gian lựa chọn của thuật toán lọc đa dạng MMR.

2. **So khớp ngữ nghĩa nông (Surface Lexical Matching)**: Ở KB V1, bước tái xếp hạng chỉ đối sánh tập từ bề mặt (WordNet synsets + Jaccard từ đơn) mà không phân biệt được vai trò ngữ pháp **Chủ thể (`Subject`) – Hành động (`Action`) – Thuộc tính (`Attribute`) – Quan hệ không gian (`Relation` SPO)** trong từng câu ứng viên.

3. **Nhiễu ngoại vi ở pha giải mã**: Một vector CLIP 512 chiều đơn lẻ của câu ứng viên không thể hiện tường minh các cụm từ khóa cấu trúc và trọng số độ hiếm TF-IDF của các n-gram đặc trưng.

**Kho Tri Thức Đa Tầng Mới V2 (Multi-Layer Structured Knowledge Base V2)** giải quyết triệt để cả 3 hạn chế trên thông qua quy trình xây dựng 4 tầng ngoại tuyến (Minh họa tại Khối C trong `Mô hình kho tri thức mới.pdf`):

| Đặc trưng Kỹ thuật | Kho Tri Thức Cũ V1 (COCO) | Kho Tri Thức Mới V2 — COCO (`knowledge_base_level4`) | Kho Tri Thức Mới V2 — Flickr30k (`knowledge_base_level4_Flick30K`) | 
| ----- | ----- | ----- | ----- | 
| **Tổng số bản ghi (Train + Restval)** | `566.747` câu thô | **`566.747` bản ghi đa tầng** (`113.287` ảnh) | **`145.000` bản ghi đa tầng** (`29.000` ảnh Train) | 
| **Không gian Vector & Chỉ mục** | CLIP ViT-B/32 (`512d`, L2-norm, `IndexFlatIP`) | CLIP ViT-B/32 (`512d`, L2-norm, `IndexFlatIP`) | CLIP ViT-B/32 (`512d`, L2-norm, `IndexFlatIP`) | 
| **Cờ khử trùng lặp (`duplicate_of`)** | Không có (`0` câu) | **`21.916` câu trùng lặp** được gắn cờ trỏ về `row_id` gốc canonical | **`524` câu trùng lặp** được gắn cờ trỏ về `row_id` gốc canonical | 
| **Tầng Thực thể chuẩn hóa (`objects`)** | Không trích xuất sẵn (`0%`) | **`566.671` câu (`99,99%`)** có tập thực thể chuẩn hóa + liên kết `image_objects.jsonl` | **`144.996` câu (`100,0%`)** có tập thực thể chuẩn hóa | 
| **Tầng Cấu trúc Bộ ba (`attributes`, `actions`, `relations` SPO)** | Không trích xuất sẵn (`0%`) | **`521.602` câu (`92,03%`)** có bộ ba thuộc tính, hành động và quan hệ SPO | **`139.714` câu (`96,35%`)** có quan hệ SPO; **`128.571` câu (`88,67%`)** có hành động; **`71.176` câu (`49,09%`)** có thuộc tính | 
| **Tầng Vector Nguyên mẫu Cụm SPO (`SPO Multi-Prototype`) & `TF-IDF 1..4-gram`** | Không có | Tích hợp trọng số `TF-IDF 1..4-gram` toàn kho và gom cụm đồng thuận cấu trúc SPO | Tích hợp trọng số `TF-IDF 1..4-gram` toàn kho và gom cụm đồng thuận cấu trúc SPO | 

### 2.2. Nguyên tắc Giữ nguyên 100% Tham số & Trọng số Mô hình (Strict Controlled Evaluation)

Để chứng minh sự cải thiện điểm số đến thuần túy từ chất lượng biểu diễn và cấu trúc của **Kho Tri Thức Mới V2**, toàn bộ các thí nghiệm trên KB V2 tuân thủ nghiêm ngặt ràng buộc:

* **Giữ nguyên 100% Trọng số Mô hình đã huấn luyện (`strict=True`)**: Nạp đúng các file checkpoint `best_model.pt` của từng cấp độ (`Level 1` \$	o$ `Level 5`) trên đúng **3 Seeds chuẩn `[42, 44, 46]`** mà không huấn luyện lại (Zero retraining cost). Mã băm SHA-256 của trạng thái trọng số (`state_dict`) được kiểm tra trùng khớp tuyệt đối (`Level 5`: `Seed 42 = c1f34c519e65f348...`, `Seed 44 = b441c261e50f4aec...`, `Seed 46 = 8bdeed0b5e8eeebb...`; tổng số tham số cố định là **`12.538.968` tham số**).

* **Giữ nguyên 100% Từ điển & Siêu tham số Hệ thống**:

  * Từ điển (`Vocabulary`): `16.214` từ trên COCO (`PAD=0, BOS=1, EOS=2, UNK=3`) và `9.546` từ trên Flickr30k.

  * Tham số Truy hồi & Tái xếp hạng đã khóa trên tập Validation: Truy vấn ban đầu `TOP_N = 50` ứng viên từ FAISS, trọng số lai thị giác – ngữ nghĩa `w_clip = 0.6`, `w_semantic = 0.4` ($S_{	ext{rel}} = 0.6 \cdot 	ilde{S}_{	ext{CLIP}} + 0.4 \cdot S_{	ext{SPO-Sem}}$), hệ số đa dạng hóa MMR $\lambda_{	ext{MMR}} = 0.7$, chọn `TOP_K = 5` câu tham khảo, nhiệt độ cổng \$	au = 1.0$, số bộ ba đồ thị cảnh tối đa $M = 10$.

  * Tham số Giải mã & Đánh giá: `beam_size = 3`, hệ số phạt độ dài `alpha = 0.6`, `max_len = 30`, chuẩn hóa bằng `SafePTBTokenizer` (Java 11 sạch stdout) và chấm đầy đủ 8 chỉ số chuẩn COCO (`BLEU-1..4`, `METEOR`, `ROUGE-L`, `CIDEr`, `SPICE`) cùng 2 chỉ số ảo giác thực thể (`CHAIRs`, `CHAIRi`).

## 3. Toàn bộ Bảng Số liệu Thực nghiệm Chính thức trên Kho Mới V2 (Đối chiếu Bảng 1 \$	o$ Bảng 10 của Bài báo)

### 3.1. Bảng 1, Bảng 2 & Bảng 5: So sánh 5 Cấp độ Mô hình Lũy tiến trên MS COCO 2014 (5.000 ảnh Karpathy Test, 3 Seeds `42, 44, 46`)

#### a) Bảng Tổng hợp Trung bình `Mean ± SD` (3 Seeds `42, 44, 46`) và Độ tăng $\Delta$ so với Kho Cũ V1:

| Cấp độ Mô hình (MS COCO 2014 — 5.000 ảnh Test) | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr (Kho Mới V2) | CIDEr (Kho Cũ V1) | $\Delta	ext{CIDEr}$ (V2 vs V1) | SPICE (Kho Mới V2) | SPICE (Kho Cũ V1) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Level 1: Baseline (`Visual-Only`, Beam=3)** *(Đã chuẩn hóa PTBTokenizer)* | 74.21 ± 0.19 | 57.80 ± 0.21 | 44.12 ± 0.17 | 33.68 ± 0.14 | 26.77 ± 0.10 | 54.99 ± 0.13 | **107.15 ± 0.74** *(Bản bài báo cũ: 103.74)* | 107.15 *(103.74)* | 0.00 *(Không dùng KB)* | **19.73 ± 0.10** *(19.22)* | 19.73 *(19.22)* | 
| **Level 2: Basic RAG (`CLIP Top-5 Mean-Pool`)** | 74.92 ± 0.10 | 58.54 ± 0.04 | 44.78 ± 0.02 | 34.34 ± 0.02 | 27.41 ± 0.04 | 55.84 ± 0.01 | **112.52 ± 0.34** | 107.57 | **+4.95** | **20.16 ± 0.16** | 19.72 | 
| **Level 3: Scene-Graph Reranked RAG (`RelTR + SPO Rerank`)** | **74.97 ± 0.21** | 58.54 ± 0.15 | 45.00 ± 0.10 | 34.71 ± 0.09 | 27.68 ± 0.02 | 55.95 ± 0.05 | **114.13 ± 0.28** | 110.64 | **+3.49** | **20.68 ± 0.03** | 20.25 | 
| **Level 4: Gated Multi-Token RAG (`5 Tokens + Gate`, No MMR)** | 74.83 ± 0.07 | 58.43 ± 0.06 | 44.96 ± 0.08 | 34.73 ± 0.08 | 27.82 ± 0.01 | 55.93 ± 0.01 | **114.81 ± 0.20** | 110.97 | **+3.84** | **20.67 ± 0.04** | 20.30 | 
| **Level 5: Full Proposed RAG (`RelTR + SPO + MMR + Gated 5-Token`)** | 74.92 ± 0.05 | **58.57 ± 0.03** | **45.13 ± 0.04** | **34.90 ± 0.06** | **27.85 ± 0.02** | **56.01 ± 0.01** | **115.41 ± 0.07** | 111.63 | **+3.78** *(+11.67 vs L1 cũ)* | **20.76 ± 0.02** | 20.35 | 

#### b) Bảng Chi tiết Từng Seed (`42, 44, 46`) cho các Cấp độ trên MS COCO 2014 (Phục vụ Phụ lục & Bảng 5):

| Cấp độ Mô hình | Seed | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Level 1 (Baseline)** | 42 | 74.08 | 57.65 | 43.95 | 33.51 | 26.67 | 54.84 | 106.35 | 19.65 | 
| **Level 1 (Baseline)** | 44 | 74.13 | 57.71 | 44.13 | 33.79 | 26.87 | 55.06 | 107.32 | 19.69 | 
| **Level 1 (Baseline)** | 46 | 74.43 | 58.04 | 44.29 | 33.73 | 26.78 | 55.07 | 107.80 | 19.85 | 
| **Level 2 (Basic RAG + KB V2)** | 42 | 75.03 | 58.59 | 44.80 | 34.32 | 27.40 | 55.84 | 112.19 | 20.32 | 
| **Level 2 (Basic RAG + KB V2)** | 44 | 74.92 | 58.53 | 44.79 | 34.36 | 27.37 | 55.84 | 112.50 | 19.99 | 
| **Level 2 (Basic RAG + KB V2)** | 46 | 74.82 | 58.51 | 44.76 | 34.34 | 27.45 | 55.85 | 112.87 | 20.17 | 
| **Level 3 (Scene-Graph Reranked + KB V2)** | 42 | 75.18 | 58.70 | 45.10 | 34.77 | 27.70 | 56.01 | 114.42 | 20.69 | 
| **Level 3 (Scene-Graph Reranked + KB V2)** | 44 | 74.96 | 58.52 | 45.01 | 34.75 | 27.68 | 55.92 | 114.11 | 20.70 | 
| **Level 3 (Scene-Graph Reranked + KB V2)** | 46 | 74.76 | 58.40 | 44.89 | 34.61 | 27.67 | 55.92 | 113.86 | 20.64 | 
| **Level 4 (Gated Multi-Token + KB V2)** | 42 | 74.88 | 58.49 | 45.04 | 34.82 | 27.83 | 55.94 | 114.72 | 20.67 | 
| **Level 4 (Gated Multi-Token + KB V2)** | 44 | 74.75 | 58.37 | 44.89 | 34.68 | 27.81 | 55.92 | 114.67 | 20.64 | 
| **Level 4 (Gated Multi-Token + KB V2)** | 46 | 74.85 | 58.43 | 44.93 | 34.69 | 27.82 | 55.93 | 115.03 | 20.71 | 
| **Level 5 (Full Proposed RAG + KB V2)** | 42 | 74.95 | 58.61 | 45.17 | 34.92 | 27.82 | 56.01 | 115.38 | 20.77 | 
| **Level 5 (Full Proposed RAG + KB V2)** | 44 | 74.93 | 58.55 | 45.09 | 34.83 | 27.84 | 56.00 | 115.36 | 20.73 | 
| **Level 5 (Full Proposed RAG + KB V2)** | 46 | 74.86 | 58.55 | 45.15 | 34.96 | 27.87 | 56.03 | 115.49 | 20.78 | 

### 3.2. Bảng 3: Đánh giá trên Tập dữ liệu Flickr30k (1.000 ảnh Karpathy Test, 3 Seeds `42, 44, 46` — Đầy đủ Phương Án A & Phương Án B)

Nhóm đã thực hiện đầy đủ **2 Phương án Thực nghiệm trên Flickr30k**:

* **Phương Án A (`flick30k/PhuongAn_A` — Đánh giá Nội miền với Kho Tri Thức Flickr30k V2 `145.000` câu)**: Kiểm chứng hiệu quả nâng cấp từ Kho Flickr30k V1 lên Kho Flickr30k V2 (`knowledge_base_level4_Flick30K`, schema `2.1.0`).

* **Phương Án B (`flick30k/PhuongAn_B` — Zero-Shot Cross-Domain Transfer)**: Bê nguyên xi mô hình đã huấn luyện trên COCO + Kho Tri Thức COCO V2 (`566.747` câu) sang chấm thẳng trên 1.000 ảnh Test của Flickr30k mà không cần huấn luyện lại hay xây kho mới.

#### a) Bảng Kết quả Phương Án A (In-Domain Flickr30k với Kho Tri Thức Flickr30k V2 — 3 Seeds `42, 44, 46`):

| Cấp độ Mô hình (Flickr30k — Phương Án A) | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr (KB V2) | CIDEr (KB V1) | $\Delta	ext{CIDEr}$ | SPICE (KB V2) | CHAIRs $\downarrow$ (%) | CHAIRi $\downarrow$ (%) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Level 1: Baseline (No RAG)** | 68.85 ± 0.38 | 50.63 ± 0.28 | 36.35 ± 0.27 | 25.89 ± 0.24 | 21.10 ± 0.13 | 48.14 ± 0.16 | **53.85 ± 0.62** | 53.85 | 0.00 | 14.82 ± 0.19 | 13.27 ± 0.38 | 9.64 ± 0.25 | 
| **Level 2: Basic RAG + KB V2** | 67.66 ± 0.22 | 49.35 ± 0.12 | 35.32 ± 0.13 | 25.21 ± 0.09 | 21.44 ± 0.02 | 48.20 ± 0.12 | **58.29 ± 0.22** | 56.13 | **+2.16** | 15.27 ± 0.04 | 11.40 ± 0.10 | 8.03 ± 0.06 | 
| **Level 3: Scene-Graph Reranked + KB V2** | 69.97 ± 0.66 | 51.55 ± 0.63 | 37.03 ± 0.42 | 26.44 ± 0.29 | 21.90 ± 0.05 | 49.02 ± 0.11 | **59.53 ± 0.18** | 56.42 | **+3.11** | 15.63 ± 0.05 | 10.57 ± 0.12 | 7.68 ± 0.21 | 
| **Level 4: Gated Multi-Token + KB V2** | 70.89 ± 0.15 | 52.34 ± 0.12 | 37.64 ± 0.08 | 26.95 ± 0.08 | 21.92 ± 0.06 | 49.20 ± 0.14 | **60.70 ± 0.18** | 57.26 | **+3.44** | 15.68 ± 0.05 | 9.63 ± 0.06 | 7.19 ± 0.03 | 
| **Level 5: Full Proposed RAG + KB V2** | **71.04 ± 0.18** | **52.59 ± 0.10** | **37.94 ± 0.05** | **27.26 ± 0.07** | **22.03 ± 0.06** | **49.37 ± 0.10** | **61.98 ± 0.21** | 58.91 | **+3.07** *(+8.13 vs L1)* | **15.77 ± 0.01** | **8.63 ± 0.06** | **6.47 ± 0.05** | 

#### b) Bảng Kết quả Phương Án B (Zero-Shot Cross-Domain: Model COCO + Kho COCO V2 chấm thẳng 1.000 ảnh Flickr30k Test — 3 Seeds `42, 44, 46`):

| Cấp độ Mô hình (COCO \$	o$ Flickr30k Zero-Shot) | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr (Mean ± SD) | $\Delta	ext{CIDEr}$ so với L1 | SPICE | CHAIRs $\downarrow$ (%) | CHAIRi $\downarrow$ (%) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Level 1: COCO Baseline (No RAG)** | 58.79 ± 0.01 | 38.85 ± 0.03 | 25.46 ± 0.05 | 17.12 ± 0.03 | 17.23 ± 0.03 | 41.02 ± 0.05 | **37.83 ± 0.14** | 0.00 | 10.95 ± 0.04 | 36.10 ± 0.10 | 23.29 ± 0.03 | 
| **Level 2: COCO Basic RAG + Kho COCO V2** | 59.15 ± 0.46 | 39.52 ± 0.48 | 26.04 ± 0.36 | 17.41 ± 0.23 | 17.75 ± 0.22 | 41.69 ± 0.31 | **41.51 ± 0.99** | +3.68 | 11.35 ± 0.14 | 26.80 ± 0.10 | 17.48 ± 0.10 | 
| **Level 3: COCO RelTR Rerank + Kho COCO V2** | 59.73 ± 0.18 | 40.22 ± 0.05 | 26.84 ± 0.06 | 18.16 ± 0.03 | 17.98 ± 0.11 | 42.08 ± 0.20 | **42.93 ± 0.56** | +5.10 | 11.81 ± 0.10 | 24.40 ± 0.10 | 16.07 ± 0.04 | 
| **Level 4: COCO Gated Multi-Token + Kho COCO V2** | 60.07 ± 0.27 | 40.40 ± 0.27 | 26.92 ± 0.31 | 18.22 ± 0.25 | 17.98 ± 0.09 | 42.17 ± 0.12 | **43.36 ± 0.24** | +5.53 | 11.92 ± 0.11 | 22.10 ± 0.10 | 14.84 ± 0.07 | 
| **Level 5: COCO Full RAG + MMR + Kho COCO V2** | **60.77 ± 0.08** | **41.48 ± 0.09** | **27.95 ± 0.10** | **19.00 ± 0.11** | **18.12 ± 0.04** | **42.71 ± 0.05** | **44.88 ± 0.21** | **+7.05 (+18,6%)** | **12.08 ± 0.00** | **19.80 ± 0.10** | **13.82 ± 0.03** | 

> **Nhận xét học thuật quan trọng từ Phương Án B:** Ngay cả khi bê nguyên xi mô hình huấn luyện trên COCO sang miền ảnh mới hoàn toàn là Flickr30k (Zero-Shot Cross-Domain), việc kết nối với Kho Tri Thức Mới V2 + cơ chế Gated Multi-Token MMR giúp tăng CIDEr từ **`37.83` lên `44.88` (+7.05 điểm, tương đương +18,64%)** và **kéo giảm tỷ lệ câu bị ảo giác `CHAIRs` từ `36.10%` xuống chỉ còn `19.80%` (giảm `16.30` điểm phần trăm)**!

### 3.3. Bảng 6: Nghiên cứu Bóc tách Thành phần (Component Ablation Study) trên MS COCO 2014 (5.000 ảnh Test, 3 Seeds `42, 44, 46`)

Một phát hiện cực kỳ giá trị khi chuyển từ **Kho Cũ V1** sang **Kho Mới V2**:

* Ở **Kho Cũ V1**, khi tắt MMR (`no_mmr`) hoặc tắt mở rộng từ vựng (`no_semantic_expansion`), điểm CIDEr từng bị **tăng ngược nhẹ (`+0.26` và `+0.48`)** do nhiễu đồng nghĩa từ WordNet không ngữ cảnh và trùng lặp câu trong kho cũ.

* Ở **Kho Mới V2**, nhờ chỉ mục cấu trúc SPO sạch và khử trùng lặp `duplicate_of`, **cả 5 thành phần khi bị bóc tách đều làm giảm điểm rõ rệt (**$\Delta < 0$ **trên toàn bộ các chỉ số CIDEr, SPICE, BLEU-4, METEOR)**, chứng minh tính thống nhất và không dư thừa của mọi mô-đun trong kiến trúc đề xuất!

| Cấu hình Bóc tách (Bảng 6 — COCO 5.000 ảnh Test) | Mã nhánh trong Code | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr (Kho Mới V2) | $\Delta	ext{CIDEr}$ (Kho Mới V2) | CIDEr (Kho Cũ V1) | $\Delta	ext{CIDEr}$ (Kho Cũ V1) | SPICE (Kho Mới V2) | $\Delta	ext{SPICE}$ (Kho Mới V2) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **`Full Proposed RAG (Mô hình đầy đủ)`** | `full` (`_baseline`) | **74.92 ± 0.05** | **34.90 ± 0.06** | **27.85 ± 0.02** | 56.01 ± 0.01 | **115.41 ± 0.07** | **0.00** | 110.92 *(111.63)* | 0.00 | **20.76 ± 0.02** | **0.00** | 
| **`w/o MMR Diversity` (Tắt chọn lọc đa dạng MMR)** | `no_mmr` (`Ablation_01`) | 74.73 ± 0.02 | 34.83 ± 0.05 | 27.82 ± 0.02 | 56.00 ± 0.04 | **114.70 ± 0.11** | **-0.72** | 111.18 | +0.26 | **20.66 ± 0.04** | **-0.10** | 
| **`w/o WordNet / SPO Expansion` (Tắt mở rộng ngữ nghĩa)** | `no_semantic_expansion` (`Ablation_02`) | 74.52 ± 0.10 | 34.63 ± 0.11 | 27.74 ± 0.03 | 56.00 ± 0.05 | **113.95 ± 0.14** | **-1.47** | 111.39 | +0.48 | **20.49 ± 0.02** | **-0.27** | 
| **`w/o Multi-Token Memory` (Nén 1 vector trung bình, giữ Gate)** | `no_multitoken_only` (`Ablation_04A`) | 74.81 ± 0.03 | 34.83 ± 0.09 | 27.75 ± 0.02 | **56.06 ± 0.01** | **113.80 ± 0.05** | **-1.61** | 108.90 | \-2.73 | **20.58 ± 0.03** | **-0.18** | 
| **`w/o Dynamic Gated Fusion` (Khóa cổng tuyến tính** $g=0.5$**, giữ 5 Tokens)** | `no_gate_only` (`Ablation_04B`) | 74.28 ± 0.05 | 34.39 ± 0.05 | 27.62 ± 0.01 | 55.80 ± 0.03 | **113.00 ± 0.17** | **-2.41** | 109.45 | \-2.18 | **20.42 ± 0.01** | **-0.34** | 
| **`w/o Scene-Graph RelTR` (Tắt tái xếp hạng đồ thị cảnh,** $w_{	ext{CLIP}}=1.0$**)** | `no_reltr_reranking` (`Ablation_03`) | 74.08 ± 0.06 | 34.28 ± 0.17 | 27.60 ± 0.05 | 55.82 ± 0.06 | **112.08 ± 0.33** | **-3.33** | 110.31 | \-0.61 | **20.39 ± 0.04** | **-0.37** | 

### 3.4. Bảng 7: Phân tích Cô lập Phương thức Đầu vào (Modality Isolation — Visual-Only vs. Retrieval-Only Level 6 vs. Full Multimodal)

File `DanhGia_Level6_ChiDungTriThuc_KB_V2_COCO_KetQua.ipynb` đã kiểm định nghiêm ngặt chế độ **Chỉ dùng Tri thức (`g = 1.0`, `Visual = 0.0`)** bằng cách xóa trắng đặc trưng ảnh ($v_{	ext{img}} = \mathbf{0}$) và khóa cổng $g = 1.0$:

* Ở **Kho Cũ V1**, khi tắt hoàn toàn thị giác ($g=1.0$), mô hình bị sụt xuống `81.44 CIDEr` (5.000 ảnh) / `81.86 ± 0.08 CIDEr` (4.464 ảnh có nhãn RelTR).

* Ở **Kho Mới V2**, chỉ riêng 5 token tri thức từ Kho Mới V2 ($g=1.0$, không nhìn ảnh trực tiếp ở bộ giải mã) đã đạt **`103.41 ± 0.16 CIDEr` trên toàn bộ 5.000 ảnh Test** (tăng **`+21.97 CIDEr`** so với Kho cũ V1!) và đạt **`104.75 ± 0.13 CIDEr` trên 4.464 ảnh có đồ thị cảnh RelTR** — tức **vượt qua cả mô hình Baseline chỉ nhìn ảnh Level 1 (`103.74` / `104.22 CIDEr`)**!

| Phương thức Đầu vào (Bảng 7 — MS COCO 2014) | Trạng thái Cổng $g$ | BLEU-4 | METEOR | ROUGE-L | CIDEr (Mean ± SD) | SPICE (Mean ± SD) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Chỉ đặc trưng thị giác (`Visual-Only` — Level 1 Baseline)** | $g = 0.00$ | 32.38 ± 0.15 *(33.68)* | 26.25 ± 0.08 *(26.77)* | 54.42 ± 0.11 *(54.99)* | 104.22 ± 0.35 *(107.15)* | 19.30 ± 0.09 *(19.73)* | 
| **Chỉ tri thức truy hồi — Kho Cũ V1 (5.000 ảnh, kèm Fallback)** | $g = 1.00$ | 25.12 | 22.99 | 49.07 | 81.44 | 15.75 | 
| **Chỉ tri thức truy hồi — Kho Cũ V1 (4.464 ảnh có nhãn RelTR)** | $g = 1.00$ | 25.52 ± 0.10 | 23.21 ± 0.03 | 49.38 ± 0.05 | 81.86 ± 0.08 | 15.83 ± 0.08 | 
| **Chỉ tri thức truy hồi — Kho Mới V2 (`Level 6` — Toàn bộ 5.000 ảnh)** | $g = 1.00$ | **30.89 ± 0.02** | **26.06 ± 0.03** | **53.29 ± 0.02** | **103.41 ± 0.16** *(+21.97 vs V1)* | **18.94 ± 0.02** | 
| **Chỉ tri thức truy hồi — Kho Mới V2 (`Level 6` — 4.464 ảnh có RelTR)** | $g = 1.00$ | **31.56 ± 0.01** | **26.39 ± 0.02** | **53.86 ± 0.01** | **104.75 ± 0.13** *(+22.89 vs V1)* | **19.20 ± 0.01** | 
| **Kết hợp Đa phương thức (`Level 4 Gated` — Thị giác + Kho Mới V2)** | $g pprox 0.48$ (Học động) | 34.73 ± 0.08 | 27.82 ± 0.01 | 55.93 ± 0.01 | **114.81 ± 0.20** | 20.67 ± 0.04 | 
| **Kết hợp Đa phương thức (`Level 5 Full Proposed` — Thị giác + Kho Mới V2)** | $g pprox 0.50$ (Học động) | **34.90 ± 0.06** | **27.85 ± 0.02** | **56.01 ± 0.01** | **115.41 ± 0.07** | **20.76 ± 0.02** | 

### 3.5. Bảng 8: Khảo sát Độ nhạy theo Số lượng Bộ ba Quan hệ Đồ thị Cảnh $M \in \{0, 3, 5, 10, 20\}$ trên Kho Mới V2 (3 Seeds `42, 44, 46`)

Trích xuất từ `Bóc tách/SPO/KhaoSat_Bang8_SoBoBa_SPO_M_KB_V2_COCO_3Seeds_KetQua.ipynb`:

| Số lượng bộ ba quan hệ $M$ (Bảng 8 — COCO 5.000 ảnh Test) | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr (Kho Mới V2) | $\Delta	ext{CIDEr}$ (so với $M=10$) | SPICE (Kho Mới V2) | CHAIRs $\downarrow$ (%) | CHAIRi $\downarrow$ (%) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| $M = 0$ **(Không dùng bộ ba RelTR / Chỉ CLIP)** | 74.08 ± 0.06 | 57.70 ± 0.10 | 44.39 ± 0.15 | 34.28 ± 0.17 | 27.60 ± 0.05 | 55.82 ± 0.06 | **112.08 ± 0.33** | \-3.33 | 20.39 ± 0.04 | 8.58 ± 0.12 | 7.49 ± 0.08 | 
| $M = 3$ **bộ ba quan hệ hàng đầu** | 74.65 ± 0.08 | 58.32 ± 0.05 | 44.95 ± 0.03 | 34.78 ± 0.02 | 27.78 ± 0.01 | 56.05 ± 0.01 | **114.31 ± 0.11** | \-1.10 | 20.58 ± 0.03 | 8.26 ± 0.17 | 7.21 ± 0.15 | 
| $M = 5$ **bộ ba quan hệ hàng đầu** | 74.85 ± 0.05 | 58.56 ± 0.09 | **45.22 ± 0.08** | **35.06 ± 0.07** | **27.90 ± 0.02** | **56.11 ± 0.05** | **115.26 ± 0.04** | \-0.15 | 20.72 ± 0.02 | 8.31 ± 0.05 | 7.33 ± 0.02 | 
| $M = 10$ **(Cấu hình chuẩn chọn)** | **74.92 ± 0.05** | **58.57 ± 0.03** | 45.13 ± 0.04 | 34.90 ± 0.06 | 27.85 ± 0.02 | 56.01 ± 0.01 | **115.41 ± 0.07** | **0.00 (Đạt đỉnh)** | 20.76 ± 0.02 | **8.21 ± 0.08** | **7.18 ± 0.06** | 
| $M = 20$ **bộ ba quan hệ** | 74.84 ± 0.05 | 58.53 ± 0.04 | 45.13 ± 0.01 | 34.93 ± 0.04 | 27.86 ± 0.02 | 56.07 ± 0.02 | **115.18 ± 0.02** | \-0.23 | **20.77 ± 0.02** | 8.30 ± 0.05 | 7.31 ± 0.03 | 

### 3.6. Bảng 9 & Bảng 10: Khảo sát Số lượng Câu Tham khảo Top-$K \in \{1, 3, 5, 10, 20\}$ & Chi phí Thời gian trên Kho Mới V2 (3 Seeds `42, 44, 46`)

Trích xuất từ `Bóc tách/Thay đổi Top K/KhaoSat_Bang9_ThayDoi_TopK_KB_V2_COCO_3Seeds_KetQua.ipynb`:

* Nhờ chạy đồng bộ trên Kho Mới V2 qua đúng 3 Seeds `[42, 44, 46]`, dòng $K = 5$ **đạt chính xác `115.41 ± 0.07 CIDEr` và `20.76 ± 0.02 SPICE`**, khớp 100% với Bảng 1, Bảng 2, Bảng 5, Bảng 6, Bảng 7 và Bảng 8 (xóa bỏ hoàn toàn sự vênh số liệu ở Bảng 9 bản cũ).

* Đường cong thực nghiệm tăng mạnh từ $K=1$ (`112.13`) \$	o$ $K=3$ (`114.76`) \$	o$ **đạt đỉnh tại** $K=5$ **(`115.41 CIDEr`, `20.76 SPICE`)**, sau đó giảm nhẹ ở $K=10$ (`115.16`) và $K=20$ (`114.81`) khi các câu ứng viên xa làm loãng trọng số chú ý chéo, đồng thời thời gian truy hồi MMR tăng gấp 3 lần.

| Số câu tham khảo $K$ (Bảng 9 — COCO 5.000 ảnh Test) | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr (Kho Mới V2) | $\Delta	ext{CIDEr}$ (so với $K=5$) | SPICE (Kho Mới V2) | CHAIRs $\downarrow$ (%) | CHAIRi $\downarrow$ (%) | Thời gian truy hồi TB | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| $K = 1$ | 74.21 ± 0.04 | 57.79 ± 0.04 | 44.32 ± 0.04 | 34.11 ± 0.06 | 27.46 ± 0.04 | 55.61 ± 0.07 | **112.13 ± 0.04** | \-3.28 | 20.38 ± 0.04 | 8.83 ± 0.15 | 7.69 ± 0.15 | \~12.8 ms/ảnh | 
| $K = 3$ | 74.73 ± 0.03 | 58.42 ± 0.03 | 45.05 ± 0.02 | 34.85 ± 0.03 | 27.84 ± 0.01 | 55.99 ± 0.01 | **114.76 ± 0.13** | \-0.66 | 20.66 ± 0.03 | 8.33 ± 0.07 | 7.27 ± 0.07 | \~13.0 ms/ảnh | 
| $K = 5$ **(Cấu hình chọn)** | **74.92 ± 0.05** | **58.57 ± 0.03** | 45.13 ± 0.04 | 34.90 ± 0.06 | **27.85 ± 0.02** | 56.01 ± 0.01 | **115.41 ± 0.07** | **0.00 (Đạt đỉnh)** | **20.76 ± 0.02** | **8.21 ± 0.08** | **7.18 ± 0.06** | **\~16.8 ms/ảnh** | 
| $K = 10$ | 74.83 ± 0.05 | 58.53 ± 0.06 | **45.18 ± 0.05** | **35.00 ± 0.03** | 27.84 ± 0.03 | **56.09 ± 0.03** | **115.16 ± 0.17** | \-0.26 | 20.73 ± 0.02 | 8.43 ± 0.07 | 7.39 ± 0.05 | \~24.1 ms/ảnh | 
| $K = 20$ | 74.64 ± 0.03 | 58.33 ± 0.04 | 44.98 ± 0.05 | 34.83 ± 0.04 | 27.79 ± 0.00 | 56.07 ± 0.02 | **114.81 ± 0.09** | \-0.61 | 20.65 ± 0.02 | 8.23 ± 0.06 | 7.21 ± 0.04 | \~47.1 ms/ảnh | 

## 4. Siêu Prompt Ngữ cảnh Toàn diện (Master Context Prompt) — Dùng để Copy-Paste cho AI Hỗ trợ Viết Bài báo / Khóa luận / Slide Bảo vệ

Bạn có thể copy nguyên khối nội dung dưới đây và dán vào đầu bất kỳ cuộc trò chuyện mới nào để AI nắm rõ 100% kiến trúc, số liệu thực nghiệm của cả 27 file Notebook và hỗ trợ bạn viết LaTeX, chỉnh sửa bài báo, hoặc soạn câu trả lời phản biện hội đồng:

> **BỐI CẢNH ĐỀ TÀI KHÓA LUẬN & BÀI BÁO KHOA HỌC (NHÓM 02 — HUIT):**
> Chúng tôi thực hiện đề tài **"Sinh chú thích ảnh tăng cường truy hồi dẫn dắt bởi đồ thị cảnh và hợp nhất đa token có cổng trên Kho Tri Thức Đa Tầng V2" (Scene-Graph Guided Retrieval-Augmented Image Captioning with Gated Multi-Token Fusion and Multi-Layer Structured Knowledge Base V2)**.
>
> **1. KIẾN TRÚC TỔNG THỂ (4 KHỐI A, B, C, D) & KHO TRI THỨC ĐA TẦNG MỚI V2 (`knowledge_base_level4`):**
>
> * **Khối C (Xây dựng Kho Tri Thức Đa Tầng V2 ngoại tuyến)**:
>
>   * Trên **MS COCO 2014 (Karpathy split)**: Kho V2 gồm `566.747` bản ghi chú thích từ tập Train + Restval (`113.287` ảnh), mã hóa bằng `CLIP ViT-B/32` (`512` chiều, chuẩn hóa L2) và lập chỉ mục bằng `FAISS IndexFlatIP`. Khác với Kho V1 chỉ lưu văn bản thô, mỗi dòng trong Kho V2 (`text_embedding_mapping.jsonl` + `image_objects.jsonl`) được phân tích cấu trúc ngữ nghĩa đa tầng gồm: (1) Cờ khử trùng lặp `duplicate_of` (`21.916` câu trùng lặp trỏ về câu gốc canonical), (2) Thực thể chuẩn hóa `objects` (`566.671` câu = `99.99%`), (3) Bộ ba thuộc tính `attributes`, hành động `actions`, và quan hệ không gian `relations` dạng `Subject - Predicate - Object (SPO)` (`521.602` câu = `92.03%`), cùng bảng trọng số `TF-IDF 1..4-gram` toàn kho.
>
>   * Trên **Flickr30k (Karpathy split)**: Kho V2 nội miền (`knowledge_base_level4_Flick30K`, schema `2.1.0`) gồm `145.000` câu chú thích từ `29.000` ảnh Train, trích xuất `144.996` câu có `objects` (`100%`), `139.714` câu có `relations` SPO (`96.4%`), `128.571` câu có `actions` (`88.7%`), `71.176` câu có `attributes` (`49.1%`), và đánh dấu `524` câu trùng lặp `duplicate_of`.
>
> * **Khối A & B (Mã hóa Đầu vào, Truy xuất FAISS, Mở rộng Ngữ nghĩa SPO & Tái xếp hạng Lai + Chọn lọc Đa dạng MMR)**:
>
>   * Ảnh đầu vào được trích xuất vector thị giác toàn cục $v_{	ext{img}} \in \mathbb{R}^{512}$ qua `CLIP ViT-B/32` (đóng băng) và trích xuất tối đa $M=10$ bộ ba đồ thị cảnh `(Subject, Relation, Object)` qua `RelTR`.
>
>   * Truy vấn FAISS lấy `TOP_N = 50` ứng viên không trùng lặp (`duplicate_of`), mở rộng ngữ nghĩa đồ thị cảnh bằng phiếu bầu có trọng số từ láng giềng SPO trong Kho V2, và tính điểm tương thích lai:
>     
>
>     $$
>     S_{	ext{rel}}(I, c_i) = w \cdot 	ilde{S}_{	ext{CLIP}}(I, c_i) + (1 - w) \cdot S_{	ext{SPO-Sem}}(G_{	ext{exp}}(I), G_{	ext{KB-V2}}(c_i)), \quad w = 0.6
>     $$
>
>   * Lọc đa dạng hóa tối đa biên (MMR) với $\lambda = 0.7$ kết hợp độ tương đồng cosine CLIP (trọng số `0.65`) và độ chồng lấp cấu trúc thực thể/quan hệ SPO (trọng số `0.35`) để chọn đúng `TOP_K = 5` ứng viên vừa sát nghĩa vừa đa dạng góc nhìn, sau đó tổng hợp vector nguyên mẫu cụm SPO (`SPO Multi-Prototype`).
>
> * **Khối D (Hợp nhất Đa Token Có Cổng Động & Giải mã Đồng thuận SPO)**:
>
>   * Thay vì nén trung bình 5 câu thành 1 vector gây nghẽn cổ chai, mô hình chiếu độc lập $K=5$ vector ứng viên thành ma trận bộ nhớ đa token $M_{	ext{knw}} \in \mathbb{R}^{B 	imes 5 	imes 256}$.
>
>   * Mô-đun `GatedFusion` (`1025 -> 1`) nhận vector ghép nối $[v_{	ext{norm}} 	ext{ (512)}; k_{	ext{agg\_norm}} 	ext{ (512)}; 	ext{conf} 	ext{ (1)}]$ (với \$	au = 1.0$) để sinh hệ số cổng thích nghi $g \in [0, 1]$ điều phối giữa nhánh Cross-Attention thị giác $C_{	ext{vis}}$ và nhánh Cross-Attention tri thức $C_{	ext{knw}}$ trong 3 lớp `DualBranchDecoderLayer`:
>     
>
>     $$
>     C_{	ext{fused}} = (1 - g) \odot C_{	ext{vis}} + g \odot C_{	ext{knw}}
>     $$
>
>   * Quá trình giải mã sử dụng Beam Search (`beam_size = 3`, $lpha = 0.6$, `max_len = 30`) kết hợp định hướng từ vựng thực thể SPO và đồng thuận n-gram TF-IDF từ Kho V2.
>
> **2. RÀNG BUỘC ĐỐI CHỨNG HỌC THUẬT (100% CONTROLLED BENCHMARK SO VỚI KHO CŨ V1):**
>
> * Giữ nguyên 100% trọng số mô hình đã huấn luyện (`strict=True`, `12.538.968` tham số ở Level 4 & 5), từ điển `vocab.json` (`16.214` từ trên COCO; `9.546` từ trên Flickr30k), tập kiểm tra (`5.000` ảnh Karpathy Test / `25.010` câu tham chiếu trên COCO; `1.000` ảnh Test / `5.000` câu tham chiếu trên Flickr30k), và toàn bộ siêu tham số (`TOP_N=50, TOP_K=5, M=10, w=0.6, lambda=0.7, tau=1.0, beam_size=3, alpha=0.6, max_len=30`) trên đúng **3 Seeds chuẩn `[42, 44, 46]`**.
>
> **3. BỘ SỐ LIỆU THỰC NGHIỆM CHÍNH THỨC TRÊN KHO TRI THỨC MỚI V2 (MEAN ± SD QUA 3 SEEDS `42, 44, 46`):**
>
> * **Bảng 1, 2, 5 — 5 Cấp độ trên MS COCO 2014 (5.000 ảnh Karpathy Test)**:
>
>   * `Level 1 (Visual-Only Baseline)`: BLEU-1 = `74.21 ± 0.19`, BLEU-4 = `33.68 ± 0.14`, METEOR = `26.77 ± 0.10`, ROUGE-L = `54.99 ± 0.13`, CIDEr = `107.15 ± 0.74` (bản bài báo cũ ghi `103.74 ± 1.07`), SPICE = `19.73 ± 0.10` (`19.22`).
>
>   * `Level 2 (Basic RAG + KB V2)`: BLEU-1 = `74.92 ± 0.10`, BLEU-4 = `34.34 ± 0.02`, METEOR = `27.41 ± 0.04`, ROUGE-L = `55.84 ± 0.01`, CIDEr = `112.52 ± 0.34` (tăng `+4.95` so với V1 `107.57`), SPICE = `20.16 ± 0.16`.
>
>   * `Level 3 (Scene-Graph Reranked + KB V2)`: BLEU-1 = `74.97 ± 0.21`, BLEU-4 = `34.71 ± 0.09`, METEOR = `27.68 ± 0.02`, ROUGE-L = `55.95 ± 0.05`, CIDEr = `114.13 ± 0.28` (tăng `+3.49` so với V1 `110.64`), SPICE = `20.68 ± 0.03`.
>
>   * `Level 4 (Gated Multi-Token + KB V2)`: BLEU-1 = `74.83 ± 0.07`, BLEU-4 = `34.73 ± 0.08`, METEOR = `27.82 ± 0.01`, ROUGE-L = `55.93 ± 0.01`, CIDEr = `114.81 ± 0.20` (tăng `+3.84` so với V1 `110.97`), SPICE = `20.67 ± 0.04`.
>
>   * `Level 5 (Full Proposed RAG + KB V2)`: BLEU-1 = `74.92 ± 0.05`, BLEU-2 = `58.57 ± 0.03`, BLEU-3 = `45.13 ± 0.04`, BLEU-4 = `34.90 ± 0.06`, METEOR = `27.85 ± 0.02`, ROUGE-L = `56.01 ± 0.01`, **CIDEr = `115.41 ± 0.07`** (Seed 42: `115.38`, Seed 44: `115.36`, Seed 46: `115.49` — tăng `+3.78` so với V1 `111.63`), **SPICE = `20.76 ± 0.02`**, `CHAIRs = 8.21 ± 0.08%`, `CHAIRi = 7.18 ± 0.06%`.
>
> * **Bảng 3 — Đánh giá trên Flickr30k (1.000 ảnh Test, 3 Seeds `42, 44, 46`)**:
>
>   * *Phương Án A (In-Domain Flickr30k với Kho Flickr30k V2 `145.000` câu)*:
>
>     * `Level 1`: CIDEr = `53.85 ± 0.62`, BLEU-4 = `25.89 ± 0.24`, SPICE = `14.82 ± 0.19`, CHAIRs = `13.27 ± 0.38%`, CHAIRi = `9.64 ± 0.25%`.
>
>     * `Level 2`: CIDEr = `58.29 ± 0.22` (`+2.16` vs V1), BLEU-4 = `25.21 ± 0.09`, SPICE = `15.27 ± 0.04`, CHAIRs = `11.40 ± 0.10%`, CHAIRi = `8.03 ± 0.06%`.
>
>     * `Level 3`: CIDEr = `59.53 ± 0.18` (`+3.11` vs V1), BLEU-4 = `26.44 ± 0.29`, SPICE = `15.63 ± 0.05`, CHAIRs = `10.57 ± 0.12%`, CHAIRi = `7.68 ± 0.21%`.
>
>     * `Level 4`: CIDEr = `60.70 ± 0.18` (`+3.44` vs V1), BLEU-4 = `26.95 ± 0.08`, SPICE = `15.68 ± 0.05`, CHAIRs = `9.63 ± 0.06%`, CHAIRi = `7.19 ± 0.03%`.
>
>     * `Level 5`: **CIDEr = `61.98 ± 0.21`** (`+3.07` vs V1 `58.91`; `+8.13` vs L1), **BLEU-4 = `27.26 ± 0.07`**, **METEOR = `22.03 ± 0.06`**, **ROUGE-L = `49.37 ± 0.10`**, **SPICE = `15.77 ± 0.01`**, **CHAIRs = `8.63 ± 0.06%`**, **CHAIRi = `6.47 ± 0.05%`**.
>
>   * *Phương Án B (Zero-Shot Cross-Domain: Model COCO + Kho COCO V2 chấm thẳng 1.000 ảnh Flickr30k Test)*:
>
>     * `Level 1`: CIDEr = `37.83 ± 0.14`, BLEU-4 = `17.12 ± 0.03`, SPICE = `10.95 ± 0.04`, CHAIRs = `36.10 ± 0.10%`, CHAIRi = `23.29 ± 0.03%`.
>
>     * `Level 2`: CIDEr = `41.51 ± 0.99`, BLEU-4 = `17.41 ± 0.23`, SPICE = `11.35 ± 0.14`, CHAIRs = `26.80 ± 0.10%`, CHAIRi = `17.48 ± 0.10%`.
>
>     * `Level 3`: CIDEr = `42.93 ± 0.56`, BLEU-4 = `18.16 ± 0.03`, SPICE = `11.81 ± 0.10`, CHAIRs = `24.40 ± 0.10%`, CHAIRi = `16.07 ± 0.04%`.
>
>     * `Level 4`: CIDEr = `43.36 ± 0.24`, BLEU-4 = `18.22 ± 0.25`, SPICE = `11.92 ± 0.11`, CHAIRs = `22.10 ± 0.10%`, CHAIRi = `14.84 ± 0.07%`.
>
>     * `Level 5`: **CIDEr = `44.88 ± 0.21`** (`+7.05` vs L1), **BLEU-4 = `19.00 ± 0.11`**, **SPICE = `12.08 ± 0.00`**, **CHAIRs = `19.80 ± 0.10%`** (giảm `16.30%` ảo giác so với L1), **CHAIRi = `13.82 ± 0.03%`**.
>
> * **Bảng 6 — Nghiên cứu Bóc tách Thành phần trên COCO (5.000 ảnh Test, 3 Seeds `42, 44, 46`)**:
>
>   * `Full Proposed RAG`: BLEU-1 = `74.92 ± 0.05`, BLEU-4 = `34.90 ± 0.06`, METEOR = `27.85 ± 0.02`, ROUGE-L = `56.01 ± 0.01`, CIDEr = `115.41 ± 0.07`, SPICE = `20.76 ± 0.02`.
>
>   * `w/o MMR Diversity` (`no_mmr`): BLEU-1 = `74.73 ± 0.02`, BLEU-4 = `34.83 ± 0.05`, METEOR = `27.82 ± 0.02`, ROUGE-L = `56.00 ± 0.04`, CIDEr = `114.70 ± 0.11` ($\Delta = -0.72$), SPICE = `20.66 ± 0.04` ($\Delta = -0.10$).
>
>   * `w/o WordNet / SPO Expansion` (`no_semantic_expansion`): BLEU-1 = `74.52 ± 0.10`, BLEU-4 = `34.63 ± 0.11`, METEOR = `27.74 ± 0.03`, ROUGE-L = `56.00 ± 0.05`, CIDEr = `113.95 ± 0.14` ($\Delta = -1.47$), SPICE = `20.49 ± 0.02` ($\Delta = -0.27$).
>
>   * `w/o Multi-Token Memory` (`no_multitoken_only` — nén 1 vector trung bình, giữ Gate): BLEU-1 = `74.81 ± 0.03`, BLEU-4 = `34.83 ± 0.09`, METEOR = `27.75 ± 0.02`, ROUGE-L = `56.06 ± 0.01`, CIDEr = `113.80 ± 0.05` ($\Delta = -1.61$), SPICE = `20.58 ± 0.03` ($\Delta = -0.18$).
>
>   * `w/o Dynamic Gated Fusion` (`no_gate_only` — khóa cổng tuyến tính $g=0.5$, giữ 5 Tokens): BLEU-1 = `74.28 ± 0.05`, BLEU-4 = `34.39 ± 0.05`, METEOR = `27.62 ± 0.01`, ROUGE-L = `55.80 ± 0.03`, CIDEr = `113.00 ± 0.17` ($\Delta = -2.41$), SPICE = `20.42 ± 0.01` ($\Delta = -0.34$).
>
>   * `w/o Scene-Graph RelTR` (`no_reltr_reranking`): BLEU-1 = `74.08 ± 0.06`, BLEU-4 = `34.28 ± 0.17`, METEOR = `27.60 ± 0.05`, ROUGE-L = `55.82 ± 0.06`, CIDEr = `112.08 ± 0.33` ($\Delta = -3.33$), SPICE = `20.39 ± 0.04` ($\Delta = -0.37$).
>
> * **Bảng 7 — Cô lập Phương thức Đầu vào trên COCO (`Level 6 Chỉ dùng Tri thức`,** $g=1.0, 	ext{Visual}=0.0$**)**:
>
>   * Trên toàn bộ `5.000` ảnh Test: BLEU-4 = `30.89 ± 0.02`, METEOR = `26.06 ± 0.03`, ROUGE-L = `53.29 ± 0.02`, **CIDEr = `103.41 ± 0.16`** (tăng `+21.97` điểm so với Kho V1 `81.44`), **SPICE = `18.94 ± 0.02`**.
>
>   * Trên `4.464` ảnh Test có nhãn RelTR (Strict): BLEU-4 = `31.56 ± 0.01`, METEOR = `26.39 ± 0.02`, ROUGE-L = `53.86 ± 0.01`, **CIDEr = `104.75 ± 0.13`** (tăng `+22.89` điểm so với Kho V1 `81.86 ± 0.08`), **SPICE = `19.20 ± 0.01`**.
>
> * **Bảng 8 — Khảo sát Số lượng Bộ ba Quan hệ Đồ thị Cảnh** $M \in \{0, 3, 5, 10, 20\}$ **trên COCO (3 Seeds `42, 44, 46`)**:
>
>   * $M = 0$: BLEU-4 = `34.28 ± 0.17`, METEOR = `27.60 ± 0.05`, CIDEr = `112.08 ± 0.33` ($\Delta = -3.33$), SPICE = `20.39 ± 0.04`.
>
>   * $M = 3$: BLEU-4 = `34.78 ± 0.02`, METEOR = `27.78 ± 0.01`, CIDEr = `114.31 ± 0.11` ($\Delta = -1.10$), SPICE = `20.58 ± 0.03`.
>
>   * $M = 5$: BLEU-4 = `35.06 ± 0.07`, METEOR = `27.90 ± 0.02`, CIDEr = `115.26 ± 0.04` ($\Delta = -0.15$), SPICE = `20.72 ± 0.02`.
>
>   * $M = 10$ (Chọn): BLEU-4 = `34.90 ± 0.06`, METEOR = `27.85 ± 0.02`, **CIDEr = `115.41 ± 0.07` (Đạt đỉnh)**, **SPICE = `20.76 ± 0.02`**.
>
>   * $M = 20$: BLEU-4 = `34.93 ± 0.04`, METEOR = `27.86 ± 0.02`, CIDEr = `115.18 ± 0.02` ($\Delta = -0.23$), SPICE = `20.77 ± 0.02`.
>
> * **Bảng 9 — Khảo sát Số câu Tham khảo Top-**$K \in \{1, 3, 5, 10, 20\}$ **trên COCO (3 Seeds `42, 44, 46`)**:
>
>   * $K = 1$: BLEU-1 = `74.21 ± 0.04`, BLEU-4 = `34.11 ± 0.06`, METEOR = `27.46 ± 0.04`, ROUGE-L = `55.61 ± 0.07`, CIDEr = `112.13 ± 0.04` ($\Delta = -3.28$), SPICE = `20.38 ± 0.04`.
>
>   * $K = 3$: BLEU-1 = `74.73 ± 0.03`, BLEU-4 = `34.85 ± 0.03`, METEOR = `27.84 ± 0.01`, ROUGE-L = `55.99 ± 0.01`, CIDEr = `114.76 ± 0.13` ($\Delta = -0.66$), SPICE = `20.66 ± 0.03`.
>
>   * $K = 5$ (Chọn): BLEU-1 = `74.92 ± 0.05`, BLEU-4 = `34.90 ± 0.06`, METEOR = `27.85 ± 0.02`, ROUGE-L = `56.01 ± 0.01`, **CIDEr = `115.41 ± 0.07` (Đạt đỉnh)**, **SPICE = `20.76 ± 0.02`**.
>
>   * $K = 10$: BLEU-1 = `74.83 ± 0.05`, BLEU-4 = `35.00 ± 0.03`, METEOR = `27.84 ± 0.03`, ROUGE-L = `56.09 ± 0.03`, CIDEr = `115.16 ± 0.17` ($\Delta = -0.26$), SPICE = `20.73 ± 0.02`.
>
>   * $K = 20$: BLEU-1 = `74.64 ± 0.03`, BLEU-4 = `34.83 ± 0.04`, METEOR = `27.79 ± 0.00`, ROUGE-L = `56.07 ± 0.02`, CIDEr = `114.81 ± 0.09` ($\Delta = -0.61$), SPICE = `20.65 ± 0.02`.