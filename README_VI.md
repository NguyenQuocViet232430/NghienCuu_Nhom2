# SemRAG: Mô Hình Sinh Chú Thích Ảnh Tăng Cường Truy Hồi Dẫn Dắt Bởi Đồ Thị Cảnh Và Hợp Nhất Đa Token Có Cổng Trên Kho Tri Thức Đa Tầng

> 🌐 **Ngôn ngữ / Language:** **Tiếng Việt** | [🇺🇸 **Click here for English Version (README.md)**](README.md)

[![Language](https://img.shields.io/badge/Ng%C3%B4n%20ng%E1%BB%AF-English-blue.svg)](README.md)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%201.7%2B-blue.svg)](https://github.com/facebookresearch/faiss)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-purple.svg)](https://github.com/openai/CLIP)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19%20%7C%20Vite-61dafb.svg)](https://react.dev/)
[![Giấy phép](https://img.shields.io/badge/Gi%E1%BA%A5y%20ph%C3%A9p-MIT-green.svg)](LICENSE)

> **Kho lưu trữ nghiên cứu chính thức** phục vụ Khóa luận tốt nghiệp ngành Công nghệ thông tin tại **Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT)**.  
> **Đề tài:** Phát triển mô hình chú thích ảnh tăng cường truy hồi tri thức ngữ nghĩa (*SemRAG*)  
> **Sinh viên thực hiện (Nhóm 02):**  
> 1. **Ngô Quốc Bình** — MSSV: `2001230056` — Lớp: `14DHTH11`  
> 2. **Nguyễn Thị Quyên** — MSSV: `2001230745` — Lớp: `14DHTH12`  
> 3. **Nguyễn Quốc Việt** — MSSV: `2001231038` — Lớp: `14DHTH11`  
> **Giảng viên hướng dẫn:** ThS. Đinh Thị Mận & TS. Nguyễn Văn Thịnh  
> **Thời gian đánh giá:** Tháng 11 năm 2026  

---

## 📖 Mục Lục
- [1. Giới Thiệu & Động Lực Nghiên Cứu](#1-giới-thiệu--động-lực-nghiên-cứu)
- [2. Kiến Trúc Hệ Thống (SemRAG)](#2-kiến-trúc-hệ-thống-semrag)
  - [Khối A: Mã hóa Đa phương thức Đầu vào](#khối-a-mã-hóa-đa-phương-thức-đầu-vào)
  - [Khối B: Tái Xếp Hạng Ngữ Nghĩa & Lọc Đa Dạng MMR](#khối-b-tái-xếp-hạng-ngữ-nghĩa--lọc-đa-dạng-mmr)
  - [Khối C: Kho Tri Thức Cấu Trúc Đa Tầng (SemKB)](#khối-c-kho-tri-thức-cấu-trúc-đa-tầng-semkb)
  - [Khối D: Bộ Giải Mã Hợp Nhất Đa Token Có Cổng Động](#khối-d-bộ-giải-mã-hợp-nhất-đa-token-có-cổng-động)
- [3. Kết Quả Thực Nghiệm Đối Chứng](#3-kết-quả-thực-nghiệm-đối-chứng)
  - [Thực Nghiệm Đối Chứng Các Mô Hình: VOB, WF-RAG, RR-RAG, GM-RAG và SemRAG trên MS COCO 2014](#thực-nghiệm-đối-chứng-các-mô-hình-vob-wf-rag-rr-rag-gm-rag-và-semrag-trên-ms-coco-2014)
  - [Thực nghiệm trên Tập Dữ Liệu Flickr30k (Phương Án A & B)](#thực-nghiệm-trên-tập-dữ-liệu-flickr30k-phương-án-a--b)
  - [Nghiên cứu Bóc Tách Thành Phần (Ablation Study)](#nghiên-cứu-bóc-tách-thành-phần-ablation-study)
  - [Phân Tích Cô Lập Phương Thức Đầu Vào (Chế độ Chỉ Dùng Tri Thức tại Decoder)](#phân-tích-cô-lập-phương-thức-đầu-vào-chế-độ-chỉ-dùng-tri-thức-tại-decoder)
  - [Khảo Sát Độ Nhạy Siêu Tham Số ($K$ và $M$)](#khảo-sát-độ-nhạy-siêu-tham-số-k-và-m)
- [4. Ứng Dụng Website Demo Tương Tác](#4-ứng-dụng-website-demo-tương-tác)
- [5. Cấu Trúc Thư Mục & Bản Đồ Notebook](#5-cấu-trúc-thư-mục--bản-đồ-notebook)
- [6. Hướng Dẫn Cài Đặt & Tái Hiện Kết Quả](#6-hướng-dẫn-cài-đặt--tái-hiện-kết-quả)
  - [Thiết lập Môi trường](#thiết-lập-môi-trường)
  - [Tải Checkpoint & Kho Tri Thức từ Google Drive](#tải-checkpoint--kho-tri-thức-từ-google-drive)
  - [Thực thi các Notebook Đánh giá](#thực-thi-các-notebook-đánh-giá)
  - [Khởi chạy Ứng dụng Website Demo](#khởi-chạy-ứng-dụng-website-demo)
- [7. Trích Dẫn Khóa Luận (BibTeX)](#7-trích-dẫn-khóa-luận-bibtex)
- [8. Giấy Phép & Lời Cảm Ơn](#8-giấy-phép--lời-cảm-ơn)

---

## 1. Giới Thiệu & Động Lực Nghiên Cứu

Mô hình sinh chú thích ảnh truyền thống dựa thuần túy trên thị giác thường gặp hai nhược điểm cốt lõi:
1. **Ảo giác thực thể (Object Hallucination):** Tự sinh ra các đối tượng hoặc thuộc tính sai lệch do thiên kiến thống kê của bộ giải mã ngôn ngữ.
2. **Câu chú thích chung chung, thiếu ngữ cảnh:** Xu hướng tạo ra các câu văn rập khuôn bề mặt, không nêu bật được mối quan hệ tương tác sinh động giữa các đối tượng trong cảnh.

Kỹ thuật Sinh tăng cường truy hồi (*Retrieval-Augmented Generation — RAG*) mở ra hướng tiếp cận triển vọng bằng cách truy hồi các câu mô tả chuẩn của con người để làm giàu ngữ cảnh. Tuy nhiên, các hệ thống RAG cơ sở (Kho tri thức cũ (Old SemKB)) bộc lộ ba nút thắt nghiêm trọng:
* **Nhiễu trùng lặp ứng viên:** Các câu tương tự nhau về mặt từ vựng chiếm giữ toàn bộ danh sách truy hồi FAISS, làm triệt tiêu không gian chọn lọc đa dạng.
* **So khớp ngữ nghĩa nông (Surface Matching):** Việc so khớp chỉ dựa trên cosine vector hoặc từ đơn Jaccard mà bỏ qua vai trò ngữ pháp: Chủ thể – Hành động – Quan hệ vị ngữ – Đối tượng (*Subject – Predicate – Object, SPO*).
* **Nghẽn cổ chai biểu diễn:** Ép trung bình cộng nhiều câu tham chiếu thành 1 vector duy nhất (`Mean-Pooling`) làm mất mát các chi tiết ngữ nghĩa độc lập.

### Giải pháp Đề xuất: Kiến trúc SemRAG
Để giải quyết triệt để các hạn chế trên, đề tài đề xuất mô hình **SemRAG** tích hợp:
1. **Kho Tri Thức Cấu Trúc Đa Tầng (SemKB):** Bổ sung cờ khử trùng lặp `duplicate_of`, chuẩn hóa tập thực thể `objects`, gán nhãn tường minh quan hệ không gian `relations` (SPO), hành động `actions`, thuộc tính `attributes` và trọng số đồng thuận `TF-IDF 1..4-gram` toàn kho.
2. **Tái xếp hạng Lai dẫn dắt bởi Đồ thị cảnh RelTR & Lọc đa dạng MMR:** Kết hợp độ tương đồng CLIP với độ tương thích đồ thị cảnh SPO, kết hợp thuật toán Chọn lọc Biên Tối đa (MMR, $\lambda=0.7$) nhằm chọn ra $K=5$ câu vừa sát ngữ cảnh vừa phong phú về góc nhìn.
3. **Bộ giải mã Hợp nhất Đa Token Có Cổng Động:** Lưu giữ trọn vẹn $K=5$ vector tham chiếu trong bộ nhớ độc lập ($K \times 256$), kết hợp cổng học thích nghi $g \in [0, 1]$ để cân bằng mềm dẻo giữa đặc trưng thị giác và tri thức bên ngoài.

---

## 2. Kiến Trúc Hệ Thống (SemRAG)

Kiến trúc tổng thể của hệ thống gồm 4 khối chức năng (minh họa tại file `Mô hình kho tri thức mới.pdf`):

```
+---------------------------------------------------------------------------------------------+
|                                     KIẾN TRÚC SEMRAG                                        |
+---------------------------------------------------------------------------------------------+
|                                                                                             |
|   +-----------------------+                         +-----------------------------------+   |
|   |   KHỐI A: ĐẦU VÀO     |                         |   KHỐI C: KHO TRI THỨC ĐA TẦNG    |   |
|   |                       |                         |              (SemKB)              |   |
|   |  - Ảnh truy vấn       |                         |                                   |   |
|   |  - CLIP ViT-B/32      |                         |  - 566.747 bản ghi đa tầng        |   |
|   |    Nhúng đặc trưng    |                         |  - Cờ khử trùng lặp duplicate_of  |   |
|   |    512 chiều          |                         |  - Tập thực thể chuẩn hóa objects |   |
|   |  - RelTR trích xuất   |                         |  - Bộ ba quan hệ không gian SPO   |   |
|   |    M=10 bộ ba SPO     |                         |  - Siêu dữ liệu Actions & Attrs   |   |
|   +-----------+-----------+                         |  - Trọng số TF-IDF 1..4-gram      |   |
|               |                                     +-----------------+-----------------+   |
|               | (Vector ảnh & Đồ thị cảnh SPO)                        |                     |
|               v                                                       |                     |
|   +-----------+-------------------------------------------------------+                 |   |
|   |   KHỐI B: TRUY HỒI, MỞ RỘNG NGỮ NGHĨA & TÁI XẾP HẠNG LAI                            |   |
|   |                                                                                     |   |
|   |  1. FAISS IndexFlatIP truy vấn Top-N = 50 câu ứng viên                              |   |
|   |  2. Mở rộng ngữ nghĩa qua láng giềng SPO đồng thuận trong Kho SemKB                    |   |
|   |  3. Điểm tái xếp hạng lai: S_rel = 0.6 * S_CLIP + 0.4 * S_SPO_Sem                   |   |
|   |  4. Lọc đa dạng hóa biên tối đa (MMR, lambda = 0.7) chọn Top-K = 5 câu              |   |
|   |  5. Tổng hợp nguyên mẫu cụm SPO (SPO Multi-Prototype)                               |   |
|   +-----------------------------------+-------------------------------------------------+   |
|                                       |                                                     |
|                                       | (K=5 vector câu tham khảo & ngữ cảnh tổng hợp)      |
|                                       v                                                     |
|   +-----------------------------------+-------------------------------------------------+   |
|   |   KHỐI D: BỘ GIẢI MÃ HỢP NHẤT ĐA TOKEN CÓ CỔNG ĐỘNG                                 |   |
|   |                                                                                     |   |
|   |  - Chiếu bộ nhớ Đa Token: M_knw in R^(B x 5 x 256)                                  |   |
|   |  - Chiếu bộ nhớ Thị giác: M_vis in R^(B x 1 x 256)                                  |   |
|   |  - Cổng thích nghi GatedFusion (tau = 1.0):                                         |   |
|   |     g = sigmoid(Linear([v_norm; k_agg_norm; conf]))                                 |   |
|   |  - Chú ý chéo nhánh kép 3 lớp (Dual-Branch Cross-Attention):                         |   |
|   |     C_fused = (1 - g) * C_vis + g * C_knw                                           |   |
|   |  - Giải mã Beam Search (Beam=3, Phạt độ dài alpha=0.6, max_len=30)                  |   |
|   +-------------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------------+
```

### Khối A: Mã hóa Đa phương thức Đầu vào
* **Đặc trưng thị giác toàn cục:** Ảnh đầu vào $I$ được trích xuất vector nhúng chuẩn hóa L2 có kích thước $512$ chiều $v_{\text{img}} \in \mathbb{R}^{512}$ thông qua mạng `CLIP ViT-B/32` (được đóng băng trọng số).
* **Đồ thị cảnh quan hệ RelTR:** Mô hình biến đổi đồ thị cảnh `RelTR` trích xuất tối đa $M = 10$ bộ ba quan hệ `(Chủ thể, Vị ngữ, Đối tượng)` có độ tin cậy cao nhất $G(I) = \{(s_m, p_m, o_m)\}_{m=1}^M$.

### Khối B: Tái Xếp Hạng Ngữ Nghĩa & Lọc Đa Dạng MMR
* **Truy xuất vector dày đặc:** FAISS (`IndexFlatIP`) truy vấn ban đầu $N = 50$ câu ứng viên dựa trên tích vô hướng cosine $\tilde{S}_{\text{CLIP}}(I, c_i)$.
* **Tái xếp hạng lai:** Tính toán độ tương thích ngữ nghĩa $S_{\text{SPO-Sem}}$ giữa đồ thị cảnh ảnh truy vấn với các bộ ba có sẵn trong Kho SemKB:
  $$S_{\text{rel}}(I, c_i) = w \cdot \tilde{S}_{\text{CLIP}}(I, c_i) + (1 - w) \cdot S_{\text{SPO-Sem}}(G(I), G_{\text{KB}}(c_i)), \quad w = 0.6$$
* **Lọc đa dạng hóa tối đa biên (MMR):** Chọn đúng $K = 5$ ứng viên đa dạng và loại trừ lặp ngữ nghĩa:
  $$\text{MMR} = \arg\max_{c_i \in C \setminus S} \left[ \lambda S_{\text{rel}}(I, c_i) - (1 - \lambda) \max_{c_j \in S} \text{Sim}(c_i, c_j) \right], \quad \lambda = 0.7$$
  với $\text{Sim}(c_i, c_j) = 0.65 \cdot \text{Sim}_{\text{CLIP}} + 0.35 \cdot \text{Jaccard}_{\text{SPO}}$.

### Khối C: Kho Tri Thức Cấu Trúc Đa Tầng (SemKB)
| Thông số Kỹ thuật | Kho COCO (SemKB) | Kho Flickr30k (SemKB) |
| :--- | :--- | :--- |
| **Tổng số bản ghi** | **566.747 câu** (Train + Restval, 113.287 ảnh) | **145.000 câu** (Train, 29.000 ảnh) |
| **Chỉ mục Vector** | CLIP ViT-B/32 (512 chiều, chuẩn hóa L2, `IndexFlatIP`) | CLIP ViT-B/32 (512 chiều, chuẩn hóa L2, `IndexFlatIP`) |
| **Khử trùng lặp (`duplicate_of`)** | **21.916 câu trùng lặp** được trỏ về `row_id` gốc | **524 câu trùng lặp** được trỏ về `row_id` gốc |
| **Thực thể chuẩn hóa (`objects`)** | **566.671 câu (99,99%)** | **144.996 câu (100,0%)** |
| **Bộ ba quan hệ SPO (`relations`)** | **521.602 câu (92,03%)** | **139.714 câu (96,35%)** |
| **Hành động & Thuộc tính** | Tách lớp hành động và thuộc tính tường minh | Hành động: 88,67%, Thuộc tính: 49,09% |
| **Đồng thuận N-gram** | Trọng số TF-IDF 1..4-gram toàn kho | Trọng số TF-IDF 1..4-gram toàn kho |

### Khối D: Bộ Giải Mã Hợp Nhất Đa Token Có Cổng Động
* **Bộ nhớ Đa Token:** Giữ nguyên từng vector độc lập trong số $K = 5$ câu tham chiếu:
  $$M_{\text{knw}} \in \mathbb{R}^{B \times 5 \times 256}$$
* **Hợp nhất cổng thích nghi:** Mạng nơ-ron cổng tính toán hệ số cân bằng $g \in [0, 1]$:
  $$g = \sigma\left(W_g [v_{\text{norm}}; k_{\text{agg\_norm}}; \text{conf}] + b_g\right)$$
* **Chú ý chéo phân nhánh kép:**
  $$C_{\text{fused}} = (1 - g) \odot \text{CrossAttn}(Q, M_{\text{vis}}) + g \odot \text{CrossAttn}(Q, M_{\text{knw}})$$
* **Suy luận:** Tìm kiếm chùm Beam Search (`beam_size = 3`, hệ số phạt độ dài $\alpha = 0.6$, chiều dài tối đa 30 từ).

---

## 3. Kết Quả Thực Nghiệm Đối Chứng

Toàn bộ các thực nghiệm tuân thủ **quy trình kiểm chứng đối chứng nghiêm ngặt (100% Controlled Benchmark)**:
- Cố định 100% trọng số checkpoint mô hình (`best_model.pt`, 12.538.968 tham số, SHA-256 xác thực).
- Cố định từ điển (`vocab.json`, 16.214 từ trên COCO; 9.546 từ trên Flickr30k).
- Chạy lặp lại và tính trung bình $\text{Mean} \pm \text{SD}$ trên đúng **3 Seeds chuẩn độc lập: `[42, 44, 46]`** trên 5.000 ảnh Karpathy Test (COCO) và 1.000 ảnh Test (Flickr30k).

### Thực Nghiệm Đối Chứng Các Mô Hình: VOB, WF-RAG, RR-RAG, GM-RAG và SemRAG trên MS COCO 2014

Đánh giá trên tập 5.000 ảnh Karpathy Test của MS COCO 2014 (25.010 câu tham chiếu, chuẩn hóa bằng `SafePTBTokenizer`):

| Mô hình nghiên cứu & Cấu hình | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr (SemKB) | CIDEr (Old SemKB) | $\Delta$ CIDEr | SPICE (SemKB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VOB (Visual-Only Baseline)** (Beam=3, không dùng KB) | 74.21±0.19 | 57.80±0.21 | 44.12±0.17 | 33.68±0.14 | 26.77±0.10 | 54.99±0.13 | **107.15 ± 0.74** | 107.15 | 0.00 | 19.73 ± 0.10 |
| **WF-RAG (Basic RAG)** (CLIP Top-5 Mean-Pool) | 74.92±0.10 | 58.54±0.04 | 44.78±0.02 | 34.34±0.02 | 27.41±0.04 | 55.84±0.01 | **112.52 ± 0.34** | 107.57 | **+4.95** | 20.16 ± 0.16 |
| **RR-RAG (Scene-Graph Reranked RAG)** (RelTR SPO) | 74.97±0.21 | 58.54±0.15 | 45.00±0.10 | 34.71±0.09 | 27.68±0.02 | 55.95±0.05 | **114.13 ± 0.28** | 110.64 | **+3.49** | 20.68 ± 0.03 |
| **GM-RAG (Gated Multi-Token RAG)** (5 Token, không MMR) | 74.83±0.07 | 58.43±0.06 | 44.96±0.08 | 34.73±0.08 | 27.82±0.01 | 55.93±0.01 | **114.81 ± 0.20** | 110.97 | **+3.84** | 20.67 ± 0.04 |
| **SemRAG (Mô hình đề xuất hoàn chỉnh)** (RelTR+MMR+Gated) | **74.92±0.05** | **58.57±0.03** | **45.13±0.04** | **34.90±0.06** | **27.85±0.02** | **56.01±0.01** | **115.41 ± 0.07** | 111.63 | **+3.78** | **20.76 ± 0.02** |

* Chi tiết điểm CIDEr từng Seed của mô hình SemRAG: Seed 42 = **115.38**, Seed 44 = **115.36**, Seed 46 = **115.49**.
* Tỷ lệ ảo giác thực thể trên tập COCO Test: $\text{CHAIR}_s = \mathbf{8.21 \pm 0.08\%}$, $\text{CHAIR}_i = \mathbf{7.18 \pm 0.06\%}$.

---

### Thực nghiệm trên Tập Dữ Liệu Flickr30k (Phương Án A & B)

Đánh giá trên tập 1.000 ảnh Karpathy Test của Flickr30k qua 3 Seeds `[42, 44, 46]`:

#### Phương Án A: Đánh giá Nội Miền (Huấn luyện trên Flickr30k + Kho Tri Thức Flickr30k SemKB)
| Mô hình nghiên cứu | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr (SemKB) | $\Delta$ so với VOB | SPICE | $\text{CHAIR}_s \downarrow$ (%) | $\text{CHAIR}_i \downarrow$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VOB (Mô hình cơ sở)** | 68.85±0.38 | 25.89±0.24 | 21.10±0.13 | 48.14±0.16 | **53.85 ± 0.62** | 0.00 | 14.82±0.19 | 13.27±0.38 | 9.64±0.25 |
| **WF-RAG (Basic RAG)** | 67.66±0.22 | 25.21±0.09 | 21.44±0.02 | 48.20±0.12 | **58.29 ± 0.22** | +4.44 | 15.27±0.04 | 11.40±0.10 | 8.03±0.06 |
| **RR-RAG (SPO Rerank)**| 69.97±0.66 | 26.44±0.29 | 21.90±0.05 | 49.02±0.11 | **59.53 ± 0.18** | +5.68 | 15.63±0.05 | 10.57±0.12 | 7.68±0.21 |
| **GM-RAG (Gated Multi-Token)**| 70.89±0.15 | 26.95±0.08 | 21.92±0.06 | 49.20±0.14 | **60.70 ± 0.18** | +6.85 | 15.68±0.05 | 9.63±0.06 | 7.19±0.03 |
| **SemRAG (Mô hình đề xuất hoàn chỉnh)**| **71.04±0.18** | **27.26±0.07** | **22.03±0.06** | **49.37±0.10** | **61.98 ± 0.21** | **+8.13** | **15.77±0.01** | **8.63 ± 0.06** | **6.47 ± 0.05** |

#### Phương Án B: Thử Thách Zero-Shot Liên Miền (Model COCO + Kho COCO SemKB chấm thẳng 1.000 ảnh Flickr30k)
| Mô hình nghiên cứu | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | $\Delta$ so với VOB | SPICE | $\text{CHAIR}_s \downarrow$ (%) | $\text{CHAIR}_i \downarrow$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VOB (COCO Baseline)** | 58.79±0.01 | 17.12±0.03 | 17.23±0.03 | 41.02±0.05 | **37.83 ± 0.14** | 0.00 | 10.95±0.04 | 36.10±0.10 | 23.29±0.03 |
| **WF-RAG (COCO Basic RAG)** | 59.15±0.46 | 17.41±0.23 | 17.75±0.22 | 41.69±0.31 | **41.51 ± 0.99** | +3.68 | 11.35±0.14 | 26.80±0.10 | 17.48±0.10 |
| **RR-RAG (COCO RelTR Rerank)**| 59.73±0.18 | 18.16±0.03 | 17.98±0.11 | 42.08±0.20 | **42.93 ± 0.56** | +5.10 | 11.81±0.10 | 24.40±0.10 | 16.07±0.04 |
| **GM-RAG (COCO Gated Multi-Token)** | 60.07±0.27 | 18.22±0.25 | 17.98±0.09 | 42.17±0.12 | **43.36 ± 0.24** | +5.53 | 11.92±0.11 | 22.10±0.10 | 14.84±0.07 |
| **SemRAG (COCO đề xuất hoàn chỉnh)** | **60.77±0.08** | **19.00±0.11** | **18.12±0.04** | **42.71±0.05** | **44.88 ± 0.21** | **+7.05 (+18,6%)**| **12.08±0.00** | **19.80 ± 0.10** | **13.82 ± 0.03** |

> **Ý nghĩa thực nghiệm then chốt:** Không cần huấn luyện lại trên miền ảnh mới, SemRAG giúp tăng CIDEr từ **`37.83` lên `44.88` (+7.05 điểm, tương đương +18,64%)** và **kéo giảm tỷ lệ câu bị ảo giác $\text{CHAIR}_s$ từ `36.10%` xuống chỉ còn `19.80%` (giảm tới `16.30` điểm phần trăm)**.

---

### Nghiên cứu Bóc Tách Thành Phần (Ablation Study)

Đánh giá trên 5.000 ảnh Karpathy Test của MS COCO 2014 qua 3 Seeds `[42, 44, 46]`:

| Cấu hình Bóc tách | Tên Nhánh Notebook | BLEU-4 | METEOR | CIDEr (SemKB) | $\Delta$ CIDEr | SPICE (SemKB) | $\Delta$ SPICE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SemRAG (Mô hình đề xuất hoàn chỉnh)** | Pipeline đề xuất đầy đủ | **34.90 ± 0.06** | **27.85 ± 0.02** | **115.41 ± 0.07** | **0.00** | **20.76 ± 0.02** | **0.00** |
| $\text{w/o MMR Diversity}$ (Tắt MMR) | `Ablation_01 (no_mmr)` | 34.83 ± 0.05 | 27.82 ± 0.02 | **114.70 ± 0.11** | **-0.72** | 20.66 ± 0.04 | **-0.10** |
| $\text{w/o WordNet / SPO Expansion}$ | `Ablation_02 (no_semantic_expansion)` | 34.63 ± 0.11 | 27.74 ± 0.03 | **113.95 ± 0.14** | **-1.47** | 20.49 ± 0.02 | **-0.27** |
| $\text{w/o Multi-Token Memory}$ (Mean-Pool)| `Ablation_04A (no_multitoken_only)` | 34.83 ± 0.09 | 27.75 ± 0.02 | **113.80 ± 0.05** | **-1.61** | 20.58 ± 0.03 | **-0.18** |
| $\text{w/o Dynamic Gated Fusion}$ ($g=0.5$)| `Ablation_04B (no_gate_only)` | 34.39 ± 0.05 | 27.62 ± 0.01 | **113.00 ± 0.17** | **-2.41** | 20.42 ± 0.01 | **-0.34** |
| $\text{w/o Scene-Graph RelTR Reranking}$ | `Ablation_03 (no_reltr_reranking)` | 34.28 ± 0.17 | 27.60 ± 0.05 | **112.08 ± 0.33** | **-3.33** | 20.39 ± 0.04 | **-0.37** |

*Tất cả 5 thành phần khi bị lược bỏ đều khiến hiệu năng suy giảm rõ rệt ($\Delta < 0$ trên mọi chỉ số).*

---

### Phân Tích Cô Lập Phương Thức Đầu Vào (Chế độ Chỉ Dùng Tri Thức tại Decoder)

Thử nghiệm xóa trắng hoàn toàn đặc trưng thị giác ($v_{\text{img}} = \mathbf{0}, g = 1.0$) nhằm kiểm chứng sức mạnh biểu diễn của Kho tri thức SemKB:

| Chế độ Phương thức Đầu vào | Giá trị Cổng $g$ | Tập ảnh Kiểm tra | BLEU-4 | METEOR | CIDEr (Mean ± SD) | SPICE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VOB (Mô hình cơ sở chỉ dùng ảnh)** |  = 0.00$ | 5.000 ảnh Test | 33.68 ± 0.14 | 26.77 ± 0.10 | **107.15 ± 0.74** | 19.73 ± 0.10 |
| **Chỉ tri thức truy hồi (Old SemKB)** |  = 1.00$ | 5.000 ảnh Test | 25.12 | 22.99 | **81.44** | 15.75 |
| **Chỉ tri thức truy hồi (Old SemKB)** |  = 1.00$ | 4.464 ảnh RelTR | 25.52 ± 0.10 | 23.21 ± 0.03 | **81.86 ± 0.08** | 15.83 ± 0.08 |
| **Chỉ tri thức truy hồi (SemKB)** |  = 1.00$ | 5.000 ảnh Test | 30.89 ± 0.02 | 26.06 ± 0.03 | **103.41 ± 0.16** *(+21.97 so với Old SemKB)* | 18.94 ± 0.02 |
| **Chỉ tri thức truy hồi (SemKB)** |  = 1.00$ | 4.464 ảnh RelTR | 31.56 ± 0.01 | 26.39 ± 0.02 | **104.75 ± 0.13** *(+22.89 so với Old SemKB)* | 19.20 ± 0.01 |
| **SemRAG (Mô hình đề xuất hoàn chỉnh)** |  \approx 0.50$ (học động) | 5.000 ảnh Test | **34.90 ± 0.06** | **27.85 ± 0.02** | **115.41 ± 0.07** | **20.76 ± 0.02** |

---

### Khảo Sát Độ Nhạy Siêu Tham Số ($K$ và $M$)

* **Số câu tham khảo được chọn $K \in \{1, 3, 5, 10, 20\}$:**
  * $K=1 \rightarrow 112.13 \pm 0.04$ CIDEr (thời gian: 12.8 ms/ảnh)
  * $K=3 \rightarrow 114.76 \pm 0.13$ CIDEr (thời gian: 13.0 ms/ảnh)
  * **$K=5 \rightarrow \mathbf{115.41 \pm 0.07}$ CIDEr (thời gian: 16.8 ms/ảnh) — Cấu hình Tối ưu Đạt Đỉnh**
  * $K=10 \rightarrow 115.16 \pm 0.17$ CIDEr (thời gian: 24.1 ms/ảnh)
  * $K=20 \rightarrow 114.81 \pm 0.09$ CIDEr (thời gian: 47.1 ms/ảnh)
* **Số bộ ba quan hệ đồ thị cảnh $M \in \{0, 3, 5, 10, 20\}$:**
  * $M=0 \rightarrow 112.08 \pm 0.33$ CIDEr
  * $M=3 \rightarrow 114.31 \pm 0.11$ CIDEr
  * $M=5 \rightarrow 115.26 \pm 0.04$ CIDEr
  * **$M=10 \rightarrow \mathbf{115.41 \pm 0.07}$ CIDEr — Cấu hình Tối ưu Đạt Đỉnh**
  * $M=20 \rightarrow 115.18 \pm 0.02$ CIDEr

---

## 4. Ứng Dụng Website Demo Tương Tác

Thư mục `UngDung/` cung cấp mã nguồn ứng dụng web hoàn chỉnh phục vụ biểu diễn trực quan:

* **Backend:** FastAPI, Uvicorn, PyTorch, FAISS, CLIP ViT-B/32, SafePTBTokenizer.
* **Frontend:** React 19, Vite, Tailwind CSS, Lucide Icons.
* **Tính năng tiêu biểu:**
  - Hỗ trợ tải ảnh linh hoạt đa định dạng: `.png`, `.jpg`, `.jpeg`, `.webp`.
  - Chế độ so sánh đối chiếu song song: **Baseline chỉ dùng ảnh ($g = 0$)** so với **SemRAG tích hợp SemKB ($g \text{ thích nghi})$**.
  - Kiểm tra và trực quan hóa giá trị cổng đóng/mở $g$ cùng thanh trượt can thiệp thủ công.
  - Hiển thị danh sách $K=5$ câu tham khảo kèm điểm số và các bộ ba quan hệ RelTR SPO tương ứng.
  - Bảng chẩn đoán kỹ thuật (độ trễ xử lý, xác suất từ khóa, thông số bộ nhớ).

---

## 5. Cấu Trúc Thư Mục & Bản Đồ Notebook

```
NghienCuu_Nhom2/
├── .gitignore                                                      # Quy tắc loại trừ tệp rác và tệp lớn
├── requirements.txt                                                # Danh mục thư viện Python toàn diện
├── README.md                                                       # Bản tài liệu tiếng Anh chuẩn GitHub
├── README_VI.md                                                    # Bản tài liệu học thuật tiếng Việt (tệp này)
├── Mô hình kho tri thức mới.pdf                                    # Sơ đồ vector kiến trúc 4 khối chi tiết
├── readme_va_prompt_ngucanh_toanbo_duan_kb_v2_nhom02.md           # Báo cáo kiểm định và siêu prompt ngữ cảnh
│
├── Coco/                                                           # Thực nghiệm trên MS COCO 2014 (8 Notebooks)
│   ├── Knowledge_Base_V2_Improved_Full.ipynb                       # Xây dựng SemKB (566.747 câu)
│   ├── Baseline_Level_1_COCO2014_MultiSeed_BeamSearch_k3_Seed42.ipynb
│   ├── Baseline_Level_1_COCO2014_MultiSeed_BeamSearch_k3_Seed44.ipynb
│   ├── Baseline_Level_1_COCO2014_MultiSeed_BeamSearch_k3_Seed46.ipynb
│   ├── DanhGia_Level2_3Seeds_CaiTien_COCO_KetQua3.ipynb            # Đánh giá WF-RAG (Seeds 42, 44, 46)
│   ├── DanhGia_Level3_3Seeds_CaiTien_COCO_KetQua3.ipynb            # Đánh giá RR-RAG (Seeds 42, 44, 46)
│   ├── DanhGia_Level4_3Seeds_CaiTien_COCO_KetQua6.ipynb            # Đánh giá GM-RAG (Seeds 42, 44, 46)
│   └── DanhGia_Level5_3Seeds_CaiTien_COCO_KetQua5.ipynb            # Đánh giá SemRAG Đề xuất Hoàn chỉnh (Seeds 42, 44, 46)
│
├── flick30k/                                                       # Thực nghiệm trên Flickr30k (11 Notebooks)
│   ├── PhuongAn_A/                                                 # Phương án A: Đánh giá nội miền Flickr30k
│   │   ├── Tao_Kho_Tri_Thuc_V2_Flickr30k_CaiTien_KetQua.ipynb     # Xây dựng Kho Flickr30k SemKB (145.000 câu)
│   │   ├── DanhGia_Level1_3Seeds_Baseline_Flickr30k_KetQua.ipynb   # VOB Baseline Flickr30k (Seeds 42, 44, 46)
│   │   ├── DanhGia_Level2_3Seeds_CaiTien_Flickr30k_KetQua.ipynb   # WF-RAG Flickr30k (Seeds 42, 44, 46)
│   │   ├── DanhGia_Level3_3Seeds_CaiTien_Flickr30k_KetQua.ipynb   # RR-RAG Flickr30k (Seeds 42, 44, 46)
│   │   ├── DanhGia_Level4_3Seeds_CaiTien_Flickr30k_KetQua.ipynb   # GM-RAG Flickr30k (Seeds 42, 44, 46)
│   │   └── DanhGia_Level5_3Seeds_CaiTien_Flickr30k_KetQua.ipynb   # SemRAG Flickr30k Đề xuất Hoàn chỉnh (Seeds 42, 44, 46)
│   │
│   └── PhuongAn_B/                                                 # Phương án B: Zero-Shot Liên miền (COCO -> Flickr30k)
│       ├── PA_B_BeNguyenCOCO_ChamFlickr30k_Level1_3Seeds_KetQua.ipynb
│       ├── PA_B_BeNguyenCOCO_ChamFlickr30k_Level2_3Seeds_KetQua.ipynb
│       ├── PA_B_BeNguyenCOCO_ChamFlickr30k_Level3_3Seeds_KetQua.ipynb
│       ├── PA_B_BeNguyenCOCO_ChamFlickr30k_Level4_3Seeds_KetQua.ipynb
│       └── PA_B_BeNguyenCOCO_ChamFlickr30k_Level5_3Seeds_KetQua.ipynb
│
├── Boc_Tach/                                                       # Bóc tách thành phần & Độ nhạy (8 Notebooks)
│   ├── Chi_Dung_TriThuc/
│   │   └── DanhGia_Level6_ChiDungTriThuc_KB_V2_COCO_KetQua.ipynb   # Cô lập phương thức (Chế độ Chỉ dùng tri thức tại Decoder)
│   ├── Tach_Dimersity/
│   │   └── Ablation_01_Tach_MMR_Diversity_KB_V2_COCO_KetQua.ipynb # Lược bỏ MMR Diversity
│   ├── Tach_Sematic/
│   │   └── Ablation_02_Tach_Semantic_Expansion_KB_V2_COCO_KetQua.ipynb # Lược bỏ Mở rộng Ngữ nghĩa
│   ├── Tach_ReRanking/
│   │   └── Ablation_03_Tach_RelTR_SPO_Reranking_KB_V2_COCO_KetQua.ipynb # Lược bỏ Tái xếp hạng RelTR
│   ├── Tach_MultiToken/
│   │   └── Ablation_04A_ChiTach_MultiToken_MeanPool_KB_V2_COCO_KetQua.ipynb # Lược bỏ Bộ nhớ Đa Token
│   ├── Tach_GateFusion/
│   │   └── Ablation_04B_ChiTach_GatedFusion_Linear_KB_V2_COCO_KetQua.ipynb  # Lược bỏ Cổng Động (g=0.5)
│   ├── SPO/
│   │   └── KhaoSat_Bang8_SoBoBa_SPO_M_KB_V2_COCO_3Seeds_KetQua.ipynb # Khảo sát số bộ ba quan hệ M
│   └── Thay đổi Top K/
│       └── KhaoSat_Bang9_ThayDoi_TopK_KB_V2_COCO_3Seeds_KetQua.ipynb # Khảo sát số câu tham khảo K
│
└── UngDung/                                                        # Ứng dụng Website Demo tương tác
    ├── scripts/
    │   ├── download_level5.py                                      # Tập lệnh tự động tải tài nguyên Google Drive
    │   └── level5_drive.json                                       # Tệp cấu hình Drive File ID & mã băm SHA-256
    ├── models/level5/                                              # Thư mục đích nạp checkpoint và kho tri thức
    └── Fontend/
        ├── image-captioning-baseline/backend/                      # Máy chủ dịch vụ Backend FastAPI
        └── webapp/                                                 # Giao diện người dùng React 19 + Vite
```

---

## 6. Hướng Dẫn Cài Đặt & Tái Hiện Kết Quả

### Thiết lập Môi trường

Khuyến nghị sử dụng Python 3.10, 3.11 hoặc 3.12:

```bash
# Sao chép kho lưu trữ
git clone https://github.com/YourUsername/SemRAG-Captioning.git
cd SemRAG-Captioning

# Khởi tạo và kích hoạt môi trường ảo
python -m venv .venv
# Trên Windows PowerShell:
.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source .venv/bin/activate

# Nâng cấp pip và cài đặt toàn bộ thư viện
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Tải Checkpoint & Kho Tri Thức từ Google Drive

Do dung lượng các ma trận nhúng của SemKB vượt quá giới hạn 100MB của GitHub, hệ thống cung cấp sẵn tập lệnh tự động tải về và kiểm tra toàn vẹn mã băm SHA-256:

```bash
cd UngDung
python scripts/download_level5.py --manifest scripts/level5_drive.json
```

**Bảng mã băm và thông tin tệp trên Google Drive (`level5_drive.json`):**
| Tên Thành phần | Tên Tệp Lưu | Dung lượng | Google Drive File ID | Mã Băm Kiểm Tra SHA-256 |
| :--- | :--- | :--- | :--- | :--- |
| **Trọng số Checkpoint** | `best_model.pt` | ~50 MB | `1SrRcYiCnzr3bnEs7aZv0xVMsbY4iFb9K` | `a708a1c41f98421d6af333a596d30d9c733034bf9ba57bf4484cce2c9357a9e4` |
| **Từ điển** | `vocab.json` | ~300 KB | `1FrnEN43-COOQ3XYWu5kqfAZpTQqPWu22` | `c47a5de5309a21a22a5e9267505c8397781584c49eb29a626a3f8c595b6bdf83` |
| **Vector nhúng CLIP** | `text_embeddings_train_restval.npy`| 1,16 GB | `1-eriJH2wm82Fz-jEFrI0zMej521bSUAU` | `a7d701abb4efbd23ea548b8499785a4fc19a26070a0686d271e0ace17f9db225` |
| **Siêu dữ liệu Mapping**| `text_embedding_mapping.jsonl` | 266 MB | `1huUhetLNK80nqs5JOAd7tP9Wj9PetTAr` | `91f8a7dd2b70f88ec14c0ab6cc67ec9de2dd2b32062d903d0698801f53b7cf32` |

### Thực thi các Notebook Đánh giá

Để tái hiện bất kỳ bảng số liệu thực nghiệm nào trong bài luận, khởi chạy giao diện Jupyter:
```bash
jupyter notebook
```
Mở tệp notebook tương ứng trong `Coco/`, `flick30k/`, hoặc `Boc_Tach/` và chọn `Restart & Run All`. Toàn bộ 27 notebook đã được lưu vết thực thi đầy đủ với trạng thái `Errors: []`, đảm bảo tính khả lặp 100%.

### Khởi chạy Ứng dụng Website Demo

1. **Khởi động Máy chủ Dịch vụ Backend (FastAPI):**
   ```bash
   cd UngDung/Fontend/image-captioning-baseline
   uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
   Tài liệu kiểm thử tương tác Swagger UI tự động kích hoạt tại `http://127.0.0.1:8000/docs`.

2. **Khởi động Giao diện Người dùng Frontend (React + Vite):**
   ```bash
   cd UngDung/Fontend/webapp
   npm install
   npm run dev
   ```
   Truy cập vào ứng dụng tại địa chỉ `http://127.0.0.1:5173`.

---

## 7. Trích Dẫn Khóa Luận (BibTeX)

Nếu mã nguồn, phương pháp SemRAG hoặc bộ dữ liệu Kho tri thức SemKB hữu ích cho công trình nghiên cứu của bạn, xin vui lòng trích dẫn:

```bibtex
@thesis{SemRAG_HUIT_2026,
  author    = {Ngo, Quoc Binh and Nguyen, Thi Quyen and Nguyen, Quoc Viet},
  title     = {Phát triển mô hình chú thích ảnh tăng cường truy hồi tri thức ngữ nghĩa},
  school    = {Khoa Công nghệ Thông tin, Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT)},
  year      = {2026},
  month     = {11},
  address   = {Thành phố Hồ Chí Minh, Việt Nam},
  type      = {Khóa luận tốt nghiệp Đại học ngành Công nghệ Thông tin},
  note      = {Giảng viên hướng dẫn: ThS. Đinh Thị Mận và TS. Nguyễn Văn Thịnh}
}
```

---

## 8. Giấy Phép & Lời Cảm Ơn

* **Giấy phép mã nguồn:** Phát hành theo giấy phép mã nguồn mở [MIT License](LICENSE).
* **Bộ dữ liệu và mô hình cơ sở:**
  - MS COCO Consortium ([Điều khoản sử dụng MS COCO](https://cocodataset.org/))
  - Flickr30k Entities Dataset ([Điều khoản Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/))
  - OpenAI CLIP ([Giấy phép MIT](https://github.com/openai/CLIP/blob/main/LICENSE))
  - RelTR ([Giấy phép Apache 2.0](https://github.com/yrcong/RelTR))
* **Lời cảm ơn:** Nhóm sinh viên xin gửi lời tri ân sâu sắc nhất tới **ThS. Đinh Thị Mận** và **TS. Nguyễn Văn Thịnh** đã tận tình hướng dẫn, định hướng khoa học và tạo mọi điều kiện thuận lợi nhất để nhóm hoàn thành xuất sắc đề tài nghiên cứu này tại HUIT.
