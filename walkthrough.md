# Kiến Trúc Covert Timing Channel (TimeGAN + NoiseGenerator)

---

## 1. Kiến Trúc 5 Mạng Neural

Hệ thống dùng 5 "bộ não" nhân tạo (mạng neural) phối hợp với nhau. Mỗi mạng có 1 nhiệm vụ riêng. Có thể hình dung như một nhà máy sản xuất tiền giả: cần người nén mẫu (Embedder), người in lại (Recovery), người dự đoán mẫu tiếp theo (Supervisor), người pha mực giả cho giống thật (NoiseGenerator), và thanh tra kiểm tra tiền giả (Discriminator). Mục tiêu cuối cùng là NoiseGenerator — kẻ pha mực — phải giỏi đến mức thanh tra không tài nào phân biệt được tiền thật và tiền giả.

Bên trong mỗi mạng đều dùng một loại "bộ nhớ tuần tự" gọi là **GRU** (Gated Recurrent Unit). GRU giống như đọc sách — nó đọc từng trang (từng timestep) và nhớ nội dung các trang trước để hiểu trang hiện tại. Nhờ vậy mỗi mạng có thể xử lý chuỗi thời gian (24 bước liên tiếp) thay vì chỉ 1 điểm đơn lẻ.

```
┌─────────────────────────────────────────────────────────┐
│  EMBEDDER (E) — Bộ nén                                  │
│  Nhận: 1 chuỗi IPD (24 giá trị, mỗi cái 1 số)         │
│  Nén thành: 1 chuỗi biểu diễn ẩn (24 bước, mỗi cái 24 số)│
│  Cấu tạo: GRU(1→24, 3 tầng) → Linear → Sigmoid        │
│  Ý nghĩa: Giống nén ảnh JPEG — giữ đặc trưng, bỏ chi tiết│
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  RECOVERY (R) — Bộ giải nén                             │
│  Nhận: biểu diễn ẩn (24 bước × 24 số)                  │
│  Giải nén lại: chuỗi IPD (24 bước × 1 số)              │
│  Cấu tạo: GRU(24→24, 3 tầng) → Linear → Sigmoid       │
│  Ý nghĩa: Giống giải nén JPEG — khôi phục gần đúng ảnh gốc│
│  Cặp E+R = Autoencoder (máy nén-giải nén)              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  SUPERVISOR (S) — Bộ dự đoán                            │
│  Nhận: biểu diễn ẩn tại bước hiện tại                   │
│  Dự đoán: biểu diễn ẩn ở bước tiếp theo                │
│  Cấu tạo: GRU(24→24, 2 tầng) → Linear → Sigmoid       │
│  Ý nghĩa: Giống dự báo thời tiết — cho hôm nay, hôm   │
│  mai sẽ ra sao? Đảm bảo chuỗi giả có liên kết thời    │
│  gian tự nhiên, không nhảy lung tung.                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  ★ NOISE GENERATOR (G) — Bộ sinh nhiễu (THAY THẾ       │
│  Generator gốc của TimeGAN)                             │
│  Nhận: 2 thứ ghép lại:                                  │
│    - Z: số ngẫu nhiên (gieo xúc xắc, seed cố định)     │
│    - C_scaled: mốc bit (0.20 nếu Bit 0, 0.40 nếu Bit 1)│
│  Sinh ra: δt — một lượng nhiễu nhỏ cộng thêm/trừ bớt  │
│  Kết quả: final_ipd = mốc_bit + δt                     │
│  Cấu tạo: GRU(2→24, 3 tầng) → Linear → Tanh × 0.4     │
│  Ý nghĩa: Mốc bit là 0.5s hoặc 1.0s — quá rõ ràng,    │
│  NIDS phát hiện ngay. G thêm nhiễu để xáo trộn, giống  │
│  như thêm "khuyết điểm tự nhiên" vào tiền giả cho giống │
│  tiền thật đã qua sử dụng. Tanh giới hạn nhiễu trong   │
│  khoảng (-0.4, +0.4) để không quá lố.                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  DISCRIMINATOR (D) — Thanh tra                          │
│  Nhận: biểu diễn ẩn (không biết là thật hay giả)       │
│  Cho điểm: 1 số — càng gần 1 = "tôi nghĩ đây là thật" │
│  Cấu tạo: Bidirectional GRU(24→24, 3 tầng) → Linear   │
│  Ý nghĩa: Thanh tra đọc chuỗi từ cả 2 chiều (trái→phải│
│  + phải→trái) để phát hiện bất thường ở bất kỳ đâu.    │
│  Nếu D không phân biệt được thật/giả → G đã thành công.│
└─────────────────────────────────────────────────────────┘
```

---

## 2. Quá Trình Training — 3 Phase

Toàn bộ quá trình huấn luyện diễn ra trước khi hệ thống hoạt động (offline). Giống đào tạo nhân viên trước khi mở cửa hàng. Chia làm 3 giai đoạn, mỗi giai đoạn 200 epoch (200 lần duyệt qua toàn bộ dữ liệu).

### Phase 1: Dạy máy nén-giải nén (Autoencoder)

Đầu tiên, ta cần cho Embedder và Recovery học cách "hiểu" traffic mạng thật trông như thế nào. Cụ thể: cho 10.000 mẫu traffic IoT giả lập (mỗi mẫu là chuỗi 24 giá trị IPD) vào Embedder để nén, rồi Recovery giải nén. So sánh bản gốc với bản giải nén — nếu khác nhau nhiều thì phạt nặng, buộc 2 mạng phải học cách nén/giải nén chính xác hơn. Sau 200 lần lặp, E và R đã "hiểu" cấu trúc traffic thật.

```
 Traffic thật x ──▶ Embedder ──▶ h (nén) ──▶ Recovery ──▶ x̃ (giải nén)
                                                              │
                    So sánh x với x̃ → Sai lệch = Loss       │
                    Loss càng nhỏ = nén-giải nén càng chính xác
```

---

### Phase 2: Dạy dự đoán bước tiếp theo (Supervisor)

Tiếp theo, ta dạy Supervisor nhìn latent state hiện tại và đoán "bước tiếp theo sẽ là gì?". Embedder lúc này đã biết nén, nên ta đóng băng nó (không thay đổi) và chỉ dạy Supervisor. Ví dụ: nếu 3 bước trước là [nhanh, nhanh, chậm] thì bước tiếp theo nhiều khả năng là [chậm]. Supervisor học quy luật này để chuỗi giả sau này sinh ra có sự liên kết logic chứ không phải số ngẫu nhiên.

```
 x ──▶ Embedder (đóng băng) ──▶ h
                                 │
                    ┌────────────┤
                    ▼            ▼
               Supervisor     h bước t+1 (đáp án đúng)
                    │            │
                    ▼            │
              ĥ (dự đoán)       │
                    │            │
                    └─── So sánh ┘ → Loss
```

---

### Phase 3: Đào tạo đối kháng (Joint Training)

Đây là giai đoạn quan trọng nhất. NoiseGenerator (G) và Discriminator (D) được đặt vào trò chơi đối kháng — giống trò "cảnh sát bắt tội phạm": G cố sinh IPD giả trông giống thật, D cố phân biệt. Mỗi vòng, G tập 2 lần còn D tập 1 lần (cho G nhiều cơ hội hơn, nếu không D quá giỏi thì G không học được).

G phải đạt 3 mục tiêu cùng lúc:
- **Lừa D:** IPD giả đi qua Embedder phải tạo ra latent mà D nghĩ là thật.
- **Khớp thống kê:** Trung bình và độ phân tán (std) của IPD giả phải giống hệt traffic thật. Đây là mục tiêu CHÍNH (trọng số ×100).
- **Đa dạng nhiễu:** Nhiễu δt phải đa dạng chứ không được lặp lại một giá trị — để phân phối đầy đặn, tránh bị phát hiện.

```
 ┌──────────── GENERATOR (G) ────────────────────┐
 │                                                │
 │  Z (ngẫu nhiên) + C (mốc bit) ──▶ G ──▶ IPD  │
 │                                    giả        │
 │  IPD giả ──▶ Embedder ──▶ h_fake ──▶ D ──▶ score│
 │                                                │
 │  Loss G = Lừa D                                │
 │         + 100 × |mean/std giả - thật|          │
 │         + Đa dạng nhiễu                        │
 └────────────────────────────────────────────────┘

 ┌──────────── DISCRIMINATOR (D) ────────────────┐
 │                                                │
 │  Traffic thật ──▶ Embedder ──▶ h_real ──▶ D ──▶ "thật!"│
 │  IPD giả     ──▶ Embedder ──▶ h_fake ──▶ D ──▶ "giả!" │
 │                                                │
 │  Loss D = Đoán đúng thật + Đoán đúng giả      │
 │  (Chỉ update khi D chưa đủ giỏi — cân bằng)  │
 └────────────────────────────────────────────────┘
```

Sau 200 vòng, G đã học sinh nhiễu δt sao cho IPD đầu ra trông giống traffic IoT thật, D không phân biệt được nữa. Lúc này ta **chỉ lưu lại G** (NoiseGenerator) — 4 mạng còn lại đã hoàn thành vai trò "thầy dạy" và không cần khi deploy.

---

## 3. Pipeline Encode / Decode

### Ý tưởng chung

Khi truyền file bí mật qua mạng, ta không gửi nội dung file trong gói tin (dễ bị bắt), mà nhúng thông tin vào **khoảng thời gian chờ giữa 2 gói tin liên tiếp** (IPD). Bit 0 = chờ khoảng 0.5 giây, Bit 1 = chờ khoảng 1.0 giây. Vấn đề là nếu chỉ gửi đúng 0.5s/1.0s thì quá rõ ràng — hệ thống giám sát mạng (NIDS) thấy traffic "đều đều 2 giá trị" sẽ nghi ngờ ngay. Giải pháp: dùng NoiseGenerator (đã train ở trên) cộng thêm nhiễu để thời gian chờ trông giống traffic bình thường, nhưng người nhận (biết cùng "mật khẩu" seed) vẫn giải mã đúng.

---

### ENCODE (Người gửi)

Người gửi lấy file bí mật, mã hóa AES (nếu muốn bảo mật thêm), thêm mã sửa lỗi BCH (phòng nhiễu mạng làm sai bit), rồi chuyển thành chuỗi bit [0,1,0,1,...]. Mỗi bit được gán một mốc thời gian cứng (Bit 0 = 0.5s, Bit 1 = 1.0s). Sau đó, NoiseGenerator nhận mốc cứng + noise Z (từ seed 2025) → sinh nhiễu δt → IPD cuối = mốc + δt. IPD đã nhúng nhiễu được ghi vào file CSV, client sẽ đọc file này và "ngủ" (sleep) đúng từng khoảng thời gian đó giữa mỗi lần gửi gói tin.

```
 File bí mật
      │
      ▼
 AES encrypt (tùy chọn, cần pass.txt)
      │
      ▼
 BCH encode: mỗi 23 bytes → 255 bits (thêm mã sửa lỗi)
      │
      ▼
 Chuyển bit → mốc cứng:  bit=0 → 0.20,  bit=1 → 0.40
      │
      │      set_seed(2025) → Z = random cố định
      │              │
      ▼              ▼
 ┌────────────────────────┐
 │    NoiseGenerator      │
 │  final_ipd = mốc + δt │
 └────────────────────────┘
      │
      ▼
 Scale về giây, rồi sang mili-giây
      │
      ▼
 Ghi ra covert_ipd.csv → Client đọc và dùng usleep()
```

---

### DECODE (Người nhận) — Shared Seed Hypothesis Testing

Người nhận đo thời gian giữa các gói tin nhận được (ví dụ 0.487s, 1.023s, 0.512s...). Bây giờ phải đoán mỗi IPD tương ứng với bit 0 hay bit 1. Vì người nhận biết cùng seed (2025) → có cùng Z → đưa Z vào cùng model G → biết chính xác model sẽ sinh nhiễu δt bao nhiêu cho bit 0 và bit 1. Nghĩa là: thử "nếu là bit 0 thì IPD lẽ ra phải là bao nhiêu?", thử "nếu là bit 1 thì bao nhiêu?", rồi so sánh với IPD thật nhận được — cái nào gần hơn thì chọn.

Điểm đặc biệt: phải đoán **tuần tự từng bit**, không đoán song song được. Lý do là GRU có "bộ nhớ" — output tại vị trí thứ 5 phụ thuộc vào input tại vị trí 1,2,3,4. Encoder biết sẵn tất cả bit nên cho vào 1 lần. Decoder không biết — phải đoán bit 1, cập nhật, đoán bit 2, cập nhật, … lần lượt.

```
 IPD nhận được: [0.487, 1.023, 0.512, ...]
      │
      │      set_seed(2025) → Z = random cố định (GIỐNG Sender)
      │              │
      ▼              ▼
 ┌────────────────────────────────────────────────┐
 │  Với mỗi bit i (đoán lần lượt):               │
 │                                                │
 │  Thử bit=0: cho model biết C[i]=0.20           │
 │         → model(Z, C) → ra expected_ipd_0      │
 │                                                │
 │  Thử bit=1: cho model biết C[i]=0.40           │
 │         → model(Z, C) → ra expected_ipd_1      │
 │                                                │
 │  So sánh:                                      │
 │    |received - expected_0| vs |received - expected_1|│
 │    Cái nào nhỏ hơn → chọn bit đó               │
 │                                                │
 │  Ghi nhớ bit vừa chọn → input cho bước tiếp    │
 └────────────────────────────────────────────────┘
      │
      ▼
 decoded_bits = [1, 0, 0, 1, ...]
      │
      ▼
 BCH decode (sửa lỗi do mạng gây ra)
      │
      ▼
 AES decrypt (nếu có)
      │
      ▼
 File gốc
```

**Ví dụ cụ thể:** Sender gửi Bit 0, mốc cứng = 0.5s, AI sinh nhiễu +0.7s → IPD gửi = 1.2s. Hệ thống cũ dùng threshold 0.75s sẽ nói "1.2s > 0.75 → Bit 1" → **SAI**. Nhưng Receiver cùng seed, chạy model cùng → biết nhiễu = +0.7s. Thử Bit 0: expected = 0.5+0.7 = 1.2s, thử Bit 1: expected = 1.0+0.7 = 1.7s. So sánh: |1.2 - 1.2| = 0 < |1.2 - 1.7| = 0.5 → chọn **Bit 0** → **ĐÚNG**.

---
---

## 4. Cơ Sở Toán Học (Mathematical Formulation)

### 4.1 Fixed MinMax Scaling

Toàn bộ hệ thống dùng một phép biến đổi tuyến tính cố định (không fit trên data) để ánh xạ giá trị IPD vật lý (giây) sang miền [0, 1]:

```
                x_phys - PHYS_MIN       x_phys - 0.0       x_phys
 x_scaled  =  ─────────────────────  =  ────────────────  = ──────
               PHYS_MAX - PHYS_MIN       2.5 - 0.0          2.5

 Inverse:  x_phys  =  x_scaled × 2.5
```

Khác với MinMaxScaler của sklearn (fit theo min/max của dataset, thay đổi khi data thay đổi), hệ thống này dùng hằng số cố định PHYS_MIN = 0.0 và PHYS_MAX = 2.5. Lý do: Sender và Receiver phải dùng chung phép biến đổi mà không cần trao đổi thông tin thống kê — chỉ cần thỏa thuận trước 2 hằng số.

Bảng ánh xạ:

```
 Giá trị vật lý     Giá trị scaled       Ý nghĩa
 ──────────────     ──────────────       ─────────
 0.50 s              0.20                Mốc Bit 0
 0.75 s              0.30                Ngưỡng phân loại (tham khảo)
 1.00 s              0.40                Mốc Bit 1
 0.30 s              0.12                Giới hạn vật lý dưới
```

### 4.2 Hàm Loss — Công Thức Toán Học

**Phase 1 — Embedding Loss:**

```
 L_E  =  10 × √( MSE(x, x̃) )

       =  10 × √( (1/N) × Σᵢ (xᵢ - x̃ᵢ)² )
```

Hệ số 10 và căn bậc hai giúp gradient không quá nhỏ khi loss gần 0, giữ cho quá trình hội tụ ổn định.

**Phase 2 — Supervisor Loss:**

```
 L_S  =  MSE( h[:,1:,:],  ĥ[:,:-1,:] )

       =  (1/N) × Σₜ ‖ h(t+1) - ĥ(t) ‖²
```

So sánh hidden state thật tại bước t+1 với dự đoán của Supervisor tại bước t. Đây là supervised learning thuần — không có adversarial.

**Phase 3 — Generator Loss (tổng hợp 3 thành phần):**

```
 L_G  =  L_GAN  +  100 × L_moment  +  L_entropy

 Trong đó:

 (1) L_GAN     =  BCE(D(E(G(z,c))), 1)
                =  -log(σ(D(E(final_ipd))))

 (2) L_moment  =  E[|σ̂_fake - σ̂_real|]  +  E[|μ̂_fake - μ̂_real|]
                   (std matching)            (mean matching)

 (3) L_entropy =  -α × Var(δt)
                =  -2.0 × Var(δt)
```

Giải thích từng thành phần:
- **L_GAN:** Binary Cross-Entropy giữa output của Discriminator trên fake data với nhãn 1 (real). G cố gắng tối thiểu hóa loss này, tức là ép D tin fake là real.
- **L_moment:** First-moment (mean) matching và second-moment (standard deviation) matching. Trọng số ×100 vì đây là mục tiêu chính đảm bảo stealth — phân phối IPD sinh ra phải khớp thống kê với cover traffic.
- **L_entropy:** Lấy cảm hứng từ Soft Actor-Critic (SAC). Dấu âm biến minimize loss thành maximize variance. Ngăn mode collapse — hiện tượng generator chỉ sinh 1-2 giá trị δt lặp lại.

**Phase 3 — Discriminator Loss:**

```
 L_D  =  BCE(D(h_real), 1)  +  BCE(D(h_fake), 0)
```

D chỉ được update khi L_D > 0.15 (xem mục 5.1).

### 4.3 Cơ Chế GRU (Gated Recurrent Unit)

GRU xử lý chuỗi tuần tự. Tại mỗi bước t, nó quyết định giữ bao nhiêu thông tin cũ và nạp bao nhiêu thông tin mới qua 2 cổng:

```
 Update gate:  zₜ = σ(W_z · [hₜ₋₁, xₜ])         ← bao nhiêu cũ giữ lại?
 Reset gate:   rₜ = σ(W_r · [hₜ₋₁, xₜ])          ← quên bao nhiêu quá khứ?
 Candidate:    h̃ₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ])    ← thông tin mới
 Output:       hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ   ← trộn cũ + mới
```

Trong hệ thống này, GRU đóng vai trò then chốt: nó tạo ra **temporal dependency** — output tại timestep t phụ thuộc vào toàn bộ input từ timestep 0 đến t. Đây chính là lý do Decoder phải giải mã tuần tự: khi thử giả thuyết bit=0 hay bit=1 tại vị trí i, output δt[i] của GRU phụ thuộc vào trạng thái ẩn tích lũy từ các bit đã giải mã trước đó (0..i-1).

### 4.4 Tại Sao Dùng Tanh Trong NoiseGenerator

```
 δt = Tanh(Linear(GRU_output)) × noise_scale

 Tanh: ℝ → (-1, +1)     ⟹     δt ∈ (-0.4, +0.4) scaled
                                      = (-1.0, +1.0) giây
```

Tanh được chọn thay vì Sigmoid (0,1) hay ReLU (0,∞) vì:
- **Có giá trị âm:** Nhiễu cần cộng và TRỪ. Sigmoid chỉ cho giá trị dương → chỉ cộng → IPD luôn bị đẩy lên.
- **Bounded:** Giới hạn biên độ nhiễu. ReLU không bounded → nhiễu có thể vô cùng lớn → phá hỏng decode.
- **Zero-centered:** Mean output của Tanh ≈ 0 khi input đối xứng → nhiễu trung bình ≈ 0 → không bias phân phối IPD sang một phía.
- **KHÔNG hard constraint tại threshold 0.75s:** Phiên bản cũ dùng clamp(output, lb, ub) để ép Bit 0 luôn dưới 0.75s. Nhưng hard constraint tạo discontinuity (vết cắt) trong phân phối → classifier dễ phát hiện. Phiên bản Tanh để output tự do, dựa vào Shared Seed để đảm bảo reliability.

### 4.5 Mã BCH — Lý Thuyết Mã Sửa Lỗi

BCH(n, k, d) = BCH(255, 191, 17) trên trường Galois GF(2⁸):

```
 n = 255      Chiều dài codeword (bits)
 k = 191      Số bit dữ liệu thực tế mỗi block
 d = 17       Khoảng cách Hamming tối thiểu
 t = 8        Khả năng sửa lỗi = ⌊(d-1)/2⌋ = 8 bits/block

 Code rate R = k/n = 191/255 ≈ 0.749
 Overhead = (n-k)/k = 64/191 ≈ 33.5%
```

Mỗi block 23 bytes dữ liệu (191 bits) được mã hóa thành 255 bits. BCH thêm 64 bits parity cho phép sửa tối đa 8 bit bị sai trong mỗi block. Lỗi này có thể xảy ra do: (1) network jitter làm IPD bị méo, (2) lỗi tích lũy khi GRU decoder bị lệch context ở bit trước đó.

---

## 5. Kỹ Thuật Ổn Định GAN Training

### 5.1 Discriminator Threshold — Cơ chế cân bằng G/D

```
 if d_loss > 0.15:
     d_loss.backward()
     optimizer_d.step()
```

GAN training thường gặp vấn đề D hội tụ quá nhanh → gradient cho G bị vanish → G không học được. Ngưỡng 0.15 hoạt động như bộ điều tốc: khi D đã "giỏi" (loss thấp) thì dừng update, cho G có khoảng trống để cải thiện. Đây là biến thể của kỹ thuật "delayed discriminator update" phổ biến trong GAN training.

### 5.2 Tỉ Lệ Training G:D = 2:1

Mỗi epoch: G được train 2 lần duyệt qua dataset, D chỉ 1 lần. Kết hợp với D threshold ở trên và learning rate nhỏ hơn cho D (lr_D = 0.5 × lr_G), hệ thống thiên vị có chủ đích cho G để tránh mode collapse.

### 5.3 Learning Rate Scheduling — Cosine Annealing

```
 lr(t) = η_min + ½(η_max - η_min)(1 + cos(πt/T_max))

 η_max = 1e-3,  η_min = 1e-5,  T_max = 200 epochs
```

Learning rate giảm dần theo hình cosine từ 1e-3 → 1e-5. Giai đoạn đầu (lr cao) → khám phá nhanh. Giai đoạn sau (lr thấp) → tinh chỉnh chính xác. Áp dụng cho cả 3 optimizer (G, D, Embedder-Recovery).

### 5.4 SAC Entropy — Chống Mode Collapse

```
 L_entropy = -α × Var(δt)    với α = 2.0
```

Lấy ý tưởng từ Soft Actor-Critic (Haarnoja et al., 2018) trong Reinforcement Learning. SAC thêm entropy bonus vào reward để khuyến khích policy đa dạng. Ở đây, ta dùng variance của nhiễu δt làm proxy cho entropy. Khi α lớn, generator bị ép phải sinh δt phân tán rộng thay vì co cụm quanh 1 giá trị → phân phối IPD đầu ra đầy đặn hơn, khó phân biệt với traffic thật.

---

## 6. Đánh Giá Và Mô Hình Đe Dọa

### 6.1 Metrics Đánh Giá

**Kolmogorov-Smirnov Test (KS Test):**
```
 KS_stat = sup_x |F_generated(x) - F_target(x)|
```
So sánh CDF (hàm phân phối tích lũy) của IPD sinh ra với phân phối mục tiêu N(0.75, 0.3). KS_stat < 0.05 và p-value > 0.05 → không đủ bằng chứng thống kê để bác bỏ giả thuyết "2 phân phối giống nhau" → stealth tốt.

**Discriminative Score:**
Huấn luyện một GRU-based binary classifier phân biệt chuỗi thật vs chuỗi giả. Score = |0.5 - accuracy|. Score → 0 nghĩa là classifier đạt accuracy ≈ 50% (random guess) → không phân biệt được → stealth hoàn hảo.

**Predictive Score:**
Huấn luyện một GRU predictor trên dữ liệu giả, đánh giá trên dữ liệu thật. MAE thấp nghĩa là temporal dynamics của chuỗi giả khớp với chuỗi thật → GRU NoiseGenerator đã học được liên kết thời gian tự nhiên.

### 6.2 Threat Model — Đối Thủ Giả Định

Hệ thống giả định đối thủ (warden/NIDS) có khả năng:
- **Quan sát toàn bộ timing traffic** giữa Sender và Receiver (passive adversary).
- **Phân tích thống kê bậc 1, 2** (mean, std) và kiểm định phân phối (KS test, chi-square).
- **Train classifier (ML-based NIDS)** để phân biệt covert traffic vs normal traffic.

Đối thủ KHÔNG có:
- Shared seed (2025).
- Model weights (generator.pth).
- Kiến thức về giao thức encode/decode.

### 6.3 Tại Sao Bỏ Generator + ConditioningNetwork Gốc Của TimeGAN

**Generator gốc (TimeGAN):** Nhận noise Z → sinh hidden state tự do → qua Supervisor → qua Recovery → ra IPD. Output không bị ràng buộc bởi mốc vật lý Bit 0/1. Trong bài toán covert channel, ta CẦN output phụ thuộc trực tiếp vào bit input (0.5s hoặc 1.0s). Generator gốc không cung cấp cơ chế nhúng bit — nó chỉ sinh dữ liệu "trông giống thật" mà không mang thông tin.

**ConditioningNetwork (MC-TimeGAN):** Biến label (0/1) thành vector điều kiện nối vào mọi mạng. Nhưng conditioning chỉ ảnh hưởng gián tiếp (steer phân phối), không đảm bảo mapping chính xác bit→IPD. Trong pilot test, conditioning accuracy chỉ đạt ~85% — không đủ cho covert channel (cần >99%).

**NoiseGenerator (thay thế):** Nhận trực tiếp mốc IPD cứng (C_scaled) + noise Z → cộng nhiễu δt → output = C + δt. Mapping bit→IPD là **by construction** (mốc cứng), nhiễu chỉ **thêm vào** chứ không thay thế → reliability 100% nếu decode đúng. Đây là sự khác biệt kiến trúc then chốt so với TimeGAN gốc.

### 6.4 Dung Lượng Kênh Truyền (Channel Capacity)

```
 1 IPD  =  1 bit thông tin
 1 BCH block  =  255 IPDs  =  191 bits data thực
 1 byte data  =  255/191 × 8  ≈  10.7 IPDs

 Throughput ≈ 1 bit / (0.75s trung bình)  ≈  1.33 bit/s
            ≈  0.17 byte/s

 Ví dụ:  File 10 bytes → 1 BCH block → 255 IPDs
         Thời gian gửi ≈ 255 × 0.75s ≈ 191 giây ≈ 3.2 phút
```

Tốc độ rất thấp — nhưng đây là trade-off cố hữu của covert timing channel: ưu tiên **undetectability** (không bị phát hiện) hơn **throughput** (tốc độ). Trong thực tế, covert channel thường chỉ cần truyền lượng nhỏ thông tin (khóa mã, lệnh điều khiển, tọa độ) nên tốc độ thấp là chấp nhận được.
