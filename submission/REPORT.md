# Lab 21 — Fine-tuning ticket CSKH: tăng điểm target nhưng chưa đạt cổng hồi quy

**Họ tên:** Phạm Hoàng Trọng  
**Mã học viên:** 2A202602765 (theo tên thư mục; cần xác nhận)  
**Ngày lập báo cáo:** 07/10/2026  
**Môi trường:** Google Colab, Tesla T4, 14,6 GB VRAM, tier T4  
**Model:** `unsloth/Qwen3.5-4B` — **Commit đã ghi nhận ở setup ban đầu:** `d27c1c0` (cần đối chiếu commit của lượt chạy lại)

> Báo cáo dùng log NB1–NB5 của lượt chạy lại mới nhất (run 2), thay thế số liệu lượt cũ. Đã nhập và đối chiếu file kết quả, adapter từ hai bản sao Drive vào checkout trên máy. Gatekeeper trên gói nộp đã cập nhật: 26 PASS, 1 WARN về phán quyết FAILED, 0 FAIL. Unit tests tại máy: 116 passed, 3 skipped; lượt Colab trước đó có 119 passed. Đánh giá bổ sung trên đủ 50 target xác nhận 33 ca fine-tune thắng, 17 ca hòa, 0 ca thua baseline (b). Không có ca thua target; phép đo bổ sung regression tìm được 5 ca có điểm thấp hơn baseline, gồm ít nhất 3 ca lệch nhiệm vụ rõ rệt. Báo cáo trình bày ca thua regression riêng; cần giảng viên xác nhận việc dùng chúng cho mục ví dụ ticket của rubric.

## 1. Mục tiêu và lựa chọn

Thí nghiệm kiểm tra liệu LoRA có cải thiện việc chuyển ticket chăm sóc khách hàng tiếng Việt thành JSON bốn trường `intent`, `urgency`, `product`, `sentiment` so với cùng model gốc được prompt tốt hay không. Điều kiện thành công còn yêu cầu giữ năng lực trả lời kiến thức/chỉ dẫn phổ thông trong ngưỡng của lab.

Sử dụng model và corpus mặc định để tập trung vào pipeline và đối chứng. T4 phù hợp với model này trong lần chạy: run chính đạt đỉnh VRAM 8,78 GB. Dataset có 250 mẫu, nhãn cấu trúc giúp chấm độ chính xác từng trường bằng scorer khách quan thay vì LLM judge. Tuy nhiên, corpus hẹp chưa đại diện đầy đủ ticket thực tế.

NB2 được chạy trước huấn luyện, lưu baseline (a) với prompt đơn giản và (b) với prompt tối ưu mặc định. Fine-tune dùng prompt đơn giản khi đánh giá để kiểm tra hành vi phân loại được học vào adapter. Không thay corpus hay prompt tối ưu trong quá trình chạy theo hướng dẫn.

## 2. Dữ liệu, template và loss mask

| Cấu hình / phép đo | Giá trị |
|---|---:|
| Tổng mẫu ban đầu | 250 |
| Train / validation, seed 42 | 225 / 25 |
| Target / regression eval | 50 / 15 |
| Token trung bình | 93,1 |
| p50 / p95 / p99 / max | 93 / 98 / 100 / 101 |
| max_length gợi ý / sử dụng | 256 / 1024 |
| MASK_MODE | assistant-only |
| EVAL_LIMIT | Không giới hạn |

p95 đo được là 98 token, nên giới hạn gợi ý 256 đã đủ khoảng dư cho corpus này. Lần chạy giữ 1024 theo tier T4 để nhất quán cấu hình giữa các run, không phải vì dữ liệu cần 1024 token. Độ dài lớn nhất 101 cho thấy giới hạn này không cắt mất nội dung corpus hiện tại. Nếu tối ưu tài nguyên ở lần thử tiếp theo, có thể dùng 256 nhất quán và ghi lại kết quả; chưa có phép đo so sánh hai giới hạn trong lần chạy này.

Template giữ được khối `<think>` trong ví dụ reasoning do NB1 dựng ra. Corpus huấn luyện có đáp án JSON, không có reasoning trace thực; kiểm tra template không chứng minh adapter học được suy luận nhiều bước.

| Bằng chứng mẫu NB1 | Kết quả |
|---|---:|
| n_supervised / n_total | 39 / 94 |
| supervised_fraction | 0,4149 |
| answer_is_supervised | true |
| question_is_masked | true |

Đoạn được tính loss trong log:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đáp án JSON nằm trong vùng supervised, câu hỏi không bị tính loss. Token đóng khối thinking và token kết thúc cũng xuất hiện trong vùng này, nên mask không chỉ chứa JSON thuần. NB1 minh họa thêm `everything` với 100% token được supervised để chỉ ra lỗi học cả prompt; các run thực tế vẫn dùng `assistant-only`. NB3 ghi nhận 9.014/20.951 token train được supervised, khoảng 43%, khác tỷ lệ mẫu minh họa là bình thường.

## 3. Huấn luyện và thiết kế đối chứng

Model gốc được giữ cố định; chỉ cập nhật LoRA. Model có 32 lớp: 24 linear attention, 8 full attention. Run chính gắn LoRA vào các module tuyến tính phần văn bản. Các run dùng cùng 225 mẫu, seed 42, batch mỗi thiết bị 1, gradient accumulation 16, batch hiệu dụng 16, max_length 1024, ngân sách 2 epochs và 30 optimizer steps. Run chính dùng cosine scheduler, warmup 3 steps, gradient checkpointing, không packing.

T4 dùng fp16 với gradient scaling, không dùng bf16. Với QLoRA, log cho thấy 496 tensor trainable được chuyển bf16 sang fp32 để tương thích fp16 GradScaler; 4-bit nói về base đã lượng tử hóa, không phải tất cả tham số adapter.

| Run | Vị trí | Rank / alpha | Tham số train | LR | Base 4-bit | Steps |
|---|---|---:|---:|---:|---|---:|
| correct | text-linear | 16 / 32 | 32.464.896 | 1e-4 | Không | 30 |
| attn_only | q,v | 283 / 566 | 32.456.704 | 1e-4 | Không | 30 |
| wrong_lr | text-linear | 16 / 32 | 32.464.896 | 1e-5 | Không | 30 |
| qlora | text-linear | 16 / 32 | 32.464.896 | 1e-4 | Có | 30 |

Ngân sách attn_only lệch 8.192 tham số, khoảng 0,0252%, dưới 5%. Rank được tăng để khớp ngân sách khi đổi vị trí; đây không phải thí nghiệm rank độc lập. wrong_lr đổi thang LR; qlora đổi chế độ lượng tử hóa và xử lý precision cần thiết. Mỗi adapter được train riêng từ base tương ứng, không train nối tiếp trên correct.

## 4. Baseline đóng băng và kết quả đánh giá

NB2 lưu `optimized_prompt_sha=719e74d3b6232053`, đủ 50 target và 15 regression, `eval_limit=null`, `smoke_mode=false`. Baseline (b) có target 0,765, vượt (a) 0,000; prompt tốt là mốc so sánh mạnh hơn thực sự.

Target là trung bình độ chính xác từng trường, không phải tỷ lệ ticket đúng toàn bộ. Regression là keyword recall trên 15 câu hỏi. Format là JSON parse được và có đủ khóa theo scorer. Latency là ms/mẫu trong phép đo generation của lab, không phải đo riêng độ trễ từng yêu cầu khi triển khai.

| Run | Target | Regression | Format | Latency ms/mẫu |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0,0000 | 0,7911 | 0,0000 | 3314,5 |
| (b) Base + optimized prompt | 0,7650 | 0,7911 | 1,0000 | 1082,5 |
| (c) LoRA correct | 0,9700 | 0,6111 | 1,0000 | 1377,9 |

Fine-tune tăng target 20,5 điểm phần trăm, giữ format 100%, nhưng latency tăng khoảng 27,3% và regression giảm 18 điểm phần trăm. Target bằng 0 đi cùng format bằng 0 ở (a) chưa chứng minh model không hiểu ticket: đầu ra không phù hợp định dạng có thể không được scorer ghi nhận các trường đúng.

## 5. Phân tích đối chứng

| Run | Loss trung bình train | Target NB5 | Format | Train giây | VRAM GB | Latency ms/mẫu |
|---|---:|---:|---:|---:|---:|---:|
| correct | 0,6271 | 0,970 | 1,000 | 399,1 | 8,78 | 1377,9 |
| attn_only | 0,5362 | 0,970 | 1,000 | 291,1 | 8,79 | 894,9 |
| wrong_lr | 1,5702 | 0,000 | 0,000 | 432,8 | 8,78 | 5248,7 |
| qlora | 0,7058 | 0,940 | 1,000 | 505,3 | 3,86 | 1818,6 |

Cột `final_loss` trong artefact là `train_loss` tổng kết của trainer, tức trung bình toàn lượt train, không phải loss đoạn cuối. correct có loss đoạn cuối 0,02913 nhưng loss tổng kết 0,6271.

### Vị trí và rank

correct và attn_only cùng đạt target 0,970 và format 1,000 trong lượt mới. Theo loss train, attn_only thấp hơn correct (0,5362 so với 0,6271), nhưng ưu thế loss không chuyển thành ưu thế target. Vì vậy, không chọn adapter chỉ dựa vào loss thấp nhất. Attention-only với rank tăng để khớp tham số hòa text-linear trên điểm tổng hợp của nhiệm vụ này; chưa biết chúng đúng hoặc sai trên cùng các mẫu. Latency attn_only thấp hơn khoảng 35,1% trong lần đo. Muốn tách tác động rank cần giữ vị trí và quét rank ở thí nghiệm riêng. Chưa đo nhiều seed nên không kết luận hai cấu hình luôn tương đương.

### Learning rate

wrong_lr giảm LR 10 lần và loss giảm chậm: từ khoảng 2,163 xuống 1,119 ở đoạn cuối, trong khi correct xuống khoảng 0,029. Với cùng 30 step, LR thấp chưa giúp đạt hành vi xuất JSON cần thiết. Nếu chỉ thấy loss giảm, có thể kết luận sai rằng model đã đủ tốt. Target và format bằng 0 cho thấy chưa đáp ứng hợp đồng đầu ra; chưa có chuỗi output đầy đủ để xác định lỗi cụ thể. Không suy ra mọi LR thấp đều kém với mọi ngân sách train.

### QLoRA

VRAM giảm từ 8,78 xuống 3,86 GB: tiết kiệm 4,92 GB, khoảng 56%. Đổi lại, target giảm 3 điểm phần trăm; thời gian train tăng khoảng 26,6% và latency tăng khoảng 32,0%. Kết quả phù hợp với lo ngại đánh đổi chất lượng do lượng tử hóa, nhưng một lần chạy chưa đủ khuyến nghị tuyệt đối không dùng QLoRA. Khi VRAM là giới hạn chính, 94% target vẫn đáng cân nhắc. NB5 không đo regression cho ba đối chứng, nên chưa thể nói attn_only hoặc qlora vượt cổng hồi quy.

## 6. Phán quyết: FAILED

`target_delta=+0,205`; `regression_delta=-0,180`; ngưỡng regression cho phép là `-0,020`. `valid_trace_rate=0,0`.

Fine-tune cải thiện nhiệm vụ phân loại nhưng chưa đạt điều kiện bảo toàn năng lực phổ thông. Chỉ nhìn target 97% sẽ bỏ qua đánh đổi mà cổng hồi quy nhằm phát hiện. Huấn luyện chỉ trên ticket có thể chuyên môn hóa model và thay đổi phản hồi ngoài miền; mức giảm phù hợp với giả thuyết quên năng lực, nhưng 15 câu hỏi và keyword recall chưa đủ khẳng định catastrophic forgetting là nguyên nhân duy nhất. Không nới ngưỡng, đổi eval hay làm yếu baseline để chuyển FAILED thành PASSED. Hướng thử tiếp theo là trộn 1–5% replay data phổ thông vào train, giữ nguyên eval và đo lại; đây là giả thuyết cải thiện, chưa được kiểm chứng. Trace rate bằng 0 trong nhiệm vụ JSON và dữ liệu không có trace chưa đủ kết luận reasoning-trace collapse.

Log có một số `grad_norm=nan`, dù loss hữu hạn và adapter được lưu. Chưa có số bước cập nhật bị bỏ qua hoặc kiểm tra chi tiết tensor, nên không khẳng định toàn bộ train ổn định số học. Fallback kernel chạy được nhưng ảnh hưởng thời gian; số tốc độ chỉ phản ánh môi trường lần chạy. NB1–NB5 lượt mới mất tổng 2.863 giây, khoảng 47,7 phút (14 + 326 + 459 + 1.393 + 671 giây), không gồm cài đặt và viết report.

Lượt đầu từng ghi nhận regression 0,7444, còn lượt mới là 0,6111 dù target đều 0,970. Báo cáo này sử dụng lượt mới nhất, không chọn lượt có regression cao hơn để trình bày. Chưa có đầu ra từng câu và thông tin đầy đủ về môi trường để giải thích chênh lệch; đây là giới hạn về tính tái lập cần khảo sát thêm.

## 7. Đánh giá định tính bổ sung

ID bắt đầu từ 0 trong tập target. Ticket và nhãn dưới đây đối chiếu corpus mặc định trong checkout; cần xác nhận checksum eval Colab khớp khi chuyển artefact. Log NB5 cắt ngắn dự đoán, nên chỉ trích thông tin nhìn thấy, không tái tạo phần bị cắt.

| ID | Ticket rút gọn | Nhãn đúng: intent / urgency / product / sentiment | Fine-tune và nhận xét |
|---|---|---|---|
| 3 | Bình giữ nhiệt, chưa thấy tiền; Khi nào tiện. Cảm ơn shop nhiều. | hoan_tien / thap / bình giữ nhiệt / tich_cuc | Score 0,75; urgency=trung_binh, sai so với thap |
| 5 | Nồi chiên không dầu thiếu phụ kiện; Khi nào tiện. Cho tôi hỏi. | san_pham_loi / thap / nồi chiên không dầu / trung_tinh | Score 0,75; urgency=trung_binh, sai so với thap |
| 12 | Áo khoác gió bị lỗi; Khi nào tiện. Cảm ơn shop nhiều. | san_pham_loi / thap / áo khoác gió / tich_cuc | Score 0,75; urgency=trung_binh, sai so với thap |
| 47 | Ốp lưng điện thoại, shipper không gọi; Hỏi cho biết thôi. Shop hỗ trợ tốt. | van_chuyen / thap / ốp lưng điện thoại / tich_cuc | Score 1,00; đúng cả bốn trường theo scorer |
| 48 | Hỏi giá ốp lưng điện thoại; Mong shop phản hồi. Nhờ shop kiểm tra. | hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh | Score 1,00; đúng cả bốn trường theo scorer |
| 49 | Ốp lưng điện thoại sai màu; Sớm nhé. Shop xem giúp. | san_pham_loi / trung_binh / ốp lưng điện thoại / trung_tinh | Score 1,00; đúng cả bốn trường theo scorer |

Ba ca sai được in ra cùng nhầm urgency thấp thành trung bình khi có cụm Khi nào tiện. Đây chưa phải thống kê toàn bộ lỗi. Ca score 1,00 chưa chắc thắng baseline, ca score 0,75 chưa chắc thua baseline.

### So sánh từng mẫu với baseline (b)

Sau NB5, chạy generation bổ sung với cùng model, prompt, tập target và hàm greedy decode của lab. Baseline tổng hợp và verdict đóng băng được giữ nguyên. Kết quả bổ sung khớp target đã ghi nhận: baseline 0,765, fine-tune 0,970. Output đầy đủ và nhãn được lưu vào `results/qualitative_comparison.json` trên Colab.

Trên 50 ticket, có **33 ca thắng, 17 ca hòa và 0 ca thua**, theo độ chính xác trung bình bốn trường của từng ticket. Vì vậy, không thể cung cấp hai ca thua target từ lần đo này. Kết quả này chỉ áp dụng cho tập target và thang đo đã dùng; không có nghĩa fine-tune không có điểm yếu, vì các lỗi urgency vẫn tồn tại và điểm regression giảm rõ rệt.

| ID | Ticket rút gọn | Trường khác biệt và nhãn đúng | Baseline (b) | Fine-tune | Điểm b / FT |
|---|---|---|---|---|---:|
| 0 | Chuột không dây, Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | intent = doi_tra | hoan_tien | doi_tra | 0,75 / 1,00 |
| 1 | Ốp lưng điện thoại, Hoàn tiền. Sớm nhé. Bực mình. | urgency = trung_binh | cao | trung_binh | 0,75 / 1,00 |
| 4 | Đèn bàn LED, Vỡ khi nhận. Gấp. Shop xem giúp. | sentiment = trung_tinh | tieu_cuc | trung_tinh | 0,75 / 1,00 |

Ở ID 0, fine-tune phân biệt yêu cầu đổi/trả với hoàn tiền. Ở ID 1, fine-tune gán đúng mức khẩn cấp trung bình theo nhãn cho cụm Sớm nhé. Ở ID 4, fine-tune không tự suy ra cảm xúc tiêu cực chỉ vì sản phẩm bị vỡ; nó khớp nhãn trung tính. Những nhận xét này dựa trên output thực tế đã gửi, không khái quát thành quy tắc cho mọi ticket.

**Giới hạn so với rubric:** đã có bằng chứng so sánh toàn bộ target và các ví dụ thắng, nhưng không tồn tại hai ca thua target trong phép đo bổ sung. Ba ca lỗi ở bảng trên không được đổi tên thành ca thua. Cần ghi nhận giới hạn này và trao đổi với giảng viên về cách chấm yêu cầu hai ca thua; không sửa eval hay prompt để tạo thất bại. Có thể bổ sung ví dụ suy giảm trên tập regression để minh họa đánh đổi, nhưng chúng không tự thay thế yêu cầu ví dụ ticket của rubric.

### Các ca thua trên regression — bằng chứng thực tế

Phép đo bổ sung trên cùng 15 câu hỏi phổ thông dùng `system=None`, `max_new_tokens=96`, greedy decode và scorer keyword recall như NB2/NB5. Điểm khớp kết quả lượt mới: base 0,7911, fine-tune 0,6111. Có 5 câu bị giảm điểm. Output và điểm từng câu được lưu trong `results/regression_comparison.json`; ví dụ trong `results/regression_examples.md`. Không thay baseline đóng băng hay verdict.

| ID | Câu hỏi | Baseline | Fine-tune | Điểm b / FT | Phân tích |
|---|---|---|---|---|---:|
| 2 | 1 km bằng bao nhiêu mét? | Trả lời 1 km = 1000 m | JSON phân loại hoi_thong_tin, không có đáp án số | 1,00 / 0,00 | Thua rõ về thực hiện nhiệm vụ |
| 9 | Một năm có bao nhiêu tháng? | Nêu 12 tháng; phần liệt kê cuối bị cắt | JSON phân loại hoi_thong_tin, không trả lời số tháng | 1,00 / 0,00 | Thua rõ; câu trả lời chính của base có đáp án cần thiết |
| 13 | Thành phố Hồ Chí Minh trước đây có tên là gì? | Nêu Saigon/Sài Gòn; có phần giải thích thêm chưa được kiểm chứng | JSON mô tả ý định câu hỏi, không nêu tên cũ | 1,00 / 0,00 | Fine-tune không trả lời thông tin được hỏi; không coi mọi chi tiết base là đúng |
| 14 | Giải thích ngắn gọn quang hợp là gì | Giải thích dùng ánh sáng, có từ cây | JSON có response giải thích phù hợp bằng thực vật, không có từ cây | 1,00 / 0,50 | Thua theo keyword recall; chưa phải bằng chứng kiến thức quang hợp kém hơn |
| 11 | Kể tên một loại trái cây nhiệt đới | Lặp dâu tây, tự phủ định và bị cắt; log chấm 0,20 | JSON mô tả ý định, không nêu trái cây | 0,20 / 0,00 | Cả hai đầu ra đều chưa đáp ứng tốt; scorer nhầm từ đưa với dứa sau bỏ dấu, tạo điểm 0,20 dù chưa có đáp án đúng |

Hai ca thua rõ rệt được trích nguyên output fine-tune:

**ID 2 — 1 km bằng bao nhiêu mét?** Baseline có đáp án `1 km = 1000 m`. Fine-tune trả:

```json
{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}
```

**ID 9 — Một năm có bao nhiêu tháng?** Baseline nêu `12 tháng`. Fine-tune trả:

```json
{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}
```

Ở cả hai câu, vấn đề không phải sai một phép tính: adapter áp hành vi phân loại ticket vào câu hỏi kiến thức dù không có system prompt yêu cầu phân loại. Đây là bằng chứng về lệch hành vi ngoài miền sau fine-tune, chưa chứng minh kiến thức tương ứng đã bị xóa khỏi trọng số. ID 13 lặp lại mẫu lệch nhiệm vụ này. Ba ca này có thể dùng để minh họa cụ thể lý do chưa triển khai adapter như trợ lý đa năng.

ID 14 cho thấy giới hạn scorer: thực vật là cách diễn đạt phù hợp, nhưng không khớp từ khóa cây. Vì vậy, giảm keyword recall không luôn tương đương giảm độ đúng nội dung. ID 11 cũng không nên được trình bày như một câu trả lời base hoàn toàn đúng: đoạn in ra không cho thấy ví dụ nhiệt đới hợp lệ và điểm 0,20 là false positive: sau bỏ dấu, từ đưa trong hãy để tôi đưa ra khớp substring dua của từ khóa dứa. Phân tích này giữ cả điểm đo được và chất lượng nội dung, tránh chọn lọc chỉ số thuận lợi.

Các ví dụ trên là regression, không phải ca thua target. Chúng bổ sung bằng chứng đánh đổi thực tế; chưa tự bảo đảm nhận điểm mục rubric yêu cầu ví dụ ticket.
## 8. Kết luận và điều học được

LoRA có giá trị đối với tác vụ ticket trong corpus này: target tăng từ 76,5% của base được prompt tốt lên 97%, trong khi format giữ 100%. Tuy nhiên, tôi chưa đề xuất sử dụng adapter như trợ lý đa năng vì điểm regression giảm quá ngưỡng lab, đồng thời latency cao hơn baseline prompt tốt. Với ứng dụng chỉ phân loại, vẫn cần đánh giá ticket thực tế, cách diễn đạt ngoài corpus và mức nghiêm trọng của lỗi trước khi triển khai. Các lỗi urgency có thể làm sai thứ tự ưu tiên xử lý yêu cầu khách hàng.

Đòn bẩy rõ trong lần chạy là thang LR: cùng ngân sách cập nhật, LR thấp chưa đạt định dạng đầu ra cần thiết. Hai vị trí adapter đạt cùng target khi khớp ngân sách tham số trong lượt mới, chưa có bằng chứng ưu tiên text-linear về độ chính xác. QLoRA tiết kiệm bộ nhớ đáng kể nhưng đánh đổi điểm và tốc độ. Mask đúng là điều kiện nền tảng làm phép so sánh có ý nghĩa; lần chạy này chưa trực tiếp đo ảnh hưởng mask sai lên chất lượng cuối. Nếu có thêm thời gian, tôi sẽ kiểm tra bất ổn số học, thêm replay data, đo nhiều seed và mở rộng eval ngoài miền. Tôi sẽ giữ nguyên mốc đánh giá và chấp nhận FAILED nếu đánh đổi chưa đạt, thay vì chọn lọc chỉ số để trình bày chiến thắng.

**Phản tư dưới đây là bản diễn đạt đề xuất; người học cần xác nhận và chỉnh theo trải nghiệm cá nhân:**

1. Tôi hiểu model gốc được giữ cố định và chỉ adapter được cập nhật. Các đối chứng train riêng từ base, không cộng cả bốn adapter khi dùng.
2. Tôi phân biệt LR nhỏ với máy chạy chậm: wrong_lr có thời gian gần correct nhưng kết quả khác rõ rệt.
3. Tôi hiểu loss thấp khác điểm tác vụ cao, và fine-tune trả lời đúng khác fine-tune thắng prompt tốt. Kết luận cần cả target và regression.

AI assistant hỗ trợ giải thích LoRA, hướng dẫn chạy từng NB, đọc log và soạn report. Người học cần đối chiếu artefact và xác nhận cách xử lý yêu cầu hai ca thua với giảng viên trước khi nộp.

## 9. Hoàn tất bài nộp

Nguồn bằng chứng cần kèm: results/template_check.json, mask_proof.json, token_stats.json, baselines_frozen.json, runs.csv, verdict.json, autopsy.json, qualitative.json, qualitative_comparison.json, regression_comparison.json, regression_examples.md; cùng adapters/correct/ và notebook theo định dạng nộp.

- [x] Chạy NB1–NB5 với eval đầy đủ theo log.
- [x] Soạn số liệu, phân tích và phán quyết FAILED.
- [x] Chuyển artefact Colab về máy; đối chiếu verdict, runs, split, checksum và bằng chứng bonus.
- [x] Chạy so sánh định tính bổ sung: 33 thắng, 17 hòa, 0 thua; lưu output đầy đủ trên Colab.
- [x] Bổ sung 5 ví dụ regression giảm điểm, phân biệt ca lệch nhiệm vụ và giới hạn keyword scorer.
- [ ] Xác nhận với giảng viên việc dùng ca thua regression cho mục ví dụ ticket khi không có ca thua target.
- [ ] Xác nhận thông tin học viên và phản tư.
- [x] Chạy gatekeeper trước khi thay report: 119 tests qua; mask, checksum, prompt và đối chứng đạt.
- [x] Kiểm tra gói nộp có report cập nhật bằng gatekeeper: 26 PASS, 1 WARN, 0 FAIL.
- [x] Notebook của gói nộp đã clear output; gói core+B1+B2 được đóng ZIP riêng.

B1 đã hoàn thành kiểm tra merge và hoán đổi hai adapter trên cùng một base theo bằng chứng Colab. B2 đã tạo dataset, chạy NB1/NB2/NB3/NB5 và bổ sung review AI cùng kiểm tra toàn bộ 288 mẫu ngày 07/10/2026; không ghi nhận review độc lập của con người. Bằng chứng B2 đã chuẩn bị để giảng viên xét tiêu chí chất lượng dữ liệu tổng hợp. B5 đã đăng adapter chính công khai lên HuggingFace Hub, kiểm tra truy cập không đăng nhập và checksum trọng số. B4 đã quét r=8,16,64 và đối chiếu artefact gốc, code/dữ liệu/baseline khớp giữa ba rank; có phân tích rank so với vị trí/LR. B3 chưa hoàn thành reasoning-trace contrast.




## 10. Bonus B1 — merge và hoán đổi adapter đã hoàn tất

NB6 đánh giá đủ 50 target: trước merge 0,9700, sau merge 0,9700, delta 0,0000, đạt ngưỡng không giảm quá 0,01. Model merged đã được lưu; bằng chứng điểm nằm trong results/merge_check.json trên Colab. Generation mất khoảng 70 giây trước merge và 43 giây sau merge trong lần đo này; đây chỉ là thời gian quan sát, không phải benchmark lặp lại.

Lần chạy NB6 ban đầu gặp ValueError yêu cầu offload_dir khi nạp base cho phần hoán đổi. Code còn giữ biến model tham chiếu tới base đã merge, có thể giữ VRAM; đã sửa để xóa cả model và merged trước khi nạp lại. Phần hoán đổi được thực hiện thành công riêng bằng scripts/bonus_hotswap.py trong process mới, không cần train hay merge lại.

Bằng chứng results/hotswap_check.json ghi nhận same_loaded_model=true và hai adapter correct, attn_only. Cả hai được nạp trên cùng một base; model.set_adapter chọn adapter lần lượt cho cùng ticket. Đây là hoán đổi tuần tự adapter, không phải thử nghiệm phục vụ nhiều request đồng thời.

Ticket kiểm tra: “Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.” Cả hai adapter đều sinh:

```json
{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
```

Hai adapter cho cùng output trên ví dụ này không phủ định việc chuyển adapter: bằng chứng là tên adapter được chọn trong cùng đối tượng model. Chưa đo sự khác biệt trên toàn bộ tập khi hoán đổi và không dùng một ticket để kết luận hai adapter luôn tương đương.

Đã hoàn thành hai yêu cầu B1: điểm sau merge không giảm quá 0,01 và đổi ít nhất hai adapter trên cùng base. Đề nghị xét +3 điểm B1 theo rubric; điểm cuối do giảng viên chấm. Các file kết quả đã được người học sao lưu vào Google Drive tại Lab21_run2.

Merge gộp phần điều chỉnh vào trọng số, loại bỏ nhánh LoRA riêng trong suy luận. Đổi lại, việc chuyển nhanh giữa các adapter trên một base sạch không còn thuận tiện và artefact model đầy đủ lớn hơn adapter. Nên giữ adapter riêng khi cần nhiều tác vụ, khách hàng, phiên bản hoặc khả năng rollback trên một base chung. Merge không tự khắc phục suy giảm regression của adapter.



## 11. Bonus B2 — dataset hỗ trợ sinh viên UIT và kết quả thực nghiệm

Đã tạo dataset tổng hợp gồm 240 mẫu train_seed và 48 eval_target, sáu intent học phí, đăng ký môn, lịch học, tài khoản, giấy tờ và học vụ. Có 72 tình huống gốc, mỗi tình huống bốn biến thể; 60 tình huống nguồn train và 12 tình huống eval không giao nhau. Mỗi intent có 40 mẫu nguồn train và 8 eval. Kiểm tra máy xác nhận không có ticket trùng sau chuẩn hóa, nhãn JSON nhất quán và không có scenario_id giao nhau. Dữ liệu dùng tên dịch vụ trong trường product để tương thích pipeline.

Nguồn là dữ liệu do AI assistant soạn và sinh bằng quy tắc, không phải hồ sơ sinh viên hoặc dataset UIT chính thức. Tên dịch vụ và câu urgency/sentiment được thể hiện trực tiếp, nên tác vụ có thể dễ hơn ticket thực tế. Không khẳng định base chưa từng thấy chủ đề sinh viên. Ngày 07/10/2026, Codex AI assistant review ngữ nghĩa 72 tình huống và chính sách nhãn, kết hợp kiểm tra tự động toàn bộ 288 mẫu; không phát hiện nhãn trái quy tắc công bố, sửa 0 mẫu. Đây là review AI sau thí nghiệm, không phải review thủ công độc lập của người học. Corpus, split, prompt và số đo giữ nguyên; không cần train lại vì không sửa dữ liệu. Tài liệu tại bonus/B2_UIT/data/CUSTOM_DATASET.md và SEMANTIC_REVIEW.json; data/uit_b2/ có bản sao tài liệu review.

### Thiết lập và kết quả B2

Chạy trong repo riêng /content/Lab21_UIT_B2, lưu ở Drive Lab21_UIT_B2, không ghi đè bài chính. Baseline B2 được đóng băng trước train với prompt taxonomy sinh viên, SHA 3eff71c568b0c8f1. Eval đủ 48 target và 15 regression, không dùng smoke mode. Bộ 240 train_seed chia train/validation 216/24 theo NB1, đã xác nhận bằng file split. Các biến thể cùng tình huống có thể nằm ở cả train và validation; tập target tách scenario là bằng chứng chính.

Run correct B2 dùng text-linear, r=16, alpha=32, LR=1e-4, fp16, assistant-only, 32.464.896 tham số train, 28 optimizer steps với ngân sách 2 epochs. Số step khác bài chính do kích thước train khác; không dùng hai miền để so tác động số step. Train 405,2 giây, VRAM đỉnh 8,88 GB, loss trung bình 0,5634. Không chạy lại ba đối chứng NB4 trong phạm vi B2.

| Run B2 | Target | Regression | Format | Latency ms/mẫu |
|---|---:|---:|---:|---:|
| Base + naive prompt | 0,0000 | 0,7911 | 0,0000 | 3381,3 |
| Base + optimized prompt UIT | 0,9583 | 0,7911 | 1,0000 | 1036,7 |
| LoRA correct UIT | 1,0000 | 0,5800 | 1,0000 | 1470,2 |

**Phán quyết B2: FAILED.** Target tăng 0,0416667 (khoảng 4,17 điểm phần trăm), regression giảm 0,2111111 (khoảng 21,11 điểm phần trăm), vượt ngưỡng 0,020. Valid trace rate 0,0; dataset JSON không có trace nên không coi đây là thí nghiệm B3.

Fine-tune đạt đúng toàn bộ trường theo scorer trên 48 target của phép đo này, nhưng không chứng minh khả năng giải quyết mọi yêu cầu sinh viên. Baseline prompt tốt đã đạt 95,83%, cho thấy taxonomy và cue trực tiếp làm tác vụ khá dễ; mức cải thiện nhỏ hơn bài chính. Adapter giảm điểm phổ thông mạnh và latency cao hơn khoảng 41,8%, nên chưa đạt điều kiện dùng làm trợ lý đa năng. Có thể tiếp tục thử replay data, ticket không ghi trực tiếp tên dịch vụ và tình huống nhiều ý; chưa thực hiện các thử nghiệm đó.

### Bằng chứng để xét B2

Dataset 240 mẫu nguồn train, 48 target, manifest, quality_checks, CUSTOM_DATASET.md và SEMANTIC_REVIEW.json đã có trong gói B2. NB1, NB2, NB3, NB5 đã chạy và kết quả được sao lưu. Đề nghị xét +3 B2 dựa trên dataset miền riêng, mô tả khử nhiễm, review AI và bằng chứng thực nghiệm; FAILED của model không tự động phủ định bonus. Đây là 240 biến thể từ 60 tình huống train, không phải 240 tình huống độc lập; tiêu chí chất lượng dữ liệu tổng hợp và điểm cuối do giảng viên xét.

Review ghi nhận các ca biên được phân loại theo ngữ cảnh: biên lai học phí thuộc hoc_phi, giấy xác nhận thuộc giay_to; trùng giờ khi đăng ký môn thuộc dang_ky_mon, trùng lịch thi thuộc lich_hoc. Lời khen hỗ trợ trước đây được gán sentiment tích cực dù ticket đang nêu sự cố. Không phát hiện lỗi nhãn theo instruction. Tuy nhiên, hoc_vu-00 trong train (bảo lưu) gần nghĩa hoc_vu-10 trong eval (nghỉ học tạm thời); không tuyên bố tách hết tương đồng ngữ nghĩa. Kiểm tra split xác nhận 19 scenario có biến thể ở cả train và validation; điểm validation không chứng minh tổng quát hóa độc lập. SHA-256 corpus và ID 288 mẫu kiểm tra được lưu trong SEMANTIC_REVIEW.json.

Gói nộp Option B giữ riêng bonus/B2_UIT/data/ (gồm CUSTOM_DATASET.md, SEMANTIC_REVIEW.json, manifest, quality_checks, corpus, checksum), bonus/B2_UIT/results/ và cấu hình prompt B2. Adapter B2 được giữ ở checkout local, không đưa trọng số vào ZIP Option B. File train_seed có 240 mẫu; train split thực tế có 216 mẫu, validation có 24 mẫu. Review_status trong quality_checks ghi rõ AI semantic review complete; independent human review not recorded, không đại diện cho xác nhận thủ công của người học.



## 12. Bonus B3 — tiếp tục kiểm tra hai chế độ đánh giá

Đã tạo gói thí nghiệm riêng Lab21_B3 với 240 mẫu số học tổng hợp chứa lời giải ngắn và đáp án nguyên, 24 bài target không trùng câu train, cùng15 câu regression của lab. Hai run assistant-only và response-only sẽ dùng cùng base Qwen3.5-4B, dữ liệu, seed42, rank16, text-linear, LR1e-4 và30 step. Trước train phải chứng minh mask khác và đo baseline thinking. Thí nghiệm bật thinking khi đánh giá, dùng scorer đáp án số thay vì scorer ticket và ghi rõ thẻ mở think do template cung cấp nếu có.

Probe tokenizer trên Colab đã xác nhận assistant-only học lời giải và đáp án, response-only chỉ học đáp án. Pilot base cho target0, trace0, regression0,7911; hai output kiểm tra có tính toán đúng nhưng chưa đóng think trong512 token. Phát hiện thêm lỗi xuất JSON ghi đè trường đáp án chuẩn bởi dự đoán; đã tách answer và predicted_answer. Chuẩn bị protocol v2 trước train, tăng budget2048 và bỏ thẻ literal khỏi instruction, giữ pilot riêng; chưa có baseline v2 hoặc adapter B3 được train. Chưa có valid_trace_rate của hai run, chưa khai báo quan sát reasoning-trace collapse hoặc hoàn thành B3. Chiều tác động phải lấy từ số đo thực tế; cấu trúc trace không chứng minh suy luận nội tại. Protocol tại scripts/B3_PROTOCOL.md và sẽ được sao chép vào data của repo B3.


B3 calibration v2: câu(101+17)×3 lặp suy luận tới2048 token; câu(102+18)×4 có output đệm vision_pad và chưa trích được đáp án. Chưa train. Chuẩn bị decode v3: cắt tại EOS trước decode, sampling theo các tham số thinking tham khảo model card Qwen, presence penalty chỉ trên token đã sinh, batch1 và seed cố định theo index cho cả ba model. Giữ pilot v2; chờ calibration và baseline v3. Không xem trace0 bị truncation là bằng chứng collapse.


**Phạm vi B3 hiện tại:** Tiếp tục B3 theo yêu cầu người học, giữ thinking cho phép đo trace và thêm lượt tắt thinking để đo đáp án trực tiếp. Pilot cũ được lưu riêng; chuẩn bị calibration decode v3 trước train. Chưa có baseline cuối hoặc hai adapter B3, chưa đề nghị điểm thưởng. Trace rate bằng0 khi chủ động tắt thinking không được dùng làm bằng chứng collapse. B1 và B2 giữ nguyên.



B3 calibration v3 mới có closing think và không chạm budget ở hai câu, đáp án cuối354/480 đúng trong cả hai chế độ. Parser quá chặt gây None do model có giải thích; sửa trước freeze baseline: exact-match dòng số nguyên cuối sau trace, giữ strict format rate riêng. Cùng scorer sẽ áp dụng base và hai adapter; giữ output calibration và kết quả parser cũ. Chưa train hoặc có bảng so sánh B3 cuối.

## 13. Bonus B5 — adapter công khai trên HuggingFace Hub

Ngày 07/10/2026 đã đăng adapter chính correct lên [trongph/lab21-qwen35-triage-vi](https://huggingface.co/trongph/lab21-qwen35-triage-vi). Base là unsloth/Qwen3.5-4B, LoRA r=16, alpha=32; đây là adapter ticket CSKH của bài chính, không phải adapter UIT B2. Không train lại để thực hiện B5.

Repo chứa adapter_model.safetensors, adapter_config.json, tokenizer, chat template, model card có ví dụ sử dụng và các artefact evaluation. Model card báo cáo target 0,9700, regression 0,6111 và verdict FAILED; việc public adapter không thay đổi phán quyết thí nghiệm.

Commit đã kiểm tra: `9d6f0e554f4cf7224004a2cc3c3456d0d5b27b87`. Đã xác nhận repo public bằng API không dùng token, đủ 10 file đã upload và SHA-256 trọng số trên Hub khớp file local. Bằng chứng tại results/hub_check.json, script tái upload tại scripts/publish_b5.py; LINKS.md ghi URL GitHub và Hub. Không chạy thêm inference trên máy local vì phép đăng và kiểm tra artefact không yêu cầu tải lại base 4B; kết quả inference là phép đo Colab đã báo cáo ở trên. Đề nghị xét +2 B5 theo rubric.

## 14. Bonus B4 — quét rank có kiểm soát và bằng chứng đã đối chiếu

Đã train và đánh giá r=8, r=64, rồi chạy lại r=16 trong môi trường Colab mới, với ba thư mục riêng. ZIP B4_evidence.zip đã nhập và kiểm tra bằng scripts/verify_b4.py. runs.csv và adapter_config xác nhận text-linear, LR=1e-4, fp16 không lượng tử hóa, assistant-only, 30 step, alpha=2r và cùng 12 target_modules. Đã đối chiếu SHA-256: code, requirements, dữ liệu và baseline giống nhau giữa ba rank; dữ liệu/baseline cũng khớp core local. Có 225 train, 25 validation, 50 target và 15 regression. Bảng lấy từ verdict.json gốc từng rank, không từ số nhập lại thủ công; r16 mới dùng riêng cho sweep, không thay kết quả core. Bằng chứng tại bonus/B4/r8/, r16/, r64/ và results/b4_comparison.json.

| Rank | Alpha | Target | Regression | Format | Latency ms/mẫu | VRAM GB | Tham số train | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 16 | 0,8650 | 0,7022 | 1,0000 | 1422,7 | 8,51 | 16.232.448 | FAILED |
| 16 | 32 | 0,9700 | 0,5889 | 1,0000 | 1428,0 | 8,78 | 32.464.896 | FAILED |
| 64 | 128 | 1,0000 | 0,0667 | 1,0000 | 1385,8 | 10,47 | 129.859.584 | FAILED |

Biên độ target quan sát khi đổi rank là 1,000 - 0,865 = 0,135 (13,5 điểm phần trăm). Đối chứng vị trí khớp ngân sách ở NB4 cho correct=0,970 và attn_only=0,970, biên độ 0,000. Đổi LR từ 1e-4 xuống 1e-5 cho target từ 0,970 xuống 0,000, biên độ 0,970. Xếp hạng trên các phép đo này: LR (0,970) > rank (0,135) > vị trí (0,000). Không dùng final_loss để xếp hạng, không suy rộng rằng vị trí luôn vô tác dụng; đây là một lượt đo mỗi cấu hình, không có khoảng tin cậy.

Tăng r16 mới lên r64 chỉ cải thiện target 0,030 nhưng regression giảm khoảng 0,5222 theo số đã làm tròn. So với baseline prompt tốt, r64 tăng target 0,235 nhưng regression giảm 0,7244444, vượt xa tolerance 0,020. Cả ba rank đều FAILED. Rank là đòn bẩy cho điểm tác vụ trong các lượt đo này, nhưng rank cao không đồng nghĩa chất lượng tổng thể cao. Dataset 225 mẫu train chưa cung cấp bằng chứng rằng r64 sử dụng hết năng lực bổ sung hoặc tổng quát hóa tốt; cần dữ liệu đa dạng và eval khó hơn để kiểm tra nhận định đó. Regression là keyword recall, nên cần đọc output để giải thích ca thua; trace rate khi tắt thinking không chứng minh B3.

Giới hạn: environment_now.txt ghi pip freeze sau ba run, không phải snapshot độc lập từng run. Adapter config không pin revision base, nên không chứng minh cùng immutable commit chỉ từ tên model. Các đối chứng LR/vị trí từ phiên core cũ, vì vậy xếp hạng ảnh hưởng là so biên độ quan sát, không phải thí nghiệm factorial trong cùng phiên. ZIP bằng chứng gọn có adapter_config nhưng không chứa trọng số sweep; hai backup đầy đủ r8/r64 được giữ riêng tại backups/runs/B4/ của checkout. Gói Option B dùng link Hub cho adapter core, không chứa trọng số local. Không đổi dữ liệu hay số đo để làm đẹp kết quả. Đã có ba rank và phân tích theo yêu cầu B4; đề nghị xét +3 điểm, điểm cuối do giảng viên chấm.

## 15. Cấu trúc nộp bài theo Option B

Nộp trực tiếp bằng URL GitHub trong LINKS.md theo Option B của rubric, cùng adapter HuggingFace công khai. Ở gốc repo có submission/REPORT.md, toàn bộ results/ và LINKS.md; có thêm src/labkit/, notebooks/, tests/, scripts/, dữ liệu đóng băng và bonus/B2_UIT/, bonus/B4/ để đối chiếu. Không cần thư mục repo lồng trong submission/ hay ZIP nộp riêng. Notebook ipynb đã clear output. Trọng số local, cache, file tải dở, token và backups/ được loại khỏi Git; adapter chính truy cập qua Hub. Các file kết quả và split đóng băng được đưa vào Git để người chấm kiểm tra số liệu. B3 chưa hoàn thành, không đề nghị bonus B3.
