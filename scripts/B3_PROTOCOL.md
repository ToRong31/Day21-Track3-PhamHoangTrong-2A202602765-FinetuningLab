# B3 — hai loss mask trên base có thinking

Thí nghiệm riêng, không thay adapter hay kết quả bài chính/B2. Base giữ unsloth/Qwen3.5-4B theo tier T4. Setup/probe phải xác nhận template thinking và hai mask thực sự khác trước train.

Dữ liệu tổng hợp: 240 bài số học, ba dạng, mỗi đáp án chứa lời giải phép tính ngắn trong think và số nguyên cuối. Phép tính được tạo bằng số nguyên Python, không dùng nhãn model đoán. Eval 24 bài dùng toán hạng ngoài khoảng train, không trùng câu; có cùng dạng bài nên không chứng minh tổng quát hóa reasoning ngoài dạng này. Không dùng eval vào train.

Đây là hướng dẫn giải ngắn do bộ sinh tạo, không phải trace nội bộ của một LLM. Dataset có nhiều template lặp; cần review. Nghiên cứu tác động loss mask trên bài số học không tự tương đương phép tái lập mọi kết luận của deck.

assistant-only tính loss cả lời giải và đáp án; response-only chỉ tính phần sau tag đóng think. Input, dataset, base, seed 42, rank16, vị trí text-linear, LR1e-4, 2epochs, batch hiệu dụng16 và ngân sách 30 step giữ bằng nhau. Số token supervised khác nhau là biến can thiệp chủ đích. Prompt train được bật thinking=True để giữ prefix nhất quán.

Đo base trước train. Eval target là exact-match số nguyên cuối, không dùng scorer ticket. Bật thinking=True cho cả base và hai adapter, max_new_tokens2048, sampling, batch1, seed42+index cho từng câu. Đánh giá toàn bộ24 bài; không dùng generation mặc định thinking=False của NB5 vì nó làm trace rate không diễn giải được. Eval regression giữ15 câu, system=None,max_new_tokens96 như bài chính.

Nếu template cung cấp thẻ mở think trong prompt, output token mới không chứa opener. Khi tính valid_trace_rate, nối opener do template cung cấp với continuation, công bố riêng raw_continuation và template_supplied_opener. Token đệm sau EOS bị loại trước decode. Chỉ coi hợp lệ khi có body không rỗng dài ít nhất10 ký tự và closing tag trong continuation. Trace rate này đo cấu trúc output, không chứng minh nội dung lời giải đúng hoặc mô tả suy luận nội tại. Output chạm2048 token có thể mất closing/final answer; phải xem raw output trước kết luận collapse.

Không dự đoán trước chiều tác động. Nếu hai rate bằng nhau hoặc base rate đã0, ghi trung thực không quan sát được collapse trong thiết lập này. Nếu target tăng nhưng trace giảm, vẫn cần đọc output và xem token budget, truncation, số học trước khẳng định nguyên nhân. Không sửa eval/ngưỡng sau khi xem kết quả. Frozen source hashes nằm trong results/data_hashes.json.

Kết quả cần lưu: mask_probe.json, baseline.json, train_assistant-only.json, train_response-only.json, score_assistant-only.json, score_response-only.json, b3_comparison.json, dataset/checksum/protocol và hai adapter nếu nộp. Chưa có kết quả GPU khi tạo gói. Không tự ghi bonus hoàn tất trước khi hai run và đánh giá hoàn thành.


## Sửa protocol trước train (v2)

Pilot v1 ghi nhận target=0, trace=0; hai output kiểm tra có phép tính đúng nhưng chưa đóng think trong512 token. Giữ pilot riêng, loại bỏ tên thẻ think khỏi task instruction để tránh model phân tích định dạng dài, tăng budget2048 cho cả base và hai adapter. Chưa có adapter được train lúc thay protocol. Phải chạy calibration hai câu rồi freeze baseline v2. Không coi pilot là baseline cuối hoặc bằng chứng mất năng lực reasoning.

Sửa lỗi lưu output: answer là đáp án chuẩn của dataset; predicted_answer là số trích từ output. Bảnv1 có va chạm tên trường khi xuất JSON, làm đáp án chuẩn hiển thị None, nhưng target được tính trước khi ghép dictionary nên không bị lỗi này trực tiếp. Giữ generated_tokens và hit_token_budget để phân biệt truncation với trace collapse.


## Decode v3 trước train

Pilot v2 vẫn có câu lặp đến2048 token. Có output đệm vision_pad sau EOS vì decode chưa cắt continuation tại EOS; sửa trước parsing. Đổi math generation sang sampling temperature1.0,top_p0.95,top_k20,repetition_penalty1.0 và presence_penalty1.5 chỉ trên token đã sinh (custom LogitsProcessor), tham khảo model card Qwen3.5-4B https://huggingface.co/Qwen/Qwen3.5-4B#best-practices. Batch1, seed42+index cho cùng câu ở cả base và hai adapter. Regression giữ greedy như bài chính. Chỉ lấy một mẫu theo seed cố định; không đủ đánh giá variance sampling. Không tăng budget vô hạn hoặc buộc đóng think để tạo trace hợp lệ. Cần calibration, baseline mới trước train; giữ pilot v2 riêng.


## Đo đáp án không thinking bổ sung

Giữ thinking=True cho phép đo trace. Đo thêm cùng24 câu với thinking=False, greedy và96 token, cùng TASK cho base và hai adapter; exact-match số nguyên được lưu target_no_thinking. Không gộp hai cấu hình decode hoặc dùng trace0 của lượt tắt thinking để kết luận collapse. Calibration v3 kiểm tra cả hai chế độ trên hai câu trước freeze baseline. Nếu thinking output vẫn chạm2048, công bố số mẫu truncation; direct accuracy có thể đo được nhưng chưa đủ bằng chứng trace collapse.


## Chấm đáp án cuối trước freeze baseline

Calibration v3 có closing think, không chạm budget và đáp án cuối đúng354/480, nhưng parser cũ yêu cầu toàn bộ phần trả lời chỉ gồm số nên trả None. Tách accuracy khỏi format: bỏ trace và token kết thúc, yêu cầu dòng không rỗng cuối cùng là số nguyên đứng riêng, exact-match với đáp án chuẩn. Không lấy số trong trace hoặc biểu thức khi thiếu dòng đáp án. Ghi strict_answer_format_rate riêng nếu toàn bộ phần sau trace chỉ gồm số. Cùng parser cho base và hai adapter. Rescore calibration đã lưu, giữ bản parser cũ, không chọn lại output thuận lợi. Đây là chỉnh protocol trước train và baseline cuối, không đổi scorer sau khi thấy kết quả adapter.
