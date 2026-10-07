# B2 — Phân loại yêu cầu hỗ trợ sinh viên UIT

## Nguồn và phạm vi

Dữ liệu tổng hợp do AI assistant soạn trong phiên làm lab, sinh bằng `prepare_uit_b2.py`; không thu thập ticket thật, không chứa hồ sơ hoặc thông tin định danh sinh viên. Đây không phải dataset chính thức của UIT. Nội dung là yêu cầu hỗ trợ giả lập, không khẳng định học phí, hạn chót hoặc quy định thực tế của trường. Người học cần đọc, chỉnh và xác nhận chất lượng ngữ nghĩa trước khi nộp.

Mục tiêu: ticket -> JSON intent, urgency, product, sentiment. Sáu intent: hoc_phi, dang_ky_mon, lich_hoc, tai_khoan, giay_to, hoc_vu. Trường product được tái sử dụng thành tên dịch vụ để tương thích scorer lab; không mang nghĩa sản phẩm thương mại. Quy tắc và ví dụ được công bố trong instruction/prompt.

## Kích thước và cách tạo

72 tình huống được soạn riêng: 12 cho mỗi intent. Mỗi tình huống có 4 biến thể lời mở đầu, urgency và sentiment. 60 tình huống tạo 240 mẫu train_seed; 12 tình huống còn lại tạo 48 mẫu eval_target. Mỗi intent có 40 train_seed và 8 eval. Chỉ 60 tình huống train riêng biệt; không trình bày 240 biến thể thành 240 vấn đề độc lập. NB1 tiếp tục chia train_seed theo seed 42 thành train/validation; do chia ngẫu nhiên, validation có thể chứa biến thể cùng tình huống train và không phải thước đo tổng quát hóa chính.

Urgency: cao khi cần hôm nay/trước sáng mai; trung_binh khi cần vài ngày/tuần này; thap khi hỏi trước/không vội. Sentiment theo lời thể hiện trực tiếp của người gửi, không suy ra chỉ từ sự cố. Product sao chép tên sau Dịch vụ. Nhãn tạo bằng quy tắc minh bạch; không phải đánh giá thủ công độc lập.

## Khử nhiễm và đóng băng

Chia theo scenario_id trước khi tạo biến thể: mọi biến thể một tình huống nằm cùng train_seed hoặc eval_target. Không có scenario_id giao nhau; không có ticket trùng sau chuẩn hóa Unicode NFKC, chữ thường và khoảng trắng. manifest.json ghi split và ID để kiểm tra. quality_checks.json kiểm tra schema, output khớp label, tên dịch vụ có trong ticket, số mẫu và phân bố intent.

Các nhóm có chung template, từ chỉ dịch vụ và cách diễn đạt urgency/sentiment. Vì vậy, không tuyên bố loại bỏ mọi tương đồng ngữ nghĩa hoặc mọi dạng leakage. Eval giữ tình huống khác train nhưng vẫn dễ hơn ticket thực tế không có cue trực tiếp. Các biến thể không thay thế kiểm tra con người; người học nên đọc đủ 288 mẫu và sửa ca mơ hồ trước NB2, sau đó đóng băng corpus và prompt.

Tập eval_regression phổ thông giữ nguyên từ lab chính; không đưa câu regression vào train. Không dùng holdout_secret thương mại cũ làm bằng chứng chất lượng miền sinh viên. Checksum được tạo trước NB2 trong repo B2; sau NB2 không sửa dữ liệu để làm đẹp kết quả.

## Khác phân phối so với corpus mặc định và base

Corpus mặc định là ticket mua sắm với ý định đổi/trả/vận chuyển. Corpus này đổi sang yêu cầu học vụ sinh viên và taxonomy sáu nhóm dịch vụ. Ánh xạ nhãn do bài lab định nghĩa là tín hiệu riêng của tác vụ, đủ để thử thích nghi miền. Không biết dữ liệu pretraining của base, nên không khẳng định model chưa từng thấy chủ đề UIT hoặc hỗ trợ sinh viên. Dữ liệu tổng hợp có thể gần kiến thức phổ thông model đã biết; hiệu quả fine-tune phải so với base được prompt tốt trong NB2 mới của B2.

## Tiêu chí xác nhận chất lượng và giới hạn

Hoàn tất kiểm tra máy chưa đồng nghĩa đủ tiêu chí ≥200 mẫu chất lượng của rubric. Cần người học xác nhận intent không mơ hồ, urgency/sentiment nhất quán, câu tự nhiên và không có dữ kiện chính sách bịa. Ghi lại số mẫu sửa và cách review trong report. Cụm Dịch vụ làm extraction và routing đơn giản; nên có tập stress riêng bỏ cue hoặc thêm yêu cầu nhiều ý ở nghiên cứu tiếp theo. Điểm trên dataset này không chứng minh sẵn sàng phục vụ sinh viên thật.

## Chạy B2

Repo B2 `/content/Lab21_UIT_B2` tách khỏi repo chính, có prompt taxonomy mới và kết quả riêng. Giữ tier T4, 2 epochs, assistant-only, EVAL_LIMIT không giới hạn. Chạy NB1, NB2 trước train NB3, rồi NB5 để đo bản correct; ba đối chứng NB4 không cần lặp cho B2 trừ khi muốn thực hiện core đầy đủ trên miền mới. NB5 sẽ bỏ qua adapter đối chứng chưa có. Không gộp baseline/adapter khác miền với bài chính. Nếu chạy gatekeeper B2, WARN thiếu đối chứng phản ánh phạm vi B2, không được nhận là hoàn thành core B2.

NB1/NB2/NB3/NB5 đã chạy trên T4. Baseline tối ưu đạt target 0,9583; adapter đạt 1,0000 trên 48 target, nhưng regression giảm từ 0,7911 xuống 0,5800 nên verdict FAILED. Artefact thực nghiệm nằm tại bonus/B2_UIT/ của checkout và gói nộp.

## Review bổ sung ngày 07/10/2026

Codex AI assistant đã review ngữ nghĩa 72 tình huống và chính sách nhãn, kết hợp kiểm tra tự động toàn bộ 288 mẫu thực tế. Không phát hiện nhãn trái quy tắc đã công bố; không sửa mẫu dữ liệu, split, prompt hoặc kết quả đo. Đây là review bằng AI, không phải xác nhận người học đã đọc toàn bộ dữ liệu hay gán nhãn độc lập bởi con người. SEMANTIC_REVIEW.json lưu ID từng mẫu, SHA-256 corpus đo và các giới hạn; scripts/review_uit_b2.py tái chạy kiểm tra.

Các ca biên: biên lai học phí thuộc hoc_phi, giấy xác nhận sinh viên thuộc giay_to; trùng giờ khi đăng ký lớp thuộc dang_ky_mon, trùng lịch thi thuộc lich_hoc. Sentiment tích cực theo lời khen hỗ trợ trước đây dù người gửi đang có sự cố. Những nhãn này nhất quán khi giữ nguyên ngữ cảnh và instruction.

Tách scenario_id không bảo đảm tách hoàn toàn ngữ nghĩa: hoc_vu-00 (train, bảo lưu) gần hoc_vu-10 (eval, nghỉ học tạm thời). Có 19 scenario xuất hiện ở cả train và validation do NB1 chia ngẫu nhiên các biến thể; không dùng validation làm bằng chứng tổng quát hóa độc lập. Tên dịch vụ và cue cảm xúc/thời hạn vẫn làm bài toán dễ. Không tuyên bố dữ liệu mới hoàn toàn so với pretraining, hoặc hiệu quả trên ticket thật.

Bằng chứng B2 đã được chuẩn bị để xét bonus: 240 mẫu nguồn train (216 train thực tế), tài liệu nguồn/thu thập/khử nhiễm/phân phối, review AI và kết quả thực nghiệm. Việc chấp nhận dữ liệu tổng hợp và điểm thưởng do giảng viên xét.
