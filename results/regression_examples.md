# Ví dụ regression — phép đo bổ sung

Đây là câu hỏi phổ thông, không phải ticket target. Điểm là keyword recall; cần đọc câu trả lời để phân biệt lỗi nội dung với khác cách diễn đạt.

Base: 0.7911; fine-tune: 0.6111; số ca thua: 5.

## Câu 2 — THUA

1 km bằng bao nhiêu mét?

Từ khóa chấm: 1000

Baseline (1.0000):

```text
Trong hệ đo lường quốc tế, tiền tố "kilo" (ký hiệu là **km**) có nghĩa là **1000**. Do đó, 1 kilômét tương đương với 1000 mét.

Kết quả: **1 km = 1000 m**.
```

Fine-tune (0.0000):

```text
{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}
```

## Câu 9 — THUA

Một năm có bao nhiêu tháng?

Từ khóa chấm: 12

Baseline (1.0000):

```text
Một năm bình thường có **12 tháng**.

Cụ thể, các tháng trong năm là:
1. Tháng 1
2. Tháng 2
3. Tháng 3
4. Tháng 4
5. Tháng 5
6. Tháng 6
7. Tháng 7
8. Tháng 8
9. Tháng 9
10. Tháng 10
11. Tháng 11
12. Tháng
```

Fine-tune (0.0000):

```text
{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}
```

## Câu 13 — THUA

Thành phố Hồ Chí Minh trước đây có tên là gì?

Từ khóa chấm: Sài Gòn

Baseline (1.0000):

```text
Thành phố Hồ Chí Minh trước đây có tên là **Saigon** (viết là Sài Gòn trong tiếng Việt).

Thành phố này được đặt tên theo tên của một dòng sông lớn chảy qua khu vực, tên gọi này đã được sử dụng rộng rãi từ thế kỷ 17 cho đến năm 1976. Vào ngày 2 tháng 7 năm 1976, theo quyết định của Quốc hội nước Cộng hòa Xã hội Chủ nghĩa Việt Nam
```

Fine-tune (0.0000):

```text
{"intent": "hoi_thong_tin", "confidence": 0.95, "urgency": "thap", "intent_text": "Người dùng đang hỏi về tên cũ của thành phố Hồ Chí Minh.", "urgency_text": "Câu hỏi này không cấp bách.", "sentiment": "trung_tinh", "sentiment_text": "Câu hỏi mang tính chất tìm hiểu kiến thức chung, không có cảm xúc tiêu cực hay tích cực
```

## Câu 14 — THUA

Giải thích ngắn gọn quang hợp là gì.

Từ khóa chấm: ánh sáng, cây

Baseline (1.0000):

```text
**Quang hợp** là quá trình sinh học quan trọng mà thực vật, tảo và một số vi khuẩn sử dụng năng lượng ánh sáng mặt trời để chuyển hóa khí carbon dioxide ($CO_2$) và nước ($H_2O$) thành **đường glucose** (nguồn thức ăn) và giải phóng khí **oxy ($O_2$)**.

Quá trình này diễn ra chủ yếu ở lá cây nhờ sắc tố **diệp lục**.
```

Fine-tune (0.5000):

```text
{"intent": "definition", "response": "Quang hợp là quá trình thực vật, tảo và một số vi khuẩn sử dụng năng lượng ánh sáng mặt trời để chuyển hóa khí cacbon dioxide (CO₂) và nước (H₂O) thành chất hữu cơ (như đường glucose) và giải phóng khí oxy (O₂). Đây là quá trình sống còn, cung cấp nguồn năng lượng chính cho hầu hết các sinh vật trên Trái Đất."}
```

## Câu 11 — THUA

Kể tên một loại trái cây nhiệt đới.

Từ khóa chấm: xoài, chuối, dứa, sầu riêng, mít

Baseline (0.2000):

```text
Một loại trái cây nhiệt đới phổ biến và nổi tiếng là **dâu tây** (dù thực ra dâu tây không phải là trái cây nhiệt đới điển hình, tôi có thể hiểu nhầm). Hãy để tôi đưa ra một ví dụ chính xác hơn:

**Dâu tây** không phải là trái cây nhiệt đới. Một loại trái cây nhiệt đới điển hình là **dâu tây**... không, tôi xin lỗi, tôi đang lặp lại.

Đáp án
```

Fine-tune (0.0000):

```text
{"intent": "hoi_thong_tin", "confidence": 0.95, "intent_text": "Người dùng đang yêu cầu liệt kê một loại trái cây nhiệt đới.", "urgency": "thap", "sentiment": "trung_tinh"}
```
