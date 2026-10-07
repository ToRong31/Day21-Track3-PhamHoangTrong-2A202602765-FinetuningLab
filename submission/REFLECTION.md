# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Model đạt điểm tác vụ cao hơn chưa chắc tốt hơn để sử dụng. Trong B4, r64 đạt target 1,0000 nhưng regression chỉ còn 0,0667; r8 có target thấp hơn, 0,8650, nhưng regression đạt 0,7022. Tôi thấy rõ việc chỉ nhìn điểm phân loại ticket sẽ bỏ qua sự suy giảm ở tác vụ phổ thông. Target ở đây là độ đúng trung bình bốn trường JSON, không phải bằng chứng model giải quyết được mọi ticket thực tế.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Phần khiến tôi phải thao tác và sửa lại nhiều là chuẩn bị môi trường Colab, chuyển artefact và giữ các lượt chạy nhất quán. Khi chuyển sang tài khoản Colab khác, tôi không có Drive cũ nên phải upload gói repo để khôi phục dữ liệu và baseline. Tôi còn chạy lại r16 trong môi trường mới để so với r8 và r64. Tôi không đo tổng thời gian từng việc, nhưng việc chuẩn bị và kiểm tra chiếm nhiều công sức hơn tôi nghĩ lúc đầu; không chỉ bấm train là xong.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi không còn xem tăng rank hoặc giảm loss là cách bảo đảm model tốt hơn. Rank cao có thể tăng điểm trên corpus hẹp nhưng làm regression giảm mạnh. Tôi cũng không mặc định fine-tune là bước đầu tiên: baseline prompt tối ưu đã đạt target 0,7650 ở bài chính và 0,9583 ở B2. Cần đo lợi ích so với prompt tốt, cùng khả năng giữ năng lực phổ thông, trước khi quyết định dùng adapter.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI để đọc code, giải thích mask/LoRA, hướng dẫn chạy Colab, tổng hợp kết quả, chuẩn bị dataset B2, viết report và sắp xếp repo. AI từng hướng dẫn dựa trên đường dẫn Drive cũ trong khi tôi dùng tài khoản khác; cell kiểm tra ZIP cũng báo lỗi vì file chưa có ở đường dẫn giả định. Một hướng dẫn đổi rank đưa đoạn SPECS riêng khiến tôi chạy thiếu phần import và gặp NameError. Tôi phải gửi lỗi để nhận cell đầy đủ. AI cũng chuẩn bị ZIP trước khi tôi nói rõ nộp bằng repo. Những việc này cho thấy tôi cần kiểm tra giả định và chạy từng bước, không chỉ làm theo hướng dẫn. Review B2 do AI thực hiện cũng không thay thế review độc lập của con người.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ xác định tác vụ, tiêu chí thành công và những năng lực bắt buộc phải giữ, rồi xây tập đánh giá từ tình huống thực tế được phép sử dụng. Tôi sẽ tách train/eval theo nguồn hoặc nhóm tình huống để hạn chế rò rỉ, sau đó đo baseline với prompt tốt trước khi train. Chỉ khi baseline chưa đáp ứng yêu cầu, tôi mới thử fine-tune và kiểm tra cả target, regression, format và latency. Tôi sẽ không đưa r64 hiện tại vào phục vụ chỉ vì target đạt 1,0000.
