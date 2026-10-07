"""Build synthetic UIT-support B2 data; --colab prepares an isolated lab copy."""
import argparse
import collections
import hashlib
import json
import pathlib
import re
import shutil
import unicodedata

SCENARIOS = {
    "hoc_phi": [
        "Em đã chuyển khoản học phí nhưng trạng thái thanh toán vẫn chưa cập nhật.",
        "Em muốn xem chi tiết các khoản phải đóng trong học kỳ này.",
        "Biên lai học phí của em ghi sai nội dung chuyển khoản, em cần được kiểm tra.",
        "Em cần hỏi thủ tục đề nghị gia hạn thời gian đóng học phí.",
        "Em đã thanh toán hai lần cùng một khoản học phí và muốn kiểm tra khoản dư.",
        "Em muốn biết cách tải chứng từ xác nhận đã đóng học phí.",
        "Số tiền học phí hiển thị khác với thông báo em đã nhận.",
        "Em cần hướng dẫn cách ghi nội dung chuyển khoản học phí để nhận diện đúng.",
        "Em muốn hỏi nơi tiếp nhận yêu cầu điều chỉnh thông tin trên biên lai.",
        "Em không thấy khoản học phí mới xuất hiện trong danh sách thanh toán.",
        "Em cần xác nhận khoản tiền đã nộp được phân bổ cho học kỳ nào.",
        "Em muốn hỏi thủ tục kiểm tra học phí sau khi thay đổi số tín chỉ đăng ký.",
    ],
    "dang_ky_mon": [
        "Em chọn lớp môn học nhưng hệ thống báo lớp đã hết chỗ.",
        "Em muốn đổi sang lớp khác của cùng môn trong đợt đăng ký hiện tại.",
        "Em đã hoàn thành môn trước nhưng vẫn bị chặn bởi điều kiện tiên quyết.",
        "Môn em cần đăng ký không xuất hiện trong danh sách lựa chọn.",
        "Em đã lưu đăng ký nhưng lớp vừa chọn không xuất hiện trong kết quả.",
        "Em muốn hỏi thủ tục đăng ký học lại một môn chưa đạt.",
        "Em bị báo vượt số tín chỉ khi thêm môn vào danh sách đăng ký.",
        "Em muốn rút một môn đã đăng ký và cần biết nơi tiếp nhận yêu cầu.",
        "Em muốn đăng ký hai lớp nhưng hệ thống báo trùng giờ và không cho lưu.",
        "Em chọn nhầm nhóm thực hành và cần điều chỉnh đăng ký nhóm.",
        "Em muốn hỏi cách đăng ký môn tương đương thay cho môn không mở lớp.",
        "Em cần biết quy trình xin đăng ký bổ sung sau khi cổng đăng ký đóng.",
    ],
    "lich_hoc": [
        "Thời khóa biểu của em hiển thị hai phòng khác nhau cho cùng một buổi học.",
        "Em cần xác nhận buổi học tuần tới được chuyển sang phòng nào.",
        "Em muốn hỏi lịch học bù của buổi giảng viên đã thông báo nghỉ.",
        "Ngày thi trên lịch cá nhân khác với thông báo lịch thi em nhận được.",
        "Em cần biết giờ bắt đầu của buổi thực hành đã được dời lịch.",
        "Em muốn hỏi cách xem thời khóa biểu theo tuần.",
        "Lớp đã đăng ký không xuất hiện trên lịch học cá nhân của em.",
        "Em cần xác nhận một buổi học được tổ chức trực tuyến hay tại lớp.",
        "Em thấy lịch thi hai môn trùng giờ và muốn được hướng dẫn xử lý.",
        "Thông báo đổi lịch học không ghi rõ ngày áp dụng, em cần xác nhận.",
        "Em muốn hỏi lịch học có thay đổi trong tuần có ngày nghỉ lễ không.",
        "Em cần tìm lịch thi lại và địa điểm thi của một môn đã học.",
    ],
    "tai_khoan": [
        "Em quên mật khẩu tài khoản sinh viên và không đăng nhập được.",
        "Email sinh viên của em không nhận được thư khôi phục mật khẩu.",
        "Em bị khóa tài khoản sau nhiều lần nhập sai mật khẩu.",
        "Em đăng nhập cổng học tập thì liên tục bị chuyển về màn hình đăng nhập.",
        "Em cần đổi số điện thoại dùng để xác thực tài khoản.",
        "Em không nhận được mã xác thực khi đăng nhập email sinh viên.",
        "Em có tài khoản sinh viên nhưng chưa được cấp quyền vào hệ thống học tập.",
        "Tên hiển thị trong hồ sơ tài khoản của em bị sai.",
        "Em muốn đăng xuất tài khoản khỏi một thiết bị đã mất.",
        "Em cần hỏi quy trình báo cáo tài khoản bị người khác sử dụng.",
        "Em không thể kích hoạt tài khoản sinh viên từ thư mời đã nhận.",
        "Em cần thay đổi địa chỉ email dự phòng cho tài khoản sinh viên.",
    ],
    "giay_to": [
        "Em muốn xin giấy xác nhận đang là sinh viên để bổ sung hồ sơ.",
        "Em cần hỏi cách đề nghị cấp bảng điểm có xác nhận.",
        "Em muốn xin bản sao giấy chứng nhận đã hoàn thành chương trình.",
        "Giấy xác nhận của em ghi sai họ tên và em cần đề nghị sửa.",
        "Em muốn hỏi nơi nhận giấy tờ đã đăng ký cấp trước đó.",
        "Em cần xin xác nhận sinh viên để làm thủ tục đi xe buýt.",
        "Em cần biết hồ sơ phải nộp khi xin cấp lại thẻ sinh viên bị mất.",
        "Em muốn hỏi thủ tục xác nhận bản dịch bảng điểm.",
        "Em cần xin giấy giới thiệu phục vụ đợt thực tập.",
        "Em đã gửi yêu cầu giấy xác nhận nhưng không thấy trạng thái xử lý.",
        "Em muốn hỏi có thể ủy quyền người khác nhận giấy tờ hay không.",
        "Em cần kiểm tra thông tin ngày sinh trên giấy tờ được cấp.",
    ],
    "hoc_vu": [
        "Em muốn hỏi thủ tục đề nghị bảo lưu kết quả học tập.",
        "Em cần hướng dẫn cách đăng ký học trở lại sau thời gian bảo lưu.",
        "Em muốn biết nơi nộp yêu cầu xem lại điểm một học phần.",
        "Điểm tổng kết của em khác với điểm giảng viên đã thông báo.",
        "Em cần hỏi quy trình công nhận tín chỉ đã học ở chương trình khác.",
        "Em muốn được hướng dẫn kiểm tra các điều kiện xét tốt nghiệp.",
        "Em nhận thông báo cảnh báo học vụ và cần được giải thích cách xử lý.",
        "Em muốn hỏi thủ tục đề nghị chuyển chương trình đào tạo.",
        "Em cần tìm người phụ trách tư vấn kế hoạch học tập cá nhân.",
        "Em muốn hỏi cách đăng ký xét tốt nghiệp trong đợt hiện tại.",
        "Em muốn hỏi thủ tục xin nghỉ học tạm thời vì lý do cá nhân.",
        "Em cần kiểm tra việc cập nhật kết quả rèn luyện trong hồ sơ học vụ.",
    ],
}
SERVICES = dict(zip(SCENARIOS, ["học phí", "đăng ký môn", "lịch học", "tài khoản sinh viên", "giấy tờ sinh viên", "học vụ"]))
URGENCY = {
    "cao": ["Em cần xử lý trong hôm nay vì hạn chót là tối nay.", "Em cần hỗ trợ trước sáng mai vì hạn nộp hồ sơ sắp hết."],
    "trung_binh": ["Em mong nhận phản hồi trong vài ngày tới.", "Em muốn được hướng dẫn sớm trong tuần này."],
    "thap": ["Em hỏi để chuẩn bị trước, hiện chưa có hạn chót.", "Em không vội, khi thuận tiện xin hướng dẫn giúp em."],
}
SENTIMENT = {
    "tieu_cuc": ["Em rất thất vọng về trải nghiệm hỗ trợ vừa rồi.", "Em khá bực vì vấn đề này chưa được giải quyết."],
    "trung_tinh": ["Nhờ bộ phận phụ trách kiểm tra giúp em.", "Em xin cung cấp thông tin để được hướng dẫn."],
    "tich_cuc": ["Em cảm ơn vì các lần hỗ trợ trước rất hữu ích.", "Em rất hài lòng với sự hỗ trợ trước đây và mong được giúp tiếp."],
}
INSTRUCTION = """Phân loại yêu cầu hỗ trợ sinh viên UIT. Chỉ trả về JSON có đúng 4 khóa: intent, urgency, product, sentiment.
intent thuộc hoc_phi | dang_ky_mon | lich_hoc | tai_khoan | giay_to | hoc_vu.
urgency thuộc cao | trung_binh | thap. cao khi cần trong hôm nay/trước sáng mai; trung_binh khi cần trong vài ngày/tuần này; thap khi chuẩn bị trước/không vội.
sentiment thuộc tieu_cuc | trung_tinh | tich_cuc, theo cảm xúc được người gửi thể hiện.
product là tên dịch vụ sau 'Dịch vụ:' trong ticket, sao chép nguyên văn và bỏ dấu chấm kết thúc. Đây là trường tương thích pipeline, không phải sản phẩm mua sắm.
Không trả lời hay suy đoán quy định của trường."""
OPTIMIZED = INSTRUCTION + """
Hướng dẫn intent: học phí/thanh toán -> hoc_phi; thêm/đổi/rút/đăng ký học phần -> dang_ky_mon; giờ/phòng/lịch học hoặc lịch thi -> lich_hoc; đăng nhập/mật khẩu/quyền tài khoản -> tai_khoan; cấp/sửa/nhận giấy xác nhận, bảng điểm, thẻ, giấy giới thiệu -> giay_to; bảo lưu, điểm học tập, tốt nghiệp, tín chỉ, chương trình, tư vấn học tập -> hoc_vu.
Ví dụ: 'Dịch vụ: học phí. Em muốn kiểm tra khoản thanh toán. Em không vội. Nhờ kiểm tra.' -> {"intent":"hoc_phi","urgency":"thap","product":"học phí","sentiment":"trung_tinh"}.
Chỉ xuất object JSON, không markdown hay giải thích."""

def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

def build(out):
    out.mkdir(parents=True, exist_ok=True)
    train, evaluation, manifest = [], [], []
    for ci, (intent, scenarios) in enumerate(SCENARIOS.items()):
        for si, scenario in enumerate(scenarios):
            split = "train_seed" if si < 10 else "eval_target"
            for variant in range(4):
                urgency = list(URGENCY)[(ci + si + variant) % 3]
                sentiment = list(SENTIMENT)[(ci + 2*si + variant) % 3]
                opener = ["Chào bộ phận hỗ trợ sinh viên UIT.", "Em xin gửi yêu cầu hỗ trợ.", "Nhờ anh/chị phụ trách hỗ trợ em.", "Em có vấn đề cần được hướng dẫn."][variant]
                ticket = f"{opener} Dịch vụ: {SERVICES[intent]}. {scenario} {URGENCY[urgency][variant%2]} {SENTIMENT[sentiment][variant%2]}"
                label = {"intent": intent, "urgency": urgency, "product": SERVICES[intent], "sentiment": sentiment}
                row = {"instruction": INSTRUCTION, "input": ticket, "output": json.dumps(label, ensure_ascii=False), "label": label}
                (train if split == "train_seed" else evaluation).append(row)
                manifest.append({"id": f"{intent}-{si:02}-{variant}", "scenario_id": f"{intent}-{si:02}", "split": split, "label": label})
    assert len(train) == 240 and len(evaluation) == 48
    assert len({normalized(r['input']) for r in train + evaluation}) == 288
    a = {r['scenario_id'] for r in manifest if r['split'] == 'train_seed'}
    b = {r['scenario_id'] for r in manifest if r['split'] == 'eval_target'}
    assert not a & b
    for row in train + evaluation:
        assert json.loads(row['output']) == row['label']
        assert row['label']['product'] in row['input']
    for name, rows in [('train_seed', train), ('eval_target', evaluation)]:
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    stats = {'train_seed':len(train), 'eval_target':len(evaluation), 'train_scenarios':len(a), 'eval_scenarios':len(b),
             'exact_duplicates':0, 'shared_scenarios':0,
             'train_intents':dict(collections.Counter(r['label']['intent'] for r in train)),
             'eval_intents':dict(collections.Counter(r['label']['intent'] for r in evaluation)),
             'review_status':'Rule validation complete; learner semantic review pending.',
             'limitations':'Synthetic templates and explicit service/urgency/sentiment cues; not real UIT requests.'}
    (out/'quality_checks.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colab', action='store_true')
    args = parser.parse_args()
    if args.colab:
        source = pathlib.Path('/content/Day21-Track3-Finetuning-Lab')
        dest = pathlib.Path('/content/Lab21_UIT_B2')
        assert (source/'src/labkit/config.py').exists()
        if dest.exists():
            raise RuntimeError('Lab21_UIT_B2 already exists; inspect it instead of overwriting results.')
        shutil.copytree(source, dest, ignore=shutil.ignore_patterns('.git', '.venv', '.env', '__pycache__', 'adapters', 'results', 'submission', 'split', 'Day21-Track3-Finetuning-Lab', '*.crdownload'))
        (dest/'adapters').mkdir()
        (dest/'results').mkdir()
        (dest/'submission').mkdir()
        generated = dest/'data'
        stats = build(generated)
        config = dest/'src/labkit/config.py'
        text = config.read_text(encoding='utf-8')
        text, count = re.subn(r'OPTIMIZED_PROMPT = """.*?"""', lambda _: 'OPTIMIZED_PROMPT = '+repr(OPTIMIZED), text, count=1, flags=re.S)
        assert count == 1
        config.write_text(text, encoding='utf-8')
        # Keep the general-capability and secret datasets unchanged, update checksums for declared corpus.
        checksums = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in generated.glob('*.jsonl') if p.name != 'manifest.jsonl'}
        (generated/'checksums.json').write_text(json.dumps(checksums, indent=2), encoding='utf-8')
        doc = pathlib.Path(__file__).with_name('UIT_CUSTOM_DATASET.md')
        assert doc.exists(), 'Upload UIT_CUSTOM_DATASET.md alongside this script.'
        shutil.copy2(doc, generated/'CUSTOM_DATASET.md')
        print('B2 isolated repo:', dest)
        print('Run %cd /content/Lab21_UIT_B2 before B2 notebooks.')
    else:
        repo = pathlib.Path(__file__).resolve().parents[1]
        generated = repo/'data/uit_b2'
        stats = build(generated)
        shutil.copy2(pathlib.Path(__file__).with_name('UIT_CUSTOM_DATASET.md'), generated/'CUSTOM_DATASET.md')
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
