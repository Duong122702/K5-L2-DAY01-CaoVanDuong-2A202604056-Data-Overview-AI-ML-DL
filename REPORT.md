# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** vì tôi thấy mức độ xếp hạng(rank) và score(xác suất) của từng record, thứ hai tôi không thấy tọa đổ của từng vật thể.
3. Danh sách tên lớp (class list) là các node lá được ánh xạ từ cây texonomy do con người thiết kế.
4. Vì ưcả `class_id`, `class_name` và `taxonomy_name`phục vụ cho các mục đích khác nhau. `class_id` phục vụ cho tính toán truy vấn của máy tính, `class_name` phục vụ cho việc con người đọc, còn `taxonomy_name` dùng để định vị ngữ nghĩa logic nhằm tránh trùng lặp nghĩa.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ bộ quy tắc để lựa chọn vật. Đầu tiên phải ưu tiên vật có diện tích lớn nhất, nằm ở tiền cảnh hoặc được lấy nét rõ nhất. Thứ hai khi các vật tương đương diện tích ưu tiên gán theo thứ tự quan trọng của bài toán(ưu tiên nghiệp vụ). Thứ ba nếu ảnh quá phân tán hoặc không xác định được thì Skip.
6. Không được lấy `score` của model làm ground truth bởi vì score chỉ phản ánh mực độ tự tin của model khi phân loại chứ đó chưa chắc đã là sự thật khách quan của thế giới thực.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=oven ; `score`=0.686791 ; `bbox_xyxy`=[0.11,187.54,195.96,292.81] ; `bbox_width`=195.85 ; `bbox_height`=105.26
2. Giải thích box bằng lời thường:vật ở góc trên-trái bắt đầu từ (0.11, 187,54) và góc dưới-phải (195.96, 292.81), rộng 195.85px và cao 105.26px
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Ngưỡng 0.6 có 6 prediction, ngưỡng 0.2 có 17 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm các việc như: xóa hộp bao rác(bị đánh nhầm), nắn chỉnh tọa độ, xóa box trùng lặp, kiểm tra lại classname.
5. **Một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này) là hộp phải tiếp xúc chính xác với các pixel ngoài cùng của vật, không cắt cụt, không thừa nền.
6. Trên ảnh `kitchen`, vật có classname:bowl, class_id:45 bị che khuất. Tôi sẽ viết thành quy tắc trong guideline vì trường hợp này có thể lặp lại nhiều lần.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-010 ; `class_name`=bowl ; `score`=0.372406 ; số điểm polygon=`polygon_point_count`=12 ; 3–5 điểm đầu của `polygon_xy`=
[155.0,170.0],[155.0,177.0],[156.0,178.0],[156.0,179.0],[161.0,184.0],
2. So với box, polygon cho thêm thông tin biên dạng lỗi lòm, diện tích và chu vi thực, hướng xoay và tư thế, cấu trúc rỗng và khoét lỗ bên trong về hình dạng vật thể.
3. `instance_id` dùng để phân biệt một cá thể riêng trong từng class? Nó **không** phải class ID và **không** phải tracking ID vì nó phản ánh một ca thể cụ thể không phải toàn bộ loại vật thể và trong không gian tĩnh.
4. **Một quy tắc vẽ biên mask** để người gán nhãn làm theo là loại bỏ bóng đổ, chỉ lấy bề mặt vật thể thực tế.
5. Trên ảnh `kitchen`, vật có classname:bowl, class_id:45 bị che khuất bị che. Tôi dùng guideline vì thực tế cũng sẽ gặp nhiều trường hợp bị khuất như vậy.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| Phân loại ảnh | Một (hoặc nhiều) nhãn cho cả bức ảnh, ví dụ `"label": "cat"`. Đơn vị nhãn: ảnh. |Ảnh có nhiều phương tiện nên model gặp khó khăn khi xác định các loại xe tương đối giống nhau như car, minibus | Xem toàn ảnh, chọn lớp đúng theo hướng dẫn gán nhãn; gắn cờ "không chắc" hoặc "ảnh lỗi" nếu cần. | Đối chiếu nhãn với nội dung ảnh; kiểm tra nhãn có thuộc danh sách lớp hợp lệ, có thiếu nhãn phụ không, lớp nào hay bị nhầm. |
| Phát hiện vật thể | Bounding box cho mỗi đối tượng kèm lớp, ví dụ `"bbox": [x, y, w, h]`, `"category_id": 3`. Đơn vị nhãn: từng đối tượng. | Các box nhỏ như bus [570.72, 71.2, 608.67, 111.73] (0,593, kích thước 38×41) gần như vuông, giống hệt cỡ của car [582.27, 123.38, 623.06, 158.96] (0,606) ngay bên dưới. Annotator khó phân biệt bus và car ở vật nhỏ, xa. | Vẽ một box cho từng đối tượng, chọn lớp, áp dụng quy tắc cho vật bị che và vật nhỏ. | Vẽ chồng box lên ảnh để tìm box lệch/thiếu/thừa/trùng; kiểm tra nhãn lớp, tọa độ có nằm trong ảnh và đúng định dạng. |
| Instance segmentation | Mask hoặc polygon riêng cho từng thể hiện kèm lớp, ví dụ `"segmentation": [[x1, y1, x2, y2, ...]]`. Đơn vị nhãn: từng vật (mức pixel). | traffic-015 (car, 0,609): có cạnh dài (338,246)→(372,205) khoảng 53 px, rồi một “cái gai” (375,236)→(331,201)→(375,236), nên polygon tự cắt/bow-tie. traffic-013 (bus): đỉnh (372,217) bị lặp lại, tạo vòng nhỏ ở góc dưới phải | Vẽ polygon hoặc dùng công cụ hỗ trợ (SAM, brush) để tạo mask chính xác, tách riêng từng thể hiện. | Chồng mask lên ảnh để kiểm tra viền; tìm mask chồng lấn, thể hiện bị gộp/tách sai, polygon không hợp lệ, lớp bị gán nhầm. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? Không bao giờ coi score trở thành ground truth.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ báo cho mentor hoặc leader.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
