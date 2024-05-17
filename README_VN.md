<p align="center">
  <br>
  <img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/a32226e8-4ae0-4377-b2f5-e926580a7ed8" style="width:28%;">
</p>
<br>

<h2>
  Power Load Classification and Prediction System Based on Deep Learning Algorithms
  <br><br>
  Hệ Thống Phân Loại và Dự Đoán Tải Điện Dựa Trên Các Thuật Toán Học Sâu
</h2>

#### Ngôn ngữ của tài liệu này

[**English**](https://github.com/duyu09/Powerload-Classification-and-Prediction-System/blob/main/README.md) | [**简体中文**](https://github.com/duyu09/Powerload-Classification-and-Prediction-System/blob/main/README_SC.md) | [**Tiếng Việt**](https://github.com/duyu09/Powerload-Classification-and-Prediction-System/blob/main/README_VN.md)

_Văn bản này được dịch bởi mô hình ChatGPT-4o của OpenAI, nội dung chỉ mang tính chất tham khảo. Vui lòng lấy phiên bản tiếng Trung giản thể làm chuẩn._

### Giới thiệu Dự án

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sử dụng kinh tế số để hỗ trợ tiết kiệm năng lượng và giảm phát thải, sử dụng công nghệ cao để giảm chi phí và tăng hiệu quả là xu hướng tất yếu của mọi ngành nghề. Tuy nhiên, việc sử dụng tài nguyên điện chưa hợp lý ở nước ta hiện nay vẫn còn rất nghiêm trọng và cần được giải quyết. Theo điều tra của Trung tâm Chứng nhận Sản phẩm Tiết kiệm Năng lượng Trung Quốc, việc thiết bị gia đình ở một hộ gia đình bình thường ở các thành phố phải ở chế độ chờ trung bình mỗi ngày sẽ gây ra một lượng tiêu thụ năng lượng nhất định. 400 triệu chiếc tivi trên cả nước một năm tiêu thụ lượng điện chờ lên tới 2,92 tỷ kWh, tương đương với 1/3 sản lượng điện hàng năm của nhà máy điện hạt nhân Đại Á. Vì vậy, phương pháp giám sát điện năng để phát hiện và kiểm soát lãng phí tài nguyên điện ngày càng trở nên cần thiết, trong đó công nghệ không xâm lấn là trọng tâm chính. Hệ thống phân loại và dự đoán tải điện dựa trên công nghệ không xâm lấn của nhóm chúng tôi là minh chứng cho việc sử dụng công nghệ trí tuệ nhân tạo để giảm chi phí điện năng cho gia đình và doanh nghiệp nhỏ, thúc đẩy tiết kiệm tài nguyên điện.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dựa trên bối cảnh trên, chúng tôi đã thiết kế một hệ thống phân loại và dự đoán tải điện cho người dùng gia đình và doanh nghiệp nhỏ. Chức năng "phân loại" có thể thông báo cho người dùng về trạng thái hoạt động của các thiết bị điện trong thời gian thực; chức năng "dự đoán" có thể dự đoán mức tiêu thụ điện năng của người dùng trong thời gian ngắn trong tương lai. Cả gia đình thông thường và nhà máy doanh nghiệp đều rất cần một hệ thống thông minh và thuật toán để giám sát trạng thái hoạt động thời gian thực của các thiết bị điện, nhận diện mức tiêu thụ điện không cần thiết. Điều này không chỉ giúp tiết kiệm năng lượng mà còn giảm chi phí điện năng. Chức năng dự đoán tiêu thụ điện năng ngắn hạn còn giúp các doanh nghiệp lên kế hoạch sử dụng điện hiệu quả hơn, đặc biệt là khi áp dụng cơ chế giá điện bậc thang.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Người dùng gia đình khi đăng nhập vào hệ thống của chúng tôi có thể xem biểu đồ sóng công suất tiêu thụ điện năng thời gian thực trong nhà, tổng lượng điện tiêu thụ trong 50 phút gần đây, và các thiết bị đang hoạt động được mô hình dự đoán (như: "máy nước nóng", "máy tính", hoặc là kết hợp của nhiều thiết bị). Người dùng cũng có thể xem tình hình tiêu thụ điện năng trong nhà trong 20 phút tiếp theo do mô hình AI dự đoán (đường cong công suất dự đoán).

### Phân tích sản phẩm cạnh tranh

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Đối mặt với quy mô ngày càng lớn của hệ thống điện, các sản phẩm giám sát tải điện xâm lấn hiện vẫn không thể giải quyết vấn đề, và dần dần bị loại bỏ. Hiện tại, các sản phẩm liên quan trên thị trường chủ yếu sử dụng các thuật toán phân loại không xâm lấn truyền thống như cây quyết định, thuật toán rừng ngẫu nhiên, SVM (máy vector hỗ trợ), hoặc sử dụng các thuật toán học tăng cường để tích hợp chúng. Các thuật toán dự đoán tải chủ yếu bao gồm thuật toán phân tích hồi quy, thuật toán ARIMA và thuật toán LSTM dựa trên học sâu. Các thuật toán trên đều không đạt được độ chính xác dự đoán lý tưởng và không có tính chuyên biệt cho các kịch bản tải điện - chúng chỉ giải quyết các vấn đề dự đoán hoặc phân loại chuỗi thời gian thông thường mà không xem xét các tính chất và phương pháp xử lý đặc biệt của dữ liệu công suất điện. Dưới đây là biểu đồ đường cong công suất dự đoán dựa trên các thuật toán ARIMA, LSTM và hệ thống của chúng tôi trên cùng một tập huấn luyện và kiểm tra:

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/8d73555e-64fe-4597-bfb9-5e2b85530933" style="width: 80%;" />

### Ý tưởng thiết kế hệ thống

- **Mô hình AI:** Sử dụng cấu trúc mô hình kết hợp, chia nhiệm vụ thành hai mô-đun phân loại và dự đoán. Phân loại sử dụng mô hình mạng nơ-ron hoàn toàn kết nối sâu để xử lý, dựa trên đặc trưng của khung thời gian để thực hiện phân loại; nhiệm vụ phân giải được thực hiện thông qua nhiệm vụ phân loại; mô hình dự đoán được thiết kế dựa trên cảm hứng từ mô hình ngôn ngữ sinh và sử dụng cấu trúc `Transformer` làm cốt lõi với hai mô hình mạng nơ-ron sâu liên kết thông qua nhãn loại thiết bị điện.

- **Kiến trúc hệ thống:** Sử dụng kiến trúc hợp tác đám mây “phân phối ứng dụng, tập trung tính toán” phù hợp với nền tảng dịch vụ siêu máy tính hiện đại. Bao gồm thiết bị biên (đồng hồ điện), back-end phân tán và cơ sở dữ liệu, mô hình AI phân loại và dự đoán tại trung tâm siêu máy tính, và ứng dụng mini WeChat trên thiết bị di động. Thiết kế này có thể giải quyết vấn đề mà mô hình không thể suy luận kịp thời do yêu cầu đồng thời của người dùng quy mô lớn được đề cập ở trên.

- **Thiết kế front-end và back-end:** Chúng tôi chủ yếu sử dụng công cụ phát triển WeChat để phát triển front-end, biểu đồ sóng sử dụng thành phần biểu đồ đường do thư viện ECharts cung cấp. Front-end truy vấn dữ liệu bằng cách thăm dò tuần tự. Để nâng cao hiệu suất, chương trình back-end sử dụng C++17 để xây dựng Server, dựa trên kiến trúc đơn Reactor đa luồng. Hệ thống lõi của server bao gồm các mô-đun như ghi nhật ký, quản lý pool luồng, ghép kênh I/O, xử lý HTTP, quản lý bộ đệm và hàng đợi blocking. Thông báo yêu cầu HTTP được đọc bằng cách đọc phân tán và được phân tích hiệu quả bằng máy trạng thái hữu hạn và biểu thức chính quy.

| Lưu đồ làm việc của mô hình | Kiến trúc hệ thống tổng quan |
| ------ | ------ |
| <img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/94326052-6266-42b5-b519-80ea3f5eb622" style="width: 65%;"/> | <img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/246ece1e-53d2-4eaf-9263-dbed7752608b" style="width: 65%;"/> |

### Thiết kế chi tiết mô hình

- **Mô hình phân loại phân giải:** Về mặt kỹ thuật đặc trưng: trước tiên xử lý dữ liệu đầu vào bằng cửa sổ, sau đó nhóm các giá trị công suất. Tiếp theo, thống kê số lượng mẫu rơi vào mỗi nhóm trong mỗi cửa sổ thời gian để có được phân phối tần suất. Để có được vector đặc trưng, chúng tôi chuẩn hóa tần suất của mỗi cửa sổ thời gian bằng cách chia cho độ dài cửa sổ. Cuối cùng, sử dụng thuật toán PCA để nén ma trận đặc trưng. Về mặt thiết kế mô hình: chọn sử dụng mạng nơ-ron hoàn toàn kết nối `FNN`, đầu vào là vector đặc trưng tương ứng của cửa sổ thời gian, đầu ra là mã one-hot của loại thiết bị điện. Chúng tôi sử dụng mô hình mạng nơ-ron hoàn toàn kết nối 3 lớp `(28, 14, 7)` để thực hiện nhiệm vụ phân loại dữ liệu (7 loại thiết bị đơn); sử dụng mô hình mạng nơ-ron hoàn toàn kết nối 3 lớp `(84, 42, 21)` để thực hiện nhiệm vụ phân giải nhãn công suất (21 loại kết hợp hai thiết bị). Khi chọn mô hình, chúng tôi so sánh hiệu suất của mạng hoàn toàn kết nối và `GRU` (đơn vị cổng lặp). Sau khi điều chỉnh

 siêu tham số, hai mô hình lần lượt đạt được độ chính xác cao nhất là 91% và 90%, tức là, hiệu quả của cả hai đều tốt và không có sự khác biệt lớn. Tuy nhiên, thử nghiệm và triển khai thực tế cho thấy, `GRU` tiêu tốn nhiều năng lượng suy luận hơn do phải quan tâm đến các cửa sổ thời gian khác. Mặt khác, giả định rằng mỗi cửa sổ thời gian không liên quan đến nhau về mặt thứ tự thời gian, mô hình có thể tập trung xử lý các đặc trưng trong mỗi cửa sổ thời gian và đưa ra quyết định, và truyền dữ liệu nhãn cho lớp nhúng của mô hình dự đoán để xử lý tiếp, mô hình dự đoán sau đó sử dụng các đặc trưng thứ tự để dự đoán, mỗi mô hình làm nhiệm vụ riêng của mình. Với hai lý do này, chúng tôi chọn sử dụng mạng hoàn toàn kết nối. Dưới đây là sơ đồ cấu trúc mô hình phân loại:

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/b4d5f10e-2e82-4483-9029-08061f71b341" style="width: 70%;" />

<br><br>

- **Mô hình dự đoán công suất tải:** Đối với nhiệm vụ dự đoán chuỗi thời gian công suất điện, chúng tôi thiết kế cấu trúc mô hình như sau. Khi xử lý dữ liệu, đầu tiên chúng tôi phân khung dữ liệu, sau đó phân cửa sổ trong mỗi khung dữ liệu. Mỗi cửa sổ thời gian sẽ tính toán một vector đặc trưng, nhiều vector đặc trưng của các cửa sổ thời gian tạo thành một ma trận đặc trưng. Các vector đặc trưng này bao gồm đặc trưng chuyển tiếp có biến động lớn và đặc trưng trạng thái ổn định khó thay đổi. Đặc trưng chuyển tiếp được tạo thành từ biên độ và pha tương ứng với các tần số thu được thông qua biến đổi Fourier nhanh (do thành phần DC không chứa bất kỳ thông tin hữu ích nào, và giá trị trung bình mẫu của tín hiệu sóng công suất được phản ánh bởi đặc trưng trạng thái ổn định, nên chúng tôi loại bỏ nó); theo lý thuyết liên quan đến điện, đặc trưng trạng thái ổn định bao gồm giá trị lớn nhất, giá trị nhỏ nhất, giá trị trung bình số học, giá trị hiệu dụng năng lượng và hệ số đỉnh sóng. Ngoài ra, mô hình ánh xạ các loại thiết bị điện rời rạc thành các vector nhúng liên tục thông qua lớp Embedding, như một phần quan trọng của đặc trưng trạng thái ổn định. Lấy cảm hứng từ mô hình ngôn ngữ sinh, ma trận đặc trưng được coi là “ma trận nhúng từ” của một câu, thông qua 5~6 lớp `Transformer Block` (cuối cùng chọn 6 lớp) và xử lý trung bình chú ý sau đó, tạo ra vector dự đoán (tương đương với vector nhúng từ của token được sinh bởi mô hình ngôn ngữ). Cuối cùng, thông qua FFT ngược để chuyển đổi đặc trưng chuyển tiếp trong vector dự đoán (vector miền tần số) thành chuỗi thời gian, hoàn thành dự đoán chuỗi công suất. Dưới đây là sơ đồ mô hình dự đoán giá trị công suất:

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/acbe5aed-2351-4396-9afe-27cabb55fbf0" style="width: 70%;" />

### Hiệu quả của mô hình trên tập dữ liệu thực nghiệm

Tập dữ liệu được sử dụng được thu thập từ các thiết bị điện thực tế, với lượng dữ liệu là 2 triệu phút, độ phân giải thời gian là 1 phút, nhiệm vụ dự đoán trong hình là dự đoán 20 phút trong tương lai dựa trên 100 phút dữ liệu đã cho. Dữ liệu được hiển thị trong hình đã được chuẩn hóa. Tập dữ liệu này liên quan đến bí mật doanh nghiệp, không được mở nguồn.

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/91d36ba6-9291-4e11-a3ad-4159cb515afa" style="width: 55%;">

### Triển khai và sử dụng

- **Khách hàng:** Dự án sử dụng ứng dụng nhỏ WeChat làm giao diện phía khách hàng. Chỉ cần mở thư mục `WeChat_client` bằng công cụ phát triển WeChat, sau đó phát hành hoặc sử dụng công cụ của bên thứ ba để đóng gói.
- **Backend phân tán (dịch vụ khách hàng):** Dự án sử dụng tiêu chuẩn `C++ 17` để phát triển mô-đun dịch vụ khách hàng backend, chương trình này được phát triển cho hệ thống `Linux`, không đa nền tảng. Sử dụng CMake để xây dựng tệp thực thi trên hệ thống Linux. Vào thư mục CPP_backend, sau đó thực hiện các lệnh tham khảo sau:
  
```bash
mkdir build
cd build
cmake ..
make
```

Trong thư mục build sẽ tạo tệp thực thi `webserver`, khi chạy cần đảm bảo tệp cơ sở dữ liệu `SQLite` là `powerload.db` nằm cùng thư mục với `webserver`. Nếu không tiện sử dụng backend C++, có thể sử dụng backend dựa trên Python Flask thay thế, cả hai đều có chức năng tương đương, tuy nhiên hiệu suất giảm nhẹ (vị trí tệp: `/Python_backend/app.py`).

- **Backend phân tán (tính toán dữ liệu) và trung tâm tính toán mô hình AI:** Mã phần này nằm trong thư mục `Python_backend`. Cung cấp các tổ hợp mã dựa trên giao thức HTTP Flask (`app-backend.py` và `app-ai.py`) và tổ hợp mã dựa trên giao thức `Rabbit MQ` (`app-backend-mq.py` và `app-ai-mq.py`), có thể chọn một trong các tổ hợp này để chạy, đề xuất khởi động trung tâm mô hình AI trước, sau đó khởi động backend phân tán. Các tệp nhị phân mô hình phân loại và dự đoán lần lượt nằm trong thư mục classify và prediction, backend sẽ tự động gọi các mô hình này.
- **Huấn luyện mô hình:** Đảm bảo thư mục `dataset` nằm trong thư mục `Python_backend`, thư mục `dataset` chứa các tệp `CSV`, tên tệp là tên nhãn (tương ứng với tên thiết bị điện hoặc tổ hợp thiết bị điện).

Thực hiện các lệnh sau để huấn luyện mô hình phân loại (thời gian: khoảng 3 phút trên môi trường CPU) và mô hình dự đoán (thời gian: khoảng 2 giờ trên môi trường CPU).

```bash
Python ./classify/Classification_Train_v6.3.py
Python ./classify/prediction.py
```

Có thể điều chỉnh siêu tham số và sửa mã tùy theo đặc điểm của tập dữ liệu và các tình huống khác:

Trong mã huấn luyện mô hình dự đoán (`prediction.py`), có thể cần thay đổi đoạn mã sau:

```python
# Đoạn mã sau nằm trong hàm chính
batches, labels_batched, categories, category2filename_dict = data_process("./dataset")
model = train(batches, labels_batched, categories, category2filename_dict, num_heads=27, num_blocks=6, \
        lr=0.00004, epochs=13, static_vector_len=7, total_number_categories=21)
torch.save(model, "model-large.pt")

# Đoạn mã sau nằm trong hàm data_process
data = get_data(os.path.join(directory, filename + ".csv"), skip_header=1, usecol=4)
```

Trong mã huấn luyện mô hình phân loại (`Classification_Train_v6.3.py`), có thể cần thay đổi đoạn mã sau:

```python
# Đoạn mã sau nằm ở đầu tệp
dc = r'../dataset'  # Đọc tất cả các tệp CSV trong thư mục này
model_save_path = r'Classification-Model-v6.3-2.2.keras'  # Đường dẫn lưu tệp mô hình
matrix_save_path = r'Classification_feature-2.2.mat'  # Đường dẫn lưu tệp ma trận đặc trưng
dict_save_path = r"label2index_dict.pkl"  # Lưu từ điển ánh xạ giữa nhãn và mã số
pca_save_path = r"pca.pkl"  # Lưu đối tượng PCA
testRate = 0.15  # Tỷ lệ tập kiểm tra trong tập dữ liệu 0.15
frameLength = 800  # Độ dài khung 800
step = 800  # Bước tính toán đặc trưng 800
max_value = 3000000  # Giá trị công suất tối đa 3000000
eps = 75  # Số lần lặp lại huấn luyện mạng nơ-ron 75
lamb = 0.001  # Hệ số phạt chuẩn hóa L1 0.001
pca_n_components = 67  # Tỷ lệ giữ lại phương sai của PCA (mức độ giữ thông tin) 0.96
es_patience = 3  # Ngưỡng dừng sớm (phát hiện hàm mất mát) 3
power_column = -1  # Trường số công suất trong tập dữ liệu (-1 là trường cuối cùng)
```

### Ví dụ giao diện GUI khách hàng

Các ví dụ sau đây lần lượt hiển thị giao diện đăng nhập người dùng, trang cá nhân, dữ liệu thời gian thực và phân loại phân tích, cũng như dữ liệu dự đoán công suất. Tổng cộng có 4 giao diện.

<img src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/9eff7d69-38a4-4127-91c4-1282b4c49c84" style="width: 70%;" />

### Video giới thiệu dự án

<video src="https://github.com/duyu09/Powerload-Classification-and-Prediction-System/assets/92843163/0684486d-e391-4301-834f-0c5af6e6ce90" style="width: 65%;"></video>

-----

### Bản quyền

Bản quyền © 2024 _Nhóm nghiên cứu và phát triển hệ thống phân loại và dự đoán tải điện dựa trên thuật toán học sâu, Khoa Khoa học và Công nghệ Máy tính, Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông)_ .

Nhóm nghiên cứu và phát triển dự án "Hệ thống phân loại và dự đoán tải điện dựa trên thuật toán học sâu" thuộc Khoa Khoa học và Công nghệ Máy tính, Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông) bảo lưu mọi quyền.

- Danh sách thành viên nhóm nghiên cứu và phát triển: **_Các bản dịch tên riêng sau đây chỉ để tham khảo_**
  - __Đỗ Vũ__ (__Yu DU__, Khoa Khoa học và Công nghệ Máy tính, Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông), No.202103180009)
  - __Khương Xuyên__ (__Chuan JIANG__, Khoa Khoa học và Công nghệ Máy tính, Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông), No.202103180020)
  - __Lý Hiểu Ngữ__ (__Xiaoyu LI__, Khoa Khoa học và Công nghệ Máy tính, Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông), No.202103180001)
  - __Lý Khánh Lón__ (__Qinglong LI__, Khoa Khoa học và Công nghệ Máy tính, Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông), No.202103180027)
  - __Trương Nhất Văn__ (__Yiwen ZHANG__, Khoa Khoa học và Công nghệ Máy tính, Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông), No.202103180051)
- Giảng viên hướng dẫn:
  - __Lecturer Giả Thụy Tường__ (__Ruixiang JIA__, Giảng viên Khoa Khoa học và Công nghệ Máy tính, Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông), Bộ môn Kỹ thuật Phần mềm)
  - __Dr. Trần Tĩnh__ (__Jing CHEN__, Trung tâm Máy tính Sơn Đông (Trung tâm Siêu máy tính Quốc gia tại Tế Nam))

Dự án này mở mã nguồn theo giấy phép mã nguồn mở tùy chỉnh của chúng tôi. Trước khi nhận mã nguồn bằng bất kỳ cách nào, vui lòng đọc và hiểu đầy đủ nội dung của giấy phép. Tệp giấy phép là tệp [LICENSE](https://github.com/duyu09/Powerload-Classification-and-Prediction-System/blob/main/LICENSE).

- Ghi chú khác:
  - **Dự án này đã tham gia Cuộc thi Thiết kế Máy tính Sinh viên Trung Quốc lần thứ 17 năm 2024 _4C2024_ trong lộ trình thi thực hành trí tuệ nhân tạo.**
  - Logo của dự án được vẽ bởi công cụ vẽ AI của Zhipu Qingyan `CogView`, đã được sửa đổi khi sử dụng làm logo. Ý nghĩa của logo: Vòng kim loại chủ thể tượng trưng cho bảng điều khiển, hệ thống của chúng tôi là hệ thống giám sát điện không xâm lấn; các sọc bên trong vòng tròn tượng trưng cho sóng dữ liệu điện, đồng thời có thể coi là các tòa nhà cao tầng trong thành phố,

 tượng trưng cho hệ thống của chúng tôi có thể phục vụ hoạt động của hệ thống điện thành phố; nền là mặt bàn gỗ, tượng trưng cho việc hệ thống của chúng tôi hỗ trợ phát triển xanh, giảm lượng khí thải carbon.

-----

### Lời cảm ơn đặc biệt

- <b>Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông).</b> (<b> _Qilu University of Technology (Shandong Academy of Sciences)_ </b>)

<img src="https://user-images.githubusercontent.com/92843163/229986960-05c91bf5-e08e-4bd3-89a4-deee3d2dbc6d.svg" style="width:40%;border-radius:20px;">

- <b>Khoa Khoa học và Công nghệ Máy tính, Trung tâm Máy tính Sơn Đông (Trung tâm Siêu máy tính Quốc gia tại Tế Nam).</b> (<b> _Faculty of Computer Science and Technology. National Supercomputing Center in Jinan_ </b>)

<img src="https://github.com/duyu09/Intelligent-Learning-Platform/assets/92843163/3a31a937-804f-4230-9585-b437430ac950" style="width:40%;border-radius:20px;">
<br><br>

- <b>Hiệp hội Phát triển Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông).</b> (<b> _Developer Association of Qilu University of Technology (Shandong Academy of Sciences)_ </b>)

<img src="https://github.com/duyu09/Intelligent-Learning-Platform/assets/92843163/7a554ca6-49b8-4099-b214-4c4ceff7c9a3" style="width:40%;border-radius:20px;">

### Liên kết hữu ích

- Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông) https://www.qlu.edu.cn/
  
- Trung tâm Máy tính Sơn Đông (Trung tâm Siêu máy tính Quốc gia tại Tế Nam) https://www.nsccjn.cn/

- Khoa Khoa học và Công nghệ Máy tính, Đại học Công nghệ Qilu (Viện Hàn lâm Khoa học Sơn Đông) http://jsxb.scsc.cn/

- Trang cá nhân của DuYu https://www.duyu09.site/

- Tài khoản GitHub của DuYu https://github.com/duyu09/

### Thống kê lượt truy cập

<div><b>Tổng số lượt truy cập (Tất cả các dự án GitHub của Duyu09): </b><br><img src="https://profile-counter.glitch.me/duyu09/count.svg" /></div> 

<div><b>Tổng số lượt truy cập (Hệ thống phân loại và dự đoán tải điện PLDA dựa trên thuật toán học sâu): </b>
<br><img src="https://profile-counter.glitch.me/duyu09-PLDA-SYSTEM/count.svg" /></div> 