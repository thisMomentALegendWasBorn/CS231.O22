## **CS231.O22 - NHẬP MÔN THỊ GIÁC MÁY TÍNH**
* **MSSV**: 22521266
* **FullName**: Trần Giang Sử
* **Report project** 

* **export.py** : Trích xuất thông tin từ dataset : tên ảnh/loại tế bào/4 tọa độ của tế bào máu sau đó chuyển thành 1 file .csv
* **get_annotate** : Chuyển từ file .csv sang thành file .txt để training
* **train_frcnn.py** : File để thực hiện training (cần có file weights của vgg16 nếu training lần đầu (có thể tìm trên mạng) hoặc file trainedweights đã huấn luyện mô hình từ trước
* **test_frcnn.py** : File dùng để thực hiện dự đoán (phân loại tế bào máu + xác định bounding box) của các tế bào máu có trong input
  
