# MScFE QPT Cheatsheet Notebook

`QPT_MScFE_Cheatsheet.ipynb` là notebook ôn tập QPT, gồm các công thức và ví dụ minh họa bằng Python cho đại số, xác suất, thống kê, hồi quy, giải tích và cấu trúc dữ liệu. Nội dung được trình bày bằng tiếng Anh và tiếng Việt.

## Notebook `.ipynb` là gì?

`.ipynb` là định dạng notebook của Jupyter. Một notebook gồm các ô (cell) chứa văn bản Markdown, công thức, code và kết quả chạy code. Có thể mở cùng một file bằng Jupyter Notebook, JupyterLab, VS Code hoặc các dịch vụ notebook trên web như Google Colab.

Code trong notebook chạy bằng một môi trường Python (kernel). Vì vậy, môi trường cần có các thư viện mà notebook sử dụng. Kết quả đã lưu trong file có thể hiển thị khi mở notebook, nhưng hãy chạy lại các ô để kiểm tra kết quả trong môi trường hiện tại.

## Chạy trên máy local

Mở Terminal tại thư mục chứa notebook. Cài Jupyter Notebook và các thư viện cần thiết:

```bash
python -m pip install notebook numpy scipy sympy
```

Sau đó khởi chạy giao diện Jupyter:

```bash
jupyter notebook
```

Trình duyệt sẽ mở trang Jupyter. Chọn `QPT_MScFE_Cheatsheet.ipynb` để mở notebook. Chạy các ô từ trên xuống bằng nút **Run** hoặc lần lượt nhấn `Shift+Enter`. Nếu cần, chọn kernel Python thuộc đúng môi trường vừa cài các thư viện.

### Dùng JupyterLab

JupyterLab là một giao diện khác trong cùng hệ sinh thái Jupyter, có thể mở nhiều tab và terminal trong một cửa sổ. Cài và chạy bằng:

```bash
python -m pip install jupyterlab numpy scipy sympy
jupyter lab
```

### Dùng VS Code

1. Cài extension **Python** và **Jupyter** trong VS Code.
2. Mở `QPT_MScFE_Cheatsheet.ipynb`.
3. Chọn Python kernel có cài `numpy`, `scipy` và `sympy`.
4. Chạy từng ô hoặc chọn **Run All**.

## Chạy trên Google Colab

1. Mở [Google Colab](https://colab.research.google.com/).
2. Chọn **File > Upload notebook** và tải `QPT_MScFE_Cheatsheet.ipynb` lên, hoặc mở file từ Google Drive.
3. Chọn **Runtime > Run all** để chạy toàn bộ notebook.

Colab thường đã có sẵn các thư viện khoa học Python được notebook sử dụng. Nếu một thư viện chưa có, thêm một ô code ở đầu notebook và chạy:

```python
%pip install numpy scipy sympy
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thaonbt-tech/dev-tool-qpt/blob/main/QPT_MScFE_Cheatsheet.ipynb)


## Lưu ý khi đổi môi trường

- Local, VS Code và Colab có thể dùng Python hoặc phiên bản thư viện khác nhau. Nếu gặp lỗi `ModuleNotFoundError`, hãy cài thư viện vào đúng môi trường/kernel đang chạy notebook.
- Chạy các ô theo thứ tự từ trên xuống vì một số ví dụ dùng biến được tạo ở các ô trước.
- Notebook này không cần dữ liệu đầu vào bên ngoài; các ví dụ được tạo trực tiếp trong code.
- Khi chia sẻ notebook, có thể dùng **Restart Kernel and Run All** (hoặc tùy chọn tương đương) để kiểm tra toàn bộ nội dung trong một phiên chạy mới.