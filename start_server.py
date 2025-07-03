import os
import sys
import subprocess
import platform

def check_gunicorn():
    """Kiểm tra Gunicorn có khả dụng không"""
    try:
        import gunicorn
        return True
    except ImportError:
        return False

def start_with_gunicorn():
    """Khởi động server với Gunicorn"""
    print("Khởi động máy chủ với Gunicorn...")
    # Xác định số workers dựa vào CPU cores
    import multiprocessing
    workers = multiprocessing.cpu_count() * 2 + 1
    workers = min(workers, 8)  # Giới hạn tối đa 8 workers
    
    # Trên Windows, chúng ta cần dùng waitress thay vì gunicorn
    if platform.system() == "Windows":
        try:
            import waitress
            from app import app
            print(f"Khởi động máy chủ với Waitress trên Windows, port 51333...")
            from waitress import serve
            serve(app, host='0.0.0.0', port=51333)
        except ImportError:
            print("Waitress không được cài đặt. Vui lòng cài đặt bằng lệnh: pip install waitress")
            print("Đang chuyển sang Flask development server...")
            start_with_flask()
    else:
        # Trên Linux/Mac chúng ta có thể dùng Gunicorn
        cmd = [
            "gunicorn",
            "--workers", str(workers),
            "--timeout", "120",
            "--bind", "0.0.0.0:51333",
            "app:app"
        ]
        subprocess.call(cmd)

def start_with_flask():
    """Khởi động server với Flask development server"""
    print("Khởi động máy chủ với Flask development server...")
    from app import app
    app.run(debug=True, host='0.0.0.0', port=51333)

if __name__ == "__main__":
    print(f"Hệ điều hành: {platform.system()} {platform.release()}")
    
    if platform.system() == "Windows":
        try:
            import waitress
            print("Waitress được tìm thấy. Sử dụng Waitress trên Windows...")
            start_with_gunicorn()  # Sử dụng waitress thay vì gunicorn trên Windows
        except ImportError:
            print("Waitress không được cài đặt. Đang cài đặt Waitress...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "waitress"])
                print("Đã cài đặt Waitress thành công!")
                start_with_gunicorn()
            except Exception as e:
                print(f"Lỗi khi cài đặt Waitress: {str(e)}")
                print("Đang chuyển sang Flask development server...")
                start_with_flask()
    else:
        if check_gunicorn():
            start_with_gunicorn()
        else:
            print("Gunicorn không được cài đặt. Đang cài đặt Gunicorn...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gunicorn"])
                print("Đã cài đặt Gunicorn thành công!")
                start_with_gunicorn()
            except Exception as e:
                print(f"Lỗi khi cài đặt Gunicorn: {str(e)}")
                print("Đang chuyển sang Flask development server...")
                start_with_flask() 