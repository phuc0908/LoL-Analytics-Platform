# 🚀 Hướng Dẫn Deploy Miễn Phí

## Tổng quan

| Component | Service | Free Tier |
|-----------|---------|-----------|
| Frontend (Next.js) | Vercel | ✅ Unlimited |
| Backend (FastAPI) | Railway | ✅ $5 credit/tháng |

---

## 📦 Bước 1: Chuẩn Bị

### 1.1 Copy models vào thư mục API

Chạy script tự động:
```powershell
.\prepare-deploy.ps1
```

Hoặc copy thủ công:
```bash
# Từ thư mục gốc project
cp -r ml/models api/models
cp -r ml/data api/data
```

Trên Windows PowerShell:
```powershell
Copy-Item -Recurse ml\models api\models
Copy-Item -Recurse ml\data api\data
```

### 1.2 Push code lên GitHub

```bash
git init
git add .
git commit -m "Initial commit - LoL Analytics Platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/lol-analytics.git
git push -u origin main
```

---

## 🔧 Bước 2: Deploy Backend (Railway)

### 2.1 Tạo tài khoản Railway
1. Truy cập https://railway.app
2. Đăng nhập bằng GitHub

### 2.2 Deploy API
1. Click **"New Project"**
2. Chọn **"Deploy from GitHub repo"**
3. Chọn repository của bạn
4. Chọn thư mục **`api`** làm root directory
5. Railway sẽ tự động detect Dockerfile và build

### 2.3 Cấu hình
1. Vào **Settings** → **Networking**
2. Click **"Generate Domain"** để có public URL
3. Copy URL (ví dụ: `https://lol-api-production.up.railway.app`)

---

## 🌐 Bước 3: Deploy Frontend (Vercel)

### 3.1 Tạo tài khoản Vercel
1. Truy cập https://vercel.com
2. Đăng nhập bằng GitHub

### 3.2 Deploy Web
1. Click **"Add New Project"**
2. Import repository từ GitHub
3. Chọn thư mục **`web`** làm Root Directory
4. Framework Preset: **Next.js**

### 3.3 Cấu hình Environment Variables
1. Vào **Settings** → **Environment Variables**
2. Thêm biến:
   - Name: `NEXT_PUBLIC_API_URL`
   - Value: `https://your-api.railway.app` (URL từ Railway)
3. Redeploy

---

## ✅ Bước 4: Kiểm Tra

1. Truy cập URL Vercel của bạn
2. Kiểm tra Dashboard hiển thị đúng data
3. Test AI Predict page

---

## 🔄 Cập Nhật

Mỗi khi push code mới lên GitHub:
- **Vercel**: Tự động redeploy
- **Railway**: Tự động redeploy

---

## 💡 Tips

### Giảm Cold Start cho Railway
Railway free tier sẽ sleep sau 30 phút không có request. Để tránh:
- Dùng UptimeRobot (free) ping API mỗi 5 phút
- URL: `https://uptimerobot.com`

### CORS Configuration
Đã được cấu hình trong `api/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 Ước Tính Chi Phí

| Service | Usage | Cost |
|---------|-------|------|
| Vercel | Frontend hosting | **$0** |
| Railway | API + ML (~500MB RAM) | **$0** (trong free tier) |
| **Total** | | **$0/tháng** |

---

## 🆘 Troubleshooting

### API không load được
- Kiểm tra Railway logs
- Đảm bảo models/ và data/ folder đã copy vào api/

### Frontend không kết nối API
- Kiểm tra NEXT_PUBLIC_API_URL đã set đúng
- Đảm bảo URL không có trailing slash

### Build failed
- Kiểm tra requirements.txt có đủ packages
- Đảm bảo Python version 3.11+

