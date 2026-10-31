# Đưa dự án lên GitHub và làm chung

Repo local đã được khởi tạo trên nhánh `main`; chưa có remote và chưa push.
Kiểm tra trước commit:

```powershell
cd D:\DL
git status
git add .
git diff --cached --stat
git commit -m "chore: initialize deep learning project scaffold"
```

Nếu Git yêu cầu danh tính, cấu hình tên/email của chính bạn; không dùng thông tin giả.
Tạo repository trên GitHub, chọn quyền riêng tư phù hợp, để repo mới rỗng (không thêm README/license/gitignore).
Sau đó dùng URL thực tế GitHub cung cấp:

```powershell
git remote add origin <URL_REPOSITORY_CUA_NHOM>
git push -u origin main
```

Thay placeholder bằng URL thật trước khi chạy. Không đăng token vào code hoặc chat nhóm.
Mời hai thành viên vào repo; thống nhất có cần để private theo quy định môn/dữ liệu.

Mỗi thành viên clone, tạo venv, cài `pip install -e .`, rồi làm trên nhánh riêng:

```powershell
git switch main
git pull --ff-only
git switch -c feat/data-pipeline
```

Mở issue theo mẫu; gán A/B/C thật, hạn và tiêu chí xong. Mở PR dẫn issue,
ghi thay đổi và cách kiểm tra. Một người khác review trước merge; tránh cùng sửa một file lớn đồng thời.
Mẫu đề tài ban đầu chưa có task chạy được: hoàn thành docs/PROPOSAL.md trước khi train.
