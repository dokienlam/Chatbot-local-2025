const express = require('express');
const fs = require('fs');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const multer = require('multer');

const app = express();
const PORT = 3000;

// Kiểm tra và tạo thư mục uploads nếu không tồn tại
const uploadsDir = path.join(__dirname, '/uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir);
}

// Cấu hình Multer để lưu file
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadsDir); // Lưu file vào thư mục uploads
    },
    filename: (req, file, cb) => {
        const fileName = `${file.fieldname}_${Date.now()}.webm`;
        cb(null, fileName); // Tạo tên file duy nhất
    }
});
const upload = multer({ storage: storage });

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' }));
app.use(express.static('./')); // Phục vụ các file tĩnh từ thư mục hiện tại

// Route chính
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'Lam.html')); // Gửi file Lam.html
});

// Endpoint để upload audio
app.post('/upload', upload.single('audio'), (req, res) => {
    if (!req.file) {
        console.error('No file uploaded.');
        return res.status(400).json({ message: 'No file uploaded.' });
    }

    console.log('Audio file uploaded successfully:', req.file);
    res.json({ message: 'Audio file saved successfully!', file: req.file.filename });
});

// Endpoint để tải file âm thanh
app.get('/downloads/:filename', (req, res) => {
    const filePath = path.join(uploadsDir, req.params.filename);

    res.download(filePath, (err) => {
        if (err) {
            console.error("File download error:", err);
            res.status(500).send("Error downloading file.");
        } else {
            console.log(`File downloaded: ${filePath}`);
        }
    });
});

// Bắt đầu server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});