const express = require('express');
const fs = require('fs');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const multer = require('multer');

const app = express();
const PORT = 3001;

// Kiểm tra và tạo thư mục outputs nếu không tồn tại
const uploadsDir = path.join(__dirname, '../outputs');
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
app.use(express.static('../templates')); // Chỉ định đường dẫn tới thư mục public

// Route chính
app.get('/', (req, res) => {
    res.send('Welcome to the ChatBot API! Use /upload to upload audio and /downloads/:filename to download audio.');
});

// Endpoint để upload audio
app.post('/outputs', upload.single('audio'), (req, res) => {
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

//////
// document.getElementById('submitText').addEventListener('click', function() {
//     const userInput = document.getElementById('textInput').value;

//     // Gửi dữ liệu đến server bằng AJAX
//     fetch('/process_text', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({ text: userInput })
//     })
//     .then(response => response.json())
//     .then(data => {
//         console.log('Server response:', data);
//         // Hiển thị phản hồi từ server (nếu có)
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });
// });