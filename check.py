from ollama import Client

# Khởi tạo Client
client = Client()

# Sử dụng hàm generate hoặc chat (tùy thuộc vào mục đích của bạn)
response = client.generate(model='llama3.1', prompt='Hello, how are you?')

# In kết quả ra
print(response.text)
