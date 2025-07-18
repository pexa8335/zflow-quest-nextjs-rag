import os
import google.generativeai as genai
from dotenv import load_dotenv

print("--- BẮT ĐẦU BÀI TEST CƠ BẢN NHẤT ---")

# 1. Tải các biến môi trường từ file .env
load_dotenv()
print("Đã tải file .env")

# 2. Lấy API Key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ LỖI: Không tìm thấy GEMINI_API_KEY trong file .env!")
    print("--- KẾT THÚC TEST ---")
    exit()

print(f"Đã tìm thấy API Key (các ký tự cuối: ...{api_key[-4:]})")

try:
    # 3. Cấu hình API
    genai.configure(api_key=api_key)
    print("Đã cấu hình genai thành công.")

    # 4. Tạo model
    # Sử dụng model gemini-pro, một model rất ổn định.
    model = genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-8b",
        system_instruction="Bạn là hướng dẫn viên du lịch chuyên nghiệp về văn hóa Huế. Hãy trả lời các câu hỏi một cách tự nhiên, thân thiện và chính xác dựa trên thông tin được cung cấp trong tài liệu tham khảo. Luôn giữ vai trò là một chuyên gia am hiểu về Huế.",
    )
    print("Đã tạo model 'gemini-pro' thành công.")

    # 5. Gửi một yêu cầu cực kỳ đơn giản và an toàn
    prompt = "áo dài truyền thống của người Huế là gì?"
    print(f"Chuẩn bị gửi prompt: '{prompt}'")
    generation_config = genai.GenerationConfig(
        temperature=0.6,
        max_output_tokens=800,
        top_p=0.95,
        top_k=40,
    )
    response = model.generate_content(
        contents=prompt, generation_config=generation_config
    )
    print("ĐÃ GỌI API XONG. Đang kiểm tra phản hồi...")

    # 6. In ra TOÀN BỘ đối tượng phản hồi để phân tích
    # ĐÂY LÀ DÒNG QUAN TRỌNG NHẤT
    print("\n--- [DEBUG] PHẢN HỒI THÔ TỪ API ---")
    print(repr(response))
    print("-------------------------------------\n")

    # 7. Phân tích kết quả
    if response.parts:
        print(f"✅ THÀNH CÔNG! AI trả lời: {response.text.strip()}")
    else:
        # Cố gắng tìm lý do tại sao phản hồi rỗng
        reason = "Không rõ"
        if response.candidates:
            reason = response.candidates[0].finish_reason.name
        print(
            f"❌ THẤT BẠI: Phản hồi từ AI bị rỗng. Lý do kết thúc (Finish Reason): {reason}"
        )
        if response.prompt_feedback:
            print(
                f"   Lý do bị chặn (Block Reason): {response.prompt_feedback.block_reason.name}"
            )

except Exception as e:
    print(f"\n❌ LỖI NGHIÊM TRỌNG XUẤT HIỆN TRONG QUÁ TRÌNH TEST: {e}")

finally:
    print("\n--- KẾT THÚC BÀI TEST CƠ BẢN NHẤT ---")
