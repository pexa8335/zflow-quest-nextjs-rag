import concurrent.futures
import time
from threading import Lock
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
from dotenv import load_dotenv


class HueChatbot:
    def __init__(self, api_key=None):
        """Initialize Hue Chatbot with API key and configuration"""
        # Load environment variables
        load_dotenv()

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set GEMINI_API_KEY in .env file or pass as parameter."
            )

        # Configure the Gemini API with your key
        genai.configure(api_key=self.api_key)

        # Initialize the Generative Model with a system instruction
        self.model = genai.GenerativeModel(
            model_name="models/gemini-2.5-flash",
            system_instruction="""
            Bạn là hướng dẫn viên du lịch chuyên nghiệp về văn hóa Huế. 
            Hãy trả lời các câu hỏi một cách tự nhiên, thân thiện và chính xác dựa trên thông tin được cung cấp trong tài liệu tham khảo. 
            Luôn giữ vai trò là một chuyên gia am hiểu về Huế, loại bỏ các thông tin thương mại, bán hàng giá cả nếu người dùng không yêu cầu.
            Trả lời ngắn gọn, súc tích và tập trung vào văn hóa, lịch sử, ẩm thực và các điểm du lịch nổi tiếng của Huế.""",
        )

        # Configuration from environment
        self.max_search_results = int(os.getenv("MAX_SEARCH_RESULTS", 2))
        self.max_history = int(os.getenv("MAX_HISTORY", 5))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", 5))

        # Chat history management
        self.history = []

        # Search cache for performance
        self.search_cache = {}
        self.cache_lock = Lock()

        # Requests session for better performance
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def _search_duckduckgo(self, query, max_results=None):
        """Search DuckDuckGo with caching for better performance"""
        if max_results is None:
            max_results = self.max_search_results

        with self.cache_lock:
            if query in self.search_cache:
                print("🚀 Dùng cache")
                return self.search_cache[query]

        try:
            with DDGS() as ddgs:
                search_query = f"{query} Huế, Việt Nam"
                results = list(
                    ddgs.text(search_query, max_results=max_results * 2, region="vn-vi")
                )
                urls = []
                for r in results:
                    url = r["href"]
                    if not any(
                        bad in url.lower()
                        for bad in [
                            ".pdf",
                            "facebook",
                            "youtube",
                            "instagram",
                            "shopee",
                            "tiki",
                        ]
                    ):
                        urls.append(url)
                    if len(urls) >= max_results:
                        break

                with self.cache_lock:
                    self.search_cache[query] = urls
                return urls
        except Exception as e:
            print(f"⚠️ Lỗi tìm kiếm: {e}")
            return []

    def _extract_article_fast(self, url, timeout=3):
        """Extract article content using requests + BeautifulSoup - FAST"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            content_selectors = [
                "article",
                ".article-content",
                ".content",
                ".post-content",
                ".entry-content",
                ".main-content",
                "main",
                ".article-body",
            ]
            text = ""
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    text = content.get_text(strip=True, separator=" ")
                    break
            if not text or len(text) < 100:
                text = soup.get_text(strip=True, separator=" ")
            if len(text) > 2000:
                text = text[:2000] + "..."
            return text
        except Exception as e:
            # print(f"⚠️ Lỗi {url[:30]}...: {str(e)[:20]}...")
            return ""

    def _process_urls_parallel(self, urls):
        """Process multiple URLs in parallel for better performance"""
        combined_text = ""
        successful_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {
                executor.submit(self._extract_article_fast, url, 3): url for url in urls
            }
            try:
                for future in concurrent.futures.as_completed(future_to_url, timeout=8):
                    try:
                        content = future.result(timeout=1)
                        if content and len(content.strip()) > 50:
                            combined_text += content + "\n\n"
                            successful_count += 1
                            print(
                                f"✅ Trích xuất thành công {successful_count}/{len(urls)}"
                            )
                            if len(combined_text) > 5000:
                                break
                    except Exception:
                        continue
            except concurrent.futures.TimeoutError:
                print(f"⏰ Timeout - xử lý được {successful_count}/{len(urls)}")
                for future in future_to_url:
                    future.cancel()
        return combined_text

    def _generate_answer(self, question, context, history=""):
        """Generate answer using Gemini AI with context and history"""
        clean_context = context.replace("\n", " ").strip()
        clean_history = history.replace("\n", " ").strip() if history else ""

        prompt = f"""
            Dưới đây là các tài liệu tham khảo được thu thập từ các trang web uy tín.
            Hãy đọc kỹ và chỉ sử dụng thông tin từ các tài liệu này để trả lời câu hỏi.

            {f"Lịch sử hội thoại trước đó: {clean_history}" if clean_history else ""}

            ---
            TÀI LIỆU THAM KHẢO:
            {clean_context}
            ---

            CÂU HỎI: {question}

            TRẢ LỜI:"""

        try:
            # Set up generation configuration
            generation_config = genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=800,
                top_p=0.95,
                top_k=40,
            )

            # Set safety settings
            safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
            }

            # Generate content using the model
            response = self.model.generate_content(
                contents=prompt,
                generation_config=generation_config,
                # safety_settings=safety_settings,
            )
            return response.text.strip()
        except Exception as e:
            return f"Lỗi khi gọi AI Gemini: {e}"

    def _rag_process(self, question, history=""):
        """Complete RAG process: search, extract, generate - TURBO MODE"""
        start_time = time.time()

        print("🔍 Tìm kiếm...")
        search_start = time.time()
        urls = self._search_duckduckgo(question)
        print(f"   ⚡ Search: {time.time() - search_start:.1f}s")

        if not urls:
            return "Không tìm thấy thông tin."

        print(f"📄 Xử lý {len(urls)} trang...")
        extract_start = time.time()
        combined_text = self._process_urls_parallel(urls)
        print(f"   ⚡ Extract: {time.time() - extract_start:.1f}s")

        if not combined_text.strip():
            return "Không trích xuất được thông tin."

        print(f"📝 Thu thập {len(combined_text)} ký tự")
        print("🤖 Tạo câu trả lời...")
        ai_start = time.time()
        answer = self._generate_answer(question, combined_text, history)
        print(f"   ⚡ AI: {time.time() - ai_start:.1f}s")

        total_time = time.time() - start_time
        print(f"🚀 TỔNG: {total_time:.1f}s")

        return answer

    def chat(self, question):
        """Single chat interaction with history context"""
        print("\n" + "=" * 50)
        print(f"👤 Bạn: {question}")
        print("=" * 50)

        # Create context from history
        history_context = ""
        if self.history:
            recent_history = self.history[-self.max_history :]
            history_context = " ".join(
                [f"Q: {h['q']} A: {h['a']}" for h in recent_history]
            )

        # Get answer
        answer = self._rag_process(question, history_context)

        # Save to history
        self.history.append({"q": question, "a": answer})

        print(f"\n🎭 Hướng dẫn viên Huế: {answer}")
        return answer

    def start(self):
        """Start interactive chatbot session"""
        print("🏛️ Chào mừng đến với Chatbot Văn hóa Huế!")
        print("💬 Hỏi tôi bất cứ điều gì về văn hóa, lịch sử, ẩm thực Huế...")
        print("⛔ Gõ 'quit' hoặc 'thoát' để kết thúc\n")

        while True:
            try:
                question = input("🎤 Hỏi về Huế: ").strip()

                if question.lower() in ["quit", "exit", "thoát", "q"]:
                    print("\n👋 Tạm biệt! Chúc bạn có chuyến du lịch Huế thú vị!")
                    break

                if not question:
                    continue

                self.chat(question)

            except KeyboardInterrupt:
                print("\n\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"❌ Lỗi: {e}")


# Initialize and start chatbot
if __name__ == "__main__":
    print("🚀 Khởi động chatbot...")
    chatbot = HueChatbot()

    # Start interactive session
    chatbot.start()
