import os
from dotenv import load_dotenv

# --- LangChain Imports ---
# Core components for building chains
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory

# LLM integration with Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# Pre-built tool for web search
from langchain_community.tools import DuckDuckGoSearchRun

# In-memory history for the chat session
from langchain.memory import ChatMessageHistory


class HueChatbotLangChain:
    def __init__(self):
        """Initialize the chatbot using LangChain components."""
        load_dotenv()
        # API Key is automatically picked up by the LangChain integration
        # if GOOGLE_API_KEY is set in the environment.
        if not os.getenv("GOOGLE_API_KEY"):
            # LangChain uses GOOGLE_API_KEY, so let's ensure it's set
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY."
                )
            os.environ["GOOGLE_API_KEY"] = api_key

        # --- 1. Create the "lego blocks" of LangChain ---

        # a. LLM: The "brain" of the operation.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
        )

        # b. Retriever: The "information gatherer".
        self.search_tool = DuckDuckGoSearchRun(
            region="vn",
            language="vi",
            max_results=5,
        )

        # c. Prompt Template: The "instruction manual" for the LLM.
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Bạn là hướng dẫn viên du lịch chuyên nghiệp về văn hóa Huế. 
                    Hãy trả lời các câu hỏi một cách tự nhiên, thân thiện và chính xác dựa trên thông tin được cung cấp trong tài liệu tham khảo. 
                    Luôn giữ vai trò là một chuyên gia am hiểu về Huế, loại bỏ các thông tin thương mại, bán hàng giá cả nếu người dùng không yêu cầu.
                    Trả lời ngắn gọn, súc tích và tập trung vào văn hóa, lịch sử, ẩm thực và các điểm du lịch nổi tiếng của Huế,
                    dựa trên lịch sử trò chuyện và ngữ cảnh tìm kiếm dưới đây.\n\n"
                    "Ngữ cảnh tìm kiếm:\n{context}",
                    """,
                ),
                ("human", "{input}"),
            ]
        )

        # d. Output Parser: Cleans up the final response.
        self.output_parser = StrOutputParser()

        # --- 2. Chaining the components into a runnable chain ---

        base_chain = (
            RunnablePassthrough.assign(
                context=lambda x: self.search_tool.run(
                    f"{x['input']} văn hóa Huế",
                )
            )
            | self.prompt
            | self.llm
            | self.output_parser
        )

        # --- 3. Storing chat history ---
        self.chat_history_store = {}

        # Using RunnableWithMessageHistory to manage chat history
        # This allows us to keep track of the conversation context.
        # It will automatically handle the input and history messages.
        # The session_id is used to differentiate between different chat sessions.

        self.conversational_chain = RunnableWithMessageHistory(
            base_chain,
            lambda session_id: self.chat_history_store.get(
                session_id, ChatMessageHistory()
            ),
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def chat(self, question, session_id="default_session", streaming=False):
        """Single chat interaction handled by the LangChain chain."""
        print("\n" + "=" * 50)
        print(f"👤 Bạn: {question}")
        print("=" * 50)

        print("🤖 Hướng dẫn viên Huế (sử dụng LangChain) đang suy nghĩ...")

        # Create a new chat history if it doesn't exist
        if session_id not in self.chat_history_store:
            self.chat_history_store[session_id] = ChatMessageHistory()

        if streaming:
            for chunk in self.conversational_chain.stream(
                {"input": question}, config={"configurable": {"session_id": session_id}}
            ):
                print(chunk, end="", flush=True)

            return None

        answer = self.conversational_chain.invoke(
            {"input": question}, config={"configurable": {"session_id": session_id}}
        )

        print(f"\n🎭 Hướng dẫn viên Huế: {answer}")
        return answer

    def start(self, streaming=False):
        """Start interactive chatbot session."""
        print("🏛️ Chào mừng đến với Chatbot Văn hóa Huế! (Phiên bản LangChain)")
        print("💬 Hỏi tôi bất cứ điều gì về văn hóa, lịch sử, ẩm thực Huế...")
        print("⛔ Gõ 'quit' hoặc 'thoát' để kết thúc\n")

        session_id = "main_chat_session"

        while True:
            try:
                question = input("🎤 Hỏi về Huế: ").strip()
                if question.lower() in ["quit", "exit", "thoát", "q"]:
                    print("\n👋 Tạm biệt! Chúc bạn có chuyến du lịch Huế thú vị!")
                    break
                if not question:
                    continue
                self.chat(question, session_id=session_id, streaming=streaming)
            except KeyboardInterrupt:
                print("\n\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"❌ Lỗi: {e}")


# Initialize and start chatbot
if __name__ == "__main__":
    print("🚀 Khởi động chatbot...")
    chatbot = HueChatbotLangChain()
    chatbot.start(streaming=False)
