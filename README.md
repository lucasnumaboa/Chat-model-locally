# ChatGPT with Local Document Context Search

![GitHub License](https://img.shields.io/github/license/yourusername/chatgpt-local-context)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)

## **Overview**

This project implements an interactive AI assistant using Flask and SocketIO, integrated with the Llama model (via `llama_cpp`). The system allows users to ask questions that are answered based on the content of local documents stored in a specified directory. The application processes these documents by dividing them into smaller chunks, which are then used as context to provide more accurate and relevant responses.

## **Key Features**

- **Interactive Web Interface:** Built with Flask and SocketIO for real-time communication between the client and server.
- **Integration with Llama Model:** Utilizes the Llama model for AI-driven response generation.
- **Document Processing:** Supports `.txt`, `.pdf`, and `.docx` files, extracting and dividing content into searchable chunks.
- **Intelligent Context Search:** Identifies relevant chunks based on user query keywords to enhance response relevance.
- **Dynamic Token Adjustment:** Adjusts the number of tokens used in responses based on query complexity and relevant context.
- **Detailed Logging:** Implements comprehensive logging for monitoring and debugging purposes.

## **How It Works**

1. **Reading and Processing Documents:**
   - The system reads files from a specified directory (`c:\temp` by default).
   - Supported formats include `.txt`, `.pdf`, and `.docx`.
   - Uses `CharacterTextSplitter` from LangChain to divide document content into chunks of 1000 characters with an overlap of 200 characters.

2. **Searching for Relevant Chunks:**
   - Upon receiving a user query, the system removes stopwords and identifies key keywords.
   - Searches through the document chunks to find those that contain at least 50% of the query keywords.
   - If relevant chunks are found, they are used as additional context for generating a more informed response.

3. **Generating Responses:**
   - Utilizes the Llama model to generate responses based on the provided context.
   - Dynamically adjusts the number of tokens (`max_response_tokens`) based on the complexity of the query and the amount of relevant context, ensuring responses are comprehensive yet concise.
   - Implements stop sequences (`stop=["Usuário:", "Assistente:"]`) to ensure responses are well-formatted and complete.

4. **Real-Time Communication:**
   - The web interface allows users to submit questions and receive answers in real-time.
   - Utilizes SocketIO for bidirectional communication between the client and server.

## **Installation**

### **Prerequisites:**

- **Python 3.8+**: Ensure Python is installed on your system.
- **Llama Model:** Download the desired Llama model (`gemma-2-2b-it-Q4_K_M.gguf`) and place it in the project directory or specify the correct path in the code.
- **Documents:** Place your `.txt`, `.pdf`, and `.docx` files in the designated directory (`c:\temp` by default).

### **Installation Steps:**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/chatgpt-local-context.git
   cd chatgpt-local-context
