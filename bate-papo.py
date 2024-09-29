import os
import logging
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from llama_cpp import Llama
import docx
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
import tiktoken

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,  # Defina como DEBUG para logs mais detalhados durante o desenvolvimento
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Logs serão salvos em app.log
        logging.StreamHandler()  # Logs também serão exibidos no console
    ]
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Caminho para o modelo GGUF local
MODEL_PATH = "gemma-2-2b-it-Q4_K_M.gguf"  #pode trocar o modelo aqui

# Inicialize o codificador para Llama
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(enc.encode(text))

# Carrega o modelo usando llama_cpp com otimização para GPU (se disponível)
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=10,         # Ajuste conforme necessário
        n_batch=1024,            # Tamanho do batch
        n_ubatch=1024,           # Tamanho do micro-batch
        use_flash_attn=False,    # Desativa flash attention se suportado
        freq_base=10000.0,       # Base de frequência
        freq_scale=1,            # Escala de frequência
        top_p=0.95,
        temperature=0.1,
        repeat_penalty=1.1,
        max_tokens=600
    )
    logging.info("Modelo Llama carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar o modelo Llama: {e}")
    llm = None  # Define llm como None em caso de falha

# Stopwords comuns em português
STOPWORDS = {"a", "e", "o", "os", "as", "de", "do", "da", "em", "para", "por", "com", "é", "que", "um", "uma", "dos", "das", "na", "no"}

# Função para extrair texto de arquivos .docx
def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = [paragraph.text for paragraph in doc.paragraphs]
        logging.debug(f"Texto extraído do arquivo DOCX: {file_path}")
        return "\n".join(full_text)
    except Exception as e:
        logging.error(f"Erro ao ler o arquivo DOCX {file_path}: {e}")
        return ""

# Função para extrair texto de arquivos .pdf
def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = [reader.pages[page_num].extract_text() for page_num in range(len(reader.pages))]
        logging.debug(f"Texto extraído do arquivo PDF: {file_path}")
        return "\n".join(text)
    except Exception as e:
        logging.error(f"Erro ao ler o arquivo PDF {file_path}: {e}")
        return ""

# Função para ler e dividir o conteúdo dos arquivos de um diretório
def read_and_split_files_from_directory(directory_path):
    file_contents = []
    chunk_size = 1000
    chunk_overlap = 200

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    logging.info(f"Iniciando a leitura e divisão dos arquivos no diretório: {directory_path}")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                if filename.lower().endswith('.txt'):
                    with open(file_path, "r", encoding='utf-8', errors='ignore') as file:
                        raw_text = file.read()
                    logging.debug(f"Arquivo TXT lido: {filename}")
                elif filename.lower().endswith('.pdf'):
                    raw_text = read_pdf(file_path)
                elif filename.lower().endswith('.docx'):
                    raw_text = read_docx(file_path)
                else:
                    logging.warning(f"Formato de arquivo não suportado: {filename}")
                    continue

                if not raw_text.strip():
                    logging.warning(f"Arquivo vazio ou erro na extração de texto: {filename}")
                    continue

                chunks = text_splitter.split_text(raw_text)
                logging.info(f"{len(chunks)} chunks gerados a partir do arquivo: {filename}")

                for chunk in chunks:
                    if len(chunk) > chunk_size:
                        refined_chunks = [chunk[i:i + chunk_size] for i in range(0, len(chunk), chunk_size)]
                        file_contents.extend(refined_chunks)
                        logging.debug(f"Chunk refinado de tamanho maior que {chunk_size} caracteres.")
                    else:
                        file_contents.append(chunk)
            except Exception as e:
                logging.error(f"Não foi possível ler o arquivo {file_path}: {e}")
    logging.info(f"Total de {len(file_contents)} chunks processados.")
    return file_contents

# Função para encontrar chunks relevantes com base na pergunta
def find_relevant_chunks(question, chunks, threshold=0.5):
    relevant_chunks = []
    question_words = set(question.lower().split()) - STOPWORDS
    num_question_words = len(question_words)

    logging.debug(f"Palavras da pergunta após remover stopwords: {question_words}")

    if num_question_words == 0:
        logging.warning("Nenhuma palavra relevante na pergunta após remover stopwords.")
        return relevant_chunks

    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        chunk_words = set(chunk_lower.split()) - STOPWORDS
        common_words = question_words.intersection(chunk_words)
        ratio = len(common_words) / num_question_words
        if ratio >= threshold:
            relevant_chunks.append(chunk)
            logging.debug(f"Chunk {idx} relevante (Taxa de correspondência: {ratio:.2f})")
        else:
            logging.debug(f"Chunk {idx} não é relevante (Taxa de correspondência: {ratio:.2f})")

    logging.info(f"{len(relevant_chunks)} chunks relevantes encontrados com base na pergunta.")
    return relevant_chunks

# Função para gerar a resposta e enviar em tempo real conforme os tokens forem gerados
def stream_response(input_text, system_message, relevant_chunks, temperature=0.2, max_response_tokens=256):
    if not llm:
        logging.error("Modelo Llama não está carregado. Não é possível gerar resposta.")
        yield "Erro interno: Modelo de linguagem não está disponível."
        return

    if relevant_chunks:
        logging.info("Chunks relevantes encontrados. Utilizando contexto adicional.")
        formatted_context = ""
        context_tokens = 0
        context_length = 512 - max_response_tokens  # Ajuste conforme necessário

        for chunk in relevant_chunks:
            chunk_length = count_tokens(chunk)
            if context_tokens + chunk_length > context_length:
                break
            formatted_context += f"{chunk}\n\n"
            context_tokens += chunk_length

        context = system_message + "\n\n" + formatted_context
    else:
        logging.info("Nenhum chunk relevante encontrado. Resposta baseada apenas na pergunta do usuário.")
        context = system_message

    prompt = context + "\n\nUsuário: " + input_text + "\nAssistente:"
    logging.info(f"Prompt enviado para o modelo: {prompt}")

    try:
        response = llm(
            prompt=prompt,
            max_tokens=max_response_tokens,
            temperature=temperature,
            stop=["Usuário:", "Assistente:"],  # Parâmetros de parada
            stream=True
        )
        for token in response:
            token_text = token['choices'][0]['text']
            logging.debug(f"Token recebido do modelo: {token_text}")
            yield token_text
    except Exception as e:
        logging.error(f"Erro ao processar a solicitação: {e}")
        yield f"Erro ao processar a solicitação: {e}\n"

# Rota principal para a interface
@app.route('/')
def index():
    return render_template('index.html')

# Evento WebSocket para lidar com o envio de mensagens do usuário
@socketio.on('send_message')
def handle_send_message(message):
    logging.info(f"Mensagem recebida do usuário: {message}")

    system_message = (
        'Você é um assistente de IA útil e amigável. Responda sempre apenas o que foi perguntado em português.'
    )

    directory_path = r"c:\temp"  # Atualize o caminho conforme necessário
    local_content_chunks = read_and_split_files_from_directory(directory_path)

    relevant_chunks = find_relevant_chunks(message, local_content_chunks, threshold=0.5)

    base_max_response_tokens = 256
    max_response_tokens = base_max_response_tokens
    if relevant_chunks:
        # Estimar tokens adicionais com base nos chunks relevantes
        additional_tokens = sum(count_tokens(chunk) for chunk in relevant_chunks)
        max_response_tokens += additional_tokens
        # Cap no número máximo de tokens do modelo (600)
        if max_response_tokens > 600:
            max_response_tokens = 600
        logging.info(f"Ajustando max_response_tokens para {max_response_tokens} com base nos chunks relevantes.")
    else:
        logging.info(f"Usando max_response_tokens padrão: {max_response_tokens}")

    temperature = 0.1

    # Emitir 'response_start' para resetar o 'currentBotMessage' no cliente
    emit('response_start', {})

    # Gerar e emitir a resposta em streaming
    for token in stream_response(message, system_message, relevant_chunks, temperature, max_response_tokens):
        emit('response', {'data': token})

    logging.info("Resposta enviada ao usuário.")

# Desabilitar o modo debug para produção
if __name__ == '__main__':
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)
