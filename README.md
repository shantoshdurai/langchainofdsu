# University Chatbot with LangChain and Ollama

A powerful AI chatbot that can answer questions about your university using your own documents. Built with LangChain and powered by Ollama's LLM (llama3.1:8b).

## Features

- **Document Support**: Upload PDF, TXT, DOCX, and MD files
- **Conversation Memory**: Remembers previous messages in the conversation
- **Source Citation**: Shows which documents were used to generate answers
- **Easy to Extend**: Add more documents anytime by placing them in the `data` directory

## Prerequisites

1. Python 3.8 or higher
2. [Ollama](https://ollama.ai/) installed and running
3. The llama3.1:8b model downloaded (run `ollama pull llama3.1:8b`)

## Installation

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # OR
   source venv/bin/activate  # On macOS/Linux
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your documents (PDF, TXT, DOCX, MD) in the `data` directory
2. Start the chatbot:
   ```bash
   python chatbot.py
   ```
3. Start chatting! Type your questions and press Enter
4. Type `exit` to quit or `clear` to clear the screen

## Example Documents

You can add various university-related documents such as:
- Course catalogs
- Academic calendars
- University handbooks
- Department guides
- Event schedules
- Policy documents

## Troubleshooting

- **Ollama not running**: Make sure to run `ollama serve` in a terminal
- **Model not found**: Run `ollama pull llama3.1:8b` to download the model
- **Document not loading**: Ensure your files are in a supported format and in the `data` directory

## Customization

You can customize the chatbot's behavior by modifying the `system_message` in `chatbot.py`. This controls how the assistant responds to questions.

## License

This project is open source and available under the MIT License.
