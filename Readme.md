# FAQ Chatbot with FastAPI and LangChain

An intelligent chatbot designed to answer frequently asked questions (FAQs) related to e-commerce using LLM, FastAPI, LangChain, HuggingFace, ChromaDB, Embedding models, Openrouter, Telegram, Render, Railway.


## 📌 Demo

📽️ [YouTube](https://youtu.be/lyczfYr9RmE)


[![](https://markdown-videos.deta.dev/youtube/lyczfYr9RmE)](https://youtu.be/lyczfYr9RmE)

## 📌 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Features

- Answers e-commerce FAQs in English and Spanish.
- Integrates with Telegram for real-time interaction.
- RESTful API built with FastAPI.
- Uses LangChain for natural language processing.
- HuggingFace for embedding model.
- Openrouter for LLM model.
- Stores and retrieves FAQ data with ChromaDB.
- Containerized with Docker for easy deployment.
- Render or Railway for deploy the bot.

---

## 🔧 Technologies Used

- [**FastAPI**](https://fastapi.tiangolo.com/) - High-performance web framework for building APIs.
- [**LangChain**](https://langchain.com/) - NLP library for processing text and integrating LLMs.
- [**HuggingFace**](https://huggingface.co/) - HuggingFace for deploy the API
- [**ChromaDB**](https://www.chromadb.com/) - Vector database for fast retrieval.
- [**Docker**](https://www.docker.com/) - Containerization platform.
- [**Telegram Bot API**](https://core.telegram.org/bots/api) - API for creating interactive Telegram bots.
- [**OpenRouter AI**](https://openrouter.ai/) - LLM provider for generating responses.
- [**Render**](https://render.com/) - Deploy the bot.
- [**Railway**](https://railway.com/) - Deploy the bot.
---

## 🛠 Installation


1. **Create the bot**

   ```bash
   Go to telegram -> BotFather -> Create new bot
   You need to save the token as enviroment variable
   ```


2. **Create an account on OpenRouter**

   ```bash
   Search the model you want to use -> Get your link 
   ```


3. **Clone the repository:**

   ```bash
   git clone https://github.com/WesleyG31/faq-chatbot-fastapi-langchain.git
   cd faq-chatbot-fastapi-langchain
   ```


3. **Upload the API to HuggingFace**

   **HuggingFace** Visit `https://huggingface.co/` then upload these documents:

    ```
    faq-chatbot-fastapi-langchain/
    ├── api/
    │   ├── Dockerfile
    │   ├── fast_api_online.py
    │   ├── Ecommerce_FAQ_Chatbot_dataset.json
    │   └── requirements.txt

    Config Openrouter enviroment variable
    ```


4. **Upload the bot to Render or Railway**

   - [**Render**](https://render.com/) - Deploy the bot.
   - [**Railway**](https://railway.com/) - Deploy the bot.
    
    then upload these documents:

    ```
    faq-chatbot-fastapi-langchain/
    ├── telegram_bot/
    │   ├── Telegram_bot.py
    │   ├── Procfile
    │   └── requirements.txt

    Config API link and telegram token as enviroment variables
    ```


---

## 📌 Usage

- **Telegram Bot:** Search for your bot in Telegram and start asking questions.

---


### If you want to use Docker

- **Docker** installed on your system.
- **Docker Compose** for managing multi-container applications.

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/WesleyG31/faq-chatbot-fastapi-langchain.git
   cd faq-chatbot-fastapi-langchain
   ```

2. **Copy the environment variables file and set up credentials:**

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file to add your **Telegram Bot Token** and **OpenRouter API Key**.

3. **Build and run the services with Docker Compose:**

   ```bash
   docker-compose up --build
   ```

   This will start all the services, including the **FastAPI server**, **Telegram bot**, and **ChromaDB**.

---

## 📌 Usage

- **FastAPI API:** Visit `http://localhost:8000/docs` to test API endpoints.
- **Telegram Bot:** Search for your bot in Telegram and start asking questions.

---

## 📂 Project Structure

```
faq-chatbot-fastapi-langchain/
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── fast_api.py
├── bot/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── telegram_bot.py
├── docker-compose.yml
└── README.md
```

- `api/`: Contains FastAPI-related files.
- `bot/`: Includes the Telegram bot implementation.
- `docker-compose.yml`: Defines and runs all services together.
- `README.md`: This documentation file.

---

## 🤝 Contributing

Contributions are welcome! Follow these steps:

1. **Fork this repository.**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/new-feature
   ```
3. **Make changes and commit:**
   ```bash
   git commit -am "Add new feature"
   ```
4. **Push your branch:**
   ```bash
   git push origin feature/new-feature
   ```
5. **Open a Pull Request.**

Make sure to follow coding standards and test your code before submission.

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

